#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Web server of the virtual leader, in its own thread: serves the headset page (`vr_client/` in the workspace
repo), takes the operator's controller and headset poses over a WebSocket, and sends back camera frames and
status. The protocol is described in `vr_client/README.md`.
"""

import asyncio
import datetime
import ipaddress
import json
import logging
import secrets
import socket
import ssl
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from aiohttp import WSMsgType, web

from .controller import XRInput

logger = logging.getLogger(__name__)

JPEG_QUALITY = 70
STATUS_HZ = 10
FAILURE_LOG_PERIOD_S = 5.0
# A headset that takes longer than this to accept a message has stopped reading: it is dropped.
SEND_TIMEOUT_S = 0.5


def lan_ip() -> str:
    """This computer's address on the local network (connecting a UDP socket sends nothing)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def load_or_create_token(directory: Path) -> str:
    """The secret the page must present (`?k=`), kept across runs so a bookmarked URL keeps working."""
    path = Path(directory) / "token.txt"
    if path.is_file():
        return path.read_text().strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(3)
    path.write_text(token)
    return token


def ensure_certificate(directory: Path) -> tuple[Path, Path]:
    """A self-signed certificate (WebXR needs HTTPS off localhost), created on first use. Returns (cert, key)."""
    directory = Path(directory)
    cert_path, key_path = directory / "cert.pem", directory / "key.pem"
    if cert_path.is_file() and key_path.is_file():
        return cert_path, key_path

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "so101-leader-vr")])
    addresses = sorted({"127.0.0.1", lan_ip()})
    alt_names = [x509.DNSName("localhost"), *(x509.IPAddress(ipaddress.ip_address(a)) for a in addresses)]
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .sign(key, hashes.SHA256())
    )
    directory.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    key_path.chmod(0o600)
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


def parse_sample(message: dict, received_at: float) -> XRInput:
    """An `XRInput` from one page message. Raises ValueError, KeyError or TypeError if it's malformed."""

    def vector(key: str, length: int) -> np.ndarray:
        value = np.array(message[key], dtype=float)
        if value.shape != (length,) or not np.all(np.isfinite(value)):
            raise ValueError(f"bad {key!r}: {message[key]!r}")
        return value

    return XRInput(
        position=vector("pos", 3),
        orientation=vector("quat", 4),
        head_orientation=vector("head", 4),
        trigger=float(message["trigger"]),
        grip=bool(message["grip"]),
        precision=bool(message["a"]),
        recenter=bool(message["b"]),
        received_at=received_at,
    )


def encode_frame(name: str, frame: np.ndarray) -> bytes | None:
    """[name length][name][JPEG] for an RGB frame, or None if it can't be encoded."""
    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        return None
    ok, jpeg = cv2.imencode(
        ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not ok:
        return None
    label = name.encode()[:255]
    return bytes([len(label)]) + label + jpeg.tobytes()


@web.middleware
async def _no_cache(request, handler):
    # The headset browser would otherwise keep serving an old page after an update.
    response = await handler(request)
    response.headers["Cache-Control"] = "no-store"
    return response


class VRServer:
    def __init__(
        self,
        client_dir: Path,
        port: int,
        https: bool,
        token: str,
        hand: str,
        stream_fps: int,
        certificate: tuple[Path, Path] | None = None,
    ):
        self.client_dir = Path(client_dir)
        self.port = port  # the real port once started (0 picks a free one)
        self.https = https
        self.token = token
        self.hand = hand
        self.stream_fps = stream_fps
        self.certificate = certificate

        self._lock = threading.Lock()
        self._latest: XRInput | None = None
        self._frames: dict[str, np.ndarray] = {}
        self._frames_new = False
        self._status: dict | None = None
        self._clients: set[web.WebSocketResponse] = set()
        self._transports: dict[web.WebSocketResponse, asyncio.BaseTransport] = {}

        self._last_failure_log: dict[str, float] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None

    @property
    def url(self) -> str:
        host = lan_ip() if self.https else "localhost"
        return f"{'https' if self.https else 'http'}://{host}:{self.port}/?k={self.token}"

    def start(self) -> None:
        """Starts listening, or raises (e.g. OSError if the port is taken)."""
        self._thread = threading.Thread(target=self._run, name="so101_leader_vr server", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._error is not None:
            self._thread.join()
            raise self._error

    def stop(self) -> None:
        if self._loop is not None and self._thread is not None and self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    def latest(self) -> XRInput | None:
        """The most recent headset sample, or None if none arrived yet."""
        with self._lock:
            return self._latest

    def publish_frames(self, frames: dict[str, np.ndarray]) -> None:
        """Latest camera frames (RGB). They're encoded and sent at `stream_fps`, older ones are dropped."""
        with self._lock:
            self._frames = dict(frames)
            self._frames_new = True

    def publish_status(self, status: dict) -> None:
        with self._lock:
            self._status = dict(status)

    def _run(self) -> None:
        self._loop = loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        app = web.Application(middlewares=[_no_cache])
        app.router.add_get("/ws", self._websocket)
        app.router.add_get("/", self._index)
        app.router.add_static("/", self.client_dir, show_index=False)
        runner = web.AppRunner(app, access_log=None)
        try:
            loop.run_until_complete(runner.setup())
            ssl_context = None
            if self.https:
                ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                ssl_context.load_cert_chain(*self.certificate)
            # Plain HTTP only on this computer: that's for a headset plugged in by USB (`adb reverse`).
            host = "0.0.0.0" if self.https else "127.0.0.1"
            loop.run_until_complete(web.TCPSite(runner, host, self.port, ssl_context=ssl_context).start())
            self.port = runner.addresses[0][1]
        except BaseException as e:
            self._error = e
            loop.run_until_complete(runner.cleanup())
            loop.close()
            self._ready.set()
            return

        tasks = [loop.create_task(self._stream_frames()), loop.create_task(self._stream_status())]
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            try:
                for task in tasks:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                # Abort rather than close: closing waits on headsets that may never read again.
                for ws in list(self._clients):
                    self._drop(ws)
                loop.run_until_complete(runner.cleanup())
            finally:
                loop.close()

    async def _index(self, request: web.Request) -> web.StreamResponse:
        return web.FileResponse(self.client_dir / "index.html")

    async def _websocket(self, request: web.Request) -> web.StreamResponse:
        if not secrets.compare_digest(request.query.get("k", ""), self.token):
            raise web.HTTPForbidden(text="Wrong or missing ?k= token: open the URL that lerobot printed.")
        ws = web.WebSocketResponse(heartbeat=5.0)
        await ws.prepare(request)
        self._clients.add(ws)
        self._transports[ws] = request.transport
        logger.info(f"Headset connected from {request.remote}.")
        await ws.send_json({"type": "hello", "hand": self.hand})
        try:
            async for message in ws:
                if message.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(message.data)
                    if data.get("type") != "xr":
                        continue
                    sample = parse_sample(data, time.monotonic())
                except (ValueError, KeyError, TypeError, AttributeError) as e:
                    logger.debug(f"Ignoring a malformed headset message: {e}")
                    continue
                with self._lock:
                    self._latest = sample
        finally:
            self._clients.discard(ws)
            self._transports.pop(ws, None)
            logger.info("Headset disconnected.")
        return ws

    async def _send_to_all(self, send) -> None:
        """Sends to every headset at once, so one that stopped reading can't hold up the others."""
        await asyncio.gather(*(self._send_one(ws, send) for ws in list(self._clients)))

    async def _send_one(self, ws: web.WebSocketResponse, send) -> None:
        try:
            await asyncio.wait_for(send(ws), SEND_TIMEOUT_S)
        except (asyncio.TimeoutError, ConnectionError, RuntimeError):
            logger.warning(
                "A headset stopped reading; dropping its connection (the page reconnects by itself)."
            )
            self._drop(ws)

    def _drop(self, ws: web.WebSocketResponse) -> None:
        self._clients.discard(ws)
        transport = self._transports.pop(ws, None)
        if transport is not None:
            transport.abort()

    def _log_failure(self, what: str) -> None:
        """Logs the exception being handled, at most every few seconds per kind: the streams keep going."""
        now = time.monotonic()
        if now - self._last_failure_log.get(what, -FAILURE_LOG_PERIOD_S) >= FAILURE_LOG_PERIOD_S:
            self._last_failure_log[what] = now
            logger.exception(f"Couldn't {what}; carrying on.")

    async def _stream_frames(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(1 / self.stream_fps)
            with self._lock:
                frames = self._frames if self._frames_new else None
                self._frames_new = False
            if not frames or not self._clients:
                continue
            for name, frame in frames.items():
                try:
                    data = await loop.run_in_executor(None, encode_frame, name, frame)
                    if data is not None:
                        await self._send_to_all(lambda ws, data=data: ws.send_bytes(data))
                except Exception:
                    self._log_failure(f"send the {name!r} camera frame")

    async def _stream_status(self) -> None:
        while True:
            await asyncio.sleep(1 / STATUS_HZ)
            with self._lock:
                status = self._status
            if status is None or not self._clients:
                continue
            try:
                message = {"type": "status", **status}
                await self._send_to_all(lambda ws, message=message: ws.send_json(message))
            except Exception:
                self._log_failure("send the status")
