"""The virtual leader's web server: serves the headset page, takes headset input, streams frames and status."""

import asyncio
import json
import ssl
import time

import numpy as np
import pytest

aiohttp = pytest.importorskip("aiohttp")
cv2 = pytest.importorskip("cv2")

from lerobot.teleoperators.so101_leader_vr.server import (  # noqa: E402
    VRServer,
    ensure_certificate,
    load_or_create_token,
)

TOKEN = "abc123"
SAMPLE = {
    "type": "xr",
    "pos": [0.1, 1.2, -0.3],
    "quat": [0, 0, 0, 1],
    "head": [0, 0.38268343, 0, 0.92387953],
    "trigger": 0.25,
    "grip": True,
    "a": False,
    "b": True,
}


@pytest.fixture
def client_dir(tmp_path):
    (tmp_path / "index.html").write_text("<html>vr page</html>")
    (tmp_path / "app.js").write_text("// app")
    return tmp_path


@pytest.fixture
def server(client_dir):
    server = VRServer(client_dir, port=0, https=False, token=TOKEN, hand="right", stream_fps=50)
    server.start()
    yield server
    server.stop()


def run(coroutine):
    return asyncio.run(coroutine)


async def fetch(url, **kwargs):
    async with aiohttp.ClientSession() as session, session.get(url, **kwargs) as response:
        return response.status, await response.text()


def wait_for(condition, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_serves_the_headset_page(server):
    assert run(fetch(f"http://127.0.0.1:{server.port}/")) == (200, "<html>vr page</html>")
    assert run(fetch(f"http://127.0.0.1:{server.port}/app.js")) == (200, "// app")


def test_rejects_websockets_without_the_token(server):
    async def connect(query):
        async with aiohttp.ClientSession() as session:
            with pytest.raises(aiohttp.WSServerHandshakeError) as error:
                await session.ws_connect(f"http://127.0.0.1:{server.port}/ws{query}")
            return error.value.status

    assert run(connect("")) == 403
    assert run(connect("?k=wrong")) == 403


def test_headset_samples_become_the_latest_input(server):
    async def send_sample():
        async with (
            aiohttp.ClientSession() as session,
            session.ws_connect(f"http://127.0.0.1:{server.port}/ws?k={TOKEN}") as ws,
        ):
            hello = await ws.receive_json()
            await ws.send_str(json.dumps(SAMPLE))
            assert wait_for(lambda: server.latest() is not None)
            return hello

    before = time.monotonic()
    hello = run(send_sample())
    latest = server.latest()

    assert hello == {"type": "hello", "hand": "right"}
    assert latest.position == pytest.approx(SAMPLE["pos"])
    assert latest.orientation == pytest.approx(SAMPLE["quat"])
    assert latest.head_orientation == pytest.approx(SAMPLE["head"])
    assert (latest.trigger, latest.grip, latest.precision, latest.recenter) == (0.25, True, False, True)
    assert latest.received_at >= before


def test_malformed_samples_are_ignored(server):
    async def send_garbage():
        async with (
            aiohttp.ClientSession() as session,
            session.ws_connect(f"http://127.0.0.1:{server.port}/ws?k={TOKEN}") as ws,
        ):
            await ws.receive_json()
            await ws.send_str("not json")
            await ws.send_str(json.dumps({"type": "xr", "pos": [1, 2]}))
            await ws.send_str(json.dumps({**SAMPLE, "trigger": 0.5}))
            assert wait_for(lambda: server.latest() is not None)

    run(send_garbage())

    assert server.latest().trigger == 0.5


def test_streams_camera_frames_as_named_jpegs(server):
    red = np.zeros((48, 64, 3), dtype=np.uint8)
    red[..., 0] = 255  # lerobot frames are RGB

    async def receive_frame():
        async with (
            aiohttp.ClientSession() as session,
            session.ws_connect(f"http://127.0.0.1:{server.port}/ws?k={TOKEN}") as ws,
        ):
            await ws.receive_json()
            server.publish_frames({"front": red})
            while True:
                message = await asyncio.wait_for(ws.receive(), timeout=2)
                if message.type == aiohttp.WSMsgType.BINARY:
                    return message.data

    data = run(receive_frame())
    name_length = data[0]
    image = cv2.imdecode(np.frombuffer(data[1 + name_length :], dtype=np.uint8), cv2.IMREAD_COLOR)  # BGR

    assert data[1 : 1 + name_length].decode() == "front"
    assert image.shape == (48, 64, 3)
    assert image[24, 32, 2] > 200 and image[24, 32, 0] < 50


def test_sends_status_updates(server):
    async def receive_status():
        async with (
            aiohttp.ClientSession() as session,
            session.ws_connect(f"http://127.0.0.1:{server.port}/ws?k={TOKEN}") as ws,
        ):
            await ws.receive_json()
            server.publish_status({"engaged": True, "stale": False, "limited": False})
            while True:
                message = await asyncio.wait_for(ws.receive_json(), timeout=2)
                if message["type"] == "status":
                    return message

    assert run(receive_status()) == {"type": "status", "engaged": True, "stale": False, "limited": False}


def test_stop_frees_the_port(client_dir):
    server = VRServer(client_dir, port=0, https=False, token=TOKEN, hand="right", stream_fps=20)
    server.start()
    port = server.port
    server.stop()

    again = VRServer(client_dir, port=port, https=False, token=TOKEN, hand="right", stream_fps=20)
    again.start()
    again.stop()


def test_start_fails_clearly_when_the_port_is_taken(server, client_dir):
    clash = VRServer(client_dir, port=server.port, https=False, token=TOKEN, hand="right", stream_fps=20)

    with pytest.raises(OSError):
        clash.start()


def test_https_serves_with_a_self_signed_certificate(client_dir, tmp_path):
    cert = ensure_certificate(tmp_path / "certs")
    server = VRServer(
        client_dir, port=0, https=True, token=TOKEN, hand="right", stream_fps=20, certificate=cert
    )
    server.start()
    try:
        insecure = ssl.create_default_context()
        insecure.check_hostname = False
        insecure.verify_mode = ssl.CERT_NONE
        assert run(fetch(f"https://127.0.0.1:{server.port}/", ssl=insecure))[0] == 200
    finally:
        server.stop()


def test_certificate_and_token_are_created_once_and_reused(tmp_path):
    first = ensure_certificate(tmp_path)
    assert ensure_certificate(tmp_path) == first
    assert all(path.is_file() for path in first)

    token = load_or_create_token(tmp_path)
    assert len(token) >= 6
    assert load_or_create_token(tmp_path) == token


def test_status_stream_survives_a_status_it_cannot_send(server):
    async def receive_status():
        async with (
            aiohttp.ClientSession() as session,
            session.ws_connect(f"http://127.0.0.1:{server.port}/ws?k={TOKEN}") as ws,
        ):
            await ws.receive_json()
            server.publish_status({"engaged": object()})
            await asyncio.sleep(0.3)
            server.publish_status({"engaged": True, "stale": False, "limited": False})
            while True:
                message = await asyncio.wait_for(ws.receive_json(), timeout=2)
                if message["type"] == "status":
                    return message

    assert run(receive_status())["engaged"] is True


def test_frame_stream_survives_a_frame_it_cannot_encode(server):
    good = np.zeros((48, 64, 3), dtype=np.uint8)

    async def receive_frame():
        async with (
            aiohttp.ClientSession() as session,
            session.ws_connect(f"http://127.0.0.1:{server.port}/ws?k={TOKEN}") as ws,
        ):
            await ws.receive_json()
            server.publish_frames({"bad": np.zeros((8, 8, 3), dtype=np.float64)})
            await asyncio.sleep(0.2)
            server.publish_frames({"front": good})
            while True:
                message = await asyncio.wait_for(ws.receive(), timeout=2)
                if message.type == aiohttp.WSMsgType.BINARY:
                    return message.data

    data = run(receive_frame())

    assert data[1 : 1 + data[0]] == b"front"
