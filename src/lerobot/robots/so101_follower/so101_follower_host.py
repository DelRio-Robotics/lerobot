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
Exposes a local SO-101 follower arm over the network, so a leader arm on another computer can drive it
through `--robot.type=so101_follower_client`. Same protocol as LeKiwi's host: ZMQ PUSH/PULL, JSON messages,
JPEG frames as base64. ZMQ has no encryption or authentication, so only expose it over a VPN like Tailscale.

Example:

```shell
python -m lerobot.robots.so101_follower.so101_follower_host \
    --robot.port=/dev/cu.usbmodem5B790188621 \
    --robot.id=my_awesome_follower_arm \
    --robot.max_relative_target=10 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"
```
"""

import base64
import json
import logging
import threading
import time
from dataclasses import asdict
from pprint import pformat

import cv2
import draccus
import zmq

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.utils.utils import init_logging

from ..robot import Robot
from .config_so101_follower import SO101FollowerHostConfig
from .so101_follower import SO101Follower


class SO101FollowerHost:
    def __init__(self, config: SO101FollowerHostConfig):
        self.config = config
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://{config.bind_ip}:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://{config.bind_ip}:{config.port_zmq_observations}")

    def encode_observation(self, observation: dict, camera_keys) -> str:
        msg = {}
        for key, value in observation.items():
            if key in camera_keys:
                ok, buffer = cv2.imencode(
                    ".jpg", value, [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
                )
                msg[key] = base64.b64encode(buffer).decode("utf-8") if ok else ""
            else:
                msg[key] = float(value)
        return json.dumps(msg)

    def disconnect(self):
        self.zmq_observation_socket.close(linger=0)
        self.zmq_cmd_socket.close(linger=0)
        self.zmq_context.term()


def _control_loop(
    robot: Robot, host: SO101FollowerHost, joints: dict, lock: threading.Lock, stop: threading.Event
):
    """Applies each command as soon as it arrives and keeps `joints` up to date. The only thread using the bus."""
    cfg = host.config
    last_cmd_time = None
    watchdog_active = False
    last_read = 0.0
    n_cmds, last_stats = 0, time.perf_counter()

    while not stop.is_set():
        if host.zmq_cmd_socket.poll(5, zmq.POLLIN):
            try:
                robot.send_action(json.loads(host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)))
                n_cmds += 1
                if last_cmd_time is None or watchdog_active:
                    logging.info("Receiving commands")
                last_cmd_time = time.perf_counter()
                watchdog_active = False
            except zmq.Again:
                pass
            except Exception as e:
                logging.error("Failed to apply command: %s", e)

        now = time.perf_counter()
        if (
            last_cmd_time is not None
            and not watchdog_active
            and now - last_cmd_time > cfg.watchdog_timeout_ms / 1000
        ):
            # The servos keep their last goal position, so the arm just holds still.
            logging.warning(f"No command for {cfg.watchdog_timeout_ms} ms. Holding position.")
            watchdog_active = True

        if now - last_read >= 1 / 60:
            try:
                present = robot.bus.sync_read("Present_Position")
                with lock:
                    joints.update({f"{motor}.pos": val for motor, val in present.items()})
                last_read = now
            except Exception as e:
                logging.error("Failed to read joints: %s", e)

        if now - last_stats >= 10:
            if n_cmds:
                logging.info(f"Applying {n_cmds / (now - last_stats):.0f} commands/s")
            n_cmds, last_stats = 0, now


def run_host(robot: Robot, host: SO101FollowerHost) -> None:
    """
    Runs motor control in a background thread, so commands are applied as soon as they arrive instead of
    waiting on the cameras. This thread publishes the latest joints with fresh camera frames, at most
    `max_loop_freq_hz` times a second.
    """
    cfg = host.config
    cameras = dict(getattr(robot, "cameras", {}))
    joints: dict = {}
    lock = threading.Lock()
    stop = threading.Event()
    control = threading.Thread(target=_control_loop, args=(robot, host, joints, lock, stop), daemon=True)

    logging.info(
        f"Listening on {cfg.bind_ip}:{cfg.port_zmq_cmd} (commands) and :{cfg.port_zmq_observations} (observations)"
    )
    control.start()
    start = time.perf_counter()
    try:
        while cfg.connection_time_s is None or time.perf_counter() - start < cfg.connection_time_s:
            loop_start = time.perf_counter()

            observation = {}
            for key, cam in cameras.items():
                try:
                    observation[key] = cam.async_read()
                except Exception as e:
                    logging.warning("Camera %s: %s", key, e)
            with lock:
                observation.update(joints)

            if observation:
                try:
                    host.zmq_observation_socket.send_string(
                        host.encode_observation(observation, cameras.keys()), flags=zmq.NOBLOCK
                    )
                except zmq.Again:
                    pass  # no client connected

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(1 / cfg.max_loop_freq_hz - elapsed, 0))
    finally:
        stop.set()
        control.join()


@draccus.wrap()
def main(cfg: SO101FollowerHostConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.robot.max_relative_target is None:
        logging.warning(
            "--robot.max_relative_target is not set. Over a laggy link, commands can arrive in bursts and the "
            "arm will jump to each one at full speed. Consider --robot.max_relative_target=10."
        )

    robot = SO101Follower(cfg.robot)
    robot.connect()
    host = SO101FollowerHost(cfg)
    try:
        run_host(robot, host)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()
        host.disconnect()
        logging.info("Host stopped.")


if __name__ == "__main__":
    main()
