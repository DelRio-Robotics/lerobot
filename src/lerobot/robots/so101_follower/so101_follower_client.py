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

import base64
import json
import logging
from functools import cached_property
from typing import Any

import cv2
import numpy as np

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_so101_follower import SO101FollowerClientConfig

logger = logging.getLogger(__name__)

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


class SO101FollowerClient(Robot):
    """
    An SO-101 follower arm attached to another computer, reached through `so101_follower_host` over ZMQ.
    Works anywhere a local `so101_follower` does (teleoperate, record, policy rollout).
    """

    config_class = SO101FollowerClientConfig
    name = "so101_follower_client"

    def __init__(self, config: SO101FollowerClientConfig):
        import zmq

        self._zmq = zmq
        super().__init__(config)
        self.config = config
        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None
        self.last_observation: dict[str, Any] = {}
        self._is_connected = False

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in MOTORS}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        # Calibration lives with the arm, on the host.
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        zmq = self._zmq
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.connect(f"tcp://{self.config.remote_ip}:{self.config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.connect(
            f"tcp://{self.config.remote_ip}:{self.config.port_zmq_observations}"
        )

        observation = self._receive_observation(self.config.connect_timeout_s * 1000)
        if observation is None:
            self._close_sockets()
            raise DeviceNotConnectedError(
                f"No observation from {self.config.remote_ip}:{self.config.port_zmq_observations} within "
                f"{self.config.connect_timeout_s}s. Is so101_follower_host running there, and is the address reachable?"
            )

        missing = set(self._cameras_ft) - set(observation)
        if missing:
            self._close_sockets()
            raise ValueError(
                f"--robot.cameras declares {sorted(missing)} but the host doesn't send them. "
                "Pass the same --robot.cameras on both computers."
            )

        self.last_observation = observation
        self._is_connected = True
        logger.info(f"{self} connected to {self.config.remote_ip}.")

    def _receive_observation(self, timeout_ms: int) -> dict[str, Any] | None:
        """Waits up to `timeout_ms` for an observation and decodes it. Returns None on timeout or bad data."""
        zmq = self._zmq
        if not self.zmq_observation_socket.poll(timeout_ms, zmq.POLLIN):
            return None
        try:
            msg = json.loads(self.zmq_observation_socket.recv_string(zmq.NOBLOCK))
        except (zmq.Again, json.JSONDecodeError) as e:
            logger.warning(f"Dropping observation: {e}")
            return None

        observation: dict[str, Any] = {}
        for key, value in msg.items():
            if key in self._motors_ft:
                observation[key] = float(value)
                continue
            frame = (
                cv2.imdecode(np.frombuffer(base64.b64decode(value), dtype=np.uint8), cv2.IMREAD_COLOR)
                if value
                else None
            )
            if frame is None:
                frame = self.last_observation.get(key)
            if frame is None:
                h, w, c = self._cameras_ft.get(key, (480, 640, 3))
                frame = np.zeros((h, w, c), dtype=np.uint8)
            observation[key] = frame
        return observation

    def get_observation(self) -> dict[str, Any]:
        """Latest observation from the host, or the previous one if nothing new arrived in time."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        observation = self._receive_observation(self.config.polling_timeout_ms)
        if observation is not None:
            self.last_observation = observation
        return dict(self.last_observation)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Sends goal positions to the host. Clipping by `max_relative_target` happens there and isn't reported back."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key: float(val) for key, val in action.items() if key in self._motors_ft}
        self.zmq_cmd_socket.send_string(json.dumps(goal_pos))
        return goal_pos

    def _close_sockets(self) -> None:
        self.zmq_observation_socket.close(linger=0)
        self.zmq_cmd_socket.close(linger=0)
        self.zmq_context.term()

    def disconnect(self) -> None:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._close_sockets()
        self._is_connected = False
        logger.info(f"{self} disconnected.")
