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
A virtual SO-101 leader arm driven from a VR headset (e.g. a Meta Quest 3S):

```shell
lerobot-teleoperate \
    --robot.type=so101_follower_client --robot.remote_ip=$FOLLOWER_IP --robot.cameras="$CAMS" \
    --teleop.type=so101_leader_vr --teleop.follower_id=my_awesome_follower_arm --fps=30
```

Where `so101_leader` opens its serial port and reads the arm's motors, this starts a web server for the
headset's browser and turns the operator's hand into the same joint positions with inverse kinematics. Hold
grip to move the arm, pull the trigger to close the gripper, hold A for fine moves, press B to make the way you
face "forward". The page itself is `vr_client/` in the workspace repo.

It needs the follower's calibration file on this computer, and the robot's observations, which the teleop
loop hands over through `on_observation`.
"""

import logging
import time
from pathlib import Path

import numpy as np

import lerobot
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_so101_leader_vr import SO101LeaderVRConfig
from .controller import VRArmController
from .kinematics import DEFAULT_URDF, ArmKinematics
from .server import VRServer, ensure_certificate, load_or_create_token
from .units import MOTORS, from_follower_units, load_follower_calibration

logger = logging.getLogger(__name__)

# The page's certificate and access token, kept across runs.
STATE_DIR = HF_LEROBOT_HOME / "so101_leader_vr"
# vr_client/ next to the lerobot checkout (the workspace repo).
DEFAULT_CLIENT_DIR = Path(lerobot.__file__).resolve().parents[3] / "vr_client"
DRY_RUN_LOG_PERIOD_S = 0.5
# No new camera frame from the robot for this long: the operator is warned that the video is frozen.
VIDEO_STALE_S = 0.5


class SO101LeaderVR(Teleoperator):
    config_class = SO101LeaderVRConfig
    name = "so101_leader_vr"

    def __init__(self, config: SO101LeaderVRConfig):
        super().__init__(config)
        self.config = config
        self.server: VRServer | None = None
        self.controller: VRArmController | None = None
        self._start: dict[str, float] | None = None
        self._last_log = 0.0
        self._clock = time.monotonic
        # The last frames handed over, kept alive so a new frame can't reuse their identity.
        self._last_frames: dict[str, np.ndarray] = {}
        self._last_new_frame: float | None = None
        self.status: dict = {}

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in MOTORS}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.server is not None

    @property
    def is_calibrated(self) -> bool:
        # Nothing to calibrate here: the follower's calibration is read at connect.
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.name} already connected")

        calibration = load_follower_calibration(self.config.follower_id, self.config.follower_calibration_dir)
        client_dir = Path(self.config.client_dir or DEFAULT_CLIENT_DIR)
        if not (client_dir / "index.html").is_file():
            raise FileNotFoundError(
                f"No headset page at {client_dir}. Pass --teleop.client_dir=<workspace repo>/vr_client."
            )
        self.controller = VRArmController(
            self.config, calibration, ArmKinematics(self.config.urdf_path or DEFAULT_URDF)
        )
        server = VRServer(
            client_dir,
            port=self.config.http_port,
            https=self.config.https,
            token=load_or_create_token(STATE_DIR),
            hand=self.config.hand,
            stream_fps=self.config.stream_fps,
            certificate=ensure_certificate(STATE_DIR) if self.config.https else None,
        )
        server.start()
        self.server = server
        logger.info(f"{self.name} ready. Open this in the headset's browser: {server.url}")

    def on_observation(self, observation: dict) -> None:
        """Takes the follower's joint readings and camera frames, from the teleop loop."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")
        self.controller.observe(observation)
        if self._start is None:
            self._start = dict(self.controller.measured)
        frames = {
            key: value
            for key, value in observation.items()
            if isinstance(value, np.ndarray) and value.ndim == 3
        }
        # When nothing new arrives, the robot client hands back the same frame objects.
        if any(self._last_frames.get(name) is not frame for name, frame in frames.items()):
            self._last_new_frame = self._clock()
            self.server.publish_frames(frames)
        self._last_frames = frames

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")
        if self._start is None:
            raise RuntimeError(
                f"{self.name} needs the robot's observations: run it with lerobot-teleoperate, which hands them over "
                "(recording with it isn't supported yet)."
            )
        action = self.controller.step(self.server.latest(), time.monotonic())
        video_stale = (
            self._last_new_frame is not None and self._clock() - self._last_new_frame > VIDEO_STALE_S
        )
        self.status = {**self.controller.status, "video_stale": bool(video_stale)}
        self.server.publish_status(self.status)
        if not self.config.dry_run:
            return action

        now = time.monotonic()
        if now - self._last_log >= DRY_RUN_LOG_PERIOD_S:
            self._last_log = now
            angles, _ = from_follower_units(
                self.controller.measured, self.controller.calibration, self.config.use_degrees
            )
            tip = self.controller.kinematics.fk(angles)[:3, 3]
            would_send = {key: round(value, 1) for key, value in action.items()}
            logger.info(f"Dry run: tip at {np.round(tip, 3)} m, would send {would_send}, {self.status}")
        return dict(self._start)

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")
        self.server.stop()
        self.server = None
        logger.info(f"{self.name} stopped.")
