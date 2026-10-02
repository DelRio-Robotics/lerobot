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

from dataclasses import dataclass, field
from pathlib import Path

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("so101_leader_vr")
@dataclass
class SO101LeaderVRConfig(TeleoperatorConfig):
    """A virtual SO-101 leader driven from a VR headset. See `so101_leader_vr.py`."""

    # The follower's calibration file, as kept by lerobot on this computer:
    # <calibration dir>/robots/so101_follower/<follower_id>.json. It must match the follower's own copy.
    follower_id: str
    follower_calibration_dir: Path | None = None
    # Same as the follower's --robot.use_degrees.
    use_degrees: bool = False

    # The page the headset opens (vr_client/ in the workspace repo). None looks for it next to the lerobot
    # checkout.
    client_dir: Path | None = None
    http_port: int = 8443
    # False serves plain HTTP on 127.0.0.1 only, for a headset plugged in by USB with `adb reverse`.
    https: bool = True

    # Robot model for the IK. None uses the packaged SO-101 model.
    urdf_path: Path | None = None

    # Hand motion to gripper motion. Holding A halves it.
    motion_scale: float = 1.0
    # Turns the hand-to-arm mapping about the vertical, e.g. 180 when watching a camera that faces the arm.
    yaw_offset_deg: float = 0.0
    # Box the gripper tip stays in, in the robot's base frame (metres, x forward, y left, z up from the bottom of
    # the base). The lower z is the floor.
    ee_bounds_min: list[float] = field(default_factory=lambda: [-0.45, -0.45, 0.0])
    ee_bounds_max: list[float] = field(default_factory=lambda: [0.45, 0.45, 0.45])
    # Largest move of the target per action, so a tracking glitch can't fling the arm.
    max_ee_step_m: float = 0.02
    # The arm has 5 joints, so the tip's orientation is a soft goal next to its position (weight 1).
    orientation_weight: float = 0.1
    ik_iterations: int = 10
    # Keep body joints this many normalized units (of -100..100) inside their calibrated range.
    joint_margin: float = 3.0
    # Gripper position (0..100) with the trigger released and fully pressed.
    gripper_open: float = 100.0
    gripper_closed: float = 0.0

    # Which controller drives the arm: "right" or "left".
    hand: str = "right"
    # Camera frames sent to the headset per second.
    stream_fps: int = 20
    # Headset data older than this counts as lost: the arm holds until grip is pressed again.
    stale_ms: int = 300

    # Send the starting pose only, and log the joint targets that would have been sent.
    dry_run: bool = False

    def __post_init__(self):
        if self.hand not in ("right", "left"):
            raise ValueError(f"--teleop.hand must be 'right' or 'left', not {self.hand!r}.")
