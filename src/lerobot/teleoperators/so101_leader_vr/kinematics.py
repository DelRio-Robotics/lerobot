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

"""Forward and inverse kinematics of the SO-101 arm (5 joints, no gripper), with joint angles in degrees."""

from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics

from .units import ARM_JOINTS

DEFAULT_URDF = Path(__file__).parent / "so101_new_calib.urdf"
TIP_FRAME = "gripper_frame_link"


class ArmKinematics:
    def __init__(self, urdf_path: Path | str = DEFAULT_URDF):
        self._kinematics = RobotKinematics(str(urdf_path), TIP_FRAME, list(ARM_JOINTS))

    def fk(self, angles: dict[str, float]) -> np.ndarray:
        """4x4 pose of the gripper tip in the robot's base frame (metres)."""
        return self._kinematics.forward_kinematics(self._to_array(angles))

    def ik(
        self, seed: dict[str, float], target: np.ndarray, orientation_weight: float, iterations: int
    ) -> dict[str, float]:
        """
        Joint angles that bring the tip toward `target`, starting from `seed`. Each solver step moves part of
        the way, so a few steps per call track a moving target closely. The arm has 5 joints, so orientation is
        a soft goal: pitch and roll can follow, but yaw is tied to the shoulder pan.
        """
        angles = self._to_array(seed)
        for _ in range(iterations):
            angles = self._kinematics.inverse_kinematics(angles, target, 1.0, orientation_weight)
        return dict(zip(ARM_JOINTS, (float(a) for a in angles), strict=True))

    def pan_axis(self) -> np.ndarray:
        """(x, y) of the vertical axis the shoulder pan turns about, in the base frame."""
        return self._kinematics.robot.get_T_world_frame("shoulder_link")[:2, 3].copy()

    def forward_axis(self) -> np.ndarray:
        """Horizontal direction the arm points at the middle of every joint's range: the robot's "forward"."""
        tip = self.fk(dict.fromkeys(ARM_JOINTS, 0.0))[:3, 3]
        forward = np.array([tip[0], tip[1], 0.0])
        return forward / np.linalg.norm(forward)

    @staticmethod
    def _to_array(angles: dict[str, float]) -> np.ndarray:
        return np.array([angles[joint] for joint in ARM_JOINTS], dtype=float)
