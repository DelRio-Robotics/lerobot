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

import math
from pathlib import Path

import numpy as np

from .units import ARM_JOINTS

DEFAULT_URDF = Path(__file__).parent / "so101_new_calib.urdf"
TIP_FRAME = "gripper_frame_link"
# The gripper points along its tip frame's z axis.
APPROACH_AXIS = np.array([0.0, 0.0, 1.0])


class ArmKinematics:
    def __init__(self, urdf_path: Path | str = DEFAULT_URDF):
        try:
            import placo
        except ImportError as e:
            raise ImportError('so101_leader_vr needs placo: uv pip install -e ".[so101_leader_vr]"') from e

        self._robot = placo.RobotWrapper(str(urdf_path))
        self._solver = placo.KinematicsSolver(self._robot)
        self._solver.mask_fbase(True)
        self._solver.enable_joint_limits(True)
        # The wrist roll is set straight from the operator's hand; the IK places the tip with the other joints.
        self._solver.mask_dof("wrist_roll")
        self._solver.mask_dof("gripper")
        self._position_task = self._solver.add_position_task(TIP_FRAME, np.zeros(3))
        self._position_task.configure("tip position", "soft", 1.0)
        self._approach_task = self._solver.add_axisalign_task(
            TIP_FRAME, APPROACH_AXIS, np.array([1.0, 0.0, 0.0])
        )
        self._approach_task.configure("tip approach", "soft", 0.1)
        self._model_limits = {joint: self._robot.get_joint_limits(joint) for joint in ARM_JOINTS}

    def _set_angles(self, angles: dict[str, float]) -> None:
        for joint in ARM_JOINTS:
            self._robot.set_joint(joint, math.radians(angles[joint]))
        self._robot.update_kinematics()

    def fk(self, angles: dict[str, float]) -> np.ndarray:
        """4x4 pose of the gripper tip in the robot's base frame (metres). Its z axis is the approach."""
        self._set_angles(angles)
        return np.array(self._robot.get_T_world_frame(TIP_FRAME))

    def ik(
        self,
        seed: dict[str, float],
        position: np.ndarray,
        approach: np.ndarray,
        approach_weight: float,
        iterations: int,
    ) -> dict[str, float]:
        """
        Joint angles that bring the tip toward `position` with the gripper pointing along `approach`, starting
        from `seed`. The wrist roll stays at the seed's value. Position comes first (weight 1); the approach
        is a softer goal. Each solver step moves part of the way, so a few steps per call track closely.
        """
        # The wrist roll is fixed during the solve, so it must start inside its limits for the solve to be feasible.
        low, high = self.joint_limits("wrist_roll")
        self._set_angles({**seed, "wrist_roll": min(max(seed["wrist_roll"], low + 0.01), high - 0.01)})
        self._position_task.target_world = np.asarray(position, dtype=float)
        self._approach_task.targetAxis_world = np.asarray(approach, dtype=float) / np.linalg.norm(approach)
        self._approach_task.configure("tip approach", "soft", approach_weight)
        for _ in range(iterations):
            self._solver.solve(True)
            self._robot.update_kinematics()
        return {joint: math.degrees(self._robot.get_joint(joint)) for joint in ARM_JOINTS}

    def joint_limits(self, joint: str) -> tuple[float, float]:
        """The (lowest, highest) angle the IK may give a joint, in degrees."""
        low, high = self._robot.get_joint_limits(joint)
        return math.degrees(low), math.degrees(high)

    def limit_joints(self, ranges: dict[str, tuple[float, float]]) -> None:
        """
        Keeps IK solutions inside these (low, high) ranges in degrees, e.g. the arm's calibrated range, on top of
        the model's own limits. Each call replaces the previous ranges.
        """
        for joint, (low, high) in ranges.items():
            model_low, model_high = self._model_limits[joint]
            self._robot.set_joint_limits(
                joint, max(model_low, math.radians(low)), min(model_high, math.radians(high))
            )

    def roll_sign(self) -> float:
        """+1 or -1: the wrist roll direction that turns the tip about its approach axis right-handedly."""
        straight = dict.fromkeys(ARM_JOINTS, 0.0)
        before = self.fk(straight)[:3, :3]
        after = self.fk({**straight, "wrist_roll": 10.0})[:3, :3]
        return 1.0 if (before.T @ after)[1, 0] > 0 else -1.0

    def pan_axis(self) -> np.ndarray:
        """(x, y) of the vertical axis the shoulder pan turns about, in the base frame."""
        return np.array(self._robot.get_T_world_frame("shoulder_link"))[:2, 3]

    def forward_axis(self) -> np.ndarray:
        """Horizontal direction the arm points at the middle of every joint's range: the robot's "forward"."""
        tip = self.fk(dict.fromkeys(ARM_JOINTS, 0.0))[:3, 3]
        forward = np.array([tip[0], tip[1], 0.0])
        return forward / np.linalg.norm(forward)
