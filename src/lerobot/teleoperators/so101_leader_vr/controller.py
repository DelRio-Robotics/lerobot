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

"""Turns the operator's headset input and the follower's joint readings into follower actions. No I/O."""

from dataclasses import dataclass

import numpy as np

from lerobot.motors import MotorCalibration

from .config_so101_leader_vr import SO101LeaderVRConfig
from .kinematics import ArmKinematics
from .mapping import (
    align_yaw_with_position,
    clip_to_box,
    clutch_target,
    head_yaw,
    limit_step,
    xr_to_robot_rotation,
)
from .units import MOTORS, from_follower_units, to_follower_units

_FLOOR_TOLERANCE_M = 0.002


@dataclass
class XRInput:
    """One sample of the operator's controller and headset."""

    position: np.ndarray  # controller position, WebXR metres
    orientation: np.ndarray  # controller orientation, quaternion (x, y, z, w)
    head_orientation: np.ndarray
    trigger: float  # 0 released .. 1 fully pressed
    grip: bool
    precision: bool  # A (or X) held: half the motion
    recenter: bool  # B (or Y) held: the operator's current facing becomes robot forward
    received_at: float  # time.monotonic() when it arrived


class VRArmController:
    """
    A virtual leader arm. Holding grip engages it: the gripper tip then moves and turns from where it was by
    as much as the hand does. Releasing grip, or losing the headset's data, holds the arm where it is.
    """

    def __init__(
        self,
        config: SO101LeaderVRConfig,
        calibration: dict[str, MotorCalibration],
        kinematics: ArmKinematics,
    ):
        self.config = config
        self.calibration = calibration
        self.kinematics = kinematics
        self._forward = kinematics.forward_axis()
        self._pan_axis = kinematics.pan_axis()
        self._xr_to_robot = xr_to_robot_rotation(self._forward, 0.0, config.yaw_offset_deg)
        self._low = np.array(config.ee_bounds_min, dtype=float)
        self._high = np.array(config.ee_bounds_max, dtype=float)

        self.measured: dict[str, float] | None = None  # latest follower joint readings
        self._hold: dict[str, float] | None = None  # the action sent while the arm isn't following
        self._angles: dict[str, float] = {}  # arm angles (degrees) of `_hold`
        self._gripper = 0.0
        self._gripper_active = False

        self._engaged = False
        self._was_gripping = False
        self._was_precise = False
        self._was_recentering = False
        self._anchor: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._target = np.eye(4)
        self.status = {"engaged": False, "stale": True, "limited": False}

    def observe(self, observation: dict) -> None:
        """Takes the follower's joint readings. The first ones are where the arm starts and holds."""
        joints = {f"{motor}.pos": float(observation[f"{motor}.pos"]) for motor in MOTORS}
        self.measured = joints
        if self._hold is None:
            self._hold = joints
            self._angles, self._gripper = from_follower_units(
                joints, self.calibration, self.config.use_degrees
            )

    def step(self, xr: XRInput | None, now: float) -> dict[str, float]:
        if self._hold is None:
            raise RuntimeError("No observation from the follower yet.")
        self.status["limited"] = False
        fresh = xr is not None and now - xr.received_at <= self.config.stale_ms / 1000
        self.status["stale"] = not fresh
        if not fresh:
            # Holds. Following resumes only on a fresh press of grip (`_was_gripping` keeps its last value).
            self._engaged = self.status["engaged"] = False
            return dict(self._hold)

        if xr.recenter and not self._was_recentering:
            self._xr_to_robot = xr_to_robot_rotation(
                self._forward, head_yaw(xr.head_orientation), self.config.yaw_offset_deg
            )
            self._anchor_at(xr)
        self._was_recentering = xr.recenter

        if not xr.grip:
            self._engaged = False
        elif not self._was_gripping:
            self._engaged = self._gripper_active = True
            self._target = self.kinematics.fk(self._angles)
            self._anchor_at(xr)
        self._was_gripping = xr.grip

        if self._engaged:
            if xr.precision != self._was_precise:
                self._anchor_at(xr)  # the scale changes: continue from here instead of jumping
            self._follow(xr)
        self._was_precise = xr.precision

        if self._gripper_active:
            trigger = min(max(xr.trigger, 0.0), 1.0)
            self._gripper = (
                self.config.gripper_open + (self.config.gripper_closed - self.config.gripper_open) * trigger
            )
            self._hold = {**self._hold, "gripper.pos": min(max(self._gripper, 0.0), 100.0)}

        self.status["engaged"] = self._engaged
        return dict(self._hold)

    def _anchor_at(self, xr: XRInput) -> None:
        self._anchor = (xr.position.copy(), xr.orientation.copy(), self._target.copy())

    def _follow(self, xr: XRInput) -> None:
        c0_pos, c0_quat, t0 = self._anchor
        scale = self.config.motion_scale * (0.5 if xr.precision else 1.0)
        target = clutch_target(t0, c0_pos, c0_quat, xr.position, xr.orientation, self._xr_to_robot, scale)

        # A tip already below the floor (e.g. resting on the table) may stay there or rise, but not sink.
        low = self._low.copy()
        low[2] = min(low[2], self._target[2, 3])
        position, clipped = clip_to_box(target[:3, 3], low, self._high)
        position, stepped = limit_step(self._target[:3, 3], position, self.config.max_ee_step_m)
        target[:3, 3] = position
        target[:3, :3] = align_yaw_with_position(target[:3, :3], position, self._pan_axis)
        self._target = target

        angles = self.kinematics.ik(
            self._angles, target, self.config.orientation_weight, self.config.ik_iterations
        )
        tip_z = self.kinematics.fk(angles)[2, 3]
        # The target is already clipped to the floor; this catches IK solutions that dip well below it.
        sinks = tip_z < self._low[2] - _FLOOR_TOLERANCE_M and tip_z < self.kinematics.fk(self._angles)[2, 3]
        self.status["limited"] = bool(clipped or stepped or sinks)
        if sinks:
            return
        self._hold = to_follower_units(
            angles, self._gripper, self.calibration, self.config.joint_margin, self.config.use_degrees
        )
        self._angles, _ = from_follower_units(self._hold, self.calibration, self.config.use_degrees)
