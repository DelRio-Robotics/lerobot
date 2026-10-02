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

import math
from dataclasses import dataclass

import numpy as np

from lerobot.motors import MotorCalibration

from .config_so101_leader_vr import SO101LeaderVRConfig
from .kinematics import ArmKinematics
from .mapping import (
    approach_direction,
    clip_to_box,
    hand_twist,
    head_yaw,
    in_plane_angle,
    limit_motion,
    limit_step,
    pointing_elevation,
    xr_to_robot_rotation,
)
from .units import ARM_JOINTS, MOTORS, from_follower_units, normalized_to_degrees, to_follower_units

_FLOOR_TOLERANCE_M = 0.002
# How far the gripper's pitch target may run ahead of the pitch the arm reached: beyond what the arm can do,
# tilting the hand further does nothing (instead of dragging the tip away), and tilting back responds at once.
_MAX_PITCH_LEAD = math.radians(10)
# A solution whose tip misses the target by more than this is being pulled off by the pitch goal.
_POSITION_TOLERANCE_M = 0.002


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


@dataclass
class _Anchor:
    """Where the hand and the target were when following (re)started."""

    hand_position: np.ndarray
    position: np.ndarray


class VRArmController:
    """
    A virtual leader arm. Holding grip engages it, and from then on, relative to where grip was pressed:
    - the gripper tip moves as much as the hand does (`motion_scale`),
    - the gripper pitches as much as the hand pitches (world frame: tilting the hand down tilts the gripper down),
    - the gripper turns about its own axis as much as the hand twists about where it points.
    Turning the hand about the vertical does nothing: the arm's direction comes from where the tip is.
    Releasing grip, or losing the headset's data, holds the arm where it is. Every action moves the tip at most
    `max_ee_step_m` and each joint at most `max_joint_step_deg`.
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
        self._roll_sign = kinematics.roll_sign()
        # The IK keeps every joint inside its calibrated range (minus the margin) as well as the model's limits.
        limit = 100 - config.joint_margin
        kinematics.limit_joints(
            {
                joint: tuple(sorted(normalized_to_degrees(v, calibration[joint]) for v in (-limit, limit)))
                for joint in ARM_JOINTS
            }
        )
        self._roll_range = kinematics.joint_limits("wrist_roll")
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
        self._anchor: _Anchor | None = None
        self._last_hand_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self._last_hand_elevation = 0.0
        # The latest target: tip position, approach angle in the arm's plane (radians), wrist roll (degrees).
        self._target_position = np.zeros(3)
        self._target_angle = 0.0
        self._target_roll = 0.0
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
            pose = self.kinematics.fk(self._angles)
            self._target_position = pose[:3, 3]
            self._target_angle = in_plane_angle(pose[:3, 2], self._target_position, self._pan_axis)
            self._target_roll = float(np.clip(self._angles["wrist_roll"], *self._roll_range))
            self._anchor_at(xr)
        self._was_gripping = xr.grip

        if self._engaged:
            if xr.precision != self._was_precise:
                self._anchor_at(xr)  # the scale changes: continue from here instead of jumping
            self._follow(xr)
        self._was_precise = xr.precision

        if self._gripper_active:
            trigger = min(max(xr.trigger, 0.0), 1.0)
            span = self.config.gripper_closed - self.config.gripper_open
            self._gripper = self.config.gripper_open + span * trigger
            self._hold = {**self._hold, "gripper.pos": min(max(self._gripper, 0.0), 100.0)}

        self.status["engaged"] = self._engaged
        return dict(self._hold)

    def _anchor_at(self, xr: XRInput) -> None:
        """Following continues from the current target, so re-anchoring never jumps."""
        self._anchor = _Anchor(hand_position=xr.position.copy(), position=self._target_position.copy())
        self._last_hand_orientation = xr.orientation.copy()
        self._last_hand_elevation = pointing_elevation(xr.orientation)

    def _follow(self, xr: XRInput) -> None:
        anchor = self._anchor
        scale = self.config.motion_scale * (0.5 if xr.precision else 1.0)
        position = anchor.position + scale * self._xr_to_robot @ (xr.position - anchor.hand_position)
        # A tip already below the floor (e.g. resting on the table) may stay there or rise, but not sink.
        low = self._low.copy()
        low[2] = min(low[2], self._target_position[2])
        position, clipped = clip_to_box(position, low, self._high)
        position, stepped = limit_step(self._target_position, position, self.config.max_ee_step_m)
        self._target_position = position
        elevation = pointing_elevation(xr.orientation)
        self._target_angle += elevation - self._last_hand_elevation
        self._last_hand_elevation = elevation
        # The twist is added up tick by tick, so it never wraps at half a turn, and it slides along the wrist's
        # limits: twisting further does nothing, twisting back responds at once.
        twist = hand_twist(self._last_hand_orientation, xr.orientation)
        self._last_hand_orientation = xr.orientation.copy()
        self._target_roll = float(
            np.clip(self._target_roll + self._roll_sign * math.degrees(twist), *self._roll_range)
        )

        goal = self._solve(position, self._target_angle)
        overreached = bool(np.linalg.norm(self.kinematics.fk(goal)[:3, 3] - position) > _POSITION_TOLERANCE_M)
        if overreached:
            # The pitch asked for is out of reach here (e.g. the wrist is at its limit) and the solver is trading
            # the tip's position for it. Position comes first: keep the pitch the arm has and place the tip.
            current = self.kinematics.fk(self._angles)
            self._target_angle = in_plane_angle(current[:3, 2], current[:3, 3], self._pan_axis)
            goal = self._solve(position, self._target_angle)
        # Keep inside the calibrated range, then bound what is actually sent.
        goal, _ = from_follower_units(
            to_follower_units(goal, 0.0, self.calibration, self.config.joint_margin, self.config.use_degrees),
            self.calibration,
            self.config.use_degrees,
        )
        moved, shortened = limit_motion(
            self._angles,
            goal,
            self.config.max_joint_step_deg,
            self.config.max_ee_step_m,
            lambda angles: self.kinematics.fk(angles)[:3, 3],
        )
        # The target is already clipped to the floor; this catches moves that dip well below it.
        tip_z = self.kinematics.fk(moved)[2, 3]
        sinks = bool(
            tip_z < self._low[2] - _FLOOR_TOLERANCE_M and tip_z < self.kinematics.fk(self._angles)[2, 3]
        )
        if not sinks:
            self._hold = to_follower_units(
                moved, self._gripper, self.calibration, self.config.joint_margin, self.config.use_degrees
            )
            self._angles, _ = from_follower_units(self._hold, self.calibration, self.config.use_degrees)

        reached = self.kinematics.fk(self._angles)
        reached_angle = in_plane_angle(reached[:3, 2], reached[:3, 3], self._pan_axis)
        lead = (self._target_angle - reached_angle + math.pi) % (2 * math.pi) - math.pi
        self._target_angle = reached_angle + float(np.clip(lead, -_MAX_PITCH_LEAD, _MAX_PITCH_LEAD))
        overreached = overreached or abs(lead) > _MAX_PITCH_LEAD
        self.status["limited"] = bool(clipped or stepped or shortened or sinks or overreached)

    def _solve(self, position: np.ndarray, angle: float) -> dict[str, float]:
        approach = approach_direction(angle, position, self._pan_axis, self._forward)
        return self.kinematics.ik(
            {**self._angles, "wrist_roll": self._target_roll},
            position,
            approach,
            self.config.orientation_weight,
            self.config.ik_iterations,
        )
