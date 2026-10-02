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
Turns hand motion from the headset into a target pose for the gripper tip.

WebXR's world frame has x to the right, y up and -z forward (the way the operator faced when the session
started). The robot's base frame has z up; "forward" is the direction the arm points at the middle of its
joint ranges (see `ArmKinematics.forward_axis`).
"""

import math

import numpy as np


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a WebXR quaternion (x, y, z, w)."""
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def head_yaw(q: np.ndarray) -> float:
    """Which way the headset faces, in radians about WebXR's up axis: 0 at the start, positive to the left."""
    facing = quat_to_matrix(q) @ np.array([0.0, 0.0, -1.0])
    return math.atan2(-facing[0], -facing[2])


def _rot_about(axis: int, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    r = np.eye(3)
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


def xr_to_robot_rotation(
    forward: np.ndarray, head_yaw: float = 0.0, yaw_offset_deg: float = 0.0
) -> np.ndarray:
    """
    Rotation taking WebXR vectors into the robot's base frame: the operator's forward (as of the last
    re-centre, `head_yaw`) becomes the robot's `forward`, up stays up. `yaw_offset_deg` turns the mapping
    about the vertical, e.g. 180 when watching a camera that faces the arm.
    """
    forward = np.array([forward[0], forward[1], 0.0]) / math.hypot(forward[0], forward[1])
    up = np.array([0.0, 0.0, 1.0])
    left = np.cross(up, forward)
    # Columns: where WebXR's x (right), y (up) and z (backward) land in the robot frame.
    xr_to_robot = np.column_stack([-left, up, -forward])
    return _rot_about(2, math.radians(yaw_offset_deg)) @ xr_to_robot @ _rot_about(1, -head_yaw)


def clutch_target(
    t0: np.ndarray,
    c0_pos: np.ndarray,
    c0_quat: np.ndarray,
    c_pos: np.ndarray,
    c_quat: np.ndarray,
    xr_to_robot: np.ndarray,
    scale: float,
) -> np.ndarray:
    """
    Tip pose that has moved and turned from `t0` the way the controller moved and turned since it was at
    (`c0_pos`, `c0_quat`). Movement is scaled by `scale`; rotation is not.
    """
    target = np.eye(4)
    hand_turn = quat_to_matrix(c_quat) @ quat_to_matrix(c0_quat).T
    target[:3, :3] = xr_to_robot @ hand_turn @ xr_to_robot.T @ t0[:3, :3]
    target[:3, 3] = t0[:3, 3] + scale * xr_to_robot @ (np.asarray(c_pos) - np.asarray(c0_pos))
    return target


def clip_to_box(point: np.ndarray, low, high) -> tuple[np.ndarray, bool]:
    clipped = np.clip(point, low, high)
    return clipped, not np.allclose(clipped, point)


def limit_step(previous: np.ndarray, point: np.ndarray, max_step: float) -> tuple[np.ndarray, bool]:
    step = point - previous
    length = np.linalg.norm(step)
    if length <= max_step:
        return point, False
    return previous + step * (max_step / length), True


def align_yaw_with_position(rotation: np.ndarray, position: np.ndarray, pan_axis: np.ndarray) -> np.ndarray:
    """
    Turns a tip orientation about the vertical so its approach (the tip's z axis) lies in the vertical plane
    through the shoulder pan axis and `position`. The SO-101's pitch joints all move in that plane, so this is
    the only yaw it can reach there: without it, the orientation goal fights the pan on sideways moves.
    Pitch and roll are kept. An approach within ~10 degrees of vertical is left as is.
    """
    approach = rotation[:, 2]
    if math.hypot(approach[0], approach[1]) < 0.17:
        return rotation
    radial = np.asarray(position[:2]) - np.asarray(pan_axis)
    turn = math.atan2(radial[1], radial[0]) - math.atan2(approach[1], approach[0])
    turn = (turn + math.pi) % (2 * math.pi) - math.pi
    if abs(turn) > math.pi / 2:  # the approach points back toward the base: keep it that way
        turn -= math.copysign(math.pi, turn)
    return _rot_about(2, turn) @ rotation
