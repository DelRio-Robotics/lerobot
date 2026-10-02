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


# Below this distance from the pan axis, the arm's plane is ill-defined (the tip is above the shoulder).
MIN_RADIUS_M = 0.02


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def pointing_elevation(q: np.ndarray) -> float:
    """How far up the controller points (its -z axis), in radians: 0 level, -pi/2 straight down."""
    pointing = quat_to_matrix(q) @ np.array([0.0, 0.0, -1.0])
    return math.asin(min(max(pointing[1], -1.0), 1.0))


def hand_twist(q0: np.ndarray, q: np.ndarray) -> float:
    """
    How far the controller has turned about the axis it points along since orientation `q0`, in radians
    (right-handed about the pointing direction), ignoring any turning or pitching. Like turning a doorknob.
    """
    x0, y0, z0, w0 = np.asarray(q0, dtype=float) / np.linalg.norm(q0)
    _, _, z, w = _quat_multiply(np.array([-x0, -y0, -z0, w0]), np.asarray(q, dtype=float) / np.linalg.norm(q))
    angle = 2 * math.atan2(-z, w)  # twist about the controller's own -z
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _radial(position: np.ndarray, pan_axis: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Horizontal unit vector from the pan axis toward `position`: the direction the arm's plane faces."""
    radial = np.array([position[0] - pan_axis[0], position[1] - pan_axis[1], 0.0])
    length = np.linalg.norm(radial)
    return radial / length if length >= MIN_RADIUS_M else fallback


def in_plane_angle(approach: np.ndarray, position: np.ndarray, pan_axis: np.ndarray) -> float:
    """
    Angle of the gripper's approach in the arm's vertical plane, in radians: 0 pointing straight out, negative
    pointing down, beyond +-pi/2 folded back toward the base.
    """
    horizontal = np.array([approach[0], approach[1], 0.0])
    own = (
        horizontal / np.linalg.norm(horizontal)
        if np.linalg.norm(horizontal) > 1e-9
        else np.array([1.0, 0, 0])
    )
    radial = _radial(position, pan_axis, own)
    return math.atan2(approach[2], float(approach @ radial))


def approach_direction(
    angle: float, position: np.ndarray, pan_axis: np.ndarray, fallback: np.ndarray
) -> np.ndarray:
    """
    The gripper approach at `angle` (see `in_plane_angle`) in the arm's plane through `position`. This is the
    only kind of approach the SO-101 can reach there, and it varies smoothly, also through straight down.
    `fallback` is the plane's direction to use when `position` is above the pan axis.
    """
    return math.cos(angle) * _radial(position, pan_axis, fallback) + math.sin(angle) * np.array(
        [0.0, 0.0, 1.0]
    )


def limit_motion(
    previous: dict[str, float], goal: dict[str, float], max_joint_step: float, max_tip_step: float, tip_of
) -> tuple[dict[str, float], bool]:
    """
    Moves from `previous` toward `goal` (joint angles, degrees) by at most `max_joint_step` per joint and
    `max_tip_step` metres at the tip (`tip_of(angles)` gives its position). Returns the angles and whether
    the move was shortened.
    """
    step = {joint: goal[joint] - previous[joint] for joint in goal}
    largest = max((abs(delta) for delta in step.values()), default=0.0)
    fraction = min(1.0, max_joint_step / largest) if largest > 0 else 1.0
    start = tip_of(previous)
    for _ in range(4):  # the tip isn't linear in the joints: re-check after shrinking
        moved = {joint: previous[joint] + fraction * delta for joint, delta in step.items()}
        tip_step = float(np.linalg.norm(tip_of(moved) - start))
        if tip_step <= max_tip_step + 1e-9:
            return (goal, False) if fraction == 1.0 else (moved, True)
        fraction *= max_tip_step / tip_step
    return {joint: previous[joint] + fraction * delta for joint, delta in step.items()}, True
