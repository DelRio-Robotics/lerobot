"""Hand-motion to gripper-target math of the virtual leader. WebXR: x right, y up, -z forward."""

import math

import numpy as np
import pytest

from lerobot.teleoperators.so101_leader_vr.mapping import (
    align_yaw_with_position,
    clip_to_box,
    clutch_target,
    head_yaw,
    limit_step,
    quat_to_matrix,
    xr_to_robot_rotation,
)

FORWARD_X = np.array([1.0, 0.0, 0.0])
XR_FORWARD, XR_UP, XR_RIGHT = np.array([0, 0, -1.0]), np.array([0, 1.0, 0]), np.array([1.0, 0, 0])


def quat_about(axis, degrees):
    """WebXR quaternion (x, y, z, w) for a rotation about `axis`."""
    half = math.radians(degrees) / 2
    return np.array([*(np.asarray(axis, float) * math.sin(half)), math.cos(half)])


def rot_z(degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_operator_directions_map_to_robot_forward_up_and_right():
    m = xr_to_robot_rotation(FORWARD_X)

    assert m @ XR_FORWARD == pytest.approx([1, 0, 0])
    assert m @ XR_UP == pytest.approx([0, 0, 1])
    assert m @ XR_RIGHT == pytest.approx([0, -1, 0])  # +y is the robot's left


def test_mapping_is_a_proper_rotation():
    m = xr_to_robot_rotation(np.array([0.6, 0.8, 0.0]), head_yaw=0.7, yaw_offset_deg=33)

    assert m @ m.T == pytest.approx(np.eye(3))
    assert np.linalg.det(m) == pytest.approx(1.0)


def test_recentring_makes_the_operators_new_facing_robot_forward():
    turned_left = math.radians(90)  # the operator now faces XR -x
    m = xr_to_robot_rotation(FORWARD_X, head_yaw=turned_left)

    assert m @ np.array([-1.0, 0, 0]) == pytest.approx([1, 0, 0])


def test_yaw_offset_rotates_the_mapping_for_a_camera_facing_the_arm():
    m = xr_to_robot_rotation(FORWARD_X, yaw_offset_deg=180)

    assert m @ XR_FORWARD == pytest.approx([-1, 0, 0])


def test_head_yaw_is_positive_when_turning_left():
    assert head_yaw(quat_about([0, 1, 0], 30)) == pytest.approx(math.radians(30))


def quat_multiply(a, b):
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


def test_head_yaw_ignores_looking_down():
    turned_then_looking_down = quat_multiply(quat_about([0, 1, 0], 30), quat_about([1, 0, 0], -40))

    assert head_yaw(turned_then_looking_down) == pytest.approx(math.radians(30))


def test_quat_to_matrix_rotates_like_webxr():
    assert quat_to_matrix(quat_about([0, 1, 0], 90)) @ XR_FORWARD == pytest.approx([-1, 0, 0])


def test_clutch_target_moves_the_tip_by_the_scaled_hand_motion():
    t0 = np.eye(4)
    t0[:3, 3] = [0.2, 0.0, 0.1]
    m = xr_to_robot_rotation(FORWARD_X)
    still = quat_about([0, 1, 0], 0)

    target = clutch_target(t0, np.zeros(3), still, 0.1 * XR_FORWARD + 0.04 * XR_UP, still, m, scale=0.5)

    assert target[:3, 3] == pytest.approx([0.25, 0.0, 0.12])
    assert target[:3, :3] == pytest.approx(np.eye(3))


def test_clutch_target_turns_the_tip_with_the_hand():
    t0 = np.eye(4)
    t0[:3, :3] = rot_z(10)
    m = xr_to_robot_rotation(FORWARD_X)

    target = clutch_target(
        t0, np.zeros(3), quat_about([0, 1, 0], 0), np.zeros(3), quat_about([0, 1, 0], 30), m, 1
    )

    assert target[:3, :3] == pytest.approx(rot_z(40))


def test_clip_to_box_leaves_points_inside_alone():
    point, clipped = clip_to_box(np.array([0.1, 0.0, 0.2]), [-0.4, -0.4, 0.0], [0.4, 0.4, 0.4])

    assert point == pytest.approx([0.1, 0.0, 0.2])
    assert not clipped


def test_clip_to_box_stops_points_at_the_floor():
    point, clipped = clip_to_box(np.array([0.1, 0.0, -0.05]), [-0.4, -0.4, 0.0], [0.4, 0.4, 0.4])

    assert point == pytest.approx([0.1, 0.0, 0.0])
    assert clipped


def test_limit_step_lets_small_steps_through():
    point, limited = limit_step(np.zeros(3), np.array([0.01, 0, 0]), max_step=0.02)

    assert point == pytest.approx([0.01, 0, 0])
    assert not limited


def test_limit_step_shortens_big_steps_along_their_direction():
    point, limited = limit_step(np.zeros(3), np.array([0.3, 0.4, 0]), max_step=0.02)

    assert point == pytest.approx([0.012, 0.016, 0])
    assert limited


def approach_of(rotation):
    return rotation[:, 2]  # the tip's z axis points out of the gripper


def test_align_yaw_points_the_approach_at_the_target_position():
    pointing_forward = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])  # approach along +x

    aligned = align_yaw_with_position(
        pointing_forward, np.array([0.04, 0.25, 0.1]), pan_axis=np.array([0.04, 0])
    )

    assert approach_of(aligned) == pytest.approx([0, 1, 0], abs=1e-9)


def test_align_yaw_keeps_pitch():
    pitched = np.array([[0.5, 0, 0.866], [0, 1.0, 0], [-0.866, 0, 0.5]])  # approach 30 deg above forward

    aligned = align_yaw_with_position(pitched, np.array([0.2, 0.2, 0.1]), pan_axis=np.zeros(2))

    assert approach_of(aligned)[2] == pytest.approx(approach_of(pitched)[2])
    assert math.atan2(approach_of(aligned)[1], approach_of(aligned)[0]) == pytest.approx(math.radians(45))


def test_align_yaw_keeps_an_approach_pointing_back_toward_the_base():
    pointing_back = rot_z(180) @ np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])  # approach along -x

    aligned = align_yaw_with_position(pointing_back, np.array([0.2, 0.01, 0.1]), pan_axis=np.zeros(2))

    assert approach_of(aligned)[0] < -0.99


def test_align_yaw_leaves_a_vertical_approach_alone():
    pointing_down = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])

    assert align_yaw_with_position(pointing_down, np.array([0.1, 0.2, 0.1]), np.zeros(2)) == pytest.approx(
        pointing_down
    )
