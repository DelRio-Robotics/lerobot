"""Hand-motion to gripper-target math of the virtual leader. WebXR: x right, y up, -z forward."""

import math

import numpy as np
import pytest

from lerobot.teleoperators.so101_leader_vr.mapping import (
    approach_direction,
    clip_to_box,
    clutch_target,
    hand_twist,
    head_yaw,
    in_plane_angle,
    limit_motion,
    limit_step,
    pointing_elevation,
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


def test_pointing_elevation_is_how_far_the_controller_points_up():
    assert pointing_elevation(quat_about([0, 1, 0], 0)) == pytest.approx(0.0)
    assert pointing_elevation(quat_about([1, 0, 0], 30)) == pytest.approx(math.radians(30))
    assert pointing_elevation(quat_about([1, 0, 0], -90)) == pytest.approx(math.radians(-90))


def test_pointing_elevation_ignores_turning_and_twisting():
    turned_and_twisted = quat_multiply(quat_about([0, 1, 0], 70), quat_about([0, 0, 1], 50))

    assert pointing_elevation(turned_and_twisted) == pytest.approx(0.0, abs=1e-9)


def test_hand_twist_is_the_rotation_about_where_the_controller_points():
    start = quat_about([0, 1, 0], 0)

    # Turning about XR z turns the controller about its pointing axis (-z) the other way.
    assert hand_twist(start, quat_about([0, 0, 1], -40)) == pytest.approx(math.radians(40))


def test_hand_twist_ignores_turning_and_pitching():
    start = quat_about([0, 1, 0], 0)

    assert hand_twist(start, quat_about([0, 1, 0], 70)) == pytest.approx(0.0, abs=1e-9)
    assert hand_twist(start, quat_about([1, 0, 0], -50)) == pytest.approx(0.0, abs=1e-9)


def test_hand_twist_is_measured_in_the_controllers_own_frame():
    pointing_down = quat_about([1, 0, 0], -90)
    twisted = quat_multiply(pointing_down, quat_about([0, 0, 1], -25))

    assert hand_twist(pointing_down, twisted) == pytest.approx(math.radians(25))


def test_in_plane_angle_is_the_approach_elevation_in_the_arm_plane():
    radial = np.array([0.2, 0.2, 0.1])  # tip off to the left at 45 degrees
    outward_down = np.array([0.5, 0.5, -1 / math.sqrt(2)])

    assert in_plane_angle(outward_down, radial, pan_axis=np.zeros(2)) == pytest.approx(math.radians(-45))


def test_in_plane_angle_beyond_vertical_points_back_toward_the_base():
    backward_down = np.array([-0.5, 0.0, -0.866])

    angle = in_plane_angle(backward_down, np.array([0.2, 0.0, 0.1]), pan_axis=np.zeros(2))

    assert angle == pytest.approx(math.radians(-120), abs=1e-3)


def test_approach_direction_points_outward_at_the_given_elevation():
    direction = approach_direction(
        math.radians(-30), np.array([0.0, 0.3, 0.1]), np.zeros(2), np.array([1.0, 0, 0])
    )

    assert direction == pytest.approx([0, math.cos(math.radians(30)), -0.5])


def test_approach_direction_is_continuous_through_vertical():
    position, axis, fallback = np.array([0.25, 0.1, 0.1]), np.zeros(2), np.array([1.0, 0, 0])

    below = approach_direction(math.radians(-89.9), position, axis, fallback)
    beyond = approach_direction(math.radians(-90.1), position, axis, fallback)

    assert np.linalg.norm(below - beyond) < math.radians(0.3)


def test_approach_direction_uses_the_fallback_above_the_pan_axis():
    direction = approach_direction(0.0, np.array([0.001, 0.0, 0.3]), np.zeros(2), np.array([0.0, 1.0, 0]))

    assert direction == pytest.approx([0, 1, 0])


def linear_tip(angles):
    """A stand-in arm whose tip moves 1 cm per degree of shoulder_lift."""
    return np.array([0.01 * angles["shoulder_lift"], 0.0, 0.0])


def test_limit_motion_passes_small_moves():
    previous = {"shoulder_lift": 0.0, "elbow_flex": 0.0}
    goal = {"shoulder_lift": 1.0, "elbow_flex": -2.0}

    assert limit_motion(previous, goal, 4.0, 0.02, linear_tip) == (goal, False)


def test_limit_motion_caps_the_fastest_joint_and_scales_the_others():
    previous = {"shoulder_lift": 0.0, "elbow_flex": 0.0}

    moved, limited = limit_motion(previous, {"shoulder_lift": 1.0, "elbow_flex": 40.0}, 4.0, 1.0, linear_tip)

    assert moved == pytest.approx({"shoulder_lift": 0.1, "elbow_flex": 4.0})
    assert limited


def test_limit_motion_caps_the_tip_step():
    previous = {"shoulder_lift": 0.0, "elbow_flex": 0.0}

    moved, limited = limit_motion(previous, {"shoulder_lift": 3.0, "elbow_flex": 0.0}, 4.0, 0.02, linear_tip)

    assert np.linalg.norm(linear_tip(moved) - linear_tip(previous)) == pytest.approx(0.02)
    assert limited
