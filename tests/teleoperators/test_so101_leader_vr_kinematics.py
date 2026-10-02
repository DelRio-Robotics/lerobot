"""The virtual leader's kinematics: the packaged SO-101 model, forward and inverse kinematics."""

import numpy as np
import pytest

pytest.importorskip("placo")

from lerobot.teleoperators.so101_leader_vr.kinematics import DEFAULT_URDF, ArmKinematics  # noqa: E402

REACHABLE = {"shoulder_pan": 15.0, "shoulder_lift": -30.0, "elbow_flex": 40.0, "wrist_flex": 35.0}
REACHABLE |= {"wrist_roll": -20.0}


@pytest.fixture(scope="module")
def kinematics():
    return ArmKinematics()


def test_packaged_urdf_needs_no_mesh_files():
    assert "<mesh" not in DEFAULT_URDF.read_text()


def test_robot_forward_is_plus_x_in_the_packaged_model(kinematics):
    assert kinematics.forward_axis() == pytest.approx(np.array([1.0, 0.0, 0.0]), abs=1e-3)


def test_forward_kinematics_puts_the_rest_pose_near_the_table(kinematics):
    rest = {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 95.0, "wrist_flex": 70.0}
    rest |= {"wrist_roll": 0.0}

    tip = kinematics.fk(rest)[:3, 3]

    assert 0.1 < tip[0] < 0.25
    assert abs(tip[2]) < 0.03


def test_inverse_kinematics_reaches_a_position_and_approach(kinematics):
    pose = kinematics.fk(REACHABLE)
    seed = {joint: angle + 8.0 for joint, angle in REACHABLE.items()}
    seed["wrist_roll"] = REACHABLE["wrist_roll"]

    solution = kinematics.ik(seed, pose[:3, 3], pose[:3, 2], approach_weight=0.1, iterations=30)

    reached = kinematics.fk(solution)
    assert np.linalg.norm(reached[:3, 3] - pose[:3, 3]) < 1e-3
    assert np.degrees(np.arccos(np.clip(reached[:3, 2] @ pose[:3, 2], -1, 1))) < 1.0
    assert solution == pytest.approx(REACHABLE, abs=0.5)


def test_inverse_kinematics_keeps_the_wrist_roll_of_its_seed(kinematics):
    pose = kinematics.fk(REACHABLE)
    seed = {**REACHABLE, "wrist_roll": 55.0}

    solution = kinematics.ik(seed, pose[:3, 3] + [0.02, 0, 0], pose[:3, 2], 0.1, 10)

    assert solution["wrist_roll"] == 55.0


def test_inverse_kinematics_does_not_change_its_seed(kinematics):
    seed = dict(REACHABLE)

    kinematics.ik(seed, np.array([0.25, 0.0, 0.1]), np.array([1.0, 0, 0]), 0.1, 5)

    assert seed == REACHABLE


def test_roll_sign_turns_the_tip_about_its_approach(kinematics):
    before = kinematics.fk(REACHABLE)[:3, :3]
    after = kinematics.fk({**REACHABLE, "wrist_roll": REACHABLE["wrist_roll"] + kinematics.roll_sign() * 10})[
        :3, :3
    ]

    turn = before.T @ after  # in the tip's own frame: a rotation about its z (approach) axis
    assert turn[2, 2] == pytest.approx(1.0, abs=1e-6)
    assert np.degrees(np.arctan2(turn[1, 0], turn[0, 0])) == pytest.approx(10.0, abs=1e-3)


def test_turning_the_pan_keeps_the_tip_at_the_same_distance_from_the_pan_axis(kinematics):
    axis = kinematics.pan_axis()
    turned = {**REACHABLE, "shoulder_pan": REACHABLE["shoulder_pan"] + 40}

    radius = np.linalg.norm(kinematics.fk(REACHABLE)[:2, 3] - axis)

    assert np.linalg.norm(kinematics.fk(turned)[:2, 3] - axis) == pytest.approx(radius, abs=1e-6)


def test_inverse_kinematics_clamps_a_wrist_roll_beyond_its_limit(kinematics):
    pose = kinematics.fk(REACHABLE)
    low, high = kinematics.joint_limits("wrist_roll")

    solution = kinematics.ik({**REACHABLE, "wrist_roll": high + 30}, pose[:3, 3], pose[:3, 2], 0.1, 5)

    assert low <= solution["wrist_roll"] <= high


def test_limit_joints_keeps_solutions_inside_the_given_range():
    kinematics = ArmKinematics()
    kinematics.limit_joints({"elbow_flex": (-90.0, 20.0)})
    pose = kinematics.fk(REACHABLE)  # needs the elbow at 40

    solution = kinematics.ik({**REACHABLE, "elbow_flex": 0.0}, pose[:3, 3], pose[:3, 2], 0.1, 30)

    assert kinematics.joint_limits("elbow_flex") == pytest.approx((-90.0, 20.0))
    assert solution["elbow_flex"] <= 20.0 + 1e-6


def test_limit_joints_never_widens_past_the_model():
    kinematics = ArmKinematics()
    model = kinematics.joint_limits("wrist_flex")

    kinematics.limit_joints({"wrist_flex": (-10.0, 10.0)})
    kinematics.limit_joints({"wrist_flex": (-500.0, 500.0)})

    assert kinematics.joint_limits("wrist_flex") == pytest.approx(model)
