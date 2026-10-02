"""How the virtual leader turns headset input into follower actions (no network, real kinematics)."""

import json
import math

import numpy as np
import pytest

pytest.importorskip("placo")

from lerobot.teleoperators.so101_leader_vr.config_so101_leader_vr import SO101LeaderVRConfig  # noqa: E402
from lerobot.teleoperators.so101_leader_vr.controller import VRArmController, XRInput  # noqa: E402
from lerobot.teleoperators.so101_leader_vr.kinematics import ArmKinematics  # noqa: E402
from lerobot.teleoperators.so101_leader_vr.units import from_follower_units, to_follower_units  # noqa: E402
from tests.fixtures.so101_follower_calibration import FOLLOWER_CALIBRATION  # noqa: E402

KINEMATICS = ArmKinematics()
START = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -30.0,
    "elbow_flex": 40.0,
    "wrist_flex": 35.0,
    "wrist_roll": 0.0,
}
OBSERVATION = to_follower_units(START, 25.0, FOLLOWER_CALIBRATION, margin=0.0)
STILL = (0.0, 0.0, 0.0, 1.0)
XR_FORWARD, XR_UP, XR_RIGHT, XR_LEFT = (0, 0, -1.0), (0, 1.0, 0), (1.0, 0, 0), (-1.0, 0, 0)
DT = 1 / 30


def xr(pos=(0, 0, 0), grip=True, trigger=0.0, precision=False, recenter=False, head=STILL, at=0.0):
    return XRInput(
        position=np.array(pos, dtype=float),
        orientation=np.array(STILL),
        head_orientation=np.array(head, dtype=float),
        trigger=trigger,
        grip=grip,
        precision=precision,
        recenter=recenter,
        received_at=at,
    )


def tip_of(action):
    angles, _ = from_follower_units(action, FOLLOWER_CALIBRATION)
    return KINEMATICS.fk(angles)[:3, 3]


def quat_about_y(degrees):
    half = math.radians(degrees) / 2
    return (0.0, math.sin(half), 0.0, math.cos(half))


class Operator:
    """Feeds the controller headset samples 30 times a second."""

    def __init__(self, **config):
        self.controller = VRArmController(
            SO101LeaderVRConfig(follower_id="test", **config), FOLLOWER_CALIBRATION, KINEMATICS
        )
        self.controller.observe(OBSERVATION)
        self.t = 0.0

    def send(self, **sample):
        self.t += DT
        return self.controller.step(xr(at=self.t, **sample), now=self.t)

    def move(self, direction, distance, start=(0, 0, 0), steps=20, settle=10, **sample):
        """Moves the hand from `start` by `distance` metres along `direction` with grip held, then holds it."""
        action = None
        for i in range(steps + settle + 1):
            pos = np.array(start) + np.array(direction) * distance * min(i, steps) / steps
            action = self.send(pos=pos, **sample)
        return action


def test_holds_the_measured_pose_until_grip_is_pressed():
    operator = Operator()

    assert operator.controller.step(None, now=0.0) == OBSERVATION
    assert operator.send(grip=False, trigger=1.0) == OBSERVATION


def test_step_before_any_observation_fails():
    controller = VRArmController(SO101LeaderVRConfig(follower_id="test"), FOLLOWER_CALIBRATION, KINEMATICS)

    with pytest.raises(RuntimeError, match="observation"):
        controller.step(None, now=0.0)


@pytest.mark.parametrize(
    "hand_direction, robot_direction",
    [(XR_FORWARD, (1, 0, 0)), (XR_UP, (0, 0, 1)), (XR_RIGHT, (0, -1, 0)), (XR_LEFT, (0, 1, 0))],
)
def test_the_tip_follows_the_hand(hand_direction, robot_direction):
    operator = Operator()
    start = tip_of(OBSERVATION)

    action = operator.move(hand_direction, 0.05)

    assert tip_of(action) == pytest.approx(start + 0.05 * np.array(robot_direction), abs=3e-3)


def test_motion_scale_scales_the_hand_motion():
    operator = Operator(motion_scale=0.5)
    start = tip_of(OBSERVATION)

    action = operator.move(XR_FORWARD, 0.06)

    assert tip_of(action) == pytest.approx(start + [0.03, 0, 0], abs=3e-3)


def test_holding_precision_halves_the_motion():
    operator = Operator()
    start = tip_of(OBSERVATION)

    action = operator.move(XR_FORWARD, 0.06, precision=True)

    assert tip_of(action) == pytest.approx(start + [0.03, 0, 0], abs=3e-3)


def test_switching_precision_mid_move_does_not_jump():
    operator = Operator()
    before = operator.move(XR_FORWARD, 0.04)

    after = operator.send(pos=np.array(XR_FORWARD) * 0.04, precision=True)

    assert after == pytest.approx(before, abs=0.05)


def test_releasing_grip_holds_the_last_commanded_pose():
    operator = Operator()
    moved = operator.move(XR_FORWARD, 0.05)

    assert operator.send(pos=(0.3, 0.2, 0.1), grip=False) == moved


def test_pressing_grip_again_continues_from_where_the_arm_is():
    operator = Operator()
    moved = operator.move(XR_FORWARD, 0.05)
    operator.send(pos=(0.3, 0.2, 0.1), grip=False)

    assert operator.send(pos=(0.3, 0.2, 0.1)) == pytest.approx(moved, abs=0.05)


def test_stale_headset_data_holds_until_grip_is_pressed_again():
    operator = Operator()
    moved = operator.move(XR_FORWARD, 0.05)

    late = operator.controller.step(xr(pos=(0, 0, -0.1), at=operator.t), now=operator.t + 0.5)
    assert late == moved
    assert operator.controller.status["stale"]

    assert operator.send(pos=(0, 0.1, -0.1)) == moved  # grip still held from before the dropout
    operator.send(pos=(0, 0.1, -0.1), grip=False)
    resumed = operator.move(XR_UP, 0.03, start=(0, 0.1, -0.1))  # a fresh press: following resumes from here
    assert tip_of(resumed)[2] == pytest.approx(tip_of(moved)[2] + 0.03, abs=3e-3)


def test_gripper_follows_the_trigger_after_the_first_grip():
    operator = Operator()
    operator.send()
    operator.send(grip=False)

    assert operator.send(grip=False, trigger=0.5)["gripper.pos"] == pytest.approx(50.0)
    assert operator.send(grip=False, trigger=1.0)["gripper.pos"] == pytest.approx(0.0)


def test_the_tip_stops_at_the_floor():
    start = tip_of(OBSERVATION)
    floor = start[2] - 0.02
    operator = Operator(ee_bounds_min=[-0.45, -0.45, floor])

    action = operator.move((0, -1.0, 0), 0.1)

    assert tip_of(action)[2] == pytest.approx(floor, abs=3e-3)
    assert operator.controller.status["limited"]


def test_starting_below_the_floor_holds_still_then_can_rise():
    start = tip_of(OBSERVATION)
    operator = Operator(ee_bounds_min=[-0.45, -0.45, start[2] + 0.03])

    held = operator.move(XR_FORWARD, 0.0)
    assert tip_of(held)[2] == pytest.approx(start[2], abs=1e-3)

    risen = operator.move(XR_UP, 0.05)
    assert tip_of(risen)[2] == pytest.approx(start[2] + 0.05, abs=3e-3)


def test_recentring_makes_the_operators_new_facing_robot_forward():
    operator = Operator()
    start = tip_of(OBSERVATION)
    operator.send(grip=False, recenter=True, head=quat_about_y(90))  # turned left, now facing XR -x

    action = operator.move(XR_LEFT, 0.05)

    assert tip_of(action) == pytest.approx(start + [0.05, 0, 0], abs=3e-3)


def test_status_reports_engagement():
    operator = Operator()

    operator.send()
    assert operator.controller.status["engaged"]

    operator.send(grip=False)
    assert not operator.controller.status["engaged"]


class DivingKinematics(ArmKinematics):
    """IK that lowers the shoulder far past the solution, as a bad solve near the floor might."""

    def ik(self, seed, target, orientation_weight, iterations):
        angles = super().ik(seed, target, orientation_weight, iterations)
        return {**angles, "shoulder_lift": angles["shoulder_lift"] + 30}


def test_ik_solutions_that_dip_below_the_floor_are_refused():
    start = tip_of(OBSERVATION)
    config = SO101LeaderVRConfig(follower_id="test", ee_bounds_min=[-0.45, -0.45, start[2] - 0.01])
    controller = VRArmController(config, FOLLOWER_CALIBRATION, DivingKinematics())
    controller.observe(OBSERVATION)

    action = controller.step(xr(pos=(0, 0, -0.01), at=0.0), now=0.0)

    arm = [key for key in OBSERVATION if key != "gripper.pos"]
    assert {key: action[key] for key in arm} == {key: OBSERVATION[key] for key in arm}
    assert controller.status["limited"]


@pytest.mark.parametrize("direction, at_limit", [(XR_FORWARD, False), ((0, -1.0, 0), True)])
def test_status_is_plain_json_while_moving_and_at_limits(direction, at_limit):
    start = tip_of(OBSERVATION)
    operator = Operator(ee_bounds_min=[-0.45, -0.45, start[2] - 0.02])

    operator.move(direction, 0.1 if at_limit else 0.02)  # down 10 cm runs into the floor

    status = operator.controller.status
    assert status["limited"] == at_limit
    assert all(type(value) is bool for value in status.values())
    assert json.loads(json.dumps(status)) == status
