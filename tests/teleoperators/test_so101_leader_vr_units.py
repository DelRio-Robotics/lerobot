"""Unit conversions of the virtual leader must agree with how the follower's motor bus normalizes positions."""

import json

import draccus
import pytest

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.teleoperators.so101_leader_vr.units import (
    MOTORS,
    degrees_to_normalized,
    from_follower_units,
    load_follower_calibration,
    normalized_to_degrees,
    to_follower_units,
)
from tests.fixtures.so101_follower_calibration import FOLLOWER_CALIBRATION

CALIBRATION = FOLLOWER_CALIBRATION


def make_bus(norm_mode: MotorNormMode, calibration: dict) -> FeetechMotorsBus:
    motors = {name: Motor(cal.id, "sts3215", norm_mode) for name, cal in calibration.items()}
    return FeetechMotorsBus(port="/dev/null", motors=motors, calibration=calibration)


@pytest.mark.parametrize("drive_mode", [0, 1])
@pytest.mark.parametrize("motor", ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])
def test_normalized_to_degrees_matches_the_bus(motor, drive_mode):
    cal = {motor: MotorCalibration(**{**vars(CALIBRATION[motor]), "drive_mode": drive_mode})}
    range_bus = make_bus(MotorNormMode.RANGE_M100_100, cal)
    degrees_bus = make_bus(MotorNormMode.DEGREES, cal)
    motor_id = cal[motor].id

    # For the same raw position, our degrees must be what the follower reports with --robot.use_degrees=true.
    for raw in range(cal[motor].range_min, cal[motor].range_max + 1, 97):
        normalized = range_bus._normalize({motor_id: raw})[motor_id]
        degrees = degrees_bus._normalize({motor_id: raw})[motor_id]
        assert normalized_to_degrees(normalized, cal[motor]) == pytest.approx(degrees, abs=1e-9)


def test_degrees_to_normalized_inverts_normalized_to_degrees():
    for cal in CALIBRATION.values():
        for value in (-100.0, -37.5, 0.0, 12.25, 100.0):
            assert degrees_to_normalized(normalized_to_degrees(value, cal), cal) == pytest.approx(value)


def test_to_follower_units_clips_body_joints_inside_the_margin():
    huge = dict.fromkeys(MOTORS[:-1], 1000.0)

    values = to_follower_units(huge, gripper=50.0, calibration=CALIBRATION, margin=3.0)

    assert all(values[f"{motor}.pos"] == pytest.approx(97.0) for motor in MOTORS[:-1])


def test_to_follower_units_clips_the_gripper_to_its_range():
    zeros = dict.fromkeys(MOTORS[:-1], 0.0)

    assert to_follower_units(zeros, 150.0, CALIBRATION, margin=3.0)["gripper.pos"] == 100.0
    assert to_follower_units(zeros, -5.0, CALIBRATION, margin=3.0)["gripper.pos"] == 0.0


def test_to_follower_units_in_degrees_mode_keeps_degrees():
    angles = dict.fromkeys(MOTORS[:-1], 10.0)

    values = to_follower_units(angles, 20.0, CALIBRATION, margin=3.0, use_degrees=True)

    assert values["shoulder_lift.pos"] == pytest.approx(10.0)
    assert values["gripper.pos"] == 20.0


def test_from_follower_units_round_trips_with_to_follower_units():
    observation = {"shoulder_pan.pos": -40.0, "shoulder_lift.pos": 10.0, "elbow_flex.pos": 55.5}
    observation |= {"wrist_flex.pos": -3.0, "wrist_roll.pos": 80.0, "gripper.pos": 30.0}

    angles, gripper = from_follower_units(observation, CALIBRATION)

    assert to_follower_units(angles, gripper, CALIBRATION, margin=0.0) == pytest.approx(observation)


def test_load_follower_calibration_reads_lerobots_calibration_file(tmp_path):
    with open(tmp_path / "arm.json", "w") as f, draccus.config_type("json"):
        draccus.dump(CALIBRATION, f, indent=4)

    assert load_follower_calibration("arm", tmp_path) == CALIBRATION


def test_load_follower_calibration_explains_how_to_get_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="scp"):
        load_follower_calibration("missing_arm", tmp_path)


def test_load_follower_calibration_rejects_a_file_without_every_motor(tmp_path):
    (tmp_path / "arm.json").write_text(json.dumps({"gripper": vars(CALIBRATION["gripper"])}))

    with pytest.raises(ValueError, match="shoulder_pan"):
        load_follower_calibration("arm", tmp_path)
