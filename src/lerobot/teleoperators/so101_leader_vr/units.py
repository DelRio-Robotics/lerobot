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
Converts between the joint angles the IK works in (degrees, zero at the middle of each joint's range, as in
`so101_new_calib.urdf`) and the units the follower expects, using the follower's calibration file.

The follower's body joints are in -100..100 by default (`MotorNormMode.RANGE_M100_100`), or in degrees with
`--robot.use_degrees=true`. The gripper is always 0..100.
"""

from pathlib import Path

import draccus

from lerobot.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from lerobot.motors import MotorCalibration

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER = "gripper"
MOTORS = (*ARM_JOINTS, GRIPPER)

# STS3215 resolution, as in `FeetechMotorsBus.model_resolution_table`.
_MAX_RES = 4096 - 1


def _half_range_deg(cal: MotorCalibration) -> float:
    return (cal.range_max - cal.range_min) / 2 * 360 / _MAX_RES


def normalized_to_degrees(value: float, cal: MotorCalibration) -> float:
    """-100..100 to degrees from the middle of the range, the inverse of `MotorsBus._normalize`."""
    value = -value if cal.drive_mode else value
    return value / 100 * _half_range_deg(cal)


def degrees_to_normalized(degrees: float, cal: MotorCalibration) -> float:
    value = degrees / _half_range_deg(cal) * 100
    return -value if cal.drive_mode else value


def to_follower_units(
    angles: dict[str, float],
    gripper: float,
    calibration: dict[str, MotorCalibration],
    margin: float,
    use_degrees: bool = False,
) -> dict[str, float]:
    """
    Turns arm angles (degrees) and a gripper value (0..100) into a follower action. Body joints are kept
    `margin` normalized units inside the calibrated range, so the arm never drives into its end stops.
    """
    limit = 100 - margin
    action = {}
    for motor in ARM_JOINTS:
        value = degrees_to_normalized(angles[motor], calibration[motor])
        value = min(max(value, -limit), limit)
        action[f"{motor}.pos"] = normalized_to_degrees(value, calibration[motor]) if use_degrees else value
    action[f"{GRIPPER}.pos"] = min(max(gripper, 0.0), 100.0)
    return action


def from_follower_units(
    observation: dict[str, float], calibration: dict[str, MotorCalibration], use_degrees: bool = False
) -> tuple[dict[str, float], float]:
    """Arm angles (degrees) and gripper value (0..100) from the follower's joint readings."""
    angles = {}
    for motor in ARM_JOINTS:
        value = observation[f"{motor}.pos"]
        angles[motor] = value if use_degrees else normalized_to_degrees(value, calibration[motor])
    return angles, observation[f"{GRIPPER}.pos"]


def load_follower_calibration(
    follower_id: str, calibration_dir: Path | None = None
) -> dict[str, MotorCalibration]:
    """Reads the follower's calibration file, from the same place lerobot keeps it on the follower's computer."""
    calibration_dir = calibration_dir or HF_LEROBOT_CALIBRATION / ROBOTS / "so101_follower"
    fpath = Path(calibration_dir) / f"{follower_id}.json"
    if not fpath.is_file():
        raise FileNotFoundError(
            f"No follower calibration at {fpath}. Copy it from the follower's computer, e.g.:\n"
            f"  scp pop-os:.cache/huggingface/lerobot/calibration/robots/so101_follower/{follower_id}.json "
            f"{fpath}"
        )
    with open(fpath) as f, draccus.config_type("json"):
        calibration = draccus.load(dict[str, MotorCalibration], f)
    missing = [motor for motor in MOTORS if motor not in calibration]
    if missing:
        raise ValueError(f"{fpath} has no calibration for {missing}.")
    return calibration
