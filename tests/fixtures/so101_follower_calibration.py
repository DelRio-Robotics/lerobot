from lerobot.motors import MotorCalibration

# The real follower calibration of this setup (my_awesome_follower_arm, 2026-10-01).
FOLLOWER_CALIBRATION = {
    "shoulder_pan": MotorCalibration(id=1, drive_mode=0, homing_offset=-357, range_min=721, range_max=3176),
    "shoulder_lift": MotorCalibration(id=2, drive_mode=0, homing_offset=1300, range_min=806, range_max=3164),
    "elbow_flex": MotorCalibration(id=3, drive_mode=0, homing_offset=-891, range_min=906, range_max=3108),
    "wrist_flex": MotorCalibration(id=4, drive_mode=0, homing_offset=-159, range_min=798, range_max=3102),
    "wrist_roll": MotorCalibration(id=5, drive_mode=0, homing_offset=-838, range_min=131, range_max=3964),
    "gripper": MotorCalibration(id=6, drive_mode=0, homing_offset=382, range_min=1664, range_max=3122),
}
