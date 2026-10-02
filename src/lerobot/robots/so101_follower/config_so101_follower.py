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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("so101_follower")
@dataclass
class SO101FollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False


@dataclass
class SO101FollowerHostConfig:
    """Runs on the computer wired to the follower arm. See `so101_follower_host.py`."""

    robot: SO101FollowerConfig

    # Interface to listen on. "*" listens everywhere; pass a Tailscale/VPN IP to listen only there.
    bind_ip: str = "*"
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    # Log a warning when no command arrives for this long. The arm holds its last goal position.
    watchdog_timeout_ms: int = 500

    # The arm is moved toward the latest command this many times a second, easing in over `smoothing_ms`.
    # Smoothing hides network jitter (late or bunched-up commands) at the cost of about that much extra lag.
    # 0 applies each command as-is.
    control_freq_hz: int = 100
    smoothing_ms: int = 25

    # Max rate of observations (joints + camera frames) sent back. Commands are applied as they arrive regardless.
    max_loop_freq_hz: int = 30

    # Lower this to save upload bandwidth (each 640x480 frame is ~30-60 KB at 80).
    jpeg_quality: int = 80

    # Stop after this many seconds. None runs until Ctrl+C.
    connection_time_s: float | None = None


@RobotConfig.register_subclass("so101_follower_client")
@dataclass
class SO101FollowerClientConfig(RobotConfig):
    """Stands in for a follower arm attached to another computer running `so101_follower_host`."""

    # IP or hostname of the computer running the host
    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    # Must use the same names, width and height as the host's `--robot.cameras`. Only the shapes are used
    # here (to declare dataset features); `index_or_path` is ignored.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # How long `get_observation()` waits for a fresh observation before returning the last one.
    polling_timeout_ms: int = 15
    connect_timeout_s: int = 10
