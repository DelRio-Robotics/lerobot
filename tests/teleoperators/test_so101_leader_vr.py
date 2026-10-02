"""The virtual leader as a lerobot teleoperator: registration, connect/disconnect and the headset round trip."""

import asyncio
import json
import subprocess
import sys
import time

import draccus
import numpy as np
import pytest

pytest.importorskip("placo")
aiohttp = pytest.importorskip("aiohttp")

import lerobot.teleoperators.so101_leader_vr.so101_leader_vr as so101_leader_vr_module  # noqa: E402
from lerobot.teleoperators import make_teleoperator_from_config  # noqa: E402
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig  # noqa: E402
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader  # noqa: E402
from lerobot.teleoperators.so101_leader_vr import SO101LeaderVRConfig  # noqa: E402
from lerobot.teleoperators.so101_leader_vr.kinematics import ArmKinematics  # noqa: E402
from lerobot.teleoperators.so101_leader_vr.so101_leader_vr import SO101LeaderVR  # noqa: E402
from lerobot.teleoperators.so101_leader_vr.units import from_follower_units, to_follower_units  # noqa: E402
from tests.fixtures.so101_follower_calibration import FOLLOWER_CALIBRATION  # noqa: E402

START = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -30.0,
    "elbow_flex": 40.0,
    "wrist_flex": 35.0,
    "wrist_roll": 0.0,
}
JOINTS = to_follower_units(START, 25.0, FOLLOWER_CALIBRATION, margin=0.0)
FRAME = np.full((24, 32, 3), 128, dtype=np.uint8)
OBSERVATION = {**JOINTS, "front": FRAME}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(so101_leader_vr_module, "STATE_DIR", tmp_path / "state")
    calibration_dir = tmp_path / "follower_calibration"
    calibration_dir.mkdir()
    with open(calibration_dir / "follower.json", "w") as f, draccus.config_type("json"):
        draccus.dump(FOLLOWER_CALIBRATION, f)
    client_dir = tmp_path / "vr_client"
    client_dir.mkdir()
    (client_dir / "index.html").write_text("<html></html>")

    def make(**overrides):
        settings = {
            "follower_id": "follower",
            "follower_calibration_dir": calibration_dir,
            "client_dir": client_dir,
            "calibration_dir": tmp_path / "teleop_calibration",
            "https": False,
            "http_port": 0,
            "stream_fps": 50,
        }
        return SO101LeaderVR(SO101LeaderVRConfig(**{**settings, **overrides}))

    return make


def tip_of(action):
    angles, _ = from_follower_units(action, FOLLOWER_CALIBRATION)
    return ArmKinematics().fk(angles)[:3, 3]


async def operate(teleop, steps):
    """Connects like the headset page, then for each step sends a sample and runs one teleop-loop tick."""
    url = f"http://127.0.0.1:{teleop.server.port}/ws?k={teleop.server.token}"
    actions, frames = [], []
    async with aiohttp.ClientSession() as session, session.ws_connect(url) as ws:
        await ws.receive_json()
        for i, pos in enumerate(steps):
            sample = {"type": "xr", "pos": list(pos), "quat": [0, 0, 0, 1], "head": [0, 0, 0, 1]}
            await ws.send_str(json.dumps({**sample, "trigger": 0.0, "grip": True, "a": False, "b": False}))
            deadline = time.monotonic() + 1
            while teleop.server.latest() is None or list(teleop.server.latest().position) != list(pos):
                assert time.monotonic() < deadline
                await asyncio.sleep(0.005)
            teleop.on_observation(OBSERVATION)
            actions.append(teleop.get_action())
            if i == 0:
                frames.append(await asyncio.wait_for(ws.receive_bytes(), timeout=2))
            await asyncio.sleep(0.005)
    return actions, frames


def test_importing_the_package_loads_no_heavy_dependencies():
    code = (
        "import sys, lerobot.teleoperators.so101_leader_vr; "
        "print(sorted(m for m in ('placo', 'aiohttp', 'cryptography') if m in sys.modules))"
    )

    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout

    assert output.strip() == "[]"


def test_teleoperator_factory_builds_the_virtual_leader(setup, tmp_path):
    config = SO101LeaderVRConfig(follower_id="follower", calibration_dir=tmp_path)

    assert isinstance(make_teleoperator_from_config(config), SO101LeaderVR)


def test_actions_have_the_same_keys_as_the_leader_arm(setup, tmp_path):
    leader = SO101Leader(SO101LeaderConfig(port="/dev/null", calibration_dir=tmp_path))

    assert setup().action_features == leader.action_features


def test_rejects_an_unknown_hand():
    with pytest.raises(ValueError, match="hand"):
        SO101LeaderVRConfig(follower_id="follower", hand="middle")


def test_connect_explains_a_missing_follower_calibration(setup):
    teleop = setup(follower_id="unknown")

    with pytest.raises(FileNotFoundError, match="scp"):
        teleop.connect()
    assert not teleop.is_connected


def test_connect_explains_a_missing_headset_page(setup, tmp_path):
    teleop = setup(client_dir=tmp_path / "nowhere")

    with pytest.raises(FileNotFoundError, match="client_dir"):
        teleop.connect()


def test_get_action_before_any_observation_says_what_is_needed(setup):
    teleop = setup()
    teleop.connect()
    try:
        with pytest.raises(RuntimeError, match="lerobot-teleoperate"):
            teleop.get_action()
    finally:
        teleop.disconnect()


def test_holds_still_without_a_headset(setup):
    teleop = setup()
    teleop.connect()
    try:
        teleop.on_observation(OBSERVATION)
        assert teleop.get_action() == JOINTS
    finally:
        teleop.disconnect()
    assert not teleop.is_connected


def test_the_arm_follows_the_headset_and_frames_reach_it(setup):
    teleop = setup()
    teleop.connect()
    try:
        forward = [(0, 0, -0.002 * i) for i in range(26)]  # 5 cm forward, 2 mm per tick
        actions, frames = asyncio.run(operate(teleop, forward + [forward[-1]] * 10))
    finally:
        teleop.disconnect()

    assert tip_of(actions[-1]) == pytest.approx(tip_of(JOINTS) + [0.05, 0, 0], abs=3e-3)
    assert frames[0][1 : 1 + frames[0][0]] == b"front"


def test_dry_run_never_moves_the_arm(setup):
    teleop = setup(dry_run=True)
    teleop.connect()
    try:
        forward = [(0, 0, -0.002 * i) for i in range(26)]
        actions, _ = asyncio.run(operate(teleop, forward))
    finally:
        teleop.disconnect()

    assert all(action == JOINTS for action in actions)


def test_status_says_when_the_robots_video_stops(setup):
    teleop = setup()
    teleop.connect()
    try:
        now = [0.0]
        teleop._clock = lambda: now[0]
        frame = np.zeros((24, 32, 3), dtype=np.uint8)

        teleop.on_observation({**JOINTS, "front": frame})
        teleop.get_action()
        assert teleop.status["video_stale"] is False

        now[0] = 1.0  # the robot client hands back the same frame: nothing new arrived
        teleop.on_observation({**JOINTS, "front": frame})
        teleop.get_action()
        assert teleop.status["video_stale"] is True

        now[0] = 1.1
        teleop.on_observation({**JOINTS, "front": frame.copy()})
        teleop.get_action()
        assert teleop.status["video_stale"] is False
    finally:
        teleop.disconnect()


def test_without_cameras_the_video_is_never_reported_stale(setup):
    teleop = setup()
    teleop.connect()
    try:
        now = [0.0]
        teleop._clock = lambda: now[0]
        teleop.on_observation(JOINTS)
        now[0] = 5.0
        teleop.on_observation(JOINTS)
        teleop.get_action()
        assert teleop.status["video_stale"] is False
    finally:
        teleop.disconnect()
