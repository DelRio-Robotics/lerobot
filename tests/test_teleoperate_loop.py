"""The teleop loop hands the robot's observation to teleops that ask for it (`on_observation`), and runs
exactly as before for every other teleop."""

import pytest

import lerobot.teleoperate as teleoperate_module
from lerobot.teleoperate import teleop_loop

ACTION = {"shoulder_pan.pos": 1.0}
OBSERVATION = {"shoulder_pan.pos": 2.0}


class FakeRobot:
    def __init__(self, calls: list):
        self.calls = calls

    @property
    def action_features(self) -> dict:
        return {"shoulder_pan.pos": float}

    def get_observation(self) -> dict:
        self.calls.append("robot.get_observation")
        return dict(OBSERVATION)

    def send_action(self, action: dict) -> dict:
        self.calls.append("robot.send_action")
        return action


class FakeTeleop:
    def __init__(self, calls: list):
        self.calls = calls

    def get_action(self) -> dict:
        self.calls.append("teleop.get_action")
        return dict(ACTION)


class FakeObservingTeleop(FakeTeleop):
    def __init__(self, calls: list):
        super().__init__(calls)
        self.observations = []

    def on_observation(self, observation: dict) -> None:
        self.calls.append("teleop.on_observation")
        self.observations.append(observation)


@pytest.fixture
def rerun_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(teleoperate_module, "log_rerun_data", lambda obs, act: calls.append((obs, act)))
    return calls


def run_one_tick(teleop, robot, display_data=False):
    teleop_loop(teleop, robot, fps=1000, display_data=display_data, duration=0)


def test_observing_teleop_gets_the_observation_before_its_action():
    calls = []
    teleop = FakeObservingTeleop(calls)

    run_one_tick(teleop, FakeRobot(calls))

    assert calls == [
        "robot.get_observation",
        "teleop.on_observation",
        "teleop.get_action",
        "robot.send_action",
    ]
    assert teleop.observations == [OBSERVATION]


def test_other_teleops_never_trigger_an_observation_read():
    calls = []

    run_one_tick(FakeTeleop(calls), FakeRobot(calls))

    assert calls == ["teleop.get_action", "robot.send_action"]


def test_other_teleops_keep_the_display_data_order(rerun_calls):
    calls = []

    run_one_tick(FakeTeleop(calls), FakeRobot(calls), display_data=True)

    assert calls == ["teleop.get_action", "robot.get_observation", "robot.send_action"]
    assert rerun_calls == [(OBSERVATION, ACTION)]


def test_observing_teleop_with_display_data_reads_the_observation_once(rerun_calls):
    calls = []

    run_one_tick(FakeObservingTeleop(calls), FakeRobot(calls), display_data=True)

    assert calls.count("robot.get_observation") == 1
    assert rerun_calls == [(OBSERVATION, ACTION)]
