"""Unit tests for GymEnvironment global world-session handling."""

import unittest
from threading import Lock
from typing import Any, cast

from duckietown.sdk.robots.duckiebot import DB21M

from gym_duckiematrix.gym_environment import GymEnvironment


class _FakeWorldInput:
    def __init__(self) -> None:
        self.callback = None
        self.current_session_id = None
        self.started = 0
        self.stopped = 0

    def attach(self, callback) -> None:
        self.callback = callback

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1


class _FakeWorldOutput:
    def __init__(self) -> None:
        self.published: list = []
        self.started = 0
        self.stopped = 0

    def publish(self, message) -> None:
        self.published.append(message)

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1


class _FakeVehicle(DB21M):
    def __init__(self, name: str) -> None:
        super().__init__(name, simulated=True, gym_mode=True)


def _make_environment() -> GymEnvironment:
    environment = GymEnvironment.__new__(GymEnvironment)
    environment._all_names = ["vehicle_a", "vehicle_b", "watchtower"]
    environment._callback = None
    environment._last_completed_session_id = None
    environment._lock = Lock()
    environment._static_names = ["watchtower"]
    environment._vehicle_names = ["vehicle_a", "vehicle_b"]
    environment._vehicles = {
        "vehicle_a": _FakeVehicle("vehicle_a"),
        "vehicle_b": _FakeVehicle("vehicle_b"),
    }
    environment._world_input = _FakeWorldInput()
    environment._world_output = _FakeWorldOutput()
    return environment


class GymEnvironmentSessionTests(unittest.TestCase):
    def test_deduplicates_completed_world_sessions(self) -> None:
        environment = _make_environment()
        callbacks: list[dict] = []
        environment._callback = callbacks.append

        environment._world_input_callback({"session_id": 2, "payload": "a"})
        environment._world_input_callback({"session_id": 2, "payload": "b"})
        environment._world_input_callback(
            {"session_id": 1, "payload": "stale"},
        )
        environment._world_input_callback({"session_id": 3, "payload": "c"})

        self.assertEqual(
            callbacks,
            [
                {"session_id": 2, "payload": "a"},
                {"session_id": 3, "payload": "c"},
            ],
        )
        self.assertEqual(environment._last_completed_session_id, 3)

    def test_rejects_world_input_without_session_id(self) -> None:
        environment = _make_environment()

        with self.assertRaisesRegex(
            ValueError,
            "missing the required session_id",
        ):
            environment._world_input_callback({"payload": "vehicle-a"})

    def test_step_publishes_one_global_world_output(self) -> None:
        environment = _make_environment()
        environment._world_input.current_session_id = 7

        environment.step(
            {
                "vehicle_a": (0.5, 0.4),
                "vehicle_b": (0.4, 0.5),
                "watchtower": (1.0, 1.0),
            },
        )

        self.assertEqual(len(environment._world_output.published), 1)
        message = environment._world_output.published[0]
        self.assertEqual(message["session_id"], 7)
        entities = message["entities"]
        self.assertEqual(
            set(entities),
            {"vehicle_a", "vehicle_b"},
        )
        self.assertEqual(
            entities["vehicle_a"]["differential_pwm"]["left"],
            0.5,
        )
        self.assertEqual(
            entities["vehicle_a"]["differential_pwm"]["right"],
            0.4,
        )
        self.assertEqual(
            entities["vehicle_b"]["differential_pwm"]["left"],
            0.4,
        )
        self.assertEqual(
            entities["vehicle_b"]["differential_pwm"]["right"],
            0.5,
        )
        self.assertEqual(
            entities["vehicle_a"]["differential_pwm"]["left"],
            0.5,
        )

    def test_step_uses_vehicle_entity_output_builder(self) -> None:
        environment = _make_environment()
        environment._world_input.current_session_id = 11

        environment.step({"vehicle_a": (0.2, 0.1)})

        message = environment._world_output.published[0]
        entities = message["entities"]
        self.assertEqual(
            entities["vehicle_a"]["differential_pwm"]["left"],
            0.2,
        )
        self.assertEqual(
            entities["vehicle_a"]["differential_pwm"]["right"],
            0.1,
        )

    def test_step_requires_a_current_session(self) -> None:
        environment = _make_environment()

        with self.assertRaisesRegex(RuntimeError, "with a session_id"):
            environment.step({"vehicle_a": (0.5, 0.4)})

    def test_step_rejects_non_tuple_actions(self) -> None:
        environment = _make_environment()
        environment._world_input.current_session_id = 9

        with self.assertRaisesRegex(TypeError, "expects each action"):
            environment.step({"vehicle_a": cast("Any", 0.5)})


if __name__ == "__main__":
    unittest.main()
