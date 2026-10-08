"""Unit tests for GymEnvironment global world-session handling."""

import math
import os
import unittest
from collections.abc import Callable
from pathlib import Path
from runpy import run_path
from threading import Event, Thread, current_thread
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import numpy as np
import pytest

from gym_duckiematrix.db21j_env import DuckiematrixDB21JEnv
from gym_duckiematrix.gym_environment import GymEnvironment


class _FakeWorldInput:
    def __init__(self) -> None:
        self.callback: Callable[[dict[str, Any]], None] | None = None
        self.current_session_id: int | None = None
        self.started = 0
        self.stopped = 0

    def attach(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self.callback = callback

    def emit(self, message: dict[str, Any]) -> None:
        session_id = message.get("session_id")
        self.current_session_id = (
            session_id if isinstance(session_id, int) else None
        )
        if self.callback is None:
            pytest.fail("No world-input callback has been attached.")
        self.callback(message)

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1


class _FakeWorldOutput:
    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []
        self.started = 0
        self.stopped = 0

    def publish(self, message: dict[str, Any]) -> None:
        self.published.append(message)

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1


def _make_environment() -> tuple[
    GymEnvironment, _FakeWorldInput, _FakeWorldOutput,
]:
    world_input = _FakeWorldInput()
    world_output = _FakeWorldOutput()
    with (
        patch(
            "gym_duckiematrix.gym_environment.discover_entities",
            return_value=(["vehicle_a", "vehicle_b"], ["watchtower"]),
        ),
        patch(
            "gym_duckiematrix.gym_environment.DTPSWorldInput",
            return_value=world_input,
        ),
        patch(
            "gym_duckiematrix.gym_environment.DTPSWorldOutput",
            return_value=world_output,
        ),
    ):
        environment = GymEnvironment()
    return environment, world_input, world_output


def _check_equal(actual: object, expected: object) -> None:
    if actual != expected:
        message = f"Expected {expected!r}, received {actual!r}."
        pytest.fail(message)


class GymEnvironmentSessionTests(unittest.TestCase):
    """Check session routing and script cleanup through public APIs."""

    def test_profiling_preserves_actions_and_reports_stages(self) -> None:
        """Report timings without changing accepted world sessions."""
        environment, world_input, world_output = _make_environment()
        logger = Mock()

        def on_input(_message: dict[str, Any]) -> None:
            environment.step({"vehicle_a": (0.2, 0.1)})

        environment.enable_profiling()
        environment.attach(on_input)
        environment.start()
        try:
            for session_id in (7, 7, 8):
                world_input.emit({"session_id": session_id})
        finally:
            environment.stop()
        _check_equal(
            [message["session_id"] for message in world_output.published],
            [7, 8],
        )
        _check_equal(world_input.stopped, 1)
        _check_equal(world_output.stopped, 1)
        environment.print_profiling(logger)
        logger.info.assert_called_once()
        report = logger.info.call_args.args[-1]
        expected_stages = (
            "[gym]:step/total",
            "[gym]:step/build-world-output",
            "[gym]:step/publish-world-output",
            "[gym]:world-input-callback/to-step",
            "[gym]:world-input-callback/to-world-output-published",
            "[gym]:world-input-callback/total",
        )
        for stage in expected_stages:
            if stage not in report:
                pytest.fail(f"Missing profiling stage: {stage}")

    def test_gym_mode_stops_on_main_thread_after_last_cycle(self) -> None:
        """Stop the script outside its callback thread."""
        expected_count = 2
        environment = Mock(spec=GymEnvironment)
        environment.vehicle_names = ["vehicle_a"]
        environment.duckiebot_names = ["vehicle_a"]
        environment.static_names = []
        callback_stop = Event()
        main_thread = current_thread()

        def send_inputs() -> None:
            callback = environment.attach.call_args.args[0]
            for session_id in (1, 2, 3):
                callback({"session_id": session_id})

        def stop() -> None:
            if current_thread() is not main_thread:
                callback_stop.set()

        worker = Thread(target=send_inputs, daemon=True)
        self.addCleanup(worker.join, 2.0)
        environment.start.side_effect = worker.start
        environment.stop.side_effect = stop
        script = Path(__file__).with_name("test_gym_mode.py")
        with (
            patch.dict(
                os.environ,
                {
                    "MAX_COUNT": str(expected_count),
                    "WAIT_TIMEOUT_SECONDS": "1",
                },
            ),
            patch(
                "gym_duckiematrix.gym_environment.GymEnvironment",
                return_value=environment,
            ),
        ):
            result = run_path(str(script), run_name="__main__")
        worker.join(timeout=2.0)
        if worker.is_alive():
            pytest.fail("The callback worker did not stop.")
        if callback_stop.is_set():
            pytest.fail("Environment shutdown ran on the callback thread.")
        _check_equal(result["state"].count, expected_count)
        if not result["event"].is_set():
            pytest.fail("The final cycle did not signal completion.")
        _check_equal(environment.step.call_count, expected_count)
        environment.stop.assert_called_once_with()

    def test_gym_mode_cleans_up_after_keyboard_interrupt(self) -> None:
        """Clean up when the wait is interrupted."""
        environment = Mock(spec=GymEnvironment)
        environment.vehicle_names = ["vehicle_a"]
        environment.static_names = []
        completion = Mock(spec=Event)
        completion.wait.side_effect = KeyboardInterrupt
        script = Path(__file__).with_name("test_gym_mode.py")
        with (
            patch(
                "gym_duckiematrix.gym_environment.GymEnvironment",
                return_value=environment,
            ),
            patch("threading.Event", return_value=completion),
            pytest.raises(SystemExit) as error,
        ):
            run_path(str(script), run_name="__main__")
        _check_equal(error.value.code, 130)
        environment.stop.assert_called_once_with()
        completion.wait.assert_called_once()

    def test_world_factories_select_native_dtps_transport(self) -> None:
        """Check native transport selection in both gym environments."""
        factories = (
            (GymEnvironment, "gym_environment"),
            (DuckiematrixDB21JEnv, "db21j_env"),
        )
        for shm_base in ("", "/private/world_io"):
            for environment_cls, module_name in factories:
                module_path = f"gym_duckiematrix.{module_name}"
                with (
                    patch.dict(os.environ, {"DTSHELL_SHM_PATH": shm_base}),
                    patch(f"{module_path}.DTPSWorldInput") as input_factory,
                    patch(f"{module_path}.DTPSWorldOutput") as output_factory,
                    patch(
                        "gym_duckiematrix.gym_environment.discover_entities",
                        return_value=(["vehicle_a"], []),
                    ),
                    patch("gym_duckiematrix.db21j_env.DB21J"),
                    patch("gym_duckiematrix.db21j_env.MapInterpreter"),
                    patch("gym_duckiematrix.db21j_env.plt"),
                ):
                    environment_cls()
                for factory, suffix in (
                    (input_factory, ".world_input"),
                    (output_factory, ".world_output"),
                ):
                    factory.assert_called_once_with(
                        "127.0.0.1",
                        7501,
                        "gym",
                        "",
                        path_prefix=("robot",),
                        shm_path=shm_base + suffix if shm_base else None,
                        shm_only=bool(shm_base),
                    )

    def test_deduplicates_completed_world_sessions(self) -> None:
        """Drop repeated and stale sessions before calling the user."""
        environment, world_input, _world_output = _make_environment()
        callbacks: list[dict] = []
        environment.attach(callbacks.append)

        world_input.emit({"session_id": 2, "payload": "a"})
        world_input.emit({"session_id": 2, "payload": "b"})
        world_input.emit({"session_id": 1, "payload": "stale"})
        world_input.emit({"session_id": 3, "payload": "c"})
        world_input.emit({"session_id": 2, "payload": "late-stale"})

        _check_equal(
            callbacks,
            [
                {"session_id": 2, "payload": "a"},
                {"session_id": 3, "payload": "c"},
            ],
        )

    def test_rejects_world_input_without_session_id(self) -> None:
        """Reject inputs without a world-session identifier."""
        environment, world_input, _world_output = _make_environment()
        environment.attach(Mock())

        with pytest.raises(
            ValueError,
            match="missing the required session_id",
        ):
            world_input.emit({"payload": "vehicle-a"})

    def test_step_publishes_one_global_world_output(self) -> None:
        """Publish vehicle actions together and omit static entities."""
        environment, world_input, world_output = _make_environment()
        world_input.current_session_id = 7

        environment.step(
            {
                "vehicle_a": (0.5, 0.4),
                "vehicle_b": (0.4, 0.5),
                "watchtower": (1.0, 1.0),
            },
        )

        _check_equal(len(world_output.published), 1)
        message = world_output.published[0]
        _check_equal(message["session_id"], 7)
        entities = message["entities"]
        _check_equal(
            set(entities),
            {"vehicle_a", "vehicle_b"},
        )
        _check_equal(
            entities["vehicle_a"]["differential_pwm"]["left"],
            0.5,
        )
        _check_equal(
            entities["vehicle_a"]["differential_pwm"]["right"],
            0.4,
        )
        _check_equal(
            entities["vehicle_b"]["differential_pwm"]["left"],
            0.4,
        )
        _check_equal(
            entities["vehicle_b"]["differential_pwm"]["right"],
            0.5,
        )
        _check_equal(
            entities["vehicle_a"]["differential_pwm"]["left"],
            0.5,
        )

    def test_step_uses_vehicle_entity_output_builder(self) -> None:
        """Build the vehicle's differential-drive output."""
        environment, world_input, world_output = _make_environment()
        world_input.current_session_id = 11

        environment.step({"vehicle_a": (0.2, 0.1)})

        message = world_output.published[0]
        entities = message["entities"]
        _check_equal(
            entities["vehicle_a"]["differential_pwm"]["left"],
            0.2,
        )
        _check_equal(
            entities["vehicle_a"]["differential_pwm"]["right"],
            0.1,
        )

    def test_step_requires_a_current_session(self) -> None:
        """Require an input session before publishing world output."""
        environment, _world_input, _world_output = _make_environment()

        with pytest.raises(RuntimeError, match="with a session_id"):
            environment.step({"vehicle_a": (0.5, 0.4)})

    def test_step_rejects_non_tuple_actions(self) -> None:
        """Require a left/right action tuple for each vehicle."""
        environment, world_input, _world_output = _make_environment()
        world_input.current_session_id = 9

        with pytest.raises(TypeError, match="expects each action"):
            environment.step({"vehicle_a": cast("Any", 0.5)})

    def test_db21j_lane_coordinates_match_renderer_frame(self) -> None:
        """Convert world poses to the renderer's lane frame."""
        environment = DuckiematrixDB21JEnv.__new__(DuckiematrixDB21JEnv)
        pose: dict[str, dict[str, float] | float] = {
            "timestamp": 2.0,
            "position": {"x": 0.2, "y": 1.6, "z": 0.0},
            "rotation": {
                "w": math.sqrt(0.5),
                "x": 0.0,
                "y": 0.0,
                "z": math.sqrt(0.5),
            },
        }
        calculator = Mock()
        calculator.get_lane_pos2.return_value = (
            SimpleNamespace(dist=0.0, dot_dir=1.0)
        )
        previous_pose = dict(pose)
        previous_pose["timestamp"] = 1.0
        previous_pose["position"] = {"x": 0.1, "y": 1.6, "z": 0.0}
        observation = np.zeros((1, 1, 3), dtype=np.uint8)
        action = np.zeros(2)
        with patch.multiple(
            environment,
            create=True,
            _pose=pose,
            _previous_pose=previous_pose,
            _lane_position_calculator=calculator,
            _observation=observation,
            _publish_world_output=Mock(),
            _wait_for_world_state=Mock(),
            _display_observation=Mock(return_value=observation),
        ):
            _image, reward, terminated, truncated, _info = environment.step(
                action,
            )
        if not math.isclose(reward, 0.1) or terminated or truncated:
            pytest.fail("World-frame motion changed the lane reward.")
        arguments = calculator.get_lane_pos2.call_args.args
        coordinates = arguments[0].tolist()
        _check_equal(coordinates, [0.2, 0.0, 1.6])
        if not math.isclose(arguments[1], -math.pi / 2):
            pytest.fail("Lane lookup received an unconverted heading.")


if __name__ == "__main__":
    unittest.main()
