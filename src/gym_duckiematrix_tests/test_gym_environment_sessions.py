"""Unit tests for GymEnvironment global world-session handling."""

import json
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
from duckietown.sdk.utils.lane_position import (
    LanePositionCalculator,
    MapInterpreter,
)

from gym_duckiematrix.db21j_env import DuckiematrixDB21JEnv
from gym_duckiematrix.gym_environment import GymEnvironment


class _FakeWorldInput:
    def __init__(self) -> None:
        self.callback: Callable[[dict[str, Any]], None] | None = None
        self.current_session_id: int | None = None
        self.started = 0
        self.stopped = 0
        self.has_started = False

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
        self.has_started = True

    def stop(self) -> None:
        self.stopped += 1
        self.has_started = False


class _FakeWorldOutput:
    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []
        self.started = 0
        self.stopped = 0
        self.has_started = False

    def publish(self, message: dict[str, Any]) -> None:
        self.published.append(message)

    def start(self) -> None:
        self.started += 1
        self.has_started = True

    def stop(self) -> None:
        self.stopped += 1
        self.has_started = False


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

    def test_db21j_configurable_images_and_lazy_rendering(self) -> None:
        """Delay plot creation until human rendering is requested."""
        with (
            patch("gym_duckiematrix.db21j_env.DB21J"),
            patch("gym_duckiematrix.db21j_env.DTPSWorldInput"),
            patch("gym_duckiematrix.db21j_env.DTPSWorldOutput"),
            patch("gym_duckiematrix.db21j_env.MapInterpreter"),
            patch("gym_duckiematrix.db21j_env.import_module") as importer,
        ):
            environment = DuckiematrixDB21JEnv(
                camera_width=np.int32(4), camera_height=np.int64(3),
                render_mode="human",
            )
            _check_equal(environment.observation_space.shape, (3, 4, 3))
            importer.assert_not_called()
            plot = importer.return_value
            figure, axes = Mock(), Mock()
            plot.subplots.return_value = (figure, axes)
            image = np.zeros((3, 4, 3), dtype=np.uint8)
            with patch.object(environment, "_observation", image):
                np.testing.assert_array_equal(environment.render(), image)
                environment.render()
            importer.assert_called_once_with("matplotlib.pyplot")
            plot.subplots.assert_called_once_with()
            environment.close()
            plot.close.assert_called_once_with(figure)

    def test_db21j_headless_render_does_not_import_matplotlib(self) -> None:
        """Keep headless and array rendering free of GUI imports."""
        for render_mode in (None, "rgb_array"):
            environment = DuckiematrixDB21JEnv.__new__(DuckiematrixDB21JEnv)
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            with (
                patch.multiple(
                    environment, create=True,
                    render_mode=render_mode, _observation=image,
                ),
                patch("gym_duckiematrix.db21j_env.import_module") as importer,
            ):
                if render_mode is None:
                    if environment.render() is not None:
                        pytest.fail("Unexpected frame with render_mode=None.")
                else:
                    np.testing.assert_array_equal(environment.render(), image)
            importer.assert_not_called()

    def test_db21j_clips_actions_and_rejects_invalid_values(self) -> None:
        """Bound motor commands and reject malformed values."""
        environment = DuckiematrixDB21JEnv.__new__(DuckiematrixDB21JEnv)
        publish = Mock()
        with patch.multiple(
            environment, create=True,
            _publish_world_output=publish, _wait_for_world_state=Mock(),
            _observation=None, _pose=None,
        ):
            environment.step(np.array([5.0, -5.0]))
            arguments = publish.call_args.args
            np.testing.assert_allclose(arguments, (0.4, -0.4))
            publish.reset_mock()
            for action in ([1], [1, 2, 3], [math.nan, 0], [0, math.inf]):
                with pytest.raises(ValueError, match="two finite"):
                    environment.step(np.array(action))
            publish.assert_not_called()

    def test_db21j_validates_camera_and_render_options(self) -> None:
        """Validate options before creating transport resources."""
        for options in (
            {"camera_width": 0}, {"camera_height": -1},
            {"camera_width": True}, {"render_mode": "invalid"},
            {"camera_width": 1.5}, {"camera_height": np.bool_(1)},
            {"camera_width": np.int64(-1)},
        ):
            with pytest.raises(
                ValueError, match=r"positive integer|render_mode",
            ):
                DuckiematrixDB21JEnv(**options)

    def test_db21j_validates_the_received_camera_shape(self) -> None:
        """Match reset observations to the declared image space."""
        pose = {
            "header": {"timestamp": 1.0},
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        }
        entity = {"pose": pose, "compressed_image": {"data": b"jpeg"}}
        for shape in ((3, 4, 3), (2, 2, 3)):
            image = np.zeros(shape, dtype=np.uint8)
            with (
                patch("gym_duckiematrix.db21j_env.DB21J"),
                patch("gym_duckiematrix.db21j_env.DTPSWorldInput") as inputs,
                patch("gym_duckiematrix.db21j_env.DTPSWorldOutput"),
                patch("gym_duckiematrix.db21j_env.MapInterpreter"),
                patch(
                    "gym_duckiematrix.db21j_env.JPEG.decode",
                    return_value=image,
                ),
            ):
                inputs.return_value.get.side_effect = [
                    {"session_id": session, "entities": {"robot": entity}}
                    for session in (1, 2)
                ]
                environment = DuckiematrixDB21JEnv(
                    entity_name="robot", camera_width=4, camera_height=3,
                )
                try:
                    if shape == (3, 4, 3):
                        observation, _info = environment.reset()
                        if not environment.observation_space.contains(
                            observation,
                        ):
                            pytest.fail("Reset returned an invalid image.")
                    else:
                        with pytest.raises(ValueError, match="Camera image"):
                            environment.reset()
                finally:
                    environment.close()

    def test_transport_rejects_empty_explicit_paths(self) -> None:
        """Reject an ambiguous SHM option before connecting."""
        for environment_cls in (GymEnvironment, DuckiematrixDB21JEnv):
            for path in ("", "   "):
                with pytest.raises(ValueError, match="shm_path"):
                    environment_cls(shm_path=path)

    def test_db21j_reward_uses_shipped_map_geometry(self) -> None:
        """Reward forward travel on actual straight and corner lanes."""
        fixture_path = Path(__file__).with_name("embedded_map_fixture.json")
        map_data = json.loads(fixture_path.read_text())
        interpreter = MapInterpreter(map_data)
        calculator = LanePositionCalculator(interpreter)
        size = interpreter.road_tile_size
        diagonal = math.sqrt(0.5)
        for tile in interpreter.drivable_tiles:
            if tile["kind"] == "curve":
                lanes = [
                    (-0.5 + 0.7 * diagonal, 0.5 - 0.7 * diagonal,
                     -diagonal, -diagonal),
                    (-0.5 + 0.3 * diagonal, 0.5 - 0.3 * diagonal,
                     diagonal, diagonal),
                ]
            else:
                lanes = [(-0.2, 0.0, 0.0, 1.0), (0.2, 0.0, 0.0, -1.0)]
            cosine = math.cos(tile["angle"])
            sine = math.sin(tile["angle"])
            for local_x, local_z, local_dx, local_dz in lanes:
                world_x = size * (
                    tile["pose"]["x"] + local_x * cosine - local_z * sine
                )
                world_y = size * (
                    tile["pose"]["y"] + local_x * sine + local_z * cosine
                )
                direction_x = local_dx * cosine - local_dz * sine
                direction_y = local_dx * sine + local_dz * cosine
                heading = math.atan2(direction_y, direction_x)
                pose = {
                    "timestamp": 2.0,
                    "position": {"x": world_x, "y": world_y, "z": 0.0},
                    "rotation": {
                        "w": math.cos(heading / 2),
                        "x": 0.0,
                        "y": 0.0,
                        "z": math.sin(heading / 2),
                    },
                }
                previous_pose = dict(pose)
                previous_pose["timestamp"] = 1.0
                previous_pose["position"] = {
                    "x": world_x - 0.1 * direction_x,
                    "y": world_y - 0.1 * direction_y,
                    "z": 0.0,
                }
                environment = DuckiematrixDB21JEnv.__new__(
                    DuckiematrixDB21JEnv,
                )
                with patch.multiple(
                    environment,
                    create=True,
                    _pose=pose,
                    _previous_pose=previous_pose,
                    _lane_position_calculator=calculator,
                    _observation=np.zeros((1, 1, 3), dtype=np.uint8),
                    _publish_world_output=Mock(),
                    _wait_for_world_state=Mock(),
                    _display_observation=Mock(),
                    _out_of_road_penalty=-10,
                ):
                    _, reward, terminated, truncated, _ = environment.step(
                        np.zeros(2),
                    )
                if not math.isclose(reward, 0.1, abs_tol=1e-8):
                    pytest.fail(f"Lane reward is incorrect for {tile!r}.")
                if terminated or truncated:
                    pytest.fail(f"On-lane travel terminated on {tile!r}.")

    def test_db21j_world_state_wait_has_deadline(self) -> None:
        """Report a world-state timeout without waiting forever."""
        environment = DuckiematrixDB21JEnv.__new__(DuckiematrixDB21JEnv)
        world_input = Mock()
        world_input.get.return_value = None
        timeout = 0.1
        with (
            patch.multiple(
                environment,
                create=True,
                _world_input=world_input,
            _timeout=timeout,
                _session_id=None,
            ),
            pytest.raises(TimeoutError, match="world state"),
        ):
            environment.reset()
        world_input.get.assert_called_once()
        remaining = world_input.get.call_args.kwargs["timeout"]
        if not 0 < remaining <= timeout:
            pytest.fail("World-state read exceeded its deadline.")

    def test_db21j_rejects_incomplete_world_state(self) -> None:
        """Report a missing image or pose in the current session."""
        environment = DuckiematrixDB21JEnv.__new__(DuckiematrixDB21JEnv)
        world_input = Mock()
        world_input.get.return_value = {
            "session_id": 1,
            "entities": {"vehicle_a": {}},
        }
        observation = np.zeros((1, 1, 3), dtype=np.uint8)
        for image, pose in ((None, {}), (observation, None)):
            with (
                patch.multiple(
                    environment,
                    create=True,
                    _world_input=world_input,
                    _entity_name="vehicle_a",
                    _timeout=0.1,
                    _session_id=None,
                ),
                patch.object(
                    environment, "_parse_observation", return_value=image,
                ),
                patch.object(environment, "_parse_pose", return_value=pose),
                pytest.raises(ValueError, match="image and pose"),
            ):
                environment.reset()

    def test_db21j_map_timeout_cleans_up_components(self) -> None:
        """Stop components when a map read times out."""
        with (
            patch("gym_duckiematrix.db21j_env.DB21J") as robot_factory,
            patch("gym_duckiematrix.db21j_env.DTPSWorldInput") as inputs,
            patch("gym_duckiematrix.db21j_env.DTPSWorldOutput") as outputs,
        ):
            robot = robot_factory.return_value
            robot.map_frames.get.return_value = None
            with pytest.raises(TimeoutError, match="map layer 'frames'"):
                DuckiematrixDB21JEnv(timeout=0.1)
            robot.map_frames.get.assert_called_once_with(
                block=True, clean_up=True, timeout=0.1,
            )
            inputs.return_value.stop.assert_called_once_with()
            outputs.return_value.stop.assert_called_once_with()

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
        script = Path(__file__).parents[2] / "examples" / "gym_mode.py"
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
        script = Path(__file__).parents[2] / "examples" / "gym_mode.py"
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

    def test_examples_do_not_connect_when_imported(self) -> None:
        """Keep imports of live demos free of engine connections."""
        for name in ("gym_mode.py", "db21j_rollout.py"):
            script = Path(__file__).parents[2] / "examples" / name
            with (
                patch(
                    "gym_duckiematrix.gym_environment.GymEnvironment",
                ) as gym_factory,
                patch(
                    "gym_duckiematrix.db21j_env.DuckiematrixDB21JEnv",
                ) as db21j_factory,
            ):
                run_path(str(script))
            gym_factory.assert_not_called()
            db21j_factory.assert_not_called()

    def test_gym_mode_frequency_excludes_startup(self) -> None:
        """Measure completed intervals without startup latency."""
        environment = Mock(spec=GymEnvironment)
        environment.vehicle_names = ["vehicle_a"]
        environment.duckiebot_names = ["vehicle_a"]
        environment.static_names = []

        def send_inputs() -> None:
            callback = environment.attach.call_args.args[0]
            for session_id in (1, 2, 3):
                callback({"session_id": session_id})

        environment.start.side_effect = send_inputs
        script = Path(__file__).parents[2] / "examples" / "gym_mode.py"
        with (
            patch.dict(os.environ, {"MAX_COUNT": "3", "GYM_SHM_PATH": ""}),
            patch(
                "gym_duckiematrix.gym_environment.GymEnvironment",
                return_value=environment,
            ) as factory,
            patch("time.perf_counter", side_effect=(100.0, 101.0, 103.0)),
        ):
            result = run_path(str(script), run_name="__main__")
        factory.assert_called_once_with(shm_path=None)
        _check_equal(result["average_frequency"], 2 / 3)

    def test_db21j_rollout_empty_shm_path_selects_network(self) -> None:
        """Treat an empty example SHM variable as network transport."""
        environment = Mock()
        environment.reset.return_value = (None, {})
        environment.step.return_value = (None, 0, True, False, {})
        script = Path(__file__).parents[2] / "examples" / "db21j_rollout.py"
        with (
            patch.dict(os.environ, {"GYM_SHM_PATH": ""}),
            patch(
                "gym_duckiematrix.db21j_env.DuckiematrixDB21JEnv",
            ) as factory,
            patch("gymnasium.wrappers.TimeLimit", return_value=environment),
        ):
            run_path(str(script), run_name="__main__")
        factory.assert_called_once_with(
            "map_0/vehicle_0", -10, render_mode="human", shm_path=None,
        )
        environment.close.assert_called_once_with()

    def test_world_factories_select_native_dtps_transport(self) -> None:
        """Check native transport selection in both gym environments."""
        factories = (
            (GymEnvironment, "gym_environment"),
            (DuckiematrixDB21JEnv, "db21j_env"),
        )
        for shm_base in (None, "/private/world_io"):
            for environment_cls, module_name in factories:
                module_path = f"gym_duckiematrix.{module_name}"
                with (
                    patch.dict(
                        os.environ, {"DTSHELL_SHM_PATH": "/stale/shell/value"},
                    ),
                    patch(f"{module_path}.DTPSWorldInput") as input_factory,
                    patch(f"{module_path}.DTPSWorldOutput") as output_factory,
                    patch(
                        "gym_duckiematrix.gym_environment.discover_entities",
                        return_value=(["vehicle_a"], []),
                    ),
                    patch("gym_duckiematrix.db21j_env.DB21J"),
                    patch("gym_duckiematrix.db21j_env.MapInterpreter"),
                ):
                    environment_cls(shm_path=shm_base)
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

    def test_input_start_failure_is_retryable(self) -> None:
        """Preserve startup errors and release output."""
        environment, world_input, world_output = _make_environment()
        startup_error = RuntimeError("Input startup failed.")
        with (
            patch.object(world_input, "start", side_effect=startup_error),
            pytest.raises(RuntimeError, match="Input startup failed") as error,
        ):
            environment.start()
        _check_equal(error.value, startup_error)
        _check_equal(world_input.has_started, expected=False)
        _check_equal(world_output.has_started, expected=False)
        environment.stop()
        _check_equal(world_input.stopped, 0)
        _check_equal(world_output.stopped, 1)
        environment.start()
        _check_equal(world_input.has_started, expected=True)
        _check_equal(world_output.has_started, expected=True)
        environment.stop()
        environment.stop()
        _check_equal(world_input.stopped, 1)
        _check_equal(world_output.stopped, 2)

    def test_input_start_failure_reports_rollback_errors(self) -> None:
        """Retain both failures and allow output cleanup to retry."""
        environment, world_input, world_output = _make_environment()
        startup_error = RuntimeError("Input startup failed.")
        cleanup_error = RuntimeError("Output cleanup failed.")
        with (
            patch.object(world_input, "start", side_effect=startup_error),
            patch.object(world_output, "stop", side_effect=cleanup_error),
            pytest.raises(
                ExceptionGroup, match="startup and rollback",
            ) as error,
        ):
            environment.start()
        _check_equal(error.value.exceptions[0], startup_error)
        rollback_error = error.value.exceptions[1]
        if not isinstance(rollback_error, BaseExceptionGroup):
            pytest.fail("Expected grouped rollback failures.")
        _check_equal(rollback_error.exceptions, (cleanup_error,))
        _check_equal(world_output.has_started, expected=True)
        environment.stop()
        _check_equal(world_output.has_started, expected=False)

    def test_output_start_failure_does_not_start_input(self) -> None:
        """Leave both components stopped if output startup fails."""
        environment, world_input, world_output = _make_environment()
        startup_error = RuntimeError("Output startup failed.")
        with (
            patch.object(world_output, "start", side_effect=startup_error),
            pytest.raises(
                RuntimeError, match="Output startup failed",
            ) as error,
        ):
            environment.start()
        _check_equal(error.value, startup_error)
        environment.stop()
        _check_equal(world_input.started, 0)
        _check_equal(world_input.stopped, 0)
        _check_equal(world_output.stopped, 0)

    def test_stop_attempts_all_started_components(self) -> None:
        """Stop output even if input cleanup fails."""
        environment, world_input, world_output = _make_environment()
        environment.start()
        cleanup_error = RuntimeError("Input cleanup failed.")
        with (
            patch.object(world_input, "stop", side_effect=cleanup_error),
            pytest.raises(ExceptionGroup, match="shutdown") as error,
        ):
            environment.stop()
        _check_equal(error.value.exceptions, (cleanup_error,))
        _check_equal(world_input.has_started, expected=True)
        _check_equal(world_output.has_started, expected=False)
        environment.stop()
        _check_equal(world_input.has_started, expected=False)
        _check_equal(world_output.stopped, 1)

    def test_restart_resets_local_session_bookkeeping(self) -> None:
        """Reset the session guard only at an explicit restart."""
        environment, world_input, _world_output = _make_environment()
        callbacks: list[dict] = []
        environment.attach(callbacks.append)
        environment.start()
        world_input.emit({"session_id": 10})
        world_input.emit({"session_id": 1})
        environment.stop()
        environment.start()
        world_input.emit({"session_id": 1})
        world_input.emit({"session_id": 1})
        environment.stop()
        _check_equal(callbacks, [{"session_id": 10}, {"session_id": 1}])

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

    def test_step_answers_active_callback_session(self) -> None:
        """Keep each callback response bound to its input session."""
        environment, world_input, world_output = _make_environment()

        def on_input(message: dict[str, Any]) -> None:
            world_input.current_session_id = message["session_id"] + 1
            environment.step({"vehicle_a": (0.2, 0.1)})

        environment.attach(on_input)
        world_input.emit({"session_id": 7})
        environment.step({"vehicle_a": (0.3, 0.2)})
        sessions = [
            message["session_id"] for message in world_output.published
        ]
        _check_equal(sessions, [7, 8])

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
