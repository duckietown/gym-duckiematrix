"""Multi-entity Duckiematrix gym environment."""

__all__ = ["GymEnvironment"]

import logging
import os
import time
from collections.abc import Callable
from threading import Lock
from typing import Any

from duckietown.sdk.middleware.components import WorldInput, WorldOutput
from duckietown.sdk.middleware.dtps.components import (
    DTPSWorldInput,
    DTPSWorldOutput,
)
from duckietown.sdk.middleware.shm.components import (
    ShmWorldInput,
    ShmWorldOutput,
)
from duckietown.sdk.robots import discover_entities
from duckietown.sdk.robots.duckiebot import DB21M
from duckietown.sdk.robots.duckiebot.generic import GenericDuckiebot
from duckietown.sdk.robots.generic_vehicle import GenericVehicle
from duckietown_messages.simulation import WorldEntityOutput

from .timing_profiler import TimingProfiler

_ENGINE_HOST = "127.0.0.1"
_ENGINE_PORT = 7501
_DIFFERENTIAL_DRIVE_ACTION_SIZE = 2
_GYM_WORLD_TOPIC_NAME = "gym"

_logger = logging.getLogger(__name__)


class GymEnvironment:
    """Multi-entity gym environment backed by the Duckiematrix engine.

    Gym mode is driven by one global ``robot/gym/in`` stream and one
    global ``robot/gym/out`` stream for the full simulation session.
    Each world message contains per-entity payloads under
    ``entities``; no individual vehicle owns the gym transport.

    Example usage::

        env = GymEnvironment()

        def on_step(world_input: dict) -> None:
            actions = {name: (0.5, 0.5) for name in env.vehicle_names}
            env.step(actions)

        env.attach(on_step)
        env.start()
        # ... wait ...
        env.stop()
    """

    _active_callback_output_published: bool
    _active_callback_session_id: int | None
    _active_callback_started_at_ms: float | None
    _active_callback_step_started: bool
    _all_names: list[str]
    _callback: Callable[[dict[str, Any]], None] | None
    _last_completed_session_id: int | None
    _lock = Lock()
    _profiler: TimingProfiler
    _static_names: list[str]
    _vehicle_names: list[str]
    _vehicles: dict[str, GenericVehicle]
    _world_input: WorldInput
    _world_output: WorldOutput

    def __init__(
        self,
        host: str = _ENGINE_HOST,
        port: int = _ENGINE_PORT,
        vehicle_cls: type[GenericVehicle] = DB21M,
    ) -> None:
        """Initialise the gym environment.

        Args:
            host: Engine host address. Defaults to ``"127.0.0.1"``.
            port: Engine DTPS port. Defaults to ``7501``.
            vehicle_cls: Class to instantiate for each discovered
                vehicle. Must be a subclass of
                :py:class:`GenericVehicle`. Defaults to
                :py:class:`DB21M`.

        """
        if not issubclass(vehicle_cls, GenericVehicle):
            message = (
                "vehicle_cls must be a subclass of GenericVehicle, got "
                f"{vehicle_cls!r}"
            )
            raise TypeError(message)
        _max_attempts = 30
        _delay = 2
        for _attempt in range(1, _max_attempts + 1):
            try:
                vehicle_names, static_names = discover_entities(host, port)
            except Exception:
                if _attempt == _max_attempts:
                    raise
                _logger.info(
                    "Engine not ready yet (attempt %d/%d), retrying in %.0fs "
                    "...",
                    _attempt,
                    _max_attempts,
                    _delay,
                )
                time.sleep(_delay)
                continue
            if vehicle_names:
                break
            if _attempt < _max_attempts:
                _logger.info(
                    "Engine reachable but no vehicles registered yet "
                    "(attempt %d/%d), retrying in %.0fs ...",
                    _attempt,
                    _max_attempts,
                    _delay,
                )
                time.sleep(_delay)
        self._vehicle_names = vehicle_names
        self._static_names = static_names
        self._all_names = vehicle_names + static_names

        self._vehicles = {
            name: vehicle_cls(
                name,
                simulated=True,
                gym_mode=True,
                host=host,
                port=port,
            )
            for name in vehicle_names
        }
        self._lock = Lock()
        self._last_completed_session_id = None
        self._callback = None
        self._world_input = self._make_world_input(host, port)
        self._world_output = self._make_world_output(host, port)
        self._profiler = TimingProfiler("Gym Profiling Information")
        self._clear_active_callback()

    @staticmethod
    def _make_world_input(host: str, port: int) -> WorldInput:
        if os.environ.get("DTSHELL_SHM_PATH", ""):
            return ShmWorldInput(host, port, _GYM_WORLD_TOPIC_NAME, "")
        return DTPSWorldInput(
            host,
            port,
            _GYM_WORLD_TOPIC_NAME,
            "",
            path_prefix=("robot",),
        )

    @staticmethod
    def _make_world_output(host: str, port: int) -> WorldOutput:
        if os.environ.get("DTSHELL_SHM_PATH", ""):
            return ShmWorldOutput(host, port, _GYM_WORLD_TOPIC_NAME, "")
        return DTPSWorldOutput(
            host,
            port,
            _GYM_WORLD_TOPIC_NAME,
            "",
            path_prefix=("robot",),
        )

    @property
    def vehicle_names(self) -> list[str]:
        """Names of all vehicle (non-static) entities."""
        return list(self._vehicle_names)

    @property
    def duckiebot_names(self) -> list[str]:
        """Names of all Duckiebot entities."""
        return [
            name
            for name, v in self._vehicles.items()
            if isinstance(v, GenericDuckiebot)
        ]

    @property
    def static_names(self) -> list[str]:
        """Names of all static entities (watchtowers, etc.)."""
        return list(self._static_names)

    def enable_profiling(self, status: bool = True) -> None:
        """Enable or disable host-side gym profiling."""
        self._profiler.enable(status=status)
        for component in (self._world_input, self._world_output):
            enable_profiling = getattr(component, "enable_profiling", None)
            if callable(enable_profiling):
                enable_profiling(status=status)

    def print_profiling(self, logger: logging.Logger | None = None) -> None:
        """Log host-side gym profiling information."""
        active_logger = logger or _logger
        for component in (self._world_input, self._world_output):
            print_profiling = getattr(component, "print_profiling", None)
            if callable(print_profiling):
                print_profiling(active_logger)
        self._profiler.log(active_logger)

    def attach(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Attach a callback that fires once per simulation cycle.

        The callback receives the raw global ``WorldInput`` dictionary
        for the session. Per-entity data is available under the
        ``entities`` key. Duplicate or stale sessions are dropped
        before the callback fires.

        Args:
            callback: Called with combined world-input data each cycle.

        """
        self._callback = callback
        self._world_input.attach(self._world_input_callback)

    def step(self, actions: dict[str, tuple[float, float]]) -> None:
        """Publish one global world-output message.

        Only discovered vehicles accept commands; static entities are
        ignored if included in *actions*. This method currently assumes
        differential-drive tuples of ``(left_pwm, right_pwm)``.

        Args:
            actions: Mapping of vehicle name ->
                ``(left_pwm, right_pwm)``.

        """
        with self._profiler.profile("[gym]:step/total"):
            session_id = self._world_input.current_session_id
            if session_id is None:
                message = (
                    "Cannot publish WorldOutput before receiving a WorldInput "
                    "with a session_id."
                )
                raise RuntimeError(message)
            callback_started_at_ms = self._active_callback_started_at_ms
            if (
                self._active_callback_session_id == session_id
                and callback_started_at_ms is not None
                and not self._active_callback_step_started
            ):
                self._active_callback_step_started = True
                self._profiler.observe(
                    "[gym]:world-input-callback/to-step",
                    max(
                        0.0,
                        time.perf_counter() * 1000 - callback_started_at_ms,
                    ),
                )
            with self._profiler.profile("[gym]:step/build-world-output"):
                world_output = {"session_id": session_id, "entities": {}}
                entities = world_output["entities"]
                for name, action in actions.items():
                    vehicle = self._vehicles.get(name)
                    if vehicle is None:
                        continue
                    if (
                        not isinstance(action, tuple)
                        or len(action) != _DIFFERENTIAL_DRIVE_ACTION_SIZE
                    ):
                        message = (
                            "GymEnvironment.step expects each action to be a "
                            "(left_pwm, right_pwm) tuple."
                        )
                        raise TypeError(message)
                    entities[name] = self._world_entity_output_to_native(
                        vehicle.make_world_entity_output(
                            left_pwm=action[0],
                            right_pwm=action[1],
                        ),
                    )
            with self._profiler.profile("[gym]:step/publish-world-output"):
                self._world_output.publish(world_output)
            callback_started_at_ms = self._active_callback_started_at_ms
            if (
                self._active_callback_session_id == session_id
                and callback_started_at_ms is not None
                and not self._active_callback_output_published
            ):
                self._active_callback_output_published = True
                self._profiler.observe(
                    "[gym]:world-input-callback/to-world-output-published",
                    max(
                        0.0,
                        time.perf_counter() * 1000 - callback_started_at_ms,
                    ),
                )

    def start(self) -> None:
        """Start the global gym world-input/world-output bridge."""
        self._world_output.start()
        self._world_input.start()

    def stop(self) -> None:
        """Stop the global gym world-input/world-output bridge."""
        self._world_input.stop()
        self._world_output.stop()

    @staticmethod
    def _get_session_id(world_input: dict[str, Any]) -> int | None:
        session_id = world_input.get("session_id")
        return session_id if isinstance(session_id, int) else None

    def _clear_active_callback(self) -> None:
        self._active_callback_session_id = None
        self._active_callback_started_at_ms = None
        self._active_callback_step_started = False
        self._active_callback_output_published = False

    def _world_input_callback(self, world_input: dict[str, Any]) -> None:
        session_id = self._get_session_id(world_input)
        if session_id is None:
            message = "WorldInput is missing the required session_id field."
            raise ValueError(message)
        callback_started_at_ms = time.perf_counter() * 1000
        with self._lock:
            if (
                self._last_completed_session_id is not None
                and session_id <= self._last_completed_session_id
            ):
                _logger.debug(
                    "Dropping stale gym WorldInput in session %d; last "
                    "completed session is %d.",
                    session_id,
                    self._last_completed_session_id,
                )
                return
            self._last_completed_session_id = session_id
            callback = self._callback
            if callback is not None:
                self._active_callback_session_id = session_id
                self._active_callback_started_at_ms = callback_started_at_ms
                self._active_callback_step_started = False
                self._active_callback_output_published = False
        if callback is not None:
            try:
                callback(world_input)
            finally:
                self._profiler.observe(
                    "[gym]:world-input-callback/total",
                    max(
                        0.0,
                        time.perf_counter() * 1000 - callback_started_at_ms,
                    ),
                )
                with self._lock:
                    if self._active_callback_session_id == session_id:
                        self._clear_active_callback()

    @staticmethod
    def _rgba_to_native(rgba: Any) -> dict[str, float]:
        return {
            "r": rgba.r,
            "g": rgba.g,
            "b": rgba.b,
            "a": rgba.a,
        }

    @classmethod
    def _world_entity_output_to_native(
        cls,
        entity_output: WorldEntityOutput,
    ) -> dict[str, Any]:
        payload = {}
        differential_pwm = entity_output.differential_pwm
        if differential_pwm is not None:
            payload["differential_pwm"] = {
                "left": differential_pwm.left,
                "right": differential_pwm.right,
            }
        car_lights = entity_output.car_lights
        if car_lights is not None:
            payload["car_lights"] = {
                "front_left": cls._rgba_to_native(car_lights.front_left),
                "front_right": cls._rgba_to_native(car_lights.front_right),
                "back_left": cls._rgba_to_native(car_lights.back_left),
                "back_right": cls._rgba_to_native(car_lights.back_right),
            }
        state_reset_flag = entity_output.state_reset_flag
        if state_reset_flag is not None:
            payload["state_reset_flag"] = {"data": state_reset_flag.data}
        return payload
