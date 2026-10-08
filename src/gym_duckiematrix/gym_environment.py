"""Multi-entity Duckiematrix gym environment."""

__all__ = ["GymEnvironment"]

import logging
import os
import time
from collections.abc import Callable
from threading import Lock
from typing import Any

from duckietown.sdk.middleware.base import WorldInput, WorldOutput
from duckietown.sdk.middleware.dtps.components import (
    DTPSWorldInput,
    DTPSWorldOutput,
)
from duckietown.sdk.robots import discover_entities
from duckietown.sdk.robots.duckiebot import DB21M
from duckietown.sdk.robots.duckiebot.generic import GenericDuckiebot
from duckietown.sdk.robots.generic_vehicle import GenericVehicle
from duckietown_messages.simulation import WorldEntityOutput

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

    _all_names: list[str]
    _callback: Callable[[dict[str, Any]], None] | None
    _last_completed_session_id: int | None
    _lock = Lock()
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

    @staticmethod
    def _make_world_input(host: str, port: int) -> WorldInput:
        shm_base = os.environ.get("DTSHELL_SHM_PATH", "")
        shm_path = shm_base + ".world_input" if shm_base else None
        return DTPSWorldInput(
            host,
            port,
            _GYM_WORLD_TOPIC_NAME,
            "",
            path_prefix=("robot",),
            shm_path=shm_path,
            shm_only=shm_path is not None,
        )

    @staticmethod
    def _make_world_output(host: str, port: int) -> WorldOutput:
        shm_base = os.environ.get("DTSHELL_SHM_PATH", "")
        shm_path = shm_base + ".world_output" if shm_base else None
        return DTPSWorldOutput(
            host,
            port,
            _GYM_WORLD_TOPIC_NAME,
            "",
            path_prefix=("robot",),
            shm_path=shm_path,
            shm_only=shm_path is not None,
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
        session_id = self._world_input.current_session_id
        if session_id is None:
            message = (
                "Cannot publish WorldOutput before receiving a WorldInput "
                "with a session_id."
            )
            raise RuntimeError(message)
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
        self._world_output.publish(world_output)

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

    def _world_input_callback(self, world_input: dict[str, Any]) -> None:
        session_id = self._get_session_id(world_input)
        if session_id is None:
            message = "WorldInput is missing the required session_id field."
            raise ValueError(message)
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
            callback(world_input)

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
