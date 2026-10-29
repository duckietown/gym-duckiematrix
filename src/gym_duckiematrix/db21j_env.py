# pyright: reportMissingTypeStubs=false

"""Duckiematrix DB21J environment."""

from __future__ import annotations

import logging
import math
import time
from importlib import import_module
from typing import TYPE_CHECKING, Any

import numpy as np
from duckietown.sdk.middleware.dtps.components import (
    DTPSWorldInput,
    DTPSWorldOutput,
)
from duckietown.sdk.robots.duckiebot import DB21J
from duckietown.sdk.utils.jpeg import JPEG
from duckietown.sdk.utils.lane_position import (
    LanePositionCalculator,
    MapInterpreter,
    NotInLane,
)
from gymnasium import Env, spaces

from gym_duckiematrix.utils import quaternion_to_euler_angles

if TYPE_CHECKING:
    from types import ModuleType

    from duckietown.sdk.middleware.base import WorldInput, WorldOutput
    from duckietown_messages.colors import RGBA
    from duckietown_messages.simulation import WorldEntityOutput
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage

_ENGINE_HOST = "127.0.0.1"
_ENGINE_PORT = 7501
_GYM_WORLD_TOPIC_NAME = "gym"
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
_ACTION_SIZE = 2
_logger = logging.getLogger(__name__)


class DuckiematrixDB21JEnv(Env):
    """Duckiematrix DB21J environment."""

    _entity_name: str
    _figure: Figure | None
    _lane_position_calculator: LanePositionCalculator
    _map_interpreter: MapInterpreter
    _observation: np.ndarray | None
    _observation_received: bool
    _out_of_road_penalty: float
    _pose: dict[str, dict | float] | None
    _pose_received: bool
    _previous_pose: dict[str, dict[str, float] | float] | None
    _session_id: int | None
    _shutdown: bool
    _timeout: float
    _window: AxesImage | None
    _pyplot: ModuleType | None
    _world_input: WorldInput
    _world_output: WorldOutput
    action_space: spaces.Box
    observation_space: spaces.Box

    def __init__(  # noqa: PLR0913
        self,
        entity_name: str = "map_0/vehicle_0",
        out_of_road_penalty: float = -10,
        *,
        timeout: float = 10.0,
        camera_width: int | np.integer[Any] = DEFAULT_CAMERA_WIDTH,
        camera_height: int | np.integer[Any] = DEFAULT_CAMERA_HEIGHT,
        render_mode: str | None = None,
        shm_path: str | None = None,
    ) -> None:
        """Duckiematrix DB21J environment.

        Args:
            entity_name (str, optional): The name of the entity.
            Defaults to `"map_0/vehicle_0"`.
            out_of_road_penalty (float, optional): The penalty for being
            out of the road. Defaults to `-10`.
            timeout: Positive deadline in seconds for each map layer and
                world-state wait. Defaults to 10 seconds.
            camera_width: Expected image width from the renderer.
            camera_height: Expected image height from the renderer.
            render_mode: ``human`` opens a window lazily; ``rgb_array``
                returns images from render(). Defaults to no window.
            shm_path: SHM base path; None selects network DTPS.

        """
        if not math.isfinite(timeout) or timeout <= 0:
            message = "timeout must be finite and greater than zero."
            raise ValueError(message)
        for name, size in (
            ("camera_width", camera_width), ("camera_height", camera_height),
        ):
            if (
                not isinstance(size, (int, np.integer))
                or isinstance(size, bool)
                or size <= 0
            ):
                message = f"{name} must be a positive integer."
                raise ValueError(message)
        camera_width = int(camera_width)
        camera_height = int(camera_height)
        if render_mode not in (None, "human", "rgb_array"):
            message = "render_mode must be None, 'human', or 'rgb_array'."
            raise ValueError(message)
        if shm_path is not None and not shm_path.strip():
            message = "shm_path must be nonempty or None."
            raise ValueError(message)
        self.metadata = {"render_modes": ["human", "rgb_array"]}
        self.render_mode = render_mode
        self._timeout = timeout
        self._window = None
        self._figure = None
        self._pyplot = None
        self._shutdown = False
        self._entity_name = entity_name
        self._session_id = None
        self._observation = None
        self._observation_received = False
        self._pose = None
        self._pose_received = False
        self.robot = DB21J(entity_name, simulated=True, gym_mode=True)
        self._world_input = self._make_world_input(shm_path)
        self._world_output = self._make_world_output(shm_path)
        _logger.info(
            "Gym transport: %s",
            shm_path or f"DTPS {_ENGINE_HOST}:{_ENGINE_PORT}",
        )
        try:
            self._start_components()
            self.action_space = spaces.Box(-1, 1, (2,))
            self.observation_space = spaces.Box(
                0,
                255,
                (camera_height, camera_width, 3),
                np.uint8,
            )
            map_data = self._read_map()
            self._map_interpreter = MapInterpreter(map_data)
            self._lane_position_calculator = LanePositionCalculator(
                self._map_interpreter,
            )
        except BaseException:
            self.close()
            raise
        self._out_of_road_penalty = out_of_road_penalty
        self._previous_pose = None

    def _read_map(self) -> dict[str, dict]:
        layers = {}
        for name, component in (
            ("frames", self.robot.map_frames),
            ("tiles", self.robot.map_tiles),
            ("tile_info", self.robot.map_tile_info),
        ):
            layer = component.get(
                block=True,
                clean_up=True,
                timeout=self._timeout,
            )
            if layer is None:
                message = f"Timed out waiting for map layer {name!r}."
                raise TimeoutError(message)
            layers[name] = layer
        return layers

    @staticmethod
    def _make_world_input(shm_base: str | None) -> WorldInput:
        shm_path = shm_base + ".world_input" if shm_base else None
        return DTPSWorldInput(
            _ENGINE_HOST,
            _ENGINE_PORT,
            _GYM_WORLD_TOPIC_NAME,
            "",
            path_prefix=("robot",),
            shm_path=shm_path,
            shm_only=shm_path is not None,
        )

    @staticmethod
    def _make_world_output(shm_base: str | None) -> WorldOutput:
        shm_path = shm_base + ".world_output" if shm_base else None
        return DTPSWorldOutput(
            _ENGINE_HOST,
            _ENGINE_PORT,
            _GYM_WORLD_TOPIC_NAME,
            "",
            path_prefix=("robot",),
            shm_path=shm_path,
            shm_only=shm_path is not None,
        )

    @staticmethod
    def _rgba_to_native(rgba: RGBA) -> dict[str, float]:
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
            payload["state_reset_flag"] = {
                "data": state_reset_flag.data,
            }
        return payload

    @staticmethod
    def _get_info() -> dict[str, Any]:
        return {}

    def _display_observation(self, observation: np.ndarray) -> np.ndarray:
        if self.render_mode != "human":
            return observation
        if self._window is None:
            self._pyplot = import_module("matplotlib.pyplot")
            figure, axes = self._pyplot.subplots()
            self._figure = figure
            axes.set_axis_off()
            self._window = axes.imshow(observation)
            figure.subplots_adjust(0, 0, 1, 1)
            self._pyplot.show(block=False)
        self._window.set_data(observation)
        if self._figure is not None:
            self._figure.canvas.draw_idle()
            self._figure.canvas.start_event_loop(0.00001)
        return observation

    def _get_reward(
        self,
        pose: dict[str, dict | float],
        previous_pose: dict[str, dict | float],
    ) -> tuple[float, bool]:
        timestamp = pose["timestamp"]
        if type(timestamp) is not float:
            message = (
                "Expected 'timestamp' to be of type 'float', got "
                f"'{type(timestamp)}'."
            )
            raise TypeError(message)
        previous_timestamp = previous_pose["timestamp"]
        if type(previous_timestamp) is not float:
            message = (
                "Expected 'previous_timestamp' to be of type 'float', got"
                f" '{type(previous_timestamp)}'."
            )
            raise TypeError(message)
        delta_time = timestamp - previous_timestamp
        position = pose["position"]
        if type(position) is not dict:
            message = (
                "Expected 'position' to be of type 'dict', got "
                f"'{type(position)}'."
            )
            raise TypeError(message)
        x, y, z = position.values()
        position_array = np.array((x, z, y))
        previous_position = previous_pose["position"]
        if type(previous_position) is not dict:
            message = (
                "Expected 'previous_position' to be of type 'dict', got "
                f"'{type(previous_position)}'."
            )
            raise TypeError(message)
        x, y, z = previous_position.values()
        previous_position_array = np.array((x, z, y))
        speed = (
            np.linalg.norm(position_array - previous_position_array)
            / delta_time
            if delta_time > 0
            else 0
        )
        rotation = pose["rotation"]
        if type(rotation) is not dict:
            message = (
                "Expected 'rotation' to be of type 'dict', got "
                f"'{type(rotation)}'."
            )
            raise TypeError(message)
        w, x, y, z = rotation.values()
        euler_angles = quaternion_to_euler_angles((w, x, y, z))
        try:
            lane_position = self._lane_position_calculator.get_lane_pos2(
                position_array,
                -euler_angles[-1],
            )
            reward = speed * lane_position.dot_dir - 10 * np.abs(
                lane_position.dist,
            )
            terminated = False
        except NotInLane:
            reward = self._out_of_road_penalty
            terminated = True
        return reward, terminated

    def _start_components(self) -> None:
        self.robot.map_frames.start()
        self.robot.map_tile_info.start()
        self.robot.map_tiles.start()
        self._world_output.start()
        self._world_input.start()

    @staticmethod
    def _normalize_binary_data(
        data: bytes | bytearray | memoryview | list[int] | None,
    ) -> bytes | None:
        if isinstance(data, bytes):
            return data
        if isinstance(data, bytearray):
            return bytes(data)
        if isinstance(data, memoryview):
            return data.tobytes()
        if isinstance(data, list):
            return bytes(data)
        return None

    @classmethod
    def _parse_observation(
        cls,
        entity_input: dict[str, Any],
    ) -> np.ndarray | None:
        compressed_image = entity_input.get("compressed_image")
        if not isinstance(compressed_image, dict):
            return None
        image_data = cls._normalize_binary_data(compressed_image.get("data"))
        if image_data is None:
            return None
        return JPEG.decode(image_data)

    @staticmethod
    def _parse_pose(
        entity_input: dict[str, Any],
    ) -> dict[str, dict[str, float] | float] | None:
        pose = entity_input.get("pose")
        if not isinstance(pose, dict):
            return None
        header = pose.get("header")
        position = pose.get("position")
        rotation = pose.get("rotation")
        if not isinstance(header, dict):
            return None
        if not isinstance(position, dict):
            return None
        if not isinstance(rotation, dict):
            return None
        timestamp = header.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            return None
        return {
            "timestamp": float(timestamp),
            "position": {
                "x": float(position.get("x", 0.0)),
                "y": float(position.get("y", 0.0)),
                "z": float(position.get("z", 0.0)),
            },
            "rotation": {
                "w": float(rotation.get("w", 1.0)),
                "x": float(rotation.get("x", 0.0)),
                "y": float(rotation.get("y", 0.0)),
                "z": float(rotation.get("z", 0.0)),
            },
        }

    def _consume_world_input(self, world_input: dict[str, Any]) -> bool:
        entities = world_input.get("entities")
        if not isinstance(entities, dict):
            return False
        entity_input = entities.get(self._entity_name)
        if not isinstance(entity_input, dict):
            return False
        observation = self._parse_observation(entity_input)
        pose = self._parse_pose(entity_input)
        if observation is None or pose is None:
            return False
        if observation.shape != self.observation_space.shape:
            message = (
                f"Camera image shape {observation.shape} does not match "
                f"observation_space {self.observation_space.shape}."
            )
            raise ValueError(message)
        session_id = world_input.get("session_id")
        if not isinstance(session_id, int):
            return False
        self._session_id = session_id
        self._observation = observation
        self._pose = pose
        self._observation_received = True
        self._pose_received = True
        return True

    def _wait_for_world_state(self) -> None:
        deadline = time.monotonic() + self._timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                message = "Timed out waiting for world state."
                raise TimeoutError(message)
            world_input = self._world_input.get(
                block=True,
                timeout=remaining,
            )
            if world_input is None:
                message = "Timed out waiting for world state."
                raise TimeoutError(message)
            if not isinstance(world_input, dict):
                message = "WorldInput must be a dictionary."
                raise TypeError(message)
            session_id = world_input.get("session_id")
            if not isinstance(session_id, int) or isinstance(session_id, bool):
                message = "WorldInput requires an integer session_id."
                raise TypeError(message)
            if self._session_id is not None and session_id <= self._session_id:
                continue
            self._observation_received = False
            self._pose_received = False
            if self._consume_world_input(world_input):
                return
            message = (
                f"WorldInput session {session_id} has no complete image and "
                f"pose for {self._entity_name!r}."
            )
            raise ValueError(message)

    def _publish_world_output(
        self,
        left_pwm: float,
        right_pwm: float,
        *,
        reset_state: bool,
    ) -> None:
        if self._session_id is None:
            self._wait_for_world_state()
        entity_output = self.robot.make_world_entity_output(
            left_pwm=left_pwm,
            right_pwm=right_pwm,
            reset_state=reset_state,
        )
        self._world_output.publish(
            {
                "session_id": self._session_id,
                "entities": {
                    self._entity_name: self._world_entity_output_to_native(
                        entity_output,
                    ),
                },
            },
        )

    def _get_current_observation(self) -> np.ndarray:
        if self._observation is None:
            return np.ndarray(())
        return self._observation[:, :, [2, 1, 0]]

    def render(self) -> np.ndarray | None:
        """Render the latest image using the selected mode."""
        if self.render_mode is None or self._observation is None:
            return None
        observation = self._get_current_observation()
        return self._display_observation(observation)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and return the initial observation.

        Args:
            seed: Optional RNG seed (passed to super). Not used
                directly by this implementation.
            options: Optional reset options. Not used directly
                by this implementation.

        Returns:
            A ``(observation, info)`` tuple.

        """
        super().reset(seed=seed, options=options)
        if self._session_id is None:
            self._wait_for_world_state()
        self._publish_world_output(0.0, 0.0, reset_state=True)
        self._wait_for_world_state()
        if self._observation is None or self._pose is None:
            return np.ndarray(()), {}
        self._previous_pose = self._pose
        info = self._get_info()
        return self._get_current_observation(), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Perform one environment step.

        Args:
            action: A 2-element array ``[left, right]`` in the range
                ``[-1, 1]``.

        Returns:
            A ``(observation, reward, terminated, truncated, info)``
            tuple.

        """
        values = np.asarray(action, dtype=np.float64)
        finite = np.isfinite(values)
        if values.shape != (_ACTION_SIZE,) or not finite.all():
            message = "action must contain two finite motor values."
            raise ValueError(message)
        clipped = np.clip(values, -1, 1)
        left = 0.4 * clipped[0]
        right = 0.4 * clipped[1]
        self._publish_world_output(
            float(left),
            float(right),
            reset_state=False,
        )
        self._wait_for_world_state()
        if self._observation is None or self._pose is None:
            return np.ndarray(()), 0, False, False, {}
        if self._previous_pose is None:
            self._previous_pose = self._pose
            observation = self._get_current_observation()
            info = self._get_info()
            return observation, 0, False, False, info
        reward, terminated = self._get_reward(self._pose, self._previous_pose)
        self._previous_pose = self._pose
        observation = self._get_current_observation()
        observation = self._display_observation(observation)
        info = self._get_info()
        return observation, reward, terminated, False, info

    def close(self) -> None:
        """Release robot components and the matplotlib window."""
        if self._shutdown:
            return
        self._shutdown = True
        for component in (
            self._world_input,
            self._world_output,
            self.robot.map_frames,
            self.robot.map_tile_info,
            self.robot.map_tiles,
        ):
            if component.has_started:
                component.stop()
        if self._pyplot is not None and self._figure is not None:
            self._pyplot.close(self._figure)
