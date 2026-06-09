# pyright: reportMissingTypeStubs=false

"""Duckiematrix DB21J environment."""

import os
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from duckietown.sdk.middleware.components import WorldInput, WorldOutput
from duckietown.sdk.middleware.dtps.components import (
    DTPSWorldInput,
    DTPSWorldOutput,
)
from duckietown.sdk.middleware.shm.components import (
    ShmWorldInput,
    ShmWorldOutput,
)
from duckietown.sdk.robots.duckiebot import DB21J
from duckietown.sdk.utils.exceptions import PointError, TangentVectorError
from duckietown.sdk.utils.jpeg import JPEG
from duckietown.sdk.utils.lane_position import MapInterpreter
from duckietown_messages.colors import RGBA
from duckietown_messages.simulation import WorldEntityOutput
from gymnasium import Env, spaces
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from gym_duckiematrix.utils import quaternion_to_euler_angles

_ENGINE_HOST = "127.0.0.1"
_ENGINE_PORT = 7501
_GYM_WORLD_TOPIC_NAME = "gym"
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480


class DuckiematrixDB21JEnv(Env):
    """Duckiematrix DB21J environment."""

    _entity_name: str
    _figure: Figure
    _map_interpreter: MapInterpreter
    _observation: np.ndarray | None
    _observation_received: bool
    _out_of_road_penalty: float
    _pose: dict[str, dict | float] | None
    _pose_received: bool
    _previous_pose: dict[str, dict[str, float] | float] | None
    _session_id: int | None
    _shutdown: bool
    _window: AxesImage
    _world_input: WorldInput
    _world_output: WorldOutput
    action_space: spaces.Box
    observation_space: spaces.Box

    def __init__(
        self,
        entity_name: str = "map_0/vehicle_0",
        out_of_road_penalty: float = -10,
    ) -> None:
        """Duckiematrix DB21J environment.

        Args:
            entity_name (str, optional): The name of the entity.
            Defaults to `"map_0/vehicle_0"`.
            out_of_road_penalty (float, optional): The penalty for being
            out of the road. Defaults to `-10`.

        """
        zeros = np.zeros((DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH, 3))
        self._window = plt.imshow(zeros)
        plt.axis("off")
        self._figure = plt.figure(1)
        plt.subplots_adjust(0, 0, 1, 1)
        plt.pause(0.01)
        self._shutdown = False
        self._entity_name = entity_name
        self._session_id = None
        self._observation = None
        self._observation_received = False
        self._pose = None
        self._pose_received = False
        self.robot = DB21J(entity_name, simulated=True, gym_mode=True)
        self._world_input = self._make_world_input()
        self._world_output = self._make_world_output()
        self._start_components()
        self.action_space = spaces.Box(-1, 1, (2,))
        self.observation_space = spaces.Box(
            0,
            255,
            (DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH, 3),
            np.uint8,
        )
        self._map_interpreter = MapInterpreter(
            {
                "frames": self.robot.map_frames.get(block=True, clean_up=True),
                "tiles": self.robot.map_tiles.get(block=True, clean_up=True),
                "tile_info": self.robot.map_tile_info.get(
                    block=True,
                    clean_up=True,
                ),
            },
        )
        self._out_of_road_penalty = out_of_road_penalty
        self._previous_pose = None

    @staticmethod
    def _make_world_input() -> WorldInput:
        if os.environ.get("DTSHELL_SHM_PATH", ""):
            return ShmWorldInput(
                _ENGINE_HOST,
                _ENGINE_PORT,
                _GYM_WORLD_TOPIC_NAME,
                "",
            )
        return DTPSWorldInput(
            _ENGINE_HOST,
            _ENGINE_PORT,
            _GYM_WORLD_TOPIC_NAME,
            "",
            path_prefix=("robot",),
        )

    @staticmethod
    def _make_world_output() -> WorldOutput:
        if os.environ.get("DTSHELL_SHM_PATH", ""):
            return ShmWorldOutput(
                _ENGINE_HOST,
                _ENGINE_PORT,
                _GYM_WORLD_TOPIC_NAME,
                "",
            )
        return DTPSWorldOutput(
            _ENGINE_HOST,
            _ENGINE_PORT,
            _GYM_WORLD_TOPIC_NAME,
            "",
            path_prefix=("robot",),
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

    @property
    def cur_angle(self) -> float:
        """Compatibility heading accessor for older learning code."""
        if self._pose is None:
            return 0.0
        rotation = self._pose.get("rotation")
        if not isinstance(rotation, dict):
            return 0.0
        w = float(rotation.get("w", 1.0))
        x = float(rotation.get("x", 0.0))
        y = float(rotation.get("y", 0.0))
        z = float(rotation.get("z", 0.0))
        return quaternion_to_euler_angles((w, x, y, z))[-1]

    @property
    def cur_pos(self) -> np.ndarray:
        """Compatibility position accessor for older learning code."""
        if self._pose is None:
            return np.zeros(3, dtype=float)
        position = self._pose.get("position")
        if not isinstance(position, dict):
            return np.zeros(3, dtype=float)
        return np.array(
            (
                float(position.get("x", 0.0)),
                float(position.get("y", 0.0)),
                float(position.get("z", 0.0)),
            ),
            dtype=float,
        )

    @property
    def objects(self) -> tuple[Any, ...]:
        """Compatibility placeholder for simulator world objects."""
        return ()

    @staticmethod
    def _get_info() -> dict[str, Any]:
        return {}

    def _display_observation(self, observation: np.ndarray) -> np.ndarray:
        self._window.set_data(observation)
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
        position_array = np.array((x, y, z))
        previous_position = previous_pose["position"]
        if type(previous_position) is not dict:
            message = (
                "Expected 'previous_position' to be of type 'dict', got "
                f"'{type(previous_position)}'."
            )
            raise TypeError(message)
        x, y, z = previous_position.values()
        previous_position_array = np.array((x, y, z))
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
            lane_position = self._map_interpreter.get_lane_position(
                position_array,
                euler_angles[-1],
            )
            reward = speed * lane_position.dot_direction - 10 * np.abs(
                lane_position.distance,
            )
            terminated = False
        except (PointError, TangentVectorError):
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
        while True:
            world_input = self._world_input.get(block=True)
            if not isinstance(world_input, dict):
                continue
            session_id = world_input.get("session_id")
            if not isinstance(session_id, int):
                continue
            if self._session_id is not None and session_id <= self._session_id:
                continue
            self._observation_received = False
            self._pose_received = False
            if self._consume_world_input(world_input):
                return

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

    def closest_curve_point(
        self,
        position: np.ndarray,
        angle: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Compatibility adapter for older pure-pursuit code."""
        return self._map_interpreter.get_closest_curve_point(position, angle)

    def get_grid_coords(self, position: np.ndarray) -> tuple[int, int]:
        """Return the tile grid coordinates for a world position."""
        grid_coordinates_getter = self._map_interpreter._get_grid_coordinates  # noqa: SLF001
        return grid_coordinates_getter(position)

    def _get_tile(self, i: int, j: int) -> dict[str, Any] | None:
        """Compatibility adapter returning tile metadata."""
        return self._map_interpreter.get_tile(i, j)

    def get_tile(self, i: int, j: int) -> dict[str, Any] | None:
        """Return tile metadata for the requested grid coordinate."""
        return self._get_tile(i, j)

    def get_lane_pos2(self, position: np.ndarray, angle: float) -> Any:  # noqa: ANN401
        """Return a legacy lane-position object used by DAgger code."""
        lane_position = self._map_interpreter.get_lane_position(
            position,
            angle,
        )
        return SimpleNamespace(
            dist=lane_position.distance,
            dot_dir=lane_position.dot_direction,
            angle_deg=lane_position.angle_deg,
            angle_rad=lane_position.angle_rad,
        )

    def render(self) -> Any | None:  # noqa: ANN401
        """Render the latest camera image in the matplotlib window."""
        if self._observation is None:
            return None
        observation = self._get_current_observation()
        return self._display_observation(observation)

    def render_obs(self) -> np.ndarray:
        """Return the latest RGB camera image."""
        return self._get_current_observation()

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
        left = 0.4 * action[0]
        right = 0.4 * action[1]
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
        plt.close(self._figure)
