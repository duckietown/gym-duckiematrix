from typing import Tuple, Dict, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from duckietown.sdk.robots.duckiebot import DB21J
from duckietown.sdk.utils.lane_position import MapInterpreter, LanePositionCalculator
from .utils import quaternion_to_euler


DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480


class DuckiematrixDB21JEnv(gym.Env):
    def __init__(
            self,
            entity_name: str = "map_0/vehicle_0",
            out_of_road_penalty: float = -10.0, 
            headless: bool = False,
            camera_height: int = DEFAULT_CAMERA_HEIGHT,
            camera_width: int = DEFAULT_CAMERA_WIDTH,
        ):
        super().__init__()
        if not headless:
            import matplotlib.pyplot as plt
            plt.ion()
            self.fig = plt.figure(1)
            ax = self.fig.add_subplot(111)
            ax.axis("off")
            self.window = ax.imshow(np.zeros((camera_height, camera_width, 3)))
            self.fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
            self.fig.canvas.draw()
            plt.show(block=False)
            self.fig.canvas.flush_events()

        self.camera_height = camera_height
        self.camera_width = camera_width
        self.headless = headless
        self.entity_name = entity_name
        self.robot: DB21J = DB21J(entity_name, simulated=True)
        self._start_components() # Will start and add components to robot
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.camera_height, self.camera_width, 3), dtype=np.uint8
        )
        self.map = {"frames": None, "tiles": None, "tile_info": None}
        self.get_map()
        self.map_int = MapInterpreter(map=self.map)
        self.lp_cal = LanePositionCalculator(map_interpreter=self.map_int)
        self.out_of_road_penalty = out_of_road_penalty
        self.last_pose = None

    def _start_components(self):
        self.robot.camera.start()
        self.robot.motors.start()
        self.robot.map_frames.start()
        self.robot.map_tiles.start()
        self.robot.map_tile_info.start()
        self.robot.pose.start()
        self.robot.reset_flag.start()
    
    def _stop_components(self):
        self.robot.reset_flag.stop()
        self.robot.pose.stop()
        self.robot.map_tile_info.stop()
        self.robot.map_tiles.stop()
        self.robot.map_frames.stop()
        self.robot.motors.stop()
        self.robot.camera.stop()

    def get_map(self):
        while True:
            if self.check_map():
                break
            if self.map["frames"] is None:
                self.map["frames"] = self.robot.map_frames.capture()
            elif self.map["tiles"] is None:
                self.map["tiles"] = self.robot.map_tiles.capture()
            elif self.map["tile_info"] is None:
                self.map["tile_info"] = self.robot.map_tile_info.capture()
            
    def check_map(self):
        is_map = True
        for key in self.map.keys():
            if self.map[key] is None:
                is_map = False 
        return is_map               
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        self.robot.reset_flag.set_reset(True)

        self.last_pose = self._get_pose_blocking()
        obs = self._get_rgb_frame_blocking()
        info = self._get_info()
        
        self._draw_obs(obs)

        return obs, info

    def _get_pose_blocking(self) -> Dict:
        pose = self.robot.pose.capture(block=True)
        return pose

    def _get_rgb_frame_blocking(self) -> np.ndarray:
        bgr = self.robot.camera.capture(block=True)
        rgb = bgr[:, :, [2, 1, 0]]
        return rgb
    
    def _draw_obs(self, obs: np.ndarray):
        if not self.headless:
            self.window.set_data(obs)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def step(self, actions : Tuple) -> Tuple:
        # TODO: this is a hack to simulate rad/s to PWM conversion
        wl = actions[0]*0.4
        wr = actions[1]*0.4
        self.robot.motors.set_pwm(left=wl, right=wr)

        obs = self._get_rgb_frame_blocking()
        pose = self._get_pose_blocking()

        reward, terminated = self.reward_fn(pose)
        self.last_pose = pose
        info = self._get_info()
        self._draw_obs(obs)
        truncated = False

        return obs, reward, terminated, truncated, info

    def reward_fn(self, pose: Dict) -> Tuple[float, bool]:
        assert self.last_pose is not None, "Last pose is None in reward function! Make sure the environment was reset initially."
        last_pose = self.last_pose

        terminated = False
        delta_t = float(pose["header"]["timestamp"]) - float(last_pose["header"]["timestamp"])
        x, y, z = pose["position"]["x"], pose["position"]["y"], pose["position"]["z"]
        last_x, last_y, last_z = last_pose["position"]["x"], last_pose["position"]["y"], last_pose["position"]["z"]
        # Calculate speed based on position change and time delta
        dx = x - last_x
        dy = y - last_y
        dz = z - last_z
        speed = np.sqrt(dx*dx + dy*dy + dz*dz) / delta_t if delta_t > 0 else 0.0
        
        quat_rot = [pose["rotation"]["w"], pose["rotation"]["x"], pose["rotation"]["y"], pose["rotation"]["z"]]
        rot = quaternion_to_euler(quat_rot)
        try:
            lp = self.lp_cal.get_lane_pos2(np.array([x, y, z]), rot[-1])
            reward = +1.0 * speed * lp.dot_dir + -10 * np.abs(lp.dist)
        except Exception as e:
            reward = self.out_of_road_penalty
            terminated = True
        
        return reward, terminated

    def _get_info(self) -> Dict:
        info : Dict = {}
        return info
    
    def close(self):
        try:
            self._stop_components()
        except Exception:
            pass
        if not self.headless:
            try:
                import matplotlib.pyplot as plt
                plt.close(self.fig)
            except Exception:
                pass