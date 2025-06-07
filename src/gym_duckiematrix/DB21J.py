from typing import Tuple, Dict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from duckietown.sdk.robots.duckiebot import DB21J
from duckietown.sdk.utils.lane_position import MapInterpreter, LanePositionCalculator
from .utils import quaternion_to_euler

DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480


class DuckiematrixDB21JEnv(gym.Env):
    def __init__(self, entity_name = "map_0/vehicle_0", out_of_road_penalty = -10.0):
        import matplotlib.pyplot as plt
        # create matplot window
        self.window = plt.imshow(np.zeros((DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH, 3)))
        plt.axis("off")
        self.fig = plt.figure(1)
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        plt.pause(0.01)

        self._shutdown = False
        #create connection to the matrix engine
        self.robot: DB21J = DB21J("map_0/vehicle_0", simulated=True)
        self.initialize_sensors()
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH, 3), dtype=np.uint8
        )
        self.map = {"frames": None, "tiles": None, "tile_info": None}
        self.get_map()
        self.map_int = MapInterpreter(map=self.map)
        self.lp_cal = LanePositionCalculator(map_interpreter=self.map_int)
        self.out_of_road_penalty = out_of_road_penalty
        self.last_pose = None

    def initialize_sensors(self):
        self.robot.camera.start()
        self.robot.motors.start()
        self.robot.map_frames.start()
        self.robot.map_tiles.start()
        self.robot.map_tile_info.start()
        self.robot.pose.start()
        self.robot.reset_flag.start()
        
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
        
        
    def reward_fn(self, pose, last_pose):
        terminated = False
        if self.last_pose is not None:
            delta_t = float(pose["header"]["timestamp"]) - float(self.last_pose["header"]["timestamp"])
        
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

    def step(self, actions : Tuple) -> Tuple:
        # TODO: this is a hack to simulate rad/s to PWM conversion
        wl = actions[0]*0.4
        wr = actions[1]*0.4

        self.robot.motors.set_pwm(left=wl, right=wr)
        bgr = self.robot.camera.capture()
        
        if bgr is None:
            print("got no image.. skipping")
            return None, None, None, None, None
        
        pose = self.robot.pose.capture()
        reward, terminated = self.reward_fn(pose, self.last_pose)
        self.last_pose = pose

        rgb = bgr[:, :, [2,1,0]]
        self.window.set_data(rgb)
        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.00001)

        info = self._get_info()
        return rgb, reward, terminated, False, info

    def reset(self,):
        while True:
            self.last_pose = self.robot.pose.capture()
            if self.last_pose is not None:
                break
        
        x, y, z = self.last_pose["position"]["x"], self.last_pose["position"]["y"], self.last_pose["position"]["z"]
        print("Intial robot position: ", x, y, z)
        self.robot.reset_flag.set_reset(True)
        obs = self.robot.camera.capture()
        info = self._get_info()
        return obs, info

    def _get_reward(self) -> float:
        #TODO
        return 0.0

    def _get_info(self) -> Dict:
        """Get the info for each robot in the environment

        Returns:
            info (Dict): A info dictionary with info for each robot
        """
        info : Dict = {}
        return info

