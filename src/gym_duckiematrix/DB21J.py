from typing import Tuple, Dict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from duckietown.sdk.robots.duckiebot import DB21J

DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480


class DuckiematrixDB21JEnv(gym.Env):
    def __init__(self, entity_name = "map_0/vehicle_0"):
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
        self.robot.camera.start()
        self.robot.motors.start()
        self.robot.map.start()
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH, 3), dtype=np.uint8
        )

        self.map = self.robot.map.capture()
        
        # TODO wheel encoder data? self.robot.encoder

    def step(self, actions : Tuple) -> Tuple:
        # TODO: this is a hack to simulate rad/s to PWM conversion
        wl = actions[0]*0.4
        wr = actions[1]*0.4

        self.robot.motors.set_pwm(left=wl, right=wr)
        bgr = self.robot.camera.capture()
        if bgr is None:
            print("got no image.. skipping")
            return None, None, None, None, None
        
        print(self.map)
        rgb = bgr[:, :, [2,1,0]]
        self.window.set_data(rgb)
        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.00001)

        terminated = truncated = False
        rew = self._get_reward()
        info = self._get_info()
        return rgb, rew, terminated, truncated, info

    def reset(self,):
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

