import Tuple, Dict
import gymnasium as gym
from gymnasium import spaces

from duckietown.sdk.robots.duckiebot import DB21J

DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480


class DuckiematrixDB21JEnv(gym.Env,):
    def __init__(self, entity_name = "map_0/vehicle_0"):
        self._shutdown = False
        #create connection to the matrix engine
        self.robot: DB21J = DB21J(entity_name, simulated=True)
        self.robot.camera.start()
        self.robot.motors.start()
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH, 3), dtype=np.uint8
        )
        # TODO wheel encoder data? self.robot.encoder

    def step(self, actions : Tuple) -> Tuple:
        # TODO: this is a hack to simulate rad/s to PWM conversion
        wl = actions[0]*0.04
        wr = actions[1]*0.04

        self.robot.motors.set_pwm(left=wl, right=wr)
        obs = self.robot.camera.capture()
        terminated = truncated = False
        rew = self._get_reward()
        return obs, rew, terminated, truncated, info

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

