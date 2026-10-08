"""Test DB21J."""

import logging

from gym_duckiematrix.db21j_env import DuckiematrixDB21JEnv

ENTITY_NAME = "map_0/vehicle_0"
OUT_OF_ROAD_PENALTY = -10

env = DuckiematrixDB21JEnv(ENTITY_NAME, OUT_OF_ROAD_PENALTY)
observation, info = env.reset()
episode_over = False
total_reward: float = 0
while not episode_over:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_over = terminated or truncated
logger = logging.getLogger(__name__)
logger.info("Episode finished! Total reward: %s", total_reward)
env.close()
