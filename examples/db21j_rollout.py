"""Run a bounded random DB21J rollout against a live engine."""

import logging
import os

from gymnasium.wrappers import TimeLimit

from gym_duckiematrix.db21j_env import DuckiematrixDB21JEnv

ENTITY_NAME = "map_0/vehicle_0"
OUT_OF_ROAD_PENALTY = -10

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    env = TimeLimit(
        DuckiematrixDB21JEnv(
            ENTITY_NAME, OUT_OF_ROAD_PENALTY,
            render_mode="human",
            shm_path=os.environ.get("GYM_SHM_PATH") or None,
        ),
        max_episode_steps=100,
    )
    try:
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
    finally:
        env.close()
