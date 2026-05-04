"""Test gym mode across all entities in the map."""

import logging
import os
import time
from logging import INFO
from threading import Event
from typing import Any

from gym_duckiematrix.gym_environment import GymEnvironment

MAX_COUNT_STRING = os.environ.get("MAX_COUNT", "100")
MAX_COUNT = int(MAX_COUNT_STRING)
LEFT_PWM_STRING = os.environ.get("LEFT_PWM", "0.5")
LEFT_PWM = float(LEFT_PWM_STRING)
RIGHT_PWM_STRING = os.environ.get("RIGHT_PWM", "0.5")
RIGHT_PWM = float(RIGHT_PWM_STRING)
WAIT_TIMEOUT_SECONDS_STRING = os.environ.get("WAIT_TIMEOUT_SECONDS", "0")
WAIT_TIMEOUT_SECONDS = float(WAIT_TIMEOUT_SECONDS_STRING)

environment = GymEnvironment()
environment.enable_profiling()
event = Event()
logger = logging.getLogger(__name__)
logger.setLevel(INFO)


class _State:
    frequencies: list[float]
    count: int
    start_time: float

    def __init__(self) -> None:
        self.frequencies = []
        self.count = 0
        self.start_time = 0


state = _State()


def _callback(_: dict[str, Any]) -> None:
    if state.count >= MAX_COUNT:
        return
    actions = dict.fromkeys(
        environment.duckiebot_names,
        (LEFT_PWM, RIGHT_PWM),
    )
    environment.step(actions)
    delta_time = time.time() - state.start_time
    state.count += 1
    state.frequencies.append(1 / delta_time)
    logger.info(
        "Cycle %d/%d -- %.2f Hz -- vehicles: %s -- statics: %s",
        state.count,
        MAX_COUNT,
        state.frequencies[-1],
        environment.vehicle_names,
        environment.static_names,
    )
    if state.count == MAX_COUNT:
        environment.stop()
        event.set()
        return
    state.start_time = time.time()


if __name__ == "__main__":
    logger.info("Discovered vehicles: %s", environment.vehicle_names)
    logger.info("Discovered statics:  %s", environment.static_names)
    if not environment.vehicle_names:
        message = "No vehicle entities found in the engine map."
        raise RuntimeError(message)
    environment.attach(_callback)
    state.start_time = time.time()
    environment.start()
    completed = event.wait(
        timeout=(WAIT_TIMEOUT_SECONDS if WAIT_TIMEOUT_SECONDS > 0 else None),
    )
    if not completed:
        environment.stop()
        environment.print_profiling(logger)
        message = (
            "Timed out waiting for gym callbacks after "
            f"{WAIT_TIMEOUT_SECONDS:.1f}s "
            f"(completed {state.count}/{MAX_COUNT} cycles)."
        )
        raise TimeoutError(message)
    environment.print_profiling(logger)
    average_frequency = sum(state.frequencies) / len(state.frequencies)
    logger.info("Average frequency: %.2f Hz", average_frequency)
