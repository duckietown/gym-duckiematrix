"""Run synchronized Gym cycles across the map's Duckiebots."""

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
WAIT_TIMEOUT_SECONDS_STRING = os.environ.get("WAIT_TIMEOUT_SECONDS", "30")
WAIT_TIMEOUT_SECONDS = float(WAIT_TIMEOUT_SECONDS_STRING)

event = Event()
logger = logging.getLogger(__name__)
logger.setLevel(INFO)


class _State:
    count: int
    first_cycle_at: float
    last_cycle_at: float

    def __init__(self) -> None:
        self.count = 0
        self.first_cycle_at = 0
        self.last_cycle_at = 0


state = _State()


def _callback(_: dict[str, Any]) -> None:
    if state.count >= MAX_COUNT:
        return
    actions = dict.fromkeys(
        environment.duckiebot_names,
        (LEFT_PWM, RIGHT_PWM),
    )
    environment.step(actions)
    cycle_at = time.perf_counter()
    if state.count == 0:
        state.first_cycle_at = cycle_at
    state.last_cycle_at = cycle_at
    state.count += 1
    logger.info(
        "Cycle %d/%d -- vehicles: %s -- statics: %s",
        state.count,
        MAX_COUNT,
        environment.vehicle_names,
        environment.static_names,
    )
    if state.count == MAX_COUNT:
        event.set()


if __name__ == "__main__":
    logging.basicConfig(level=INFO)
    if MAX_COUNT <= 0:
        message = "MAX_COUNT must be greater than zero."
        raise ValueError(message)
    environment = GymEnvironment(
        shm_path=os.environ.get("GYM_SHM_PATH") or None,
    )
    environment.enable_profiling()
    logger.info("Discovered vehicles: %s", environment.vehicle_names)
    logger.info("Discovered statics:  %s", environment.static_names)
    if not environment.vehicle_names:
        message = "No vehicle entities found in the engine map."
        raise RuntimeError(message)
    environment.attach(_callback)
    try:
        environment.start()
        completed = event.wait(
            timeout=(
                WAIT_TIMEOUT_SECONDS if WAIT_TIMEOUT_SECONDS > 0 else None
            ),
        )
    except KeyboardInterrupt:
        logger.info("Interrupted after %d/%d cycles.", state.count, MAX_COUNT)
        raise SystemExit(130) from None
    finally:
        environment.stop()
        environment.print_profiling(logger)
    if not completed:
        message = (
            "Timed out waiting for gym callbacks after "
            f"{WAIT_TIMEOUT_SECONDS:.1f}s "
            f"(completed {state.count}/{MAX_COUNT} cycles)."
        )
        raise TimeoutError(message)
    elapsed = state.last_cycle_at - state.first_cycle_at
    if state.count > 1 and elapsed > 0:
        average_frequency = (state.count - 1) / elapsed
        logger.info("Average frequency: %.2f Hz", average_frequency)
    else:
        logger.info("At least two cycles are required to measure frequency.")
