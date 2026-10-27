# Gym-Duckiematrix

Run synchronized Duckiematrix simulations through either a multi-entity
callback API or a single-Duckiebot Gymnasium environment.

## Requirements

Use Python 3.12 or newer, `duckietown-sdk>=0.3.0,<0.4`, and
`duckietown-messages>=0.1.0,<0.2`. The matching SDK and messages releases must
be available before a fresh package installation can succeed. During local
development, install the matching checkouts first.

On Linux, install the `libturbojpeg0` system package. From this checkout:

```bash
python3 -m pip install -e .
```

Start a compatible Duckiematrix engine and renderer in Gym mode:

```bash
dts matrix run --gym --standalone --embedded --no-tutorial --map loop
```

The engine and renderer must support the global `robot/gym/in` and
`robot/gym/out` session protocol used by the companion
[SDK integration](https://github.com/duckietown/duckietown-sdk/pull/6).

## Multiple Vehicles

`GymEnvironment` discovers vehicles and static entities. Its callback
receives one world-input dictionary per simulation session, with sensor
payloads under `entities`. Publish all vehicle actions together from that
callback:

```python
from threading import Event

from gym_duckiematrix import GymEnvironment

environment = GymEnvironment()

def on_world_input(world_input):
    actions = dict.fromkeys(environment.duckiebot_names, (0.2, 0.2))
    environment.step(actions)

environment.attach(on_world_input)
try:
    environment.start()
    Event().wait(10)
finally:
    environment.stop()
```

Actions are `(left_pwm, right_pwm)` tuples keyed by vehicle name. Static
entities are ignored. Duplicate and older sessions do not invoke the
callback. Call `stop()` explicitly when finished.

The finite-cycle demo reports frequency between completed cycles, excluding
startup latency:

```bash
MAX_COUNT=100 WAIT_TIMEOUT_SECONDS=30 python3 examples/gym_mode.py
```

## Single-Duckiebot Gymnasium API

```python
from gymnasium.wrappers import TimeLimit

from gym_duckiematrix.db21j_env import DuckiematrixDB21JEnv

environment = TimeLimit(
    DuckiematrixDB21JEnv(entity_name="map_0/vehicle_0", timeout=10.0),
    max_episode_steps=100,
)
try:
    observation, info = environment.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action = environment.action_space.sample()
        observation, reward, terminated, truncated, info = environment.step(
            action,
        )
finally:
    environment.close()
```

This environment opens a Matplotlib camera window. It consumes an image and
pose from the same world-input session. Map-layer and world-state reads use
the positive `timeout` in seconds; unavailable data raises `TimeoutError`,
and an incomplete session raises `ValueError` identifying the entity and
session. The reward uses forward lane progress and lateral lane deviation;
leaving a supported lane terminates the episode with the configured
`out_of_road_penalty`.

Run `python3 examples/db21j_rollout.py` for a bounded random rollout. The
obsolete `gym_duckiematrix.DB21J` and `gym_duckiematrix.db21j` modules have
been removed. Import the class from `gym_duckiematrix.db21j_env`; legacy
`headless` and camera-size arguments are not part of this API.

## Shared-Memory Transport

Without `DTSHELL_SHM_PATH`, both environments use network DTPS on
`127.0.0.1:7501`. To select native shared memory, give the engine and Python
process the same base path:

```bash
export DTSHELL_SHM_PATH=/tmp/duckiematrix-gym
dts matrix run --gym --standalone --embedded --no-tutorial --map loop \
    --shm-path "$DTSHELL_SHM_PATH"
```

Keep the environment variable set when starting Python. DTPS uses separate
`<base>.world_input` and `<base>.world_output` channels. Both processes must
see the same backing files; containerized runs need a shared mount. This
requires the native shared-memory support in `dtps-http>=1.6.1,<2` and
matching engine/renderer changes. Shared-memory errors are reported without
falling back to network transport.

## Development Tests

Install `requirements.txt` after the matching SDK/messages packages are
available, then run `make test` from this checkout. The tests collect and
run without an engine. Live demos are in `examples/` and execute only when
run as scripts. Use `ruff check .` for the lint gate.
