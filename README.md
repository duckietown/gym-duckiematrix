# Gym-Duckiematrix

Run synchronized Duckiematrix simulations through either a multi-entity
callback API or a single-Duckiebot Gymnasium environment.

## Requirements

Use Python 3.12 or newer, `duckietown-sdk>=0.3.0,<0.4`, and
`duckietown-messages>=0.1.0,<0.2`. Package installation requires compatible
SDK and messages releases to be available on PyPI.

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
callback. Call `stop()` explicitly when finished. Starting a stopped
environment resets its local session guards. After an Engine restart,
restart the Python client to establish fresh connections. A lower session
ID alone is treated as stale traffic; the protocol has no Engine-generation
identifier. Recreate the environment if the Engine's entity list or map
changed.

The finite-cycle demo reports frequency between completed cycles, excluding
startup latency:

```bash
MAX_COUNT=100 WAIT_TIMEOUT_SECONDS=30 python3 examples/gym_mode.py
```

## Parallel Environments

Use matching SDK, Gym, Engine and Renderer builds. Start the Engine with
the desired number of replicas; Python discovers that count and mapping:

```bash
dts matrix run --gym --standalone --embedded --no-tutorial --map loop \
    --environments 4 --cameras-only
MAX_COUNT=100 WAIT_TIMEOUT_SECONDS=30 python3 examples/gym_mode.py
```

`GymEnvironment.environments` exposes the SDK's immutable replica
manifest in numeric index order, preserving keys such as
`environment_2/map_0/vehicle_0`. `vehicle_names` follows that order rather
than sorting replica names lexicographically. Constructor discovery waits
for every advertised vehicle and rejects missing or inconsistent metadata.
Each replicated `step(actions)` must command every advertised vehicle;
its input callback requires every vehicle's entity payload in that session.
The source `loop` map uses DB21M, matching the bridge's default SDK model.

The bridge also accepts `WorldEntityOutput` models built by its existing
SDK vehicles. Inside the same world-input callback, use either
`step(actions)` or this command path once for the complete batch:

```python
vehicles = environment.vehicles
commands = {}
for name, vehicle in vehicles.items():
    commands[name] = vehicle.make_world_entity_output(
        left_pwm=0.0, right_pwm=0.0, reset_state=True,
    )
environment.step_commands(commands)
```

Both paths use the same global session and output transport, including
when SHM is selected. `environment.vehicles` returns a copy of the vehicle
mapping without starting additional sensor or actuator streams.

This supports controlled parallel diagnostics, not a Gymnasium vector
training API. The Engine does not yet enforce an exclusive coordinator,
bind messages to a run ID, or implement reset-only transitions. Its current
reset flag resets a vehicle and then integrates it. Do not interpret that
as `NEXT_STEP` autoreset, wrap independent scalar clients in
`SyncVectorEnv`/`AsyncVectorEnv` against one Engine, or rely on independent
episode/reset guarantees. Run exactly one global Gym publisher and restart
the owned simulation and client together after an uncertain step failure.

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

The default environment opens no window and does not import Matplotlib.
With `render_mode=None`, `render()` returns `None`.
Use `render_mode="human"` for a lazily created camera window or
`render_mode="rgb_array"` to return the current RGB image from `render()`.
The rollout example explicitly selects human rendering.

Set `camera_width` and `camera_height` to the renderer's image dimensions;
the defaults are 640 by 480. Both accept positive Python or NumPy integers.
These options define the observation space and validate incoming images;
they do not resize images or reconfigure the renderer. Actions must contain
two finite values; values outside `[-1, 1]` are clipped before applying the
existing 0.4 motor scaling.

The environment consumes an image and pose from the same world-input
session. Map-layer and world-state reads use
the positive `timeout` in seconds; unavailable data raises `TimeoutError`,
and an incomplete session raises `ValueError` identifying the entity and
session. The reward uses forward lane progress and lateral lane deviation;
leaving a supported lane terminates the episode with the configured
`out_of_road_penalty`.

The scalar wrapper rejects replicated Engines before creating publishers.
Use one `GymEnvironment` bridge for parallel diagnostics. Both APIs accept
`host` and `port` for discovery, SDK map components and world transports,
for example `DuckiematrixDB21JEnv(host="engine.example", port=7601)`.

Run `python3 examples/db21j_rollout.py` for a bounded random rollout. The
obsolete `gym_duckiematrix.DB21J` and `gym_duckiematrix.db21j` modules have
been removed. Import the class from `gym_duckiematrix.db21j_env`; replace
the legacy `headless` option with `render_mode`.

## Shared-Memory Transport

Both environments default to network DTPS on `127.0.0.1:7501`, regardless
of `DTSHELL_SHM_PATH`. Select native shared memory explicitly with
`GymEnvironment(shm_path="/tmp/duckiematrix-gym")` or the same keyword on
`DuckiematrixDB21JEnv`. The selected transport is logged at INFO level.
Give the Engine and Python process the same base path. The examples accept
the separate, opt-in `GYM_SHM_PATH` variable:

```bash
export GYM_SHM_PATH=/tmp/duckiematrix-gym
dts matrix run --gym --standalone --embedded --no-tutorial --map loop \
    --shm-path "$GYM_SHM_PATH"
python3 examples/gym_mode.py
```

The environment variable affects these examples, not the environment
constructors. An unset or empty `GYM_SHM_PATH` selects network transport in
the examples. DTPS uses separate
`<base>.world_input` and `<base>.world_output` channels. Both processes must
see the same backing files; containerized runs need a shared mount. This
requires the native shared-memory support in `dtps-http>=1.6.1,<2` and
matching engine/renderer changes. Shared-memory errors are reported without
falling back to network transport.

## Profiling

Call `GymEnvironment.enable_profiling()` and `print_profiling()` to report
application and transport timings. With the matching SDK review fixes,
tables show lifetime means alongside median and p95 latency from the last
200 samples. The sample windows are labeled explicitly. The retained first
WorldInput can be old, so its Engine-send-to-receive delay is a startup age,
not steady-state transport latency. Do not compare lifetime means alone.

Three-way intersection rewards use the corrected lane geometry in the
matching SDK checkout or release; the SDK regressions cover all six routes
and their orientation in the shipped sandbox map.

## Development Tests

Install `requirements.txt` after the matching SDK/messages packages are
available, then run `make test` from this checkout. The tests collect and
run without an engine. Live demos are in `examples/` and execute only when
run as scripts. Use `ruff check .` for the lint gate.
