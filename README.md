# mj_environment

Dynamic object management for MuJoCo simulations.

## The Problem

MuJoCo models are immutable at runtime—you cannot add or remove bodies without rebuilding the entire simulation. This makes perception-driven robotics challenging: objects detected by vision systems cannot simply appear in the scene.

## The Solution

**mj_environment** pre-initializes all possible objects at load time and controls their visibility via RGBA alpha. Objects "appear" by setting alpha to 1 and "disappear" by setting it to 0, with positions updated through the physics state. This provides dynamic object behavior without regenerating XML or restarting MuJoCo.

## Installation

```bash
git clone https://github.com/personalrobotics/mj_environment.git
cd mj_environment
uv venv && source .venv/bin/activate
uv pip install -e .
```

## Quick Start

```python
from mj_environment import Environment

env = Environment(
    base_scene_xml="data/scene.xml",
    objects_dir="data/objects",
    scene_config_yaml="data/scene_config.yaml",
)

# Activate and position objects
env.update([
    {"name": "cup_0", "pos": [0.1, 0.2, 0.4], "quat": [1, 0, 0, 0]},
    {"name": "plate_0", "pos": [-0.2, 0.0, 0.4], "quat": [1, 0, 0, 0]},
])

# Step physics
env.step()
```

## Key Features

### Forking for Motion Planning

Create independent environment clones for trajectory simulation without affecting the original state:

```python
# Fork creates a lightweight clone with independent physics state
planning_env = env.fork()
planning_env.update([{"name": "cup_0", "pos": [0.5, 0.5, 0.4], "quat": [1, 0, 0, 0]}])

for _ in range(100):
    planning_env.sim.step()

# Original environment is unchanged
assert env.data.time == 0.0
```

For parallel planners:

```python
from concurrent.futures import ThreadPoolExecutor

forks = env.fork(n=4)
with ThreadPoolExecutor() as executor:
    results = list(executor.map(planner.evaluate, forks))
```

### Syncing State from Forks

Apply processed state from a fork back to the main environment:

```python
with env.fork() as perception_fork:
    perception_fork.update(filtered_detections, persist=False)
    env.sync_from(perception_fork)
```

This pattern is useful for perception processing: filter and validate detections in an isolated fork, then commit the results to the main simulation.

### Perception Aliases

Different perception systems (YCB, COCO, custom detectors) can use their own naming conventions. The AssetManager resolves aliases to object types:

```python
obj_type = env.asset_manager.resolve_alias("coffee cup", module="coco")  # Returns "cup"
```

## Architecture

```mermaid
graph TD
    A[scene.xml] -->|Base world| E(Environment)
    B[scene_config.yaml] -->|Object counts| E
    C[data/objects/*] -->|Metadata + XML| D[AssetManager]
    D --> E
    E -->|Object lifecycle| F[ObjectRegistry]
    E -->|Physics| G[Simulation]
    F -->|Show/Hide via RGBA| G
    G -->|Render| H[MuJoCo Viewer]
```

**Environment** composes the MuJoCo scene in memory from:
- `scene.xml` — base world geometry
- `scene_config.yaml` — object types and instance counts
- `data/objects/*/` — per-object XML and metadata

**ObjectRegistry** manages object lifecycle:
- All instances are pre-loaded (e.g., `cup_0`, `cup_1`, `plate_0`)
- Hidden objects have RGBA alpha = 0 and are positioned off-scene
- `activate()` makes an object visible; `hide()` reverses this
- `update()` batch-processes perception detections

## File Structure

```
data/
├── scene.xml           # Base MuJoCo world
├── scene_config.yaml   # Object counts: {cup: 3, plate: 2}
└── objects/
    ├── cup/
    │   ├── model.xml   # MuJoCo geometry
    │   └── meta.yaml   # Metadata (mass, color, perception aliases)
    └── plate/
        ├── model.xml
        └── meta.yaml
```

Example `meta.yaml`:

```yaml
name: cup
category: [kitchenware, drinkware]
mass: 0.25
color: [0.9, 0.9, 1.0, 1.0]
scale: 1.0

mujoco:
  xml_path: model.xml

perception:
  ycb:
    aliases: ["cup", "cup001", "red cup"]
  coco:
    aliases: ["cup", "mug", "coffee cup"]
```

Values in `meta.yaml` (mass, color, scale) override those in `model.xml`.

## Running Demos

```bash
./run_demo.sh demos/dynamic_kitchen_demo.py
./run_demo.sh demos/perception_update_demo.py
```

## API Reference

### Environment

| Method | Description |
|--------|-------------|
| `update(detections, persist=False)` | Batch activate/move/hide objects |
| `fork(n=None)` | Create independent clone(s) for planning |
| `sync_from(other)` | Copy state from another environment |
| `step(ctrl=None)` | Advance physics simulation |
| `reset()` | Reset simulation state |
| `save_state(path)` | Serialize state to YAML |
| `load_state(path)` | Restore state from YAML |
| `get_object_metadata(name)` | Get object properties |

### ObjectRegistry

| Method | Description |
|--------|-------------|
| `activate(obj_type, pos, quat=None)` | Show an inactive object instance |
| `hide(name)` | Hide an active object |
| `update(detections, persist=False)` | Batch process detections |

## License

BSD-3-Clause — [Personal Robotics Laboratory](https://github.com/personalrobotics), University of Washington
