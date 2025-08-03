# 🧠 MuJoCo Object Manager

A Python module for managing dynamic objects and full environments in MuJoCo. Designed for simulation scenarios with perception-driven updates, planning, and visualization. This project demonstrates how to integrate dynamic object manipulation into a stable MuJoCo simulation pipeline.

## 🎯 Problem Statement

MuJoCo scenes are **immutable** - once a model is loaded, you cannot add new objects without reinitializing the entire simulation via XML. This limitation makes it impossible to dynamically add objects during runtime, which is essential for robotics applications where objects appear and disappear based on perception or user input.

## 💡 Solution: Pre-initialization with Dynamic Activation

The `Environment` class solves this problem by:
1. **Pre-initializing all potential objects** at startup from XML definitions
2. **Hiding unused objects** by setting their mass to zero and disabling collisions
3. **Activating objects on-demand** by restoring their properties and positioning them
4. **Managing object lifecycle** through show/hide operations rather than add/remove

This approach provides the illusion of dynamic object creation while maintaining MuJoCo's performance and stability.

## ✨ Features

- ✅ **Environment management abstraction**: `Environment` class encapsulates MuJoCo model, data, and object management.
- 📦 **Dynamic object manipulation**: Add, move, or remove objects at runtime with `add_object()`, `move_object()`, and `remove_object()` methods.
- 🔄 **Batch object updates**: Update multiple objects simultaneously with the `update()` method.
- 🎯 **Persistence control**: Configurable object persistence with the `persist` parameter in `update()`.
- 📊 **State cloning**: Create independent copies of environment states with `clone()` and `update_from_clone()` methods.
- 💾 **State serialization**: Serialize and deserialize environment states with `pickle()` and `unpickle()` methods.
- 📂 **Modular scene files**: Scene and object definitions separated into XML components.
- 🎨 **Object lifecycle management**: Automatic object hiding and activation with collision and mass property preservation.

## 📁 Directory Structure

```
mj_environment/
├── 📄 README.md                    # This file - project documentation
├── 📄 pyproject.toml              # Python package configuration
├── 📄 .gitignore                  # Git ignore patterns
├── 📄 .python-version             # Python version specification
│
├── 📁 mj_environment/             # Main package directory
│   ├── 📄 __init__.py            # Package initialization
│   └── 📄 environment.py         # Core Environment class implementation
│
├── 📁 data/                       # Scene and object data files
│   ├── 📄 scene.xml              # Base scene definition (table + object includes)
│   └── 📁 objects/               # Object definitions and configurations
│       └── 📄 household.xml      # Household object collection (cups, plates, etc.)
│
├── 📁 demos/                      # Example applications and demonstrations
│   ├── 📄 object_update_demo.py  # Basic object manipulation demo
│   └── 📄 perception_update_demo.py # Threaded perception system demo
│
└── 📁 mj_environment.egg-info/    # Package metadata (auto-generated)
```

## 🏗️ Architecture Overview

### Core Components

**`Environment` Class** (`mj_environment/environment.py`)
- **Core Innovation**: Solves MuJoCo's immutability problem through pre-initialization
- Manages MuJoCo model and data instances with all objects pre-loaded
- Provides individual object manipulation methods (`add_object`, `move_object`, `remove_object`)
- Supports batch object updates with `update()` method
- Implements state cloning and serialization (`clone`, `update_from_clone`, `pickle`, `unpickle`)
- Handles object lifecycle through show/hide operations (not true add/remove)
- Preserves object properties (collision, mass) during state changes

**Scene Configuration** (`data/scene.xml`)
- Defines the base simulation environment
- Includes a table surface for object placement
- References external object definitions via XML includes

**Object Definitions** (`data/objects/household.xml`)
- Contains 10 common household objects (cup, plate, bowl, fork, spoon, knife, bottle, can, jar, mug)
- Each object has unique visual properties (color, size, mass)
- Objects use free joints for dynamic positioning

### Demo Applications

**Object Update Demo** (`demos/object_update_demo.py`)
- Demonstrates basic object manipulation
- Shows random object placement and movement
- Includes state cloning functionality for planning

**Perception Update Demo** (`demos/perception_update_demo.py`)
- Simulates a perception system with threaded updates
- Demonstrates thread-safe environment updates
- Shows configurable object persistence modes

## 🛠️ Requirements

- **Python** ≥ 3.9  
- **[MuJoCo](https://mujoco.readthedocs.io/en/stable/)** ≥ 2.1.0
- **NumPy** for numerical operations
- **`uv`** for dependency management (optional but recommended)

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/personalrobotics/mj_environment.git
   cd mj_environment
   ```

2. **Create and activate a virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. **Install in editable mode**:
   ```bash
   uv pip install -e .
   ```

> **Note**: You can substitute `uv` with `python -m venv` and `pip` if preferred.

## ▶️ Usage

### Basic Environment Setup

```python
from mj_environment import Environment

# Create environment with scene and object definitions
env = Environment("data/scene.xml", "data/objects/household.xml")
model = env.model
data = env.data
```

### Object Manipulation

```python
# Update objects in the scene
object_list = [
    {"name": "cup", "pos": [0.1, 0.1, 0.41], "quat": [1, 0, 0, 0]},
    {"name": "plate", "pos": [-0.1, -0.1, 0.41], "quat": [1, 0, 0, 0]}
]
env.update(object_list)

# Move individual objects
env.move_object("cup", [0.3, 0.3, 0.41], [1, 0, 0, 0])

# Clone environment state for planning
cloned_data = env.clone()
```

### Running Demos

**Object Update Demo**:
```bash
python demos/object_update_demo.py
```

**Perception Update Demo**:
```bash
python demos/perception_update_demo.py
```

> **Note**: The demos use the scene and object files from the `data/` directory.

### Controls

| Key / Mouse         | Action                  |
|---------------------|--------------------------|
| Left Mouse          | Rotate view             |
| Right Mouse         | Pan view                |
| Scroll Wheel        | Zoom in/out             |
| `Space`             | Pause/Resume simulation |
| `Esc`               | Close viewer            |

## 🔧 Configuration

### Object Properties

Objects in `objects/household.xml` have the following properties:
- **Geometry**: Cylindrical shape with configurable radius and height
- **Mass**: 0.2 kg for realistic physics
- **Friction**: 0.8 static, 0.1 dynamic, 0.1 rolling
- **Colors**: Unique RGBA values for visual distinction

### Scene Configuration

The base scene (`scene.xml`) includes:
- **Table**: 1.0m × 1.0m × 0.04m gray surface
- **Camera**: Pre-configured viewing angles and distance
- **Lighting**: Default MuJoCo lighting setup

## 📌 Technical Notes

- **MuJoCo Limitation**: Scenes are immutable - objects cannot be truly added/removed after initialization
- **Object Lifecycle**: "Adding" objects actually activates pre-initialized objects; "removing" hides them
- **Object Hiding**: Objects are hidden by setting mass to zero, disabling collisions, and moving to a configurable hide position
- **State Preservation**: Collision types, affinities, and mass properties are preserved per object and restored on activation
- **Joint Requirements**: Objects must have free joints and be defined in the objects XML file
- **State Management**: The `clone()` method creates independent copies, while `pickle()`/`unpickle()` provide serialization
- **Batch Updates**: The `update()` method supports both individual object updates and persistence control

## 🎯 Use Cases

- **Robotics Simulation**: Dynamic object manipulation for robot planning
- **Perception Testing**: Simulating vision systems with changing object configurations
- **Planning Research**: State cloning for motion planning algorithms
- **Educational**: Learning MuJoCo integration and object management

## 🔄 Workflow

1. **Initialization**: All potential objects are loaded from XML and hidden
2. **Runtime**: Objects are "added" by activating them and positioning them
3. **Updates**: Objects are moved or "removed" (hidden) as needed
4. **State Management**: Environment states can be cloned or serialized for planning

This workflow provides the flexibility of dynamic object management while respecting MuJoCo's architectural constraints.

## 📣 Future Extensions

- ✅ **State cloning and serialization**
- ✅ **Batch object updates with persistence control**
- ✅ **Object lifecycle management**
- ⏳ **ROS serialization support** - ROS message types and services for environment state communication
- ⏳ **Mesh-based object import** (URDF/STL)
- ⏳ **Support for multiple object types**
- ⏳ **Labeled logging and replay**
- ⏳ **Advanced physics properties**
- ⏳ **Thread-safe concurrent updates**

## 👨‍💻 Authors

**Siddhartha Srinivasa** - *siddh@cs.washington.edu*

## 🤝 Contributing

This project is part of the [Personal Robotics Laboratory](https://github.com/personalrobotics/) at the University of Washington. Contributions are welcome!

## 📄 License

This project follows the BSD-3-Clause license pattern used by other Personal Robotics Laboratory projects.
