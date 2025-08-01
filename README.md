# 🧠 MuJoCo Object Manager

A Python module for managing dynamic objects and full environments in MuJoCo. Designed for simulation scenarios with perception-driven updates, planning, and visualization. This project demonstrates how to integrate dynamic object manipulation into a stable MuJoCo simulation pipeline.

## ✨ Features

- ✅ **Environment management abstraction**: `Environment` class encapsulates model, data, object access, and update logic.
- 📦 **Dynamic object updates**: Add, move, or remove objects at runtime based on external input.
- 📊 **Automatic placement**: Computes object heights from the scene for accurate table placement.
- 🎨 **Custom visuals**: Color-coded common household objects.
- 🗾️ **3D viewer**: Passive MuJoCo viewer integration with custom camera positioning.
- 📂 **Modular scene files**: Scene and object definitions separated into XML components.
- 🔄 **State cloning**: Create independent copies of environment states for planning.
- 🧵 **Thread-safe updates**: Support for perception systems running in separate threads.
- 🎯 **Persistence control**: Configurable object persistence for different perception models.

## 📁 Directory Structure

```
mj_environment/
├── 📄 README.md                    # This file - project documentation
├── 📄 pyproject.toml              # Python package configuration
├── 📄 scene.xml                   # Base scene definition (table + object includes)
├── 📄 .gitignore                  # Git ignore patterns
├── 📄 .python-version             # Python version specification
│
├── 📁 mj_environment/             # Main package directory
│   ├── 📄 __init__.py            # Package initialization
│   └── 📄 environment.py         # Core Environment class implementation
│
├── 📁 demos/                      # Example applications and demonstrations
│   ├── 📄 object_update_demo.py  # Basic object manipulation demo
│   └── 📄 perception_update_demo.py # Threaded perception system demo
│
├── 📁 objects/                    # Object definitions and configurations
│   └── 📄 household.xml          # Household object collection (cups, plates, etc.)
│
└── 📁 mj_environment.egg-info/    # Package metadata (auto-generated)
```

## 🏗️ Architecture Overview

### Core Components

**`Environment` Class** (`mj_environment/environment.py`)
- Manages MuJoCo model and data instances
- Handles dynamic object updates and state management
- Provides object manipulation methods (add, move, remove)
- Supports state cloning for planning scenarios

**Scene Configuration** (`scene.xml`)
- Defines the base simulation environment
- Includes a table surface for object placement
- References external object definitions via XML includes

**Object Definitions** (`objects/household.xml`)
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
env = Environment("scene.xml", "objects/household.xml")
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

- **Object Hiding**: Objects are hidden when not in use by assigning zero mass and moving them far away
- **State Preservation**: Geometry, collision, and mass properties are preserved per object and restored on activation
- **Joint Assumptions**: Objects are assumed to have free joints and be defined in `household.xml`
- **Thread Safety**: The `Environment` class supports concurrent updates from multiple threads
- **Memory Management**: State cloning creates independent copies for planning without affecting the original

## 🎯 Use Cases

- **Robotics Simulation**: Dynamic object manipulation for robot planning
- **Perception Testing**: Simulating vision systems with changing object configurations
- **Planning Research**: State cloning for motion planning algorithms
- **Educational**: Learning MuJoCo integration and object management

## 📣 Future Extensions

- ✅ **Perception system integration**
- ✅ **Planner environment cloning**
- ✅ **Thread-safe updates**
- ⏳ **Mesh-based object import** (URDF/STL)
- ⏳ **Support for multiple object types**
- ⏳ **Labeled logging and replay**
- ⏳ **ROS2 integration**
- ⏳ **Advanced physics properties**

## 👨‍💻 Authors

**Siddhartha Srinivasa** - *siddh@cs.washington.edu*

## 🤝 Contributing

This project is part of the [Personal Robotics Laboratory](https://github.com/personalrobotics/) at the University of Washington. Contributions are welcome!

## 📄 License

This project follows the BSD-3-Clause license pattern used by other Personal Robotics Laboratory projects.
