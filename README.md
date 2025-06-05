# 🧠 MuJoCo Object Manager

A Python module for managing dynamic objects and full environments in MuJoCo. Designed for simulation scenarios with perception-driven updates, planning, and visualization. This project demonstrates how to integrate dynamic object manipulation into a stable MuJoCo simulation pipeline.

## ✨ Features

- ✅ **Environment management abstraction**: `Environment` class encapsulates model, data, object access, and update logic.
- 📦 **Dynamic object updates**: Add, move, or remove objects at runtime based on external input.
- 📊 **Automatic placement**: Computes object heights from the scene for accurate table placement.
- 🎨 **Custom visuals**: Color-coded common household objects.
- 🗾️ **3D viewer**: Passive MuJoCo viewer integration with custom camera positioning.
- 📂 **Modular scene files**: Scene and object definitions separated into XML components.

## 🗉 Directory Structure

```
.
├── main.py               # Entry point: runs simulation loop
├── environment.py        # Environment manager for MuJoCo
├── scene.xml             # Base scene (table, includes objects)
├── objects/
│   └── household.xml     # Household objects (cylinder geoms)
└── README.md
```

## 🛠️ Requirements

- Python ≥ 3.9  
- [MuJoCo](https://mujoco.readthedocs.io/en/stable/)
- NumPy  
- `uv` for dependency management (optional but recommended)

## 🚀 Installation

1. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate
   ```

2. Install in editable mode:
   ```bash
   uv pip install -e .
   ```

> You can substitute `uv` with `python -m venv` and `pip` if preferred.

## ▶️ Usage

Run the simulation:

```bash
mjpython main.py
```

### Controls

| Key / Mouse         | Action                  |
|---------------------|--------------------------|
| Left Mouse          | Rotate view             |
| Right Mouse         | Pan view                |
| Scroll Wheel        | Zoom in/out             |
| `Space`             | Pause/Resume simulation |
| `Esc`               | Close viewer            |

## 📌 Notes

- Objects are hidden when not in use by assigning zero mass and moving them far away.
- Geometry, collision, and mass properties are preserved per object and restored on activation.
- Objects are assumed to have free joints and be defined in `household.xml`.

## 📣 Future Extensions

- ✅ Perception system integration
- ✅ Planner environment cloning
- ⏳ Mesh-based object import (URDF/STL)
- ⏳ Support for multiple object types
- ⏳ Labeled logging and replay
