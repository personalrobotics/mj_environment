# MuJoCo Object Manager

A Python project for managing and visualizing objects in a MuJoCo physics simulation environment. The project demonstrates dynamic object positioning and physics-based interactions.

## Features

- Dynamic object positioning with random placement
- Physics-based simulation using MuJoCo
- Interactive 3D visualization
- Automatic height calculation based on object and table dimensions

## Requirements

- Python >= 3.9
- MuJoCo
- NumPy

## Installation

1. Create a virtual environment:
```bash
uv venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
uv pip install -e .
```

## Usage

Run the simulation:
```bash
mjpython main.py
```

Controls:
- Left mouse button: Rotate view
- Right mouse button: Pan view
- Scroll wheel: Zoom
- Space: Pause/Resume
- Esc: Close viewer
