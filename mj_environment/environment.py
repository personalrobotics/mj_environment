"""High-level MuJoCo environment orchestrator."""

from .simulation import Simulation
from .object_registry import ObjectRegistry
from .state_io import StateIO
from typing import List, Dict, Any


class Environment:
    """Coordinates Simulation, ObjectRegistry, and StateIO."""

    def __init__(self, scene_xml: str, objects_xml: str, hide_pos=[0, 0, -1], verbose: bool = False):
        self.sim = Simulation(scene_xml)
        self.registry = ObjectRegistry(self.sim.model, self.sim.data, objects_xml, hide_pos, verbose=verbose)
        self.state_io = StateIO()

    def update(self, object_list: List[Dict[str, Any]], persist: bool = False):
        """Batch update multiple objects in the environment."""
        names = set()
        for obj in object_list:
            name, pos, quat = obj["name"], obj["pos"], obj["quat"]
            names.add(name)
            if obj.get("status", "active") == "active":
                if name in self.registry.active_objects:
                    self.registry.move(name, pos, quat)
                else:
                    self.registry.activate(name, pos, quat)
            else:
                self.registry.hide(name)

        if not persist:
            for name in list(self.registry.active_objects):
                if name not in names:
                    self.registry.hide(name)

        self.sim.forward()

    def step(self, ctrl=None):
        """Advance simulation by one step."""
        self.sim.step(ctrl)

    def reset(self):
        """Reset simulation and all objects."""
        self.sim.reset()

    def save_state(self, path: str):
        """Serialize current simulation state to YAML."""
        self.state_io.save(self.sim.model, self.sim.data, self.registry.active_objects, path)

    def load_state(self, path: str):
        """Load simulation state from YAML."""
        self.registry.active_objects = self.state_io.load(self.sim.model, self.sim.data, path)