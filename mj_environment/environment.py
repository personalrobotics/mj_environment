"""High-level MuJoCo environment orchestrator."""

from .simulation import Simulation
from .object_registry import ObjectRegistry
from .state_io import StateIO
from .asset_manager import AssetManager
from typing import List, Dict, Any


class Environment:
    """Coordinates Simulation, ObjectRegistry, and StateIO."""

    def __init__(self, scene_xml: str, objects_xml: str, scene_config_yaml: str, verbose: bool = False):
        self.sim = Simulation(scene_xml)
        self.asset_manager = AssetManager(objects_xml, verbose=verbose)
        self.registry = ObjectRegistry(self.sim.model, self.sim.data, self.asset_manager, scene_config_yaml, verbose=verbose)
        self.state_io = StateIO()

    def update(self, object_list: List[Dict[str, Any]], persist: bool = False):
        """Batch update multiple objects in the environment."""
        self.registry.update(object_list, persist)
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