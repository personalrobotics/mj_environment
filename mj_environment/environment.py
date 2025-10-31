from .simulation import Simulation
from .object_registry import ObjectRegistry
from .state_io import StateIO

class Environment:
    """High-level orchestrator combining simulation and object registry."""
    def __init__(self, scene_xml, objects_xml, hide_pos=[0, 0, -1]):
        self.sim = Simulation(scene_xml)
        self.registry = ObjectRegistry(self.sim.model, self.sim.data, objects_xml, hide_pos)
        self.state_io = StateIO()

    def update(self, object_list, persist=False):
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