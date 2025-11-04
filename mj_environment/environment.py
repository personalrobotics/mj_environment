"""High-level MuJoCo environment orchestrator."""

import mujoco
import yaml
from xml.etree.ElementTree import Element, SubElement, tostring, parse as ETparse
from xml.dom import minidom
from typing import List, Dict, Any

from .simulation import Simulation
from .object_registry import ObjectRegistry
from .asset_manager import AssetManager
from .state_io import StateIO


class Environment:
    """
    Unified entry point for MuJoCo simulation, asset management, and object lifecycle.

    Responsibilities:
      - Load all object metadata via AssetManager
      - Dynamically compose a MuJoCo scene XML in memory
      - Initialize Simulation and ObjectRegistry
      - Provide cloning, updating, and serialization
    """

    def __init__(
        self,
        base_scene_xml: str,
        objects_dir: str,
        scene_config_yaml: str,
        hide_pos: List[float] = [0, 0, -1],
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.hide_pos = hide_pos

        # ------------------------------------------------------------------
        # 1️⃣ Asset Manager: load YAML metadata + object XMLs
        # ------------------------------------------------------------------
        self.asset_manager = AssetManager(base_dir=objects_dir, verbose=verbose)

        # ------------------------------------------------------------------
        # 2️⃣ Build in-memory XML string for the complete scene
        # ------------------------------------------------------------------
        xml_string = self._build_scene_xml_string(base_scene_xml, scene_config_yaml)

        # ------------------------------------------------------------------
        # 3️⃣ Create MuJoCo model + data directly from XML string
        # ------------------------------------------------------------------
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)

        # ------------------------------------------------------------------
        # 4️⃣ Initialize Simulation + Object Registry
        # ------------------------------------------------------------------
        self.sim = Simulation(self.model, self.data)
        self.registry = ObjectRegistry(
            self.model, self.data, self.asset_manager, scene_config_yaml, hide_pos, verbose
        )

        # ------------------------------------------------------------------
        # 5️⃣ Add state I/O helper for serialization
        # ------------------------------------------------------------------
        self.state_io = StateIO()

        if verbose:
            print(f"[Environment] Loaded scene with {len(self.registry.objects)} object types.")

    # ======================================================================
    # Internal: Scene Composition
    # ======================================================================
    def _build_scene_xml_string(self, base_scene_xml: str, scene_yaml: str) -> str:
        """
        Build a MuJoCo scene XML in memory by combining:
        - the base scene (e.g., table, lights, cameras)
        - all object instances defined in scene_config.yaml
        """
        with open(scene_yaml, "r") as f:
            cfg = yaml.safe_load(f)
        if "objects" not in cfg:
            raise ValueError(f"No 'objects' field found in {scene_yaml}")
        scene_cfg = cfg["objects"]

        mujoco_el = Element("mujoco", {"model": "autogen_scene"})
        SubElement(mujoco_el, "include", {"file": base_scene_xml})
        worldbody_el = SubElement(mujoco_el, "worldbody")

        for obj_type, entry in scene_cfg.items():
            count = entry.get("count", 1)
            if not self.asset_manager.has(obj_type):
                print(f"[WARN] Unknown asset '{obj_type}', skipping.")
                continue

            meta = self.asset_manager.get(obj_type)
            xml_path = meta["xml_path"]

            # Parse object XML to extract worldbody contents
            obj_tree = ETparse(xml_path)
            obj_root = obj_tree.getroot()
            obj_worldbody = obj_root.find("worldbody")
            
            for i in range(count):
                instance_name = f"{obj_type}_{i}"
                # Copy all bodies from the object's worldbody
                for obj_body in obj_worldbody.findall("body"):
                    # Clone the body element
                    new_body = SubElement(
                        worldbody_el,
                        "body",
                        obj_body.attrib.copy()
                    )
                    # Update position to hide position
                    new_body.set("name", instance_name)
                    new_body.set("pos", f"{self.hide_pos[0]} {self.hide_pos[1]} {self.hide_pos[2]}")
                    # Copy all children (geoms, joints, etc.)
                    for child in obj_body:
                        SubElement(new_body, child.tag, child.attrib)

        xml_bytes = tostring(mujoco_el, "utf-8")
        return minidom.parseString(xml_bytes).toprettyxml(indent="  ")

    # ======================================================================
    # Public API
    # ======================================================================
    def update(self, object_list: List[Dict[str, Any]], persist: bool = False):
        """
        Batch update multiple objects in the environment.

        Each entry in object_list should have:
          {"name": str, "pos": [x, y, z], "quat": [w, x, y, z]}
        """
        self.registry.update(object_list, persist)
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl=None):
        """Advance simulation by one step."""
        self.sim.step(ctrl)

    def reset(self):
        """Reset simulation and all objects."""
        self.sim.reset()

    # ------------------------------------------------------------------
    # ✅ Cloning and State Sync
    # ------------------------------------------------------------------
    def clone_data(self) -> mujoco.MjData:
        """Return a deep clone of the current simulation data."""
        return self.sim.clone_data()

    def update_from_clone(self, cloned_data: mujoco.MjData):
        """Restore this environment's state from a cloned MjData."""
        if cloned_data.model is not self.sim.model:
            raise ValueError("Cannot restore: cloned_data belongs to a different model.")
        Simulation.copy_data(self.sim.data, cloned_data)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save_state(self, path: str):
        """Serialize current simulation state to YAML."""
        self.state_io.save(self.model, self.data, self.registry.active_objects, path)

    def load_state(self, path: str):
        """Load simulation state from YAML."""
        self.registry.active_objects = self.state_io.load(self.model, self.data, path)