"""High-level MuJoCo environment orchestrator."""

import mujoco
import yaml
import numpy as np
import re
from xml.etree.ElementTree import Element, SubElement, tostring, parse as ETparse
from xml.dom import minidom
from typing import List, Dict, Any, Optional


def _deep_copy_element(elem: Element, parent: Optional[Element] = None) -> Element:
    """
    Recursively deep copy an XML element including all children, text, and tail.

    Args:
        elem: The element to copy
        parent: If provided, the copy will be appended to this parent

    Returns:
        A deep copy of the element
    """
    if parent is not None:
        new_elem = SubElement(parent, elem.tag, elem.attrib.copy())
    else:
        new_elem = Element(elem.tag, elem.attrib.copy())

    new_elem.text = elem.text
    new_elem.tail = elem.tail

    for child in elem:
        _deep_copy_element(child, new_elem)

    return new_elem

import os
from .simulation import Simulation
from .object_registry import ObjectRegistry
from asset_manager import AssetManager
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

        # Cache for original geom sizes (before scale overrides are applied)
        self._geom_original_size: Dict[int, np.ndarray] = {}

        # ------------------------------------------------------------------
        # 4️⃣ Apply overrides from meta.yaml (mass, color, scale)
        #    These take priority over values in XML files
        # ------------------------------------------------------------------
        self._apply_metadata_overrides(scene_config_yaml)

        # ------------------------------------------------------------------
        # 5️⃣ Initialize Simulation + Object Registry
        # ------------------------------------------------------------------
        self.sim = Simulation(self.model, self.data)
        self.registry = ObjectRegistry(
            self.model, self.data, self.asset_manager, scene_config_yaml, hide_pos, verbose
        )

        # ------------------------------------------------------------------
        # 6️⃣ Add state I/O helper for serialization
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
            if obj_type not in self.asset_manager.list():
                if self.verbose:
                    print(f"[WARN] Unknown asset '{obj_type}', skipping.")
                continue

            # Get XML path from AssetManager - relies on mujoco.xml_path in meta.yaml
            xml_path = self.asset_manager.get_path(obj_type, "mujoco")
            if xml_path is None:
                if self.verbose:
                    print(f"[WARN] No XML path found for '{obj_type}' with simulator 'mujoco', skipping.")
                continue
            
            if not os.path.exists(xml_path):
                if self.verbose:
                    print(f"[WARN] XML file not found for '{obj_type}' at {xml_path}, skipping.")
                continue

            # Parse object XML to extract worldbody contents
            obj_tree = ETparse(xml_path)
            obj_root = obj_tree.getroot()
            obj_worldbody = obj_root.find("worldbody")
            
            for i in range(count):
                instance_name = f"{obj_type}_{i}"
                # Copy all bodies from the object's worldbody
                for obj_body in obj_worldbody.findall("body"):
                    # Deep copy the entire body element tree (handles nested structures)
                    new_body = _deep_copy_element(obj_body, worldbody_el)
                    # Update name and position for this instance
                    new_body.set("name", instance_name)
                    new_body.set("pos", f"{self.hide_pos[0]} {self.hide_pos[1]} {self.hide_pos[2]}")

        xml_bytes = tostring(mujoco_el, "utf-8")
        return minidom.parseString(xml_bytes).toprettyxml(indent="  ")

    # ------------------------------------------------------------------
    # Metadata Overrides
    # ------------------------------------------------------------------
    def _apply_metadata_overrides(self, scene_config_yaml: str):
        """Apply mass, color, and scale overrides from meta.yaml to the model.
        
        These overrides take priority over values specified in the XML files.
        """
        
        with open(scene_config_yaml, "r") as f:
            cfg = yaml.safe_load(f)
        scene_cfg = cfg.get("objects", {})
        
        for obj_type, entry in scene_cfg.items():
            if obj_type not in self.asset_manager.list():
                continue
            
            # Get metadata from AssetManager
            meta = self.asset_manager.get(obj_type)
            count = entry.get("count", 1)
            
            # Apply overrides to all instances of this object type
            for i in range(count):
                instance_name = f"{obj_type}_{i}"
                try:
                    body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, instance_name)
                except (TypeError, AttributeError):
                    continue
                
                # Apply mass override
                if "mass" in meta:
                    mass_value = float(meta["mass"])
                    self.model.body_mass[body_id] = mass_value
                    if self.verbose:
                        print(f"[Override] Set mass of {instance_name} to {mass_value}")
                
                # Apply color and scale overrides to all geoms of this body
                geom_adr = self.model.body_geomadr[body_id]
                geom_num = self.model.body_geomnum[body_id]
                
                for geom_idx in range(geom_num):
                    geom_id = geom_adr + geom_idx
                    
                    # Apply color override
                    if "color" in meta:
                        color = meta["color"]
                        if isinstance(color, list) and len(color) >= 3:
                            rgba = np.array(color[:4] if len(color) >= 4 else color + [1.0], dtype=float)
                            self.model.geom_rgba[geom_id] = rgba
                            if self.verbose and geom_idx == 0:
                                print(f"[Override] Set color of {instance_name} to {rgba}")
                    
                    # Apply scale override
                    if "scale" in meta:
                        scale = float(meta["scale"])
                        # Cache original size before scaling (if not already cached)
                        if geom_id not in self._geom_original_size:
                            self._geom_original_size[geom_id] = self.model.geom_size[geom_id].copy()
                        # Apply scale to original size (not current size, to avoid compounding)
                        self.model.geom_size[geom_id] = self._geom_original_size[geom_id] * scale
                        if self.verbose and geom_idx == 0:
                            print(f"[Override] Applied scale of {scale} to {instance_name}")

    # ======================================================================
    # Public API
    # ======================================================================
    def get_object_metadata(self, instance_name: str) -> Dict[str, Any]:
        """
        Get asset metadata for an object instance.
        
        Args:
            instance_name: Instance name like "plate_1", "cup_0", "kitchen_knife_2", etc.
            
        Returns:
            Dictionary containing metadata (mass, color, scale, category, etc.)
            
        Example:
            >>> meta = env.get_object_metadata("plate_1")
            >>> print(meta["mass"], meta["color"], meta["category"])
        """
        # Find object type by looking up instance_name in the registry
        # This handles cases where object type names have underscores (e.g., "kitchen_knife")
        for obj_type, obj_info in self.registry.objects.items():
            if instance_name in obj_info["instances"]:
                return self.asset_manager.get(obj_type)
        
        # Fallback: if not found in registry, try parsing (for edge cases)
        # Remove trailing underscore and number pattern
        obj_type = re.sub(r'_\d+$', '', instance_name)
        if obj_type in self.asset_manager.list():
            return self.asset_manager.get(obj_type)
        
        raise KeyError(f"Object instance '{instance_name}' not found in registry or asset manager")
    
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
        Simulation.copy_data(self.model, self.sim.data, cloned_data)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save_state(self, path: str):
        """Serialize current simulation state to YAML."""
        self.state_io.save(self.model, self.data, self.registry.active_objects, path)

    def load_state(self, path: str):
        """Load simulation state from YAML."""
        self.registry.active_objects = self.state_io.load(self.model, self.data, path)
        # Sync visibility state to match loaded active_objects
        for name, is_active in self.registry.active_objects.items():
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self.registry._set_body_visibility(body_id, visible=is_active)