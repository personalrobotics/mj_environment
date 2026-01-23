"""High-level MuJoCo environment orchestrator."""

import mujoco
import yaml
import numpy as np
import re
from xml.etree.ElementTree import Element, SubElement, tostring, parse as ETparse
from xml.dom import minidom
from typing import List, Dict, Any, Optional, Union, overload


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
from .exceptions import ConfigurationError, ObjectNotFoundError
from .types import Detection, ObjectMetadata


class Environment:
    """
    Unified entry point for MuJoCo simulation, asset management, and object lifecycle.

    Responsibilities:
      - Load all object metadata via AssetManager
      - Dynamically compose a MuJoCo scene XML in memory
      - Initialize Simulation and ObjectRegistry
      - Provide forking, updating, and serialization
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

        # Track whether this is a fork (for potential future use)
        self._is_fork = False

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
        if not os.path.exists(scene_yaml):
            raise ConfigurationError(
                f"Scene config file not found: {scene_yaml}",
                path=scene_yaml,
                hint="Create the file or check the path passed to Environment().",
            )
        with open(scene_yaml, "r") as f:
            cfg = yaml.safe_load(f)
        if "objects" not in cfg:
            raise ConfigurationError(
                "Scene config must define 'objects' key",
                path=scene_yaml,
                hint="Add an 'objects' section with object types and counts.",
            )
        scene_cfg = cfg["objects"]

        mujoco_el = Element("mujoco", {"model": "autogen_scene"})
        # Convert to absolute path for reliable include resolution
        abs_scene_path = os.path.abspath(base_scene_xml)
        SubElement(mujoco_el, "include", {"file": abs_scene_path})
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
            if obj_worldbody is None:
                if self.verbose:
                    print(f"[WARN] No worldbody found in {xml_path}, skipping.")
                continue

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
    def get_object_metadata(self, instance_name: str) -> ObjectMetadata:
        """
        Get asset metadata for an object instance.

        Args:
            instance_name: Instance name like "plate_1", "cup_0", "kitchen_knife_2", etc.

        Returns:
            ObjectMetadata dict with keys: name, category, mass, color, scale, etc.

        Raises:
            ObjectNotFoundError: If instance_name is not in the registry.

        Example:
            >>> meta = env.get_object_metadata("plate_1")
            >>> print(meta["mass"], meta["color"], meta["category"])
        """
        # Find object type by looking up instance_name in the registry
        # This handles cases where object type names have underscores (e.g., "kitchen_knife")
        for registered_type, obj_info in self.registry.objects.items():
            if instance_name in obj_info["instances"]:
                return self.asset_manager.get(registered_type)

        # Fallback: use registry's parsing logic for edge cases
        parsed_type = self.registry._parse_object_type(instance_name)
        if parsed_type is not None and parsed_type in self.asset_manager.list():
            return self.asset_manager.get(parsed_type)
        
        all_instances = list(self.registry.active_objects.keys())
        raise ObjectNotFoundError(instance_name, all_instances)
    
    def update(
        self,
        object_list: Union[List[Detection], List[Dict[str, Any]]],
        persist: bool = False,
    ) -> None:
        """
        Batch update multiple objects in the environment.

        Args:
            object_list: List of detection dicts with keys:
                - name: Instance name (e.g., "cup_0")
                - pos: Position [x, y, z]
                - quat: (optional) Quaternion [w, x, y, z], defaults to [1, 0, 0, 0]
            persist: If False (default), hide objects not in the list.
                     If True, keep previously active objects visible.
        """
        self.registry.update(object_list, persist)  # type: ignore[arg-type]
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl: Optional[np.ndarray] = None) -> None:
        """Advance simulation by one step."""
        self.sim.step(ctrl)

    def reset(self) -> None:
        """Reset simulation and all objects."""
        self.sim.reset()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save_state(self, path: str) -> None:
        """Serialize current simulation state to YAML."""
        self.state_io.save(self.model, self.data, self.registry.active_objects, path)

    def load_state(self, path: str) -> None:
        """Load simulation state from YAML."""
        self.registry.active_objects = self.state_io.load(self.model, self.data, path)
        # Sync visibility state to match loaded active_objects
        for name, is_active in self.registry.active_objects.items():
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self.registry._set_body_visibility(body_id, visible=is_active)

    # ------------------------------------------------------------------
    # Forking for Planning
    # ------------------------------------------------------------------
    @overload
    def fork(self, n: None = None) -> 'Environment': ...
    @overload
    def fork(self, n: int) -> List['Environment']: ...

    def fork(self, n: Optional[int] = None) -> Union['Environment', List['Environment']]:
        """
        Create fully functional clone(s) with independent state for planning.

        Forked environments share immutable data (MjModel, AssetManager) but have
        independent simulation state (MjData, ObjectRegistry). This enables:
        - Motion planning without polluting the original environment
        - Multiple planners running in parallel on separate forks

        Args:
            n: Number of forks to create. If None, returns a single Environment.
               If an integer, returns a list of Environments.

        Returns:
            A single Environment if n is None, or a list of n Environments.

        Example:
            # Single fork for planning
            planning_env = env.fork()
            planning_env.update([{"name": "cup_0", "pos": [0.1, 0.2, 0.3]}])
            trajectory = planner.plan(planning_env, goal)
            # Original env unchanged

            # Multiple forks for parallel planning
            forks = env.fork(n=4)
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(planner.plan, forks))

            # Context manager for explicit scope
            with env.fork() as planning_env:
                trajectory = planner.plan(planning_env)
        """
        if n is None:
            return self._create_fork()
        return [self._create_fork() for _ in range(n)]

    def _create_fork(self) -> 'Environment':
        """Create a single forked environment."""
        fork = Environment.__new__(Environment)

        # Shared (immutable)
        fork.model = self.model
        fork.asset_manager = self.asset_manager
        fork.hide_pos = self.hide_pos
        fork.verbose = self.verbose
        fork._geom_original_size = self._geom_original_size  # Read-only cache

        # Independent state
        fork.data = self.sim.clone_data()
        fork.sim = Simulation(fork.model, fork.data)
        fork.registry = self.registry.copy(fork.data)
        fork.state_io = StateIO()

        # Mark as fork (for potential future use)
        fork._is_fork = True

        return fork

    def sync_from(self, other: 'Environment') -> None:
        """
        Synchronize this environment's state from another environment.

        Copies both simulation state (MjData) and object registry state
        (active objects, visibility). Useful for applying changes from a
        fork back to the main environment.

        Args:
            other: The environment to sync from (typically a fork).

        Example:
            # Fork for perception processing
            perception_fork = env.fork()
            perception_fork.update(filtered_detections)

            # Apply processed state back to main
            env.sync_from(perception_fork)
        """
        # Sync MjData (physics state)
        Simulation.copy_data(self.model, self.data, other.data)

        # Sync ObjectRegistry state (active objects, visibility)
        self.registry.active_objects = dict(other.registry.active_objects)
        for name, is_active in self.registry.active_objects.items():
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self.registry._set_body_visibility(body_id, visible=is_active)

    # ------------------------------------------------------------------
    # Context Manager Support
    # ------------------------------------------------------------------
    def __enter__(self) -> 'Environment':
        """Enter context manager. Returns self for use in with statements."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager. Cleanup is handled by garbage collection."""
        pass