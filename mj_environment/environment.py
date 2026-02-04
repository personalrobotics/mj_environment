"""High-level MuJoCo environment orchestrator."""

# Standard library
import logging
import os
import warnings
from typing import List, Dict, Any, Optional, Union, overload
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring, parse as ETparse

# Third-party
import mujoco
import numpy as np
import yaml

logger = logging.getLogger(__name__)


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


def _prefix_names_in_subtree(elem: Element, prefix: str) -> None:
    """
    Recursively prefix all 'name' attributes in element and descendants.

    This ensures unique names when creating multiple instances of the same object.
    For example, if prefix is "can_0" and a geom has name="can_body",
    it becomes name="can_0/can_body".

    Args:
        elem: Root element to process (modified in place)
        prefix: Prefix to add to names (e.g., "can_0")
    """
    # Skip the root body element itself (already renamed by caller)
    for child in elem.iter():
        if child is elem:
            continue
        if "name" in child.attrib:
            child.set("name", f"{prefix}/{child.attrib['name']}")

# Third-party (external package)
from asset_manager import AssetManager

# Local
from .constants import POSITION_DIM, QUATERNION_DIM
from .exceptions import ConfigurationError, ObjectNotFoundError
from .mujoco_helpers import MujocoIndexCache
from .object_registry import ObjectRegistry
from .simulation import Simulation
from .state_io import StateIO
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
        objects_dir: Optional[str] = None,
        scene_config_yaml: Optional[str] = None,
        hide_pos: List[float] = [0, 0, -1],
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.hide_pos = hide_pos

        # Configure logging based on verbose flag (for backward compatibility)
        if verbose and not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

        # ------------------------------------------------------------------
        # 1️⃣ Load scene config and determine if we have objects to manage
        # ------------------------------------------------------------------
        self._scene_cfg = self._load_scene_config(objects_dir, scene_config_yaml)
        self._has_objects = bool(self._scene_cfg)

        # ------------------------------------------------------------------
        # 2️⃣ Asset Manager: load YAML metadata + object XMLs (if objects exist)
        # ------------------------------------------------------------------
        if self._has_objects:
            self.asset_manager: Optional[AssetManager] = AssetManager(base_dir=objects_dir, verbose=verbose)
        else:
            self.asset_manager = None

        # ------------------------------------------------------------------
        # 3️⃣ Build in-memory XML string for the complete scene + assets dict for meshes
        # ------------------------------------------------------------------
        xml_string, assets_dict = self._build_scene_xml_string(base_scene_xml)

        # ------------------------------------------------------------------
        # 4️⃣ Create MuJoCo model + data directly from XML string (with assets dict for mesh files)
        # ------------------------------------------------------------------
        self.model = mujoco.MjModel.from_xml_string(xml_string, assets=assets_dict)
        self.data = mujoco.MjData(self.model)
        self.assets = assets_dict  # Store assets dict for forking

        # Cache for original geom sizes (before scale overrides are applied)
        self._geom_original_size: Dict[int, np.ndarray] = {}

        # ------------------------------------------------------------------
        # 5️⃣ Apply overrides from meta.yaml (mass, color, scale)
        #    These take priority over values in XML files
        # ------------------------------------------------------------------
        if self._has_objects:
            self._apply_metadata_overrides()

        # ------------------------------------------------------------------
        # 6️⃣ Initialize Simulation + Object Registry
        # ------------------------------------------------------------------
        self.sim = Simulation(self.model, self.data)
        if self._has_objects:
            self.registry: Optional[ObjectRegistry] = ObjectRegistry(
                self.model, self.data, self.asset_manager, self._scene_cfg, hide_pos, verbose
            )
        else:
            self.registry = None

        # ------------------------------------------------------------------
        # 7️⃣ Add state I/O helper for serialization
        # ------------------------------------------------------------------
        self.state_io = StateIO()

        # Track whether this is a fork (for potential future use)
        self._is_fork = False

        object_count = len(self.registry.objects) if self.registry else 0
        logger.info("Loaded scene with %d object types", object_count)

    def _load_scene_config(
        self, objects_dir: Optional[str], scene_config_yaml: Optional[str]
    ) -> Dict[str, Any]:
        """Load and validate scene configuration.

        Returns the objects dict from scene_config.yaml, or empty dict for robot-only scenes.
        Raises ConfigurationError if config file is provided but missing/invalid.
        """
        # Robot-only scene: no config needed
        if objects_dir is None and scene_config_yaml is None:
            return {}

        # If one is provided but not the other, treat as robot-only
        if objects_dir is None or scene_config_yaml is None:
            return {}

        # Config file provided - must exist
        if not os.path.exists(scene_config_yaml):
            raise ConfigurationError(
                f"Scene config not found: {scene_config_yaml}",
                path=scene_config_yaml,
                hint="Ensure the scene_config.yaml file exists at the specified path."
            )

        with open(scene_config_yaml, "r") as f:
            cfg = yaml.safe_load(f) or {}

        return cfg.get("objects", {})

    # ======================================================================
    # Internal: Scene Composition
    # ======================================================================
    def _build_scene_xml_string(self, base_scene_xml: str) -> tuple[str, dict[str, bytes]]:
        """
        Build a MuJoCo scene XML in memory by combining:
        - the base scene (e.g., table, lights, cameras)
        - all object instances from self._scene_cfg

        Returns:
            Tuple of (xml_string, assets_dict) where assets_dict contains mesh files for assets
        """
        assets_dict: dict[str, bytes] = {}

        mujoco_el = Element("mujoco", {"model": "autogen_scene"})
        abs_scene_path = os.path.abspath(base_scene_xml)
        SubElement(mujoco_el, "include", {"file": abs_scene_path})

        # Robot-only scene: no objects to add
        if not self._has_objects:
            SubElement(mujoco_el, "worldbody")
            xml_bytes = tostring(mujoco_el, "utf-8")
            return minidom.parseString(xml_bytes).toprettyxml(indent="  "), assets_dict

        # Parse object XMLs once and cache them
        parsed_objects = self._parse_object_xmls()

        # Collect assets and mesh files from object XMLs
        asset_el = SubElement(mujoco_el, "asset")
        self._collect_object_assets(parsed_objects, asset_el, assets_dict)

        # Add object instances to worldbody
        worldbody_el = SubElement(mujoco_el, "worldbody")
        self._add_object_instances(parsed_objects, worldbody_el)

        xml_bytes = tostring(mujoco_el, "utf-8")
        return minidom.parseString(xml_bytes).toprettyxml(indent="  "), assets_dict

    def _parse_object_xmls(self) -> Dict[str, Dict[str, Any]]:
        """
        Parse all object XMLs once and return cached data.

        Returns:
            Dict mapping obj_type to {xml_path, root, count} for each valid object
        """
        parsed = {}

        for obj_type, entry in self._scene_cfg.items():
            if self.asset_manager is None or obj_type not in self.asset_manager.list():
                logger.warning("Unknown asset '%s', skipping", obj_type)
                continue

            xml_path = self.asset_manager.get_path(obj_type, "mujoco")
            if xml_path is None:
                logger.warning("No XML path for '%s' with simulator 'mujoco', skipping", obj_type)
                continue

            if not os.path.exists(xml_path):
                logger.warning("XML file not found for '%s' at %s, skipping", obj_type, xml_path)
                continue

            obj_tree = ETparse(xml_path)
            parsed[obj_type] = {
                'xml_path': xml_path,
                'root': obj_tree.getroot(),
                'count': entry.get("count", 1),
            }

        return parsed

    def _collect_object_assets(
        self,
        parsed_objects: Dict[str, Dict[str, Any]],
        asset_el: Element,
        assets_dict: dict[str, bytes],
    ) -> None:
        """
        Collect asset definitions and mesh files from parsed object XMLs.

        Args:
            parsed_objects: Dict from _parse_object_xmls()
            asset_el: Parent <asset> element to add children to
            assets_dict: Dictionary to populate with mesh file data
        """
        included_assets: set = set()

        for obj_type, obj_data in parsed_objects.items():
            obj_asset = obj_data['root'].find("asset")
            if obj_asset is None:
                continue

            for asset_child in obj_asset:
                asset_name = asset_child.get("name", "")
                asset_key = (asset_child.tag, asset_name)
                if asset_key in included_assets:
                    continue

                _deep_copy_element(asset_child, asset_el)
                included_assets.add(asset_key)

                if asset_child.tag == "mesh":
                    self._load_mesh_file(obj_data['xml_path'], asset_child, assets_dict)

    def _load_mesh_file(
        self,
        xml_path: str,
        mesh_element: Element,
        assets_dict: dict[str, bytes],
    ) -> None:
        """
        Load a mesh file into the assets dictionary.

        Args:
            xml_path: Path to the object XML (used to resolve relative mesh paths)
            mesh_element: The <mesh> XML element containing the file attribute
            assets_dict: Dictionary to populate with mesh file data
        """
        mesh_file = mesh_element.get("file")
        if not mesh_file:
            return

        obj_dir = os.path.dirname(xml_path)
        mesh_path = os.path.join(obj_dir, mesh_file)

        if os.path.exists(mesh_path):
            with open(mesh_path, 'rb') as f:
                assets_dict[mesh_file] = f.read()
        else:
            logger.warning("Mesh file not found: %s", mesh_path)

    def _add_object_instances(
        self,
        parsed_objects: Dict[str, Dict[str, Any]],
        worldbody_el: Element,
    ) -> None:
        """
        Add object body instances to the worldbody element.

        Creates numbered instances (e.g., cup_0, cup_1) for each object type,
        positioning them at hide_pos and prefixing nested element names.

        Args:
            parsed_objects: Dict from _parse_object_xmls()
            worldbody_el: The <worldbody> element to add bodies to
        """
        for obj_type, obj_data in parsed_objects.items():
            obj_worldbody = obj_data['root'].find("worldbody")
            if obj_worldbody is None:
                logger.warning("No worldbody found in %s, skipping", obj_data['xml_path'])
                continue

            for i in range(obj_data['count']):
                instance_name = f"{obj_type}_{i}"
                for obj_body in obj_worldbody.findall("body"):
                    new_body = _deep_copy_element(obj_body, worldbody_el)
                    new_body.set("name", instance_name)
                    new_body.set("pos", f"{self.hide_pos[0]} {self.hide_pos[1]} {self.hide_pos[2]}")
                    _prefix_names_in_subtree(new_body, instance_name)

    # ------------------------------------------------------------------
    # Metadata Overrides
    # ------------------------------------------------------------------
    def _apply_metadata_overrides(self) -> None:
        """Apply mass, color, and scale overrides from meta.yaml to the model.

        These overrides take priority over values specified in the XML files.
        Uses self._scene_cfg loaded during __init__.
        """
        for obj_type, entry in self._scene_cfg.items():
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
                    logger.debug("Set mass of %s to %s", instance_name, mass_value)

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
                            if geom_idx == 0:
                                logger.debug("Set color of %s to %s", instance_name, rgba)

                    # Apply scale override
                    if "scale" in meta:
                        scale = float(meta["scale"])
                        # Cache original size before scaling (if not already cached)
                        if geom_id not in self._geom_original_size:
                            self._geom_original_size[geom_id] = self.model.geom_size[geom_id].copy()
                        # Apply scale to original size (not current size, to avoid compounding)
                        self.model.geom_size[geom_id] = self._geom_original_size[geom_id] * scale
                        if geom_idx == 0:
                            logger.debug("Applied scale of %s to %s", scale, instance_name)

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
            ObjectNotFoundError: If instance_name is not in the registry or no objects configured.

        Example:
            >>> meta = env.get_object_metadata("plate_1")
            >>> print(meta["mass"], meta["color"], meta["category"])
        """
        if self.registry is None or self.asset_manager is None:
            raise ObjectNotFoundError(instance_name, [])

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
        hide_unlisted: Optional[bool] = None,
        persist: Optional[bool] = None,
    ) -> None:
        """
        Batch update multiple objects in the environment.

        Args:
            object_list: List of detection dicts with keys:
                - name: Instance name (e.g., "cup_0")
                - pos: Position [x, y, z]
                - quat: (optional) Quaternion [w, x, y, z], defaults to [1, 0, 0, 0]
            hide_unlisted: If True (default), hide objects not in the list.
                          If False, keep previously active objects visible.
            persist: DEPRECATED. Use hide_unlisted instead.

        Raises:
            RuntimeError: If environment has no object management configured.

        Example:
            >>> # Hide objects not in list (default)
            >>> env.update([{"name": "cup_0", "pos": [0, 0, 0.5]}])
            >>> # Keep previously active objects
            >>> env.update([...], hide_unlisted=False)
        """
        # Handle parameter deprecation
        if persist is not None:
            warnings.warn(
                "The 'persist' parameter is deprecated and will be removed in v2.0. "
                "Use 'hide_unlisted' instead: persist=False -> hide_unlisted=True, "
                "persist=True -> hide_unlisted=False",
                DeprecationWarning,
                stacklevel=2
            )
            if hide_unlisted is None:
                hide_unlisted = not persist

        if self.registry is None:
            raise RuntimeError("Cannot update objects: environment has no object management configured")
        self.registry.update(object_list, hide_unlisted=hide_unlisted)  # type: ignore[arg-type]
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl: Optional[np.ndarray] = None) -> None:
        """Advance simulation by one step."""
        self.sim.step(ctrl)

    def reset(self) -> None:
        """Reset simulation and all objects."""
        self.sim.reset()

    def status(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Get current scene status for inspection and debugging.

        Args:
            verbose: If True, include positions for all objects (not just active).

        Returns:
            Dict with keys:
                - time: Current simulation time
                - active_count: Number of active objects
                - active_objects: Dict mapping name -> {pos, quat} for active objects
                - object_types: Dict mapping type -> {total, active, available}

        Example:
            >>> status = env.status()
            >>> print(f"Active: {status['active_count']} objects")
            >>> for name, state in status['active_objects'].items():
            ...     print(f"  {name}: pos={state['pos']}")
        """
        # Robot-only environment (no objects)
        if self.registry is None:
            return {
                "time": self.data.time,
                "active_count": 0,
                "active_objects": {},
                "object_types": {},
            }

        active_objects = {}
        for name, is_active in self.registry.active_objects.items():
            if is_active or verbose:
                indices = self.registry._index_cache.get_body_indices(name)
                pos = self.data.qpos[indices.qpos_adr:indices.qpos_adr+POSITION_DIM].tolist()
                quat = self.data.qpos[indices.qpos_adr+POSITION_DIM:indices.qpos_adr+POSITION_DIM+QUATERNION_DIM].tolist()
                active_objects[name] = {
                    "pos": pos,
                    "quat": quat,
                    "active": is_active,
                }

        object_types = {}
        for obj_type, info in self.registry.objects.items():
            instances = info["instances"]
            active_count = sum(1 for n in instances if self.registry.active_objects.get(n, False))
            object_types[obj_type] = {
                "total": len(instances),
                "active": active_count,
                "available": len(instances) - active_count,
            }

        return {
            "time": self.data.time,
            "active_count": sum(1 for v in self.registry.active_objects.values() if v),
            "active_objects": active_objects,
            "object_types": object_types,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save_state(self, path: str) -> None:
        """Serialize current simulation state to YAML."""
        active_objects = self.registry.active_objects if self.registry else {}
        self.state_io.save(self.model, self.data, active_objects, path)

    def load_state(self, path: str) -> None:
        """Load simulation state from YAML."""
        active_objects = self.state_io.load(self.model, self.data, path)
        if self.registry is not None:
            self.registry.active_objects = active_objects
            self.registry.sync_visibility()

    # ------------------------------------------------------------------
    # Forking for Planning
    # ------------------------------------------------------------------
    def fork(self, n: Optional[int] = None) -> Union['Environment', List['Environment']]:
        """
        Create a functional clone with independent state for planning.

        Forked environments share immutable data (MjModel, AssetManager) but have
        independent simulation state (MjData, ObjectRegistry). This enables:
        - Motion planning without polluting the original environment
        - Multiple planners running in parallel on separate forks

        Args:
            n: DEPRECATED. Use fork_many(n) for multiple forks.
               If provided, calls fork_many(n) with deprecation warning.

        Returns:
            A single Environment instance.

        Example:
            # Single fork for planning
            planning_env = env.fork()
            planning_env.update([{"name": "cup_0", "pos": [0.1, 0.2, 0.3]}])
            trajectory = planner.plan(planning_env, goal)
            # Original env unchanged

            # Context manager for explicit scope
            with env.fork() as planning_env:
                trajectory = planner.plan(planning_env)
        """
        if n is not None:
            warnings.warn(
                "fork(n=...) is deprecated and will be removed in v2.0. "
                "Use fork_many(n) for creating multiple forks instead.",
                DeprecationWarning,
                stacklevel=2
            )
            return self.fork_many(n)
        return self._create_fork()

    def fork_many(self, n: int) -> List['Environment']:
        """
        Create multiple functional clones for parallel planning.

        Each fork shares immutable data (MjModel, AssetManager) but has
        independent simulation state (MjData, ObjectRegistry).

        Args:
            n: Number of forks to create

        Returns:
            List of n independent Environment instances

        Example:
            # Multiple forks for parallel planning
            forks = env.fork_many(4)
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(planner.plan, forks))
        """
        return [self._create_fork() for _ in range(n)]

    def _create_fork(self) -> 'Environment':
        """Create a single forked environment."""
        fork = Environment.__new__(Environment)

        # Shared (immutable)
        fork.model = self.model
        fork.assets = self.assets  # Assets dict is shared (read-only)
        fork.asset_manager = self.asset_manager
        fork.hide_pos = self.hide_pos
        fork.verbose = self.verbose
        fork._geom_original_size = self._geom_original_size  # Read-only cache
        fork._has_objects = self._has_objects

        # Independent state
        fork.data = self.sim.clone_data()
        fork.sim = Simulation(fork.model, fork.data)
        fork.registry = self.registry.copy(fork.data) if self.registry else None
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

        # Sync ObjectRegistry state (active objects, visibility) if objects exist
        if self.registry is not None and other.registry is not None:
            self.registry.active_objects = dict(other.registry.active_objects)
            self.registry.sync_visibility()

    # ------------------------------------------------------------------
    # Context Manager Support
    # ------------------------------------------------------------------
    def __enter__(self) -> 'Environment':
        """Enter context manager. Returns self for use in with statements."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager. Cleanup is handled by garbage collection."""
        pass