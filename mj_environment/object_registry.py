"""
object_registry.py
Manages object lifecycle (activation, hiding, movement) in MuJoCo.
"""

import os
import yaml
import mujoco
import numpy as np
from typing import Dict, List, Any, Optional


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    """
    Normalize a quaternion to unit length.

    MuJoCo expects unit quaternions for proper physics simulation.
    This ensures quaternions are valid even if provided unnormalized.

    Args:
        quat: Quaternion array [w, x, y, z]

    Returns:
        Normalized quaternion. Returns identity [1, 0, 0, 0] if input is near-zero.
    """
    q = np.array(quat, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1, 0, 0, 0], dtype=float)
    return q / norm


class ObjectRegistry:
    """
    Manages dynamic objects in a MuJoCo environment.

    Responsibilities:
      - Preloading objects from AssetManager and YAML scene config
      - Activating, hiding, and moving objects
      - Tracking object states and reuse

    Naming and Instance Reuse Policy:
      - Objects are preinitialized as e.g. cup_0, cup_1, ...
      - Hidden objects are reusable to respect MuJoCo immutability.
      - For unique identifiers, overprovision in the YAML config.

    Thread Safety:
      - This class is NOT thread-safe for concurrent writes.
      - Multiple threads may safely READ from `objects` dict concurrently.
      - All write operations (activate, hide, update) must be synchronized
        externally if accessed from multiple threads.
      - For multi-threaded perception pipelines, use a queue to batch updates
        and process them on a single thread (see perception_update_demo.py).
    """

    def __init__(self, model, data, asset_manager, scene_config_yaml, hide_pos=[0, 0, -1], verbose=False):
        self.model = model
        self.data = data
        self.asset_manager = asset_manager
        self.hide_pos = np.array(hide_pos, dtype=float)
        self.verbose = verbose
        self.objects: Dict[str, Dict[str, Any]] = {}
        self.active_objects: Dict[str, bool] = {}
        self.geom_visibility: Dict[int, np.ndarray] = {}  # Cache original geom colors

        self._load_scene_config(scene_config_yaml)
        self._preload_objects()
        self._cache_geom_colors()

    # ------------------------------------------------------------------
    # Scene configuration
    # ------------------------------------------------------------------
    def _load_scene_config(self, yaml_path: str):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(yaml_path)
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        if "objects" not in cfg:
            raise ValueError(f"Scene config must define 'objects': {yaml_path}")
        self.scene_cfg = cfg["objects"]

    # ------------------------------------------------------------------
    # Object preloading
    # ------------------------------------------------------------------
    def _preload_objects(self):
        """Track object instances that were preloaded into the MuJoCo model by Environment."""
        for obj_type, entry in self.scene_cfg.items():
            if obj_type not in self.asset_manager.list():
                if self.verbose:
                    print(f"[WARN] Unknown asset '{obj_type}', skipping preload.")
                continue

            count = entry.get("count", 1)
            self.objects[obj_type] = {"count": count, "instances": []}

            for i in range(count):
                name = f"{obj_type}_{i}"
                # Verify object exists in the model (it should be preloaded by Environment)
                try:
                    body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                    self.objects[obj_type]["instances"].append(name)
                    self.active_objects[name] = False
                except (TypeError, AttributeError):
                    if self.verbose:
                        print(f"[WARN] Object '{name}' not found in model, skipping.")

            if self.verbose:
                print(f"[INFO] Preloaded {count} {obj_type}(s).")

    def _cache_geom_colors(self):
        """Cache the original RGBA colors for all object geoms."""
        for obj_type in self.objects:
            for instance_name in self.objects[obj_type]["instances"]:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, instance_name)
                geom_adr = self.model.body_geomadr[body_id]
                geom_num = self.model.body_geomnum[body_id]

                # Store original colors for each geom
                for i in range(geom_num):
                    geom_id = geom_adr + i
                    if geom_id not in self.geom_visibility:
                        self.geom_visibility[geom_id] = self.model.geom_rgba[geom_id].copy()

                # Make all objects invisible initially (they start inactive)
                self._set_body_visibility(body_id, visible=False)

    def copy(self, new_data: mujoco.MjData) -> 'ObjectRegistry':
        """
        Create an independent copy of this registry with new MjData.

        The copy shares the MjModel (immutable) but has independent state tracking.
        Used by Environment.fork() for creating planning environments.

        Args:
            new_data: The new MjData instance for the copy to use.

        Returns:
            A new ObjectRegistry with copied state but independent data.
        """
        clone = ObjectRegistry.__new__(ObjectRegistry)
        clone.model = self.model  # Shared (immutable)
        clone.data = new_data  # Independent
        clone.asset_manager = self.asset_manager  # Shared (read-only)
        clone.hide_pos = self.hide_pos.copy()
        clone.verbose = self.verbose

        # Deep copy mutable state
        clone.objects = {
            obj_type: {
                "count": info["count"],
                "instances": list(info["instances"])
            }
            for obj_type, info in self.objects.items()
        }
        clone.active_objects = dict(self.active_objects)
        clone.geom_visibility = {k: v.copy() for k, v in self.geom_visibility.items()}
        clone.scene_cfg = self.scene_cfg  # Read-only after init

        return clone

    def _parse_object_type(self, instance_name: str) -> Optional[str]:
        """
        Parse object type from instance name, handling underscores correctly.

        Examples:
            cup_0 -> cup
            kitchen_knife_2 -> kitchen_knife
            my_cool_object_15 -> my_cool_object (if registered)
        """
        # Check each registered object type to find a match
        for obj_type in self.objects:
            # Instance names follow pattern: {obj_type}_{index}
            if instance_name.startswith(obj_type + "_"):
                suffix = instance_name[len(obj_type) + 1:]
                # Verify suffix is a valid index (digits only)
                if suffix.isdigit():
                    return obj_type
        return None

    def _set_body_visibility(self, body_id: int, visible: bool):
        """Show or hide all geoms of a body by setting RGBA alpha channel."""
        geom_adr = self.model.body_geomadr[body_id]
        geom_num = self.model.body_geomnum[body_id]
        
        for i in range(geom_num):
            geom_id = geom_adr + i
            if geom_id in self.geom_visibility:
                if visible:
                    # Restore original alpha
                    self.model.geom_rgba[geom_id, 3] = self.geom_visibility[geom_id][3]
                else:
                    # Set alpha to 0 (invisible)
                    self.model.geom_rgba[geom_id, 3] = 0.0

    # ------------------------------------------------------------------
    # Runtime API
    # ------------------------------------------------------------------
    def activate(self, obj_type: str, pos: List[float], quat: Optional[List[float]] = None) -> Optional[str]:
        """Activate the next available hidden instance of a given type."""
        if obj_type not in self.objects:
            raise KeyError(obj_type)
        candidates = [n for n in self.objects[obj_type]["instances"] if not self.active_objects[n]]
        if not candidates:
            if self.verbose:
                print(f"[WARN] No inactive instances left for '{obj_type}'.")
            return None
        name = candidates[0]
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        joint_adr = self.model.body_jntadr[body_id]
        qpos_adr = self.model.jnt_qposadr[joint_adr]
        qvel_adr = self.model.jnt_dofadr[joint_adr]
        self.data.qpos[qpos_adr:qpos_adr+3] = np.array(pos, dtype=float)
        if quat is None:
            self.data.qpos[qpos_adr+3:qpos_adr+7] = np.array([1, 0, 0, 0], dtype=float)
        else:
            self.data.qpos[qpos_adr+3:qpos_adr+7] = _normalize_quat(quat)
        self.data.qvel[qvel_adr:qvel_adr+6] = 0
        self._set_body_visibility(body_id, visible=True)
        self.active_objects[name] = True
        if self.verbose:
            print(f"[INFO] Activated {name}.")
        return name

    def hide(self, name: str):
        """Hide an active object by moving and disabling it."""
        if name not in self.active_objects:
            raise KeyError(name)
        if not self.active_objects[name]:
            return
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        joint_adr = self.model.body_jntadr[body_id]
        qpos_adr = self.model.jnt_qposadr[joint_adr]
        qvel_adr = self.model.jnt_dofadr[joint_adr]
        self.data.qpos[qpos_adr:qpos_adr+3] = self.hide_pos
        self.data.qpos[qpos_adr+3:qpos_adr+7] = np.array([1, 0, 0, 0], dtype=float)
        self.data.qvel[qvel_adr:qvel_adr+6] = 0
        self._set_body_visibility(body_id, visible=False)
        self.active_objects[name] = False
        if self.verbose:
            print(f"[INFO] Hid {name}.")

    def update(self, updates: List[Dict[str, Any]], persist: bool = False):
        """
        Batch activate/move/hide objects based on updates.

        Args:
            updates: List of dicts with keys: name, pos, quat (optional)
            persist: If False (default), hide objects not in updates list.
                     If True, keep previously active objects visible.
        """
        active_now = set()

        for upd in updates:
            name = upd["name"]
            pos = np.array(upd["pos"], dtype=float)
            quat = _normalize_quat(upd.get("quat", [1, 0, 0, 0]))

            if name not in self.active_objects:
                # Find object type by checking registry membership
                # This handles object types with underscores (e.g., kitchen_knife_0 -> kitchen_knife)
                obj_type = self._parse_object_type(name)
                if obj_type is None:
                    if self.verbose:
                        print(f"[WARN] Could not determine object type for '{name}', skipping.")
                    continue
                new_name = self.activate(obj_type, pos, quat)
                if new_name:
                    active_now.add(new_name)
            else:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                joint_adr = self.model.body_jntadr[body_id]
                qpos_adr = self.model.jnt_qposadr[joint_adr]
                qvel_adr = self.model.jnt_dofadr[joint_adr]
                self.data.qpos[qpos_adr:qpos_adr+3] = pos
                self.data.qpos[qpos_adr+3:qpos_adr+7] = quat
                self.data.qvel[qvel_adr:qvel_adr+6] = 0
                # Make sure the object is visible (it might have been hidden previously)
                if not self.active_objects[name]:
                    self._set_body_visibility(body_id, visible=True)
                self.active_objects[name] = True
                active_now.add(name)

        if not persist:
            for name, active in list(self.active_objects.items()):
                if active and name not in active_now:
                    self.hide(name)