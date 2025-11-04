"""
object_registry.py
Manages object lifecycle (activation, hiding, movement) in MuJoCo.
"""

import os
import yaml
import mujoco
import numpy as np
from typing import Dict, List, Any, Optional


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
            if not self.asset_manager.has(obj_type):
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
            self.data.qpos[qpos_adr+3:qpos_adr+7] = np.array(quat, dtype=float)
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

    def update(self, updates: List[Dict[str, Any]], persist: bool = True):
        """Batch activate/move/hide objects based on updates."""
        active_now = set()

        for upd in updates:
            name = upd["name"]
            pos = np.array(upd["pos"], dtype=float)
            quat = np.array(upd.get("quat", [1, 0, 0, 0]), dtype=float)

            if name not in self.active_objects:
                obj_type = name.split("_")[0]
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
                self.active_objects[name] = True
                active_now.add(name)

        if not persist:
            for name, active in list(self.active_objects.items()):
                if active and name not in active_now:
                    self.hide(name)