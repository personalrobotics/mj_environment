"""ObjectRegistry — YAML-driven object instantiation and lifecycle manager for MuJoCo."""

import os
import yaml
import mujoco
import numpy as np
from typing import Dict, List, Any, Optional


class ObjectRegistry:
    """
    Manages all dynamic objects in a MuJoCo environment.

    Uses:
      - AssetManager for asset metadata (XML, colors, etc.)
      - A YAML scene manifest to specify how many of each object to preload

    Handles:
      - Preloading all object instances
      - Activating / hiding objects
      - Tracking object states
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        asset_manager,
        scene_config_yaml: str,
        hide_pos: List[float] = [0, 0, -1],
        verbose: bool = False,
    ):
        self.model = model
        self.data = data
        self.asset_manager = asset_manager
        self.hide_pos = np.array(hide_pos)
        self.verbose = verbose

        # Object state dict: {type_name: {instances: [...], count: N}}
        self.objects: Dict[str, Any] = {}
        self.active_objects: Dict[str, bool] = {}

        self._load_scene_config(scene_config_yaml)
        self._preload_objects()

    # ------------------------------------------------------------------
    # YAML loading
    # ------------------------------------------------------------------
    def _load_scene_config(self, scene_config_yaml: str):
        """Load YAML defining how many of each object to instantiate."""
        if not os.path.exists(scene_config_yaml):
            raise FileNotFoundError(f"Scene config file not found: {scene_config_yaml}")
        with open(scene_config_yaml, "r") as f:
            cfg = yaml.safe_load(f)

        if "objects" not in cfg:
            raise ValueError(f"Scene config must contain an 'objects' key: {scene_config_yaml}")

        self.scene_cfg = cfg["objects"]

    # ------------------------------------------------------------------
    # Object preloading
    # ------------------------------------------------------------------
    def _preload_objects(self):
        """Instantiate and hide all object instances according to YAML and assets."""
        for obj_type, entry in self.scene_cfg.items():
            if not self.asset_manager.has(obj_type):
                if self.verbose:
                    print(f"[WARN] Asset '{obj_type}' not found in AssetManager; skipping.")
                continue

            meta = self.asset_manager.get(obj_type)
            xml_path = meta["xml_path"]
            count = entry.get("count", 1)

            if not os.path.exists(xml_path):
                print(f"[WARN] XML file for {obj_type} not found: {xml_path}")
                continue

            xml = mujoco.MjModel.from_xml_path(xml_path)
            self.objects[obj_type] = {"count": count, "instances": []}

            for i in range(count):
                name = f"{obj_type}_{i}"
                self._insert_object(xml, name, self.hide_pos)
                self.objects[obj_type]["instances"].append(name)
                self.active_objects[name] = False

            if self.verbose:
                print(f"[INFO] Preloaded {count} instances of '{obj_type}'.")

    def _insert_object(self, obj_model: mujoco.MjModel, name: str, pos: np.ndarray):
        """Clone object geometry into main model."""
        world_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "world")
        for b in range(obj_model.nbody):
            body_name = mujoco.mj_id2name(obj_model, mujoco.mjtObj.mjOBJ_BODY, b)
            if body_name == "world":
                continue
            new_name = f"{name}::{body_name}"
            mujoco.mj_copyBody(self.model, obj_model, b, new_name, world_id)
            mujoco.mj_setBodyPos(self.model, new_name, pos)

    # ------------------------------------------------------------------
    # Runtime control
    # ------------------------------------------------------------------
    def activate(self, obj_type: str, pos: List[float], quat: Optional[List[float]] = None) -> Optional[str]:
        """Activate the next available hidden object of a given type."""
        if obj_type not in self.objects:
            raise KeyError(f"Unknown object type: {obj_type}")

        candidates = [n for n in self.objects[obj_type]["instances"] if not self.active_objects[n]]
        if not candidates:
            if self.verbose:
                print(f"[WARN] No inactive instances of '{obj_type}' left.")
            return None

        name = candidates[0]
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.data.xpos[body_id] = pos
        if quat is not None:
            self.data.xquat[body_id] = quat
        self.active_objects[name] = True
        if self.verbose:
            print(f"[INFO] Activated {name} at {pos}")
        return name

    def hide(self, name: str):
        """Hide an active object."""
        if name not in self.active_objects:
            raise KeyError(f"Unknown object: {name}")
        if not self.active_objects[name]:
            return
        mujoco.mj_setBodyPos(self.model, name, self.hide_pos)
        self.active_objects[name] = False
        if self.verbose:
            print(f"[INFO] Hid object '{name}'")

    def update(self, updates: List[Dict[str, Any]], persist: bool = True):
        """Batch update — move or activate objects from perception input."""
        active_this_cycle = set()

        for upd in updates:
            name = upd["name"]
            pos = np.array(upd["pos"])
            quat = np.array(upd.get("quat", [1, 0, 0, 0]))

            if name not in self.active_objects:
                obj_type = name.split("_")[0]
                new_name = self.activate(obj_type, pos, quat)
                if new_name:
                    active_this_cycle.add(new_name)
            else:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                self.data.xpos[body_id] = pos
                self.data.xquat[body_id] = quat
                self.active_objects[name] = True
                active_this_cycle.add(name)

        if not persist:
            for name, active in list(self.active_objects.items()):
                if name not in active_this_cycle and active:
                    self.hide(name)