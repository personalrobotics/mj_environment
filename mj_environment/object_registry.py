import mujoco
import numpy as np
from typing import Dict, Any, List

class ObjectRegistry:
    """Manages MuJoCo objects: activation, hiding, and metadata."""
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, objects_xml: str, hide_pos=[0, 0, -1]):
        import xml.etree.ElementTree as ET
        self.model = model
        self.data = data
        self.hide_pos = hide_pos
        self.objects: Dict[str, Dict[str, Any]] = {}
        self.active_objects: set[str] = set()

        root = ET.parse(objects_xml).getroot()
        for body in root.findall("body"):
            name = body.attrib.get("name")
            if not name:
                continue
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            self.objects[name] = {
                "geom_id": geom_id,
                "body_id": body_id,
                "contype": model.geom_contype[geom_id],
                "conaffinity": model.geom_conaffinity[geom_id],
                "mass": model.body_mass[body_id],
            }
            self.hide(name)

    def activate(self, name: str, pos: List[float], quat: List[float]):
        if name not in self.objects:
            raise ValueError(f"Unknown object name: '{name}'")
        obj = self.objects[name]
        self.model.geom_contype[obj["geom_id"]] = obj["contype"]
        self.model.geom_conaffinity[obj["geom_id"]] = obj["conaffinity"]
        self.model.body_mass[obj["body_id"]] = obj["mass"]
        self.move(name, pos, quat)
        self.active_objects.add(name)

    def move(self, name: str, pos: List[float], quat: List[float]):
        if name not in self.objects:
            raise ValueError(f"Unknown object name: '{name}'")
        obj = self.objects[name]
        body_id = obj["body_id"]
        joint_adr = self.model.body_jntadr[body_id]
        qpos_adr = self.model.jnt_qposadr[joint_adr]
        qvel_adr = self.model.jnt_dofadr[joint_adr]
        self.data.qpos[qpos_adr:qpos_adr+3] = pos
        self.data.qpos[qpos_adr+3:qpos_adr+7] = quat
        self.data.qvel[qvel_adr:qvel_adr+6] = 0

    def hide(self, name: str):
        if name not in self.objects:
            # Silently ignore if object doesn't exist
            return
        obj = self.objects[name]
        self.move(name, self.hide_pos, [1, 0, 0, 0])
        self.model.geom_contype[obj["geom_id"]] = 0
        self.model.geom_conaffinity[obj["geom_id"]] = 0
        self.model.body_mass[obj["body_id"]] = 0.0
        self.active_objects.discard(name)