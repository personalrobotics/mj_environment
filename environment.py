import numpy as np
import mujoco
import os
import xml.etree.ElementTree as ET
import pickle
from typing import List, Dict, Any

class Environment:
    """
    A class to manage a MuJoCo simulation environment, including loading models,
    manipulating objects, and updating simulation state.
    """
    def __init__(self, scene_xml_path: str, objects_xml_path: str, hide_pos: List[float] = [0, 0, -1]):
        self.scene_xml_path = scene_xml_path
        self.objects_xml_path = objects_xml_path
        self.hide_pos = hide_pos

        try:
            self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(scene_xml_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model from {scene_xml_path}: {e}")

        try:
            self.data: mujoco.MjData = mujoco.MjData(self.model)
        except Exception as e:
            raise RuntimeError(f"Failed to create MuJoCo data from model: {e}")

        self.objects: Dict[str, Dict[str, Any]] = {}
        self.active_objects: set[str] = set()
        object_names = self._parse_object_names(objects_xml_path)

        for name in object_names:
            try:
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                self.objects[name] = {
                    "geom_id": geom_id,
                    "body_id": body_id,
                    "contype": self.model.geom_contype[geom_id],
                    "conaffinity": self.model.geom_conaffinity[geom_id],
                    "mass": self.model.body_mass[body_id]
                }
                self._hide_object(name)
            except Exception as e:
                print(f"[Warning] Failed to initialize object '{name}': {e}")

    def _parse_object_names(self, xml_path: str) -> List[str]:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            return [body.attrib["name"] for body in root.findall("body") if "name" in body.attrib]
        except Exception as e:
            raise RuntimeError(f"Failed to parse object names from {xml_path}: {e}")

    def update(self, object_list: List[Dict[str, Any]], persist: bool = False) -> None:
        current_names = set()
        for obj in object_list:
            name = obj["name"]
            current_names.add(name)
            status = obj.get("status", "active")

            if status == "active":
                if name in self.active_objects:
                    self.move_object(name, obj["pos"], obj["quat"])
                else:
                    self.add_object(name, obj["pos"], obj["quat"])
            elif status == "inactive":
                self.remove_object(name)

        if not persist:
            for name in list(self.active_objects):
                if name not in current_names:
                    self.remove_object(name)
        mujoco.mj_forward(self.model, self.data)

    def add_object(self, name: str, pos: List[float], quat: List[float]) -> None:
        if name not in self.objects:
            raise ValueError(f"Unknown object name: {name}")
        self._activate_object(name, pos, quat)
        self.active_objects.add(name)
        print(f"Added {name}")
        mujoco.mj_forward(self.model, self.data)

    def remove_object(self, name: str) -> None:
        if name not in self.objects:
            raise ValueError(f"Unknown object name: {name}")
        self._hide_object(name)
        self.active_objects.discard(name)
        print(f"Removed {name}")
        mujoco.mj_forward(self.model, self.data)

    def move_object(self, name: str, pos: List[float], quat: List[float]) -> None:
        if name not in self.objects:
            raise ValueError(f"Unknown object name: {name}")
        self._move_object(name, pos, quat)
        mujoco.mj_forward(self.model, self.data)

    def get_active_object_names(self) -> List[str]:
        return list(self.active_objects)

    def _activate_object(self, name: str, pos: List[float], quat: List[float]) -> None:
        obj = self.objects[name]
        self.model.geom_contype[obj["geom_id"]] = obj["contype"]
        self.model.geom_conaffinity[obj["geom_id"]] = obj["conaffinity"]
        self.model.body_mass[obj["body_id"]] = obj["mass"]
        self._move_object(name, pos, quat)

    def _move_object(self, name: str, pos: List[float], quat: List[float]) -> None:
        obj = self.objects[name]
        body_id = obj["body_id"]
        joint_adr = self.model.body_jntadr[body_id]
        if joint_adr == -1:
            raise RuntimeError(f"Body '{name}' has no joint assigned")
        qpos_addr = self.model.jnt_qposadr[joint_adr]
        qvel_addr = self.model.jnt_dofadr[joint_adr]
        self.data.qpos[qpos_addr:qpos_addr+3] = pos
        self.data.qpos[qpos_addr+3:qpos_addr+7] = quat
        self.data.qvel[qvel_addr:qvel_addr+6] = 0

    def _hide_object(self, name: str) -> None:
        self._move_object(name, self.hide_pos, [1, 0, 0, 0])
        obj = self.objects[name]
        self.model.geom_contype[obj["geom_id"]] = 0
        self.model.geom_conaffinity[obj["geom_id"]] = 0
        self.model.body_mass[obj["body_id"]] = 0.0

    def clone(self) -> mujoco.MjData:
        """
        Create a new MjData instance with identical state to the environment's current data.
        """
        data_clone = mujoco.MjData(self.model)
        copy_data(data_clone, self.data, self.model)
        return data_clone

    def update_from_clone(self, other_data: mujoco.MjData) -> None:
        """
        Update this environment's internal data to match another MjData instance.
        Raises ValueError if models do not match.
        """
        if other_data.model is not self.model:
            raise ValueError("Cannot update: input data uses a different model.")
        copy_data(self.data, other_data, self.model)
        mujoco.mj_forward(self.model, self.data)

    def pickle(self) -> bytes:
        """
        Serialize the current environment state (data) into bytes.
        """
        return pickle.dumps({
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "act": self.data.act.copy(),
            "ctrl": self.data.ctrl.copy(),
            "qacc": self.data.qacc.copy(),
            "qfrc_applied": self.data.qfrc_applied.copy()
        })

    def unpickle(self, data_bytes: bytes) -> None:
        """
        Deserialize bytes into the environment's current data.
        """
        state = pickle.loads(data_bytes)
        np.copyto(self.data.qpos, state["qpos"])
        np.copyto(self.data.qvel, state["qvel"])
        np.copyto(self.data.act, state["act"])
        np.copyto(self.data.ctrl, state["ctrl"])
        np.copyto(self.data.qacc, state["qacc"])
        np.copyto(self.data.qfrc_applied, state["qfrc_applied"])
        mujoco.mj_forward(self.model, self.data)

def copy_data(dst: mujoco.MjData, src: mujoco.MjData, model: mujoco.MjModel) -> None:
    """
    Copy simulation state from src to dst for the same model.
    """
    np.copyto(dst.qpos, src.qpos)
    np.copyto(dst.qvel, src.qvel)
    np.copyto(dst.act, src.act)
    np.copyto(dst.ctrl, src.ctrl)
    np.copyto(dst.qacc, src.qacc)
    np.copyto(dst.qfrc_applied, src.qfrc_applied)
    mujoco.mj_forward(model, dst)
