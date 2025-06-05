import numpy as np
import mujoco
import os
import xml.etree.ElementTree as ET

class Environment:
    def __init__(self, scene_xml_path, objects_xml_path, hide_pos=[0, 0, -1]):
        self.scene_xml_path = scene_xml_path
        self.objects_xml_path = objects_xml_path
        self.hide_pos = hide_pos
        self.active_objects = set()

        try:
            self.model = mujoco.MjModel.from_xml_path(scene_xml_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model from {scene_xml_path}: {e}")

        try:
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            raise RuntimeError(f"Failed to create MuJoCo data from model: {e}")

        self.objects = {}
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

    def _parse_object_names(self, xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            return [body.attrib["name"] for body in root.findall("body") if "name" in body.attrib]
        except Exception as e:
            raise RuntimeError(f"Failed to parse object names from {xml_path}: {e}")

    def update(self, object_list, persist=False):
        current_names = set()
        for obj in object_list:
            name = obj["name"]
            current_names.add(name)
            if name in self.active_objects:
                self.move_object(name, obj["pos"], obj["quat"])
            else:
                self.add_object(name, obj["pos"], obj["quat"])

        if not persist:
            for name in list(self.active_objects):
                if name not in current_names:
                    self.remove_object(name)

    def add_object(self, name, pos, quat):
        if name not in self.objects:
            raise ValueError(f"Unknown object name: {name}")
        self._activate_object(name, pos, quat)
        self.active_objects.add(name)
        print(f"Added {name}")

    def remove_object(self, name):
        if name not in self.objects:
            raise ValueError(f"Unknown object name: {name}")
        self._hide_object(name)
        self.active_objects.discard(name)
        print(f"Removed {name}")

    def move_object(self, name, pos, quat):
        if name not in self.objects:
            raise ValueError(f"Unknown object name: {name}")
        self._move_object(name, pos, quat)

    def get_active_object_names(self):
        return list(self.active_objects)

    def _activate_object(self, name, pos, quat):
        obj = self.objects[name]
        self.model.geom_contype[obj["geom_id"]] = obj["contype"]
        self.model.geom_conaffinity[obj["geom_id"]] = obj["conaffinity"]
        self.model.body_mass[obj["body_id"]] = obj["mass"]
        self._move_object(name, pos, quat)

    def _move_object(self, name, pos, quat):
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

    def _hide_object(self, name):
        self._move_object(name, self.hide_pos, [1, 0, 0, 0])
        obj = self.objects[name]
        self.model.geom_contype[obj["geom_id"]] = 0
        self.model.geom_conaffinity[obj["geom_id"]] = 0
        self.model.body_mass[obj["body_id"]] = 0.0
