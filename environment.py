import mujoco
import numpy as np
import xml.etree.ElementTree as ET

class Environment:
    def __init__(self, scene_xml_path, household_xml_path):
        self.model = mujoco.MjModel.from_xml_path(scene_xml_path)
        self.data = mujoco.MjData(self.model)
        self.hide_pos = [0, 0, -1]  # Below table

        household_names = self._parse_household_names(household_xml_path)
        self.objects = {}
        self._initialize_objects(household_names)

    def _parse_household_names(self, household_xml_path):
        tree = ET.parse(household_xml_path)
        root = tree.getroot()
        return [body.attrib['name'] for body in root.findall('body')]

    def _initialize_objects(self, names):
        for name in names:
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

    def update(self, object_list, persist=False):
        current_names = set()
        for obj in object_list:
            name = obj["name"]
            current_names.add(name)
            if name in self.objects:
                self._activate_object(name, obj["pos"], obj["quat"])

        if not persist:
            for name in self.objects:
                if name not in current_names:
                    self._hide_object(name)

        mujoco.mj_forward(self.model, self.data)

    def get_active_objects(self):
        active = []
        for name, obj in self.objects.items():
            pos = self.data.qpos[self._get_qpos_addr(name):self._get_qpos_addr(name)+3]
            if not np.allclose(pos, self.hide_pos, atol=1e-4):
                active.append(name)
        return active

    def _get_qpos_addr(self, name):
        body_id = self.objects[name]["body_id"]
        joint_adr = self.model.body_jntadr[body_id]
        return self.model.jnt_qposadr[joint_adr]

    def _get_qvel_addr(self, name):
        body_id = self.objects[name]["body_id"]
        joint_adr = self.model.body_jntadr[body_id]
        return self.model.jnt_dofadr[joint_adr]

    def _activate_object(self, name, pos, quat):
        obj = self.objects[name]
        self.model.geom_contype[obj["geom_id"]] = obj["contype"]
        self.model.geom_conaffinity[obj["geom_id"]] = obj["conaffinity"]
        self.model.body_mass[obj["body_id"]] = obj["mass"]
        self._move_object(name, pos, quat)

    def _hide_object(self, name):
        obj = self.objects[name]
        self.model.geom_contype[obj["geom_id"]] = 0
        self.model.geom_conaffinity[obj["geom_id"]] = 0
        self.model.body_mass[obj["body_id"]] = 0.0
        self._move_object(name, self.hide_pos, [1, 0, 0, 0])

    def _move_object(self, name, pos, quat):
        qpos_addr = self._get_qpos_addr(name)
        qvel_addr = self._get_qvel_addr(name)
        self.data.qpos[qpos_addr:qpos_addr+3] = pos
        self.data.qpos[qpos_addr+3:qpos_addr+7] = quat
        self.data.qvel[qvel_addr:qvel_addr+6] = 0.0
