import numpy as np
import mujoco

class ObjectManager:
    def __init__(self, model, data, object_names, hide_pos=[1000, 1000, 1000]):
        self.model = model
        self.data = data
        self.hide_pos = hide_pos

        # Store body information
        self.objects = {}
        for name in object_names:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            self.objects[name] = {
                "geom_id": geom_id,
                "body_id": body_id,
                "contype": model.geom_contype[geom_id],
                "conaffinity": model.geom_conaffinity[geom_id],
                "mass": model.body_mass[body_id]
            }
            self._hide_object(name)

    def add_object(self, name, pos, quat):
        assert name in self.objects, f"Unknown object: {name}"
        self._activate_object(name, pos, quat)
        print(f"Added {name}")

    def remove_object(self, name):
        assert name in self.objects, f"Unknown object: {name}"
        self._hide_object(name)
        print(f"Removed {name}")

    def move_object(self, name, pos, quat):
        assert name in self.objects, f"Unknown object: {name}"
        self._move_object(name, pos, quat)

    def update(self, object_list, persist=False):
        current_names = set()
        for obj in object_list:
            name = obj["name"]
            current_names.add(name)
            status = obj.get("status", "active")
            if status == "active":
                self.add_object(name, obj["pos"], obj["quat"])
            elif status == "inactive":
                self.remove_object(name)

        if not persist:
            for name in self.objects:
                if name not in current_names:
                    self.remove_object(name)

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
        qpos_addr = self.model.jnt_qposadr[joint_adr]
        qvel_addr = self.model.jnt_dofadr[joint_adr]

        self.data.qpos[qpos_addr:qpos_addr+3] = pos
        self.data.qpos[qpos_addr+3:qpos_addr+7] = quat
        self.data.qvel[qvel_addr:qvel_addr+6] = 0  # Reset velocity to avoid shaking

    def _hide_object(self, name):
        self._move_object(name, self.hide_pos, [1, 0, 0, 0])
        obj = self.objects[name]
        self.model.geom_contype[obj["geom_id"]] = 0
        self.model.geom_conaffinity[obj["geom_id"]] = 0
        self.model.body_mass[obj["body_id"]] = 0.0

def get_object_names_from_model(model, prefix=""):
    exclude = {"world", "table"}
    return [model.body(i).name for i in range(model.nbody)
            if model.body(i).name.startswith(prefix) and model.body(i).name not in exclude]