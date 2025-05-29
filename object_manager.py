import numpy as np
import mujoco

class ObjectManager:
    def __init__(self, model, data, object_names, hide_pos=[1000, 1000, 1000]):
        self.model = model
        self.data = data
        self.hide_pos = hide_pos

        self.objects = {}
        for name in object_names:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            self.objects[name] = {
                "active": False,
                "name": None,
                "geom_id": geom_id,
                "body_id": body_id,
                "contype": model.geom_contype[geom_id],
                "conaffinity": model.geom_conaffinity[geom_id],
                "mass": model.body_mass[body_id]
            }
            self._hide_object(name)

    def add_object(self, name, pos, quat):
        available = [obj for obj, state in self.objects.items() if not state["active"]]
        if not available:
            print(f"Warning: No available objects to assign to {name}")
            return
        obj_name = available[0]
        state = self.objects[obj_name]
        state["active"] = True
        state["name"] = name

        self.model.geom_contype[state["geom_id"]] = state["contype"]
        self.model.geom_conaffinity[state["geom_id"]] = state["conaffinity"]
        self.model.body_mass[state["body_id"]] = state["mass"]
        self._move_object(obj_name, pos, quat)
        print(f"Added {name} as {obj_name}")

    def remove_object(self, name):
        for obj_name, state in self.objects.items():
            if state["name"] == name:
                self._hide_object(obj_name)
                state["active"] = False
                state["name"] = None
                print(f"Removed {name} (was {obj_name})")
                return
        print(f"Warning: {name} not found in active objects")

    def move_object(self, name, pos, quat):
        for obj_name, state in self.objects.items():
            if state["name"] == name:
                self._move_object(obj_name, pos, quat)
                return
        print(f"Warning: {name} not found in active objects")

    def update(self, object_list):
        current_names = set()
        for obj in object_list:
            name = obj["name"]
            current_names.add(name)
            status = obj.get("status", "active")
            if status == "active":
                if any(state["name"] == name for state in self.objects.values()):
                    self.move_object(name, obj["pos"], obj["quat"])
                else:
                    self.add_object(name, obj["pos"], obj["quat"])
            elif status == "inactive":
                self.remove_object(name)

        for obj_name, state in self.objects.items():
            if state["active"] and state["name"] not in current_names:
                self.remove_object(state["name"])

    def _move_object(self, obj_name, pos, quat):
        body_id = self.objects[obj_name]["body_id"]
        joint_adr = self.model.body_jntadr[body_id]
        qpos_addr = self.model.jnt_qposadr[joint_adr]
        self.data.qpos[qpos_addr:qpos_addr+3] = pos
        self.data.qpos[qpos_addr+3:qpos_addr+7] = quat

    def _hide_object(self, obj_name):
        self._move_object(obj_name, self.hide_pos, [1, 0, 0, 0])
        state = self.objects[obj_name]
        self.model.geom_contype[state["geom_id"]] = 0
        self.model.geom_conaffinity[state["geom_id"]] = 0
        self.model.body_mass[state["body_id"]] = 0.0

def get_object_names_from_model(model, prefix="object"):
    return [model.body(i).name for i in range(model.nbody) if model.body(i).name.startswith(prefix)]
