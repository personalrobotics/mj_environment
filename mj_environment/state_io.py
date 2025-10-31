import yaml
import numpy as np
import mujoco

class StateIO:
    """Handles serialization of MuJoCo state."""
    def save(self, model, data, active_objects, path: str):
        state = {
            "qpos": data.qpos.tolist(),
            "qvel": data.qvel.tolist(),
            "active_objects": list(active_objects),
        }
        yaml.safe_dump(state, open(path, "w"))

    def load(self, model, data, path: str):
        state = yaml.safe_load(open(path))
        data.qpos[:] = np.array(state["qpos"])
        data.qvel[:] = np.array(state["qvel"])
        mujoco.mj_forward(model, data)
        return set(state["active_objects"])