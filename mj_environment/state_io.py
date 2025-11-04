"""Handles YAML-based MuJoCo state serialization."""

import yaml
import numpy as np
import mujoco


class StateIO:
    """Serialize and restore simulation state."""

    SCHEMA_VERSION = 1

    def save(self, model, data, active_objects, path: str):
        """Save simulation state. active_objects can be dict or set."""
        # Convert to dict if needed
        if isinstance(active_objects, dict):
            active_dict = active_objects
        else:
            active_dict = {name: True for name in active_objects}
        
        state = {
            "schema_version": self.SCHEMA_VERSION,
            "qpos": data.qpos.tolist(),
            "qvel": data.qvel.tolist(),
            "active_objects": active_dict,
        }
        with open(path, "w") as f:
            yaml.safe_dump(state, f)

    def load(self, model, data, path: str):
        """Load simulation state. Returns dict of active objects."""
        with open(path, "r") as f:
            state = yaml.safe_load(f)

        if state.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("Incompatible YAML schema version.")

        qpos = np.array(state["qpos"])
        qvel = np.array(state["qvel"])

        if len(qpos) != model.nq or len(qvel) != model.nv:
            raise ValueError("State file does not match model dimensions.")

        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        
        # Return as dict
        active_list = state.get("active_objects", [])
        if isinstance(active_list, dict):
            return active_list
        return {name: True for name in active_list}