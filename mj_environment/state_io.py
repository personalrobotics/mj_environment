"""Handles YAML-based MuJoCo state serialization."""

import yaml
import numpy as np
import mujoco
from typing import Set


class StateIO:
    """Serialize and restore simulation state."""

    SCHEMA_VERSION = 1

    def save(self, model, data, active_objects: Set[str], path: str):
        state = {
            "schema_version": self.SCHEMA_VERSION,
            "qpos": data.qpos.tolist(),
            "qvel": data.qvel.tolist(),
            "active_objects": list(active_objects),
        }
        with open(path, "w") as f:
            yaml.safe_dump(state, f)

    def load(self, model, data, path: str) -> Set[str]:
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
        return set(state.get("active_objects", []))