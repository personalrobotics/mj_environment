"""Wrapper around MuJoCo simulation lifecycle."""

import mujoco
import numpy as np
import time
from typing import Optional


class Simulation:
    """Encapsulates MuJoCo model, data, and stepping."""

    def __init__(self, scene_xml: str):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)

    def step(self, ctrl: Optional[np.ndarray] = None):
        if ctrl is not None:
            self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    def clone_data(self) -> mujoco.MjData:
        """Return a deep copy of current MjData."""
        clone = mujoco.MjData(self.model)
        np.copyto(clone.qpos, self.data.qpos)
        np.copyto(clone.qvel, self.data.qvel)
        mujoco.mj_forward(self.model, clone)
        return clone

    def simulate(self, duration: float, realtime: bool = False):
        """Run the simulation for a given duration."""
        steps = int(duration / self.model.opt.timestep)
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            if realtime:
                time.sleep(self.model.opt.timestep)