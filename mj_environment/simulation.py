import mujoco
import numpy as np

class Simulation:
    """Handles MuJoCo stepping, reset, and forward integration."""
    def __init__(self, scene_xml: str):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)

    def step(self, ctrl=None):
        if ctrl is not None:
            self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    def clone_data(self):
        clone = mujoco.MjData(self.model)
        np.copyto(clone.qpos, self.data.qpos)
        np.copyto(clone.qvel, self.data.qvel)
        mujoco.mj_forward(self.model, clone)
        return clone