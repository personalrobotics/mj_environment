"""Simulation — thin wrapper around MuJoCo model and data with cloning support."""

import mujoco
import numpy as np


class Simulation:
    """Wraps MuJoCo model and data for stepping, resetting, and cloning."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData = None):
        self.model = model
        self.data = data if data is not None else mujoco.MjData(model)

    # ------------------------------------------------------------------
    # Basic stepping
    # ------------------------------------------------------------------
    def step(self, ctrl=None):
        """Advance simulation one step with optional control input."""
        if ctrl is not None:
            np.copyto(self.data.ctrl, ctrl)
        mujoco.mj_step(self.model, self.data)

    def reset(self):
        """Reset the simulation to its initial state."""
        mujoco.mj_resetData(self.model, self.data)

    # ------------------------------------------------------------------
    # ✅ Cloning / State Copying
    # ------------------------------------------------------------------
    def clone_data(self) -> mujoco.MjData:
        """
        Create a deep clone of the current simulation state.

        Returns:
            mujoco.MjData: A new data object with identical simulation state.
        """
        clone = mujoco.MjData(self.model)
        self.copy_data(self.model, clone, self.data)
        return clone

    @staticmethod
    def copy_data(model: mujoco.MjModel, dst: mujoco.MjData, src: mujoco.MjData):
        """
        Copy all MuJoCo state arrays between two MjData objects.

        Args:
            model: The MuJoCo model (required for mj_forward)
            dst: Destination MjData (already constructed for the same model)
            src: Source MjData to copy from
        """
        np.copyto(dst.qpos, src.qpos)
        np.copyto(dst.qvel, src.qvel)
        np.copyto(dst.act, src.act)
        np.copyto(dst.ctrl, src.ctrl)
        np.copyto(dst.qacc, src.qacc)
        np.copyto(dst.qfrc_applied, src.qfrc_applied)
        mujoco.mj_forward(model, dst)