"""Helper functions and utilities for MuJoCo model operations."""
from typing import NamedTuple
import mujoco


class BodyIndices(NamedTuple):
    """Cached indices for a body's joint and state arrays."""
    body_id: int
    joint_adr: int
    qpos_adr: int
    qvel_adr: int


class MujocoIndexCache:
    """
    Cache for MuJoCo body/joint indices to avoid repeated lookups.

    This eliminates the repeated pattern of:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        joint_adr = model.body_jntadr[body_id]
        qpos_adr = model.jnt_qposadr[joint_adr]
        qvel_adr = model.jnt_dofadr[joint_adr]

    Example:
        >>> cache = MujocoIndexCache(model)
        >>> indices = cache.get_body_indices("can_0")
        >>> data.qpos[indices.qpos_adr:indices.qpos_adr+3] = [0, 0, 1]
    """

    def __init__(self, model: mujoco.MjModel):
        """
        Initialize index cache.

        Args:
            model: MuJoCo model to cache indices for
        """
        self.model = model
        self._cache = {}

    def get_body_indices(self, body_name: str) -> BodyIndices:
        """
        Get all indices for a body's joint and state arrays.

        Args:
            body_name: Name of the body in the MuJoCo model

        Returns:
            BodyIndices with body_id, joint_adr, qpos_adr, qvel_adr

        Raises:
            KeyError: If body name not found in model
        """
        if body_name in self._cache:
            return self._cache[body_name]

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise KeyError(f"Body '{body_name}' not found in model")

        joint_adr = self.model.body_jntadr[body_id]
        qpos_adr = self.model.jnt_qposadr[joint_adr]
        qvel_adr = self.model.jnt_dofadr[joint_adr]

        indices = BodyIndices(body_id, joint_adr, qpos_adr, qvel_adr)
        self._cache[body_name] = indices
        return indices

    def clear(self):
        """Clear the index cache."""
        self._cache.clear()
