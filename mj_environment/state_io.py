"""Handles YAML-based MuJoCo state serialization."""

import yaml
import numpy as np
import mujoco

from .constants import STATE_IO_SCHEMA_VERSION
from .exceptions import StateError


class StateIO:
    """Serialize and restore simulation state."""

    SCHEMA_VERSION = STATE_IO_SCHEMA_VERSION

    def save(self, model, data, active_objects, path: str):
        """
        Save simulation state to YAML file.

        Args:
            model: MuJoCo model (for validation)
            data: MuJoCo data containing state
            active_objects: Dict or set of active object names
            path: Output file path

        Raises:
            StateError: If state contains NaN/Inf values or write fails
        """
        # Validate state before saving
        if not np.all(np.isfinite(data.qpos)):
            raise StateError(
                "Cannot save: qpos contains NaN or Inf values",
                hint="Check for physics explosions or invalid object poses.",
            )
        if not np.all(np.isfinite(data.qvel)):
            raise StateError(
                "Cannot save: qvel contains NaN or Inf values",
                hint="Check for physics explosions or invalid object velocities.",
            )

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
        """
        Load simulation state from YAML file.

        Args:
            model: MuJoCo model (for dimension validation)
            data: MuJoCo data to restore state into
            path: Path to state file

        Returns:
            Dict mapping object names to active status (bool)

        Raises:
            StateError: If file has incompatible schema or dimensions

        Example:
            >>> active_objects = state_io.load(model, data, "checkpoint.yaml")
            >>> env.registry.update_from_dict(active_objects)
        """
        with open(path, "r") as f:
            state = yaml.safe_load(f)

        if state.get("schema_version") != self.SCHEMA_VERSION:
            raise StateError(
                f"Incompatible schema version: found {state.get('schema_version')}, expected {self.SCHEMA_VERSION}",
                hint="This state file was saved with a different version of mj_environment.",
            )

        qpos = np.array(state["qpos"])
        qvel = np.array(state["qvel"])

        if len(qpos) != model.nq or len(qvel) != model.nv:
            raise StateError(
                f"State dimensions mismatch: qpos={len(qpos)} (expected {model.nq}), qvel={len(qvel)} (expected {model.nv})",
                hint="The state file was saved from a different model configuration.",
            )

        # Parse active_objects BEFORE modifying any state.
        # If the active_objects section is malformed, we want to fail before
        # touching qpos/qvel to avoid leaving the environment in a corrupt state.
        active_list = state.get("active_objects", [])
        if isinstance(active_list, dict):
            active_objects = active_list
        else:
            active_objects = {name: True for name in active_list}

        # All validation passed — now apply state.
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)

        return active_objects