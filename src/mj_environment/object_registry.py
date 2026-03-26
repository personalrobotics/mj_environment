"""
object_registry.py
Manages object lifecycle (activation, hiding, movement) in MuJoCo.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Union

import mujoco
import numpy as np

from .exceptions import (
    ObjectTypeNotFoundError,
    ObjectNotFoundError,
    ObjectPoolExhaustedError,
)

# Project constants
DEFAULT_HIDE_POSITION = [0, 0, -1]
HIDE_GRID_SPACING = 0.5  # meters between hidden object parking spots
IDENTITY_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
_QUAT_NORM_EPSILON = 1e-10

logger = logging.getLogger(__name__)


def _normalize_quaternion(quat: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Normalize a quaternion to unit length. Raises ValueError for near-zero input."""
    q = np.array(quat, dtype=float)
    norm = np.linalg.norm(q)
    if norm < _QUAT_NORM_EPSILON:
        raise ValueError(
            f"Cannot normalize near-zero quaternion {q}. "
            f"Magnitude {norm} is below threshold {_QUAT_NORM_EPSILON}."
        )
    return q / norm


class _BodyIndices(NamedTuple):
    """Cached indices for a body's joint and state arrays."""
    body_id: int
    joint_adr: int
    qpos_adr: int
    qvel_adr: int


class _MujocoIndexCache:
    """Cache for MuJoCo body/joint indices to avoid repeated lookups."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self._cache: Dict[str, _BodyIndices] = {}

    def get_body_indices(self, body_name: str) -> _BodyIndices:
        if body_name in self._cache:
            return self._cache[body_name]

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise KeyError(f"Body '{body_name}' not found in model")

        joint_adr = self.model.body_jntadr[body_id]
        qpos_adr = self.model.jnt_qposadr[joint_adr]
        qvel_adr = self.model.jnt_dofadr[joint_adr]

        indices = _BodyIndices(body_id, joint_adr, qpos_adr, qvel_adr)
        self._cache[body_name] = indices
        return indices

    def clear(self):
        self._cache.clear()


class ObjectRegistry:
    """
    Manages dynamic objects in a MuJoCo environment.

    Responsibilities:
      - Preloading objects from AssetManager and YAML scene config
      - Activating, hiding, and moving objects
      - Tracking object states and reuse

    Naming and Instance Reuse Policy:
      - Objects are preinitialized as e.g. cup_0, cup_1, ...
      - Hidden objects are reusable to respect MuJoCo immutability.
      - For unique identifiers, overprovision in the YAML config.

    State Access:
      - Use is_active(name) to check if an object is active
      - Use get_active_instances() to get list of active objects
      - Direct access to active_objects dict is discouraged for external use
      - Internal methods may access active_objects directly for performance

    Thread Safety:
      - This class is NOT thread-safe for concurrent writes.
      - Multiple threads may safely READ from `objects` dict concurrently.
      - All write operations (activate, hide, update) must be synchronized
        externally if accessed from multiple threads.
      - For multi-threaded perception pipelines, use a queue to batch updates
        and process them on a single thread (see perception_update_demo.py).
    """

    def __init__(
        self,
        model,
        data,
        asset_manager,
        scene_cfg: Dict[str, Any],
        hide_pos=DEFAULT_HIDE_POSITION,
    ):
        """
        Initialize ObjectRegistry.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            asset_manager: AssetManager instance
            scene_cfg: Objects dict from scene_config.yaml (already loaded by Environment)
            hide_pos: Position for hidden objects
        """
        self.model: mujoco.MjModel = model
        self.data: mujoco.MjData = data
        self.asset_manager = asset_manager
        self.hide_pos = np.array(hide_pos, dtype=float)
        self.objects: Dict[str, Dict[str, Any]] = {}
        self.active_objects: Dict[str, bool] = {}
        self.geom_visibility: Dict[int, np.ndarray] = {}  # Cache original geom colors
        self.geom_collision: Dict[int, tuple[int, int]] = {}  # Cache original (contype, conaffinity)
        self._index_cache = _MujocoIndexCache(model)  # Cache for body/joint indices

        self._hide_positions: Dict[str, np.ndarray] = {}

        self._preload_objects(scene_cfg)
        self._cache_geom_properties()
        self._compute_hide_grid()
        self._apply_hide_grid()

    # ------------------------------------------------------------------
    # Object preloading
    # ------------------------------------------------------------------
    def _preload_objects(self, scene_cfg: Dict[str, Any]):
        """Track object instances that were preloaded into the MuJoCo model by Environment."""
        for obj_type, entry in scene_cfg.items():
            if obj_type not in self.asset_manager.list():
                logger.warning("Unknown asset '%s', skipping preload", obj_type)
                continue

            names = entry["names"]  # normalized by Environment._load_scene_config
            self.objects[obj_type] = {"instances": []}

            for name in names:
                # Verify object exists in the model (it should be preloaded by Environment)
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id == -1:
                    logger.warning("Object '%s' not found in model, skipping", name)
                    continue
                self.objects[obj_type]["instances"].append(name)
                self.active_objects[name] = False

            logger.debug("Preloaded %d %s(s)", len(names), obj_type)

    def _cache_geom_properties(self):
        """Cache original RGBA colors and collision settings for all object geoms."""
        for obj_type in self.objects:
            for instance_name in self.objects[obj_type]["instances"]:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, instance_name)
                geom_adr = self.model.body_geomadr[body_id]
                geom_num = self.model.body_geomnum[body_id]

                # Store original colors and collision settings for each geom
                for i in range(geom_num):
                    geom_id = geom_adr + i
                    if geom_id not in self.geom_visibility:
                        self.geom_visibility[geom_id] = self.model.geom_rgba[geom_id].copy()
                        self.geom_collision[geom_id] = (
                            int(self.model.geom_contype[geom_id]),
                            int(self.model.geom_conaffinity[geom_id]),
                        )

                # Make all objects invisible and non-collidable initially (they start inactive)
                self._set_body_visibility(body_id, visible=False)

    def _compute_hide_grid(self):
        """Assign each instance a unique parking position on a grid underground.

        Spreading hidden objects apart prevents O(N^2) broadphase collision pairs
        that occur when all objects share a single hide position in MuJoCo.
        """
        all_instances = []
        for obj_type in self.objects:
            all_instances.extend(self.objects[obj_type]["instances"])

        n = len(all_instances)
        if n == 0:
            return

        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        for idx, name in enumerate(all_instances):
            col = idx % cols
            row = idx // cols
            x = self.hide_pos[0] + (col - (cols - 1) / 2) * HIDE_GRID_SPACING
            y = self.hide_pos[1] + (row - (rows - 1) / 2) * HIDE_GRID_SPACING
            z = self.hide_pos[2]
            self._hide_positions[name] = np.array([x, y, z], dtype=float)

    def _apply_hide_grid(self):
        """Move all inactive instances to their grid positions.

        Called once at init to spread objects that start at the shared hide_pos.
        """
        for name, pos in self._hide_positions.items():
            if not self.active_objects.get(name, False):
                indices = self._index_cache.get_body_indices(name)
                self.data.qpos[indices.qpos_adr:indices.qpos_adr + 3] = pos

    def copy(self, new_data: mujoco.MjData) -> 'ObjectRegistry':
        """
        Create an independent copy of this registry with new MjData.

        The copy shares the MjModel (immutable) but has independent state tracking.
        Used by Environment.fork() for creating planning environments.

        Args:
            new_data: The new MjData instance for the copy to use.

        Returns:
            A new ObjectRegistry with copied state but independent data.
        """
        clone = ObjectRegistry.__new__(ObjectRegistry)
        clone.model = self.model  # Shared (immutable)
        clone.data = new_data  # Independent
        clone.asset_manager = self.asset_manager  # Shared (read-only)
        clone.hide_pos = self.hide_pos.copy()
        clone._hide_positions = {k: v.copy() for k, v in self._hide_positions.items()}

        # Deep copy mutable state
        clone.objects = {
            obj_type: {"instances": list(info["instances"])}
            for obj_type, info in self.objects.items()
        }
        clone.active_objects = dict(self.active_objects)
        clone.geom_visibility = {k: v.copy() for k, v in self.geom_visibility.items()}
        clone.geom_collision = dict(self.geom_collision)  # Tuples are immutable, shallow copy OK

        # Create new index cache (shared model, so indices are the same)
        clone._index_cache = _MujocoIndexCache(clone.model)

        return clone

    def _parse_object_type(self, instance_name: str) -> Optional[str]:
        """
        Resolve object type from an instance name.

        Tries direct registry lookup first via get_type(), then falls back
        to parsing the {type}_{index} pattern for auto-activation in update().

        Examples:
            cup_0 -> cup
            kitchen_knife_2 -> kitchen_knife
            recycle_bin_right -> recycle_bin (if registered as custom name)
        """
        try:
            return self.get_type(instance_name)
        except ObjectNotFoundError:
            pass
        # Fallback: parse {type}_{index} pattern
        for obj_type in self.objects:
            if instance_name.startswith(obj_type + "_"):
                suffix = instance_name[len(obj_type) + 1:]
                if suffix.isdigit():
                    return obj_type
        return None

    def _set_body_visibility(self, body_id: int, visible: bool):
        """Show or hide all geoms of a body by setting RGBA alpha and collision flags."""
        geom_adr = self.model.body_geomadr[body_id]
        geom_num = self.model.body_geomnum[body_id]

        for i in range(geom_num):
            geom_id = geom_adr + i
            if geom_id in self.geom_visibility:
                if visible:
                    # Restore original alpha and collision settings
                    self.model.geom_rgba[geom_id, 3] = self.geom_visibility[geom_id][3]
                    contype, conaffinity = self.geom_collision[geom_id]
                    self.model.geom_contype[geom_id] = contype
                    self.model.geom_conaffinity[geom_id] = conaffinity
                else:
                    # Set alpha to 0 (invisible) and disable collisions
                    self.model.geom_rgba[geom_id, 3] = 0.0
                    self.model.geom_contype[geom_id] = 0
                    self.model.geom_conaffinity[geom_id] = 0
            else:
                logger.warning("Geom %d not in visibility cache, skipping", geom_id)

    # ------------------------------------------------------------------
    # Runtime API
    # ------------------------------------------------------------------
    def activate(
        self,
        obj_type: str,
        pos: Union[Sequence[float], np.ndarray],
        quat: Optional[Union[Sequence[float], np.ndarray]] = None,
    ) -> str:
        """
        Activate the next available hidden instance of a given type.

        Args:
            obj_type: Type of object to activate (e.g., "cup", "can")
            pos: Position [x, y, z] in meters
            quat: Orientation quaternion [w, x, y, z]. Defaults to identity.

        Returns:
            Name of the activated instance (e.g., "cup_0")

        Raises:
            ObjectTypeNotFoundError: If obj_type not in registry
            ObjectPoolExhaustedError: If all instances are already active
            ValueError: If quaternion has near-zero magnitude

        Example:
            >>> name = registry.activate("cup", [0.5, 0.0, 0.8])
            >>> print(f"Activated {name}")
            Activated cup_0
        """
        if obj_type not in self.objects:
            raise ObjectTypeNotFoundError(obj_type, list(self.objects.keys()))
        candidates = [n for n in self.objects[obj_type]["instances"] if not self.active_objects[n]]
        if not candidates:
            all_instances = self.objects[obj_type]["instances"]
            raise ObjectPoolExhaustedError(obj_type, len(all_instances), all_instances)
        name = candidates[0]
        indices = self._index_cache.get_body_indices(name)
        self.data.qpos[indices.qpos_adr:indices.qpos_adr+3] = np.array(pos, dtype=float)
        if quat is None:
            self.data.qpos[indices.qpos_adr+3:indices.qpos_adr+3+4] = IDENTITY_QUATERNION
        else:
            self.data.qpos[indices.qpos_adr+3:indices.qpos_adr+3+4] = _normalize_quaternion(quat)
        self.data.qvel[indices.qvel_adr:indices.qvel_adr+6] = 0
        self._set_body_visibility(indices.body_id, visible=True)
        self.active_objects[name] = True
        logger.debug("Activated %s", name)
        return name

    def hide(self, name: str) -> None:
        """Hide an active object by moving and disabling it."""
        if name not in self.active_objects:
            raise ObjectNotFoundError(name, list(self.active_objects.keys()))
        if not self.active_objects[name]:
            return
        indices = self._index_cache.get_body_indices(name)
        self.data.qpos[indices.qpos_adr:indices.qpos_adr+3] = self._hide_positions[name]
        self.data.qpos[indices.qpos_adr+3:indices.qpos_adr+3+4] = IDENTITY_QUATERNION
        self.data.qvel[indices.qvel_adr:indices.qvel_adr+6] = 0
        self._set_body_visibility(indices.body_id, visible=False)
        self.active_objects[name] = False
        logger.debug("Hid %s", name)

    def update(
        self,
        updates: List[Dict[str, Any]],
        hide_unlisted: Optional[bool] = None,
    ) -> None:
        """
        Batch activate/move/hide objects based on updates.

        Args:
            updates: List of dicts with keys: name, pos, quat (optional)
            hide_unlisted: If True (default), hide objects not in updates list.
                          If False, keep previously active objects visible.

        Raises:
            TypeError: If updates is not a list
            ValueError: If any update dict is missing required keys or has invalid values

        Example:
            >>> # Hide objects not in the list (default)
            >>> registry.update([{"name": "cup_0", "pos": [0, 0, 0.5]}])
            >>> # Keep previously active objects
            >>> registry.update([...], hide_unlisted=False)
        """
        # Input validation
        if not isinstance(updates, list):
            raise TypeError(f"updates must be a list, got {type(updates).__name__}")

        for i, upd in enumerate(updates):
            if not isinstance(upd, dict):
                raise TypeError(f"updates[{i}] must be a dict, got {type(upd).__name__}")
            if "name" not in upd:
                raise ValueError(f"updates[{i}] missing required key 'name'")
            if "pos" not in upd:
                raise ValueError(f"updates[{i}] missing required key 'pos' (name={upd['name']!r})")
            pos = upd["pos"]
            try:
                if len(pos) != 3:
                    raise ValueError(f"updates[{i}] 'pos' must have 3 elements, got {len(pos)} (name={upd['name']!r})")
            except TypeError:
                raise ValueError(f"updates[{i}] 'pos' must be a sequence, got {type(pos).__name__} (name={upd['name']!r})")

        # Default behavior: hide unlisted objects
        if hide_unlisted is None:
            hide_unlisted = True

        active_now = set()

        for upd in updates:
            name = upd["name"]
            pos = np.array(upd["pos"], dtype=float)
            quat = _normalize_quaternion(upd.get("quat", [1, 0, 0, 0]))

            if name not in self.active_objects:
                # Find object type by checking registry membership
                # This handles object types with underscores (e.g., kitchen_knife_0 -> kitchen_knife)
                obj_type = self._parse_object_type(name)
                if obj_type is None:
                    logger.warning("Could not determine object type for '%s', skipping", name)
                    continue
                new_name = self.activate(obj_type, pos, quat)
                active_now.add(new_name)
            else:
                indices = self._index_cache.get_body_indices(name)
                self.data.qpos[indices.qpos_adr:indices.qpos_adr+3] = pos
                self.data.qpos[indices.qpos_adr+3:indices.qpos_adr+3+4] = quat
                self.data.qvel[indices.qvel_adr:indices.qvel_adr+6] = 0
                # Make sure the object is visible (it might have been hidden previously)
                if not self.active_objects[name]:
                    self._set_body_visibility(indices.body_id, visible=True)
                self.active_objects[name] = True
                active_now.add(name)

        if hide_unlisted:
            for name, active in list(self.active_objects.items()):
                if active and name not in active_now:
                    self.hide(name)

    def is_active(self, name: str) -> bool:
        """
        Check if an object instance is currently active.

        Args:
            name: Instance name (e.g., "cup_0")

        Returns:
            True if object is active, False otherwise

        Example:
            >>> if registry.is_active("cup_0"):
            ...     print("Cup is visible")
        """
        return self.active_objects.get(name, False)

    def get_type(self, instance_name: str) -> str:
        """
        Get the object type for an instance name.

        Args:
            instance_name: Instance name (e.g., "cup_0" or a custom name)

        Returns:
            Object type string (e.g., "cup")

        Raises:
            ObjectNotFoundError: If instance_name is not in the registry

        Example:
            >>> registry.get_type("cup_0")
            'cup'
        """
        for obj_type, info in self.objects.items():
            if instance_name in info["instances"]:
                return obj_type
        raise ObjectNotFoundError(instance_name, list(self.active_objects.keys()))

    def sync_visibility(self) -> None:
        """
        Synchronize geom visibility with active_objects state.

        Updates the visual appearance (RGBA alpha) of all object geoms
        to match their active/inactive status. Call this after directly
        modifying active_objects dict (e.g., after loading state).

        This is an internal method used by Environment.load_state() and
        Environment.sync_from() to ensure visual consistency.
        """
        for name, is_active in self.active_objects.items():
            indices = self._index_cache.get_body_indices(name)
            self._set_body_visibility(indices.body_id, visible=is_active)

    def get_active_instances(self, obj_type: Optional[str] = None) -> List[str]:
        """
        Get list of currently active object instances.

        Args:
            obj_type: Optional object type to filter by (e.g., "cup")
                     If None, returns all active instances

        Returns:
            List of active instance names

        Example:
            >>> # Get all active objects
            >>> active = registry.get_active_instances()
            >>> print(f"Active: {active}")
            Active: ['cup_0', 'can_1']

            >>> # Get active objects of specific type
            >>> active_cups = registry.get_active_instances("cup")
            >>> print(f"Active cups: {active_cups}")
            Active cups: ['cup_0']
        """
        if obj_type is None:
            return [name for name, is_active in self.active_objects.items() if is_active]
        else:
            if obj_type not in self.objects:
                return []
            instances = self.objects[obj_type]["instances"]
            return [name for name in instances if self.active_objects.get(name, False)]