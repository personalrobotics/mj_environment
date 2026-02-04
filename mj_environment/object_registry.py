"""
object_registry.py
Manages object lifecycle (activation, hiding, movement) in MuJoCo.
"""

import logging
import warnings

import mujoco
import numpy as np
from typing import Dict, List, Any, Optional, Union, Sequence

from .constants import IDENTITY_QUATERNION, POSITION_DIM, QUATERNION_DIM, DOF_DIM, RGBA_ALPHA_CHANNEL
from .exceptions import (
    ObjectTypeNotFoundError,
    ObjectNotFoundError,
    ObjectPoolExhaustedError,
)
from .mujoco_helpers import MujocoIndexCache
from .quaternion import normalize_quaternion

logger = logging.getLogger(__name__)


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
        hide_pos=[0, 0, -1],
        verbose=False,
    ):
        """
        Initialize ObjectRegistry.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            asset_manager: AssetManager instance
            scene_cfg: Objects dict from scene_config.yaml (already loaded by Environment)
            hide_pos: Position for hidden objects
            verbose: Enable debug logging
        """
        self.model = model
        self.data = data
        self.asset_manager = asset_manager
        self.scene_cfg = scene_cfg
        self.hide_pos = np.array(hide_pos, dtype=float)
        self.verbose = verbose
        self.objects: Dict[str, Dict[str, Any]] = {}
        self.active_objects: Dict[str, bool] = {}
        self.geom_visibility: Dict[int, np.ndarray] = {}  # Cache original geom colors
        self.geom_collision: Dict[int, tuple[int, int]] = {}  # Cache original (contype, conaffinity)
        self._index_cache = MujocoIndexCache(model)  # Cache for body/joint indices

        self._preload_objects()
        self._cache_geom_properties()

    # ------------------------------------------------------------------
    # Object preloading
    # ------------------------------------------------------------------
    def _preload_objects(self):
        """Track object instances that were preloaded into the MuJoCo model by Environment."""
        for obj_type, entry in self.scene_cfg.items():
            if obj_type not in self.asset_manager.list():
                logger.warning("Unknown asset '%s', skipping preload", obj_type)
                continue

            count = entry.get("count", 1)
            self.objects[obj_type] = {"count": count, "instances": []}

            for i in range(count):
                name = f"{obj_type}_{i}"
                # Verify object exists in the model (it should be preloaded by Environment)
                try:
                    body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                    self.objects[obj_type]["instances"].append(name)
                    self.active_objects[name] = False
                except (TypeError, AttributeError):
                    logger.warning("Object '%s' not found in model, skipping", name)

            logger.debug("Preloaded %d %s(s)", count, obj_type)

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
        clone.verbose = self.verbose

        # Deep copy mutable state
        clone.objects = {
            obj_type: {
                "count": info["count"],
                "instances": list(info["instances"])
            }
            for obj_type, info in self.objects.items()
        }
        clone.active_objects = dict(self.active_objects)
        clone.geom_visibility = {k: v.copy() for k, v in self.geom_visibility.items()}
        clone.geom_collision = dict(self.geom_collision)  # Tuples are immutable, shallow copy OK
        clone.scene_cfg = self.scene_cfg  # Read-only after init

        # Create new index cache (shared model, so indices are the same)
        clone._index_cache = MujocoIndexCache(clone.model)

        return clone

    def _parse_object_type(self, instance_name: str) -> Optional[str]:
        """
        Parse object type from instance name, handling underscores correctly.

        Examples:
            cup_0 -> cup
            kitchen_knife_2 -> kitchen_knife
            my_cool_object_15 -> my_cool_object (if registered)
        """
        # Check each registered object type to find a match
        for obj_type in self.objects:
            # Instance names follow pattern: {obj_type}_{index}
            if instance_name.startswith(obj_type + "_"):
                suffix = instance_name[len(obj_type) + 1:]
                # Verify suffix is a valid index (digits only)
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
                    self.model.geom_rgba[geom_id, RGBA_ALPHA_CHANNEL] = self.geom_visibility[geom_id][RGBA_ALPHA_CHANNEL]
                    contype, conaffinity = self.geom_collision[geom_id]
                    self.model.geom_contype[geom_id] = contype
                    self.model.geom_conaffinity[geom_id] = conaffinity
                else:
                    # Set alpha to 0 (invisible) and disable collisions
                    self.model.geom_rgba[geom_id, RGBA_ALPHA_CHANNEL] = 0.0
                    self.model.geom_contype[geom_id] = 0
                    self.model.geom_conaffinity[geom_id] = 0

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
        self.data.qpos[indices.qpos_adr:indices.qpos_adr+POSITION_DIM] = np.array(pos, dtype=float)
        if quat is None:
            self.data.qpos[indices.qpos_adr+POSITION_DIM:indices.qpos_adr+POSITION_DIM+QUATERNION_DIM] = IDENTITY_QUATERNION
        else:
            self.data.qpos[indices.qpos_adr+POSITION_DIM:indices.qpos_adr+POSITION_DIM+QUATERNION_DIM] = normalize_quaternion(quat)
        self.data.qvel[indices.qvel_adr:indices.qvel_adr+DOF_DIM] = 0
        self._set_body_visibility(indices.body_id, visible=True)
        self.active_objects[name] = True
        logger.debug("Activated %s", name)
        return name

    def hide(self, name: str):
        """Hide an active object by moving and disabling it."""
        if name not in self.active_objects:
            raise ObjectNotFoundError(name, list(self.active_objects.keys()))
        if not self.active_objects[name]:
            return
        indices = self._index_cache.get_body_indices(name)
        self.data.qpos[indices.qpos_adr:indices.qpos_adr+POSITION_DIM] = self.hide_pos
        self.data.qpos[indices.qpos_adr+POSITION_DIM:indices.qpos_adr+POSITION_DIM+QUATERNION_DIM] = IDENTITY_QUATERNION
        self.data.qvel[indices.qvel_adr:indices.qvel_adr+DOF_DIM] = 0
        self._set_body_visibility(indices.body_id, visible=False)
        self.active_objects[name] = False
        logger.debug("Hid %s", name)

    def update(
        self,
        updates: List[Dict[str, Any]],
        hide_unlisted: Optional[bool] = None,
        persist: Optional[bool] = None,
    ) -> None:
        """
        Batch activate/move/hide objects based on updates.

        Args:
            updates: List of dicts with keys: name, pos, quat (optional)
            hide_unlisted: If True (default), hide objects not in updates list.
                          If False, keep previously active objects visible.
            persist: DEPRECATED. Use hide_unlisted instead.
                    persist=False is equivalent to hide_unlisted=True
                    persist=True is equivalent to hide_unlisted=False

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

        # Handle parameter deprecation
        if persist is not None:
            warnings.warn(
                "The 'persist' parameter is deprecated and will be removed in v2.0. "
                "Use 'hide_unlisted' instead: persist=False -> hide_unlisted=True, "
                "persist=True -> hide_unlisted=False",
                DeprecationWarning,
                stacklevel=2
            )
            if hide_unlisted is None:
                hide_unlisted = not persist  # Invert logic
            # If both provided, hide_unlisted takes precedence (ignore persist)

        # Default behavior: hide unlisted objects
        if hide_unlisted is None:
            hide_unlisted = True

        active_now = set()

        for upd in updates:
            name = upd["name"]
            pos = np.array(upd["pos"], dtype=float)
            quat = normalize_quaternion(upd.get("quat", [1, 0, 0, 0]))

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
                self.data.qpos[indices.qpos_adr:indices.qpos_adr+POSITION_DIM] = pos
                self.data.qpos[indices.qpos_adr+POSITION_DIM:indices.qpos_adr+POSITION_DIM+QUATERNION_DIM] = quat
                self.data.qvel[indices.qvel_adr:indices.qvel_adr+DOF_DIM] = 0
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