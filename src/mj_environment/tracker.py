# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""
Object tracker for persistent detection-to-instance association.

Maps raw perception detections (type + pose) to persistent MuJoCo instance
names across frames using nearest-neighbor matching. Designed as a base class
so more sophisticated trackers (Kalman, Hungarian) can be drop-in replacements.

Example::

    tracker = ObjectTracker(env.registry, max_distance=0.15)

    # Each perception frame:
    updates = tracker.associate([
        {"type": "cup", "pos": [0.1, 0.2, 0.4]},
        {"type": "plate", "pos": [0.3, -0.1, 0.4]},
    ])
    env.update(updates, hide_unlisted=True)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .object_registry import ObjectRegistry


class BaseTracker(ABC):
    """Abstract base for object trackers.

    Subclasses implement ``associate()`` to map raw detections to named
    instance updates. The interface is stable so that implementations
    using Kalman filtering, the Hungarian algorithm, or learned re-ID
    can be swapped in without changing caller code.

    Args:
        registry: ObjectRegistry that manages available instances.
        max_distance: Maximum Euclidean distance (meters) for a detection
            to match an existing tracked instance.
    """

    def __init__(self, registry: ObjectRegistry, max_distance: float = 0.15):
        self.registry = registry
        self.max_distance = max_distance

    @abstractmethod
    def associate(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Match raw detections to persistent instance names.

        Args:
            detections: List of dicts, each with at least:
                - ``"type"`` (str): object type (e.g., ``"cup"``)
                - ``"pos"`` (list[float]): ``[x, y, z]`` world position
                - ``"quat"`` (list[float], optional): ``[w, x, y, z]``
                  orientation, defaults to ``[1, 0, 0, 0]``

        Returns:
            List of dicts ready for ``env.update()``, each with:
                - ``"name"`` (str): persistent instance name (e.g., ``"cup_0"``)
                - ``"pos"`` (list[float])
                - ``"quat"`` (list[float])
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all tracking state."""
        ...


class ObjectTracker(BaseTracker):
    """Nearest-neighbor tracker that associates detections by position.

    Each frame, tracked instances are matched to the closest same-type
    detection within ``max_distance``. Unmatched detections are assigned
    to the next available hidden instance. Unmatched tracked instances
    are dropped (the caller's ``hide_unlisted=True`` will hide them).

    Args:
        registry: ObjectRegistry that manages available instances.
        max_distance: Maximum match distance in meters (default 0.15).
    """

    def __init__(self, registry: ObjectRegistry, max_distance: float = 0.15):
        super().__init__(registry, max_distance)
        # instance_name → (obj_type, last_known_position)
        self._tracked: Dict[str, Tuple[str, np.ndarray]] = {}

    def associate(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        updates: List[Dict[str, Any]] = []
        remaining = list(detections)

        # 1. Match tracked instances to nearest same-type detection
        for name, (obj_type, last_pos) in list(self._tracked.items()):
            best_idx: Optional[int] = None
            best_dist = self.max_distance

            for i, det in enumerate(remaining):
                if det.get("type") != obj_type:
                    continue
                dist = float(np.linalg.norm(np.asarray(det["pos"]) - last_pos))
                if dist < best_dist:
                    best_idx, best_dist = i, dist

            if best_idx is not None:
                det = remaining.pop(best_idx)
                pos = list(det["pos"])
                quat = det.get("quat", [1, 0, 0, 0])
                self._tracked[name] = (obj_type, np.asarray(pos))
                updates.append({"name": name, "pos": pos, "quat": quat})
            else:
                # Lost — stop tracking; caller's hide_unlisted handles hiding
                del self._tracked[name]

        # 2. Assign available instances to unmatched detections
        for det in remaining:
            obj_type = det.get("type")
            if obj_type is None:
                continue
            name = self._next_available(obj_type)
            if name is None:
                continue  # pool exhausted or unknown type
            pos = list(det["pos"])
            quat = det.get("quat", [1, 0, 0, 0])
            self._tracked[name] = (obj_type, np.asarray(pos))
            updates.append({"name": name, "pos": pos, "quat": quat})

        return updates

    def reset(self) -> None:
        self._tracked.clear()

    def _next_available(self, obj_type: str) -> Optional[str]:
        """Return the first hidden and untracked instance of ``obj_type``, or None."""
        if obj_type not in self.registry.objects:
            return None
        for name in self.registry.objects[obj_type]["instances"]:
            if not self.registry.active_objects.get(name, False) and name not in self._tracked:
                return name
        return None
