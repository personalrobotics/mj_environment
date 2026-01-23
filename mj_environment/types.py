"""
Type definitions for mj_environment.

Provides TypedDict definitions for structured data passed to the API.
These enable IDE autocompletion and type checking with mypy.

Example:
    from mj_environment.types import Detection

    detections: list[Detection] = [
        {"name": "cup_0", "pos": [0.1, 0.2, 0.3], "quat": [1, 0, 0, 0]},
        {"name": "plate_0", "pos": [-0.2, 0.0, 0.4]},
    ]
    env.update(detections)
"""

from typing import List, Dict, Any, Optional, Union
from typing_extensions import TypedDict, NotRequired
import numpy as np
import numpy.typing as npt


# Position and orientation types
Position = Union[List[float], npt.NDArray[np.floating[Any]]]
Quaternion = Union[List[float], npt.NDArray[np.floating[Any]]]


class Detection(TypedDict, total=False):
    """
    A single object detection for use with Environment.update().

    Attributes:
        name: Instance name (e.g., "cup_0", "plate_1").
        pos: Position [x, y, z] in world coordinates.
        quat: Quaternion [w, x, y, z] for orientation.
              Defaults to [1, 0, 0, 0] (identity) if not provided.

    Example:
        detection: Detection = {
            "name": "cup_0",
            "pos": [0.1, 0.2, 0.3],
            "quat": [1, 0, 0, 0],
        }
    """

    name: str
    pos: Position
    quat: NotRequired[Quaternion]


class ObjectConfigEntry(TypedDict, total=False):
    """
    Configuration entry for a single object type in scene_config.yaml.

    Attributes:
        count: Number of instances to preallocate (default: 1).

    Example in YAML:
        objects:
          cup:
            count: 5
          plate:
            count: 3
    """

    count: NotRequired[int]


class SceneConfig(TypedDict):
    """
    Root configuration for scene_config.yaml.

    Attributes:
        objects: Dictionary mapping object type names to their configs.

    Example:
        config: SceneConfig = {
            "objects": {
                "cup": {"count": 5},
                "plate": {"count": 3},
            }
        }
    """

    objects: Dict[str, ObjectConfigEntry]


class ObjectMetadata(TypedDict, total=False):
    """
    Object metadata from meta.yaml files.

    Attributes:
        name: Object type name.
        category: List of categories this object belongs to.
        mass: Mass in kg (overrides XML value).
        color: RGBA color [r, g, b, a] (overrides XML value).
        scale: Scale factor (overrides XML value).
        mujoco: MuJoCo-specific configuration.
        perception: Perception alias mappings.

    Example:
        metadata: ObjectMetadata = {
            "name": "cup",
            "category": ["kitchenware", "drinkware"],
            "mass": 0.25,
            "color": [0.9, 0.9, 1.0, 1.0],
            "scale": 1.0,
        }
    """

    name: NotRequired[str]
    category: NotRequired[List[str]]
    mass: NotRequired[float]
    color: NotRequired[List[float]]
    scale: NotRequired[float]
    mujoco: NotRequired[Dict[str, Any]]
    perception: NotRequired[Dict[str, Dict[str, List[str]]]]


class SavedState(TypedDict):
    """
    Serialized simulation state from save_state().

    Attributes:
        schema_version: Version number for forward compatibility.
        qpos: Joint positions array.
        qvel: Joint velocities array.
        active_objects: Dict mapping instance names to active status.
    """

    schema_version: int
    qpos: List[float]
    qvel: List[float]
    active_objects: Dict[str, bool]


# Type aliases for common patterns
DetectionList = List[Detection]
ActiveObjects = Dict[str, bool]
ObjectInstances = Dict[str, Dict[str, Any]]
