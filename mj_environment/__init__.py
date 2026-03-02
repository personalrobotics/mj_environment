from .environment import Environment
from .object_registry import ObjectRegistry
from .state_io import StateIO
from .tracker import BaseTracker, ObjectTracker
from .exceptions import (
    MjEnvironmentError,
    ObjectTypeNotFoundError,
    ObjectNotFoundError,
    ObjectPoolExhaustedError,
    ConfigurationError,
    StateError,
)
from .types import (
    Detection,
    DetectionList,
    ObjectMetadata,
    SceneConfig,
    SavedState,
    Position,
    Quaternion,
)

__all__ = [
    # Primary API
    'Environment',
    # Exception classes
    'MjEnvironmentError',       # Base exception for catching all errors
    'ObjectTypeNotFoundError',  # Unknown object type in registry
    'ObjectNotFoundError',      # Unknown object instance
    'ObjectPoolExhaustedError', # All instances of a type are active
    'ConfigurationError',       # Config file issues
    'StateError',               # State loading/saving issues
    # Type definitions
    'Detection',                # TypedDict for object detections
    'DetectionList',            # List[Detection] alias
    'ObjectMetadata',           # TypedDict for meta.yaml structure
    'SceneConfig',              # TypedDict for scene_config.yaml
    'SavedState',               # TypedDict for saved state files
    'Position',                 # Union[List[float], ndarray]
    'Quaternion',               # Union[List[float], ndarray]
    # Tracking
    'BaseTracker',              # ABC for custom tracker implementations
    'ObjectTracker',            # Nearest-neighbor detection-to-instance tracker
    # Advanced components
    'ObjectRegistry',
    'StateIO',
] 