# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

from .environment import Environment
from .exceptions import (
    ConfigurationError,
    MjEnvironmentError,
    ObjectNotFoundError,
    ObjectPoolExhaustedError,
    ObjectTypeNotFoundError,
    StateError,
)
from .object_registry import ObjectRegistry
from .tracker import BaseTracker, ObjectTracker

__all__ = [
    # Primary API
    "Environment",
    # Exception classes
    "MjEnvironmentError",  # Base exception for catching all errors
    "ObjectTypeNotFoundError",  # Unknown object type in registry
    "ObjectNotFoundError",  # Unknown object instance
    "ObjectPoolExhaustedError",  # All instances of a type are active
    "ConfigurationError",  # Config file issues
    "StateError",  # State loading/saving issues
    # Tracking
    "BaseTracker",  # ABC for custom tracker implementations
    "ObjectTracker",  # Nearest-neighbor detection-to-instance tracker
    # Advanced components
    "ObjectRegistry",
]
