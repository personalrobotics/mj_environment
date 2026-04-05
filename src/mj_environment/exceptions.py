# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""
Custom exceptions for mj_environment.

All exceptions inherit from MjEnvironmentError for easy catching:
    try:
        env.registry.activate("mug", [0, 0, 0.5])
    except MjEnvironmentError as e:
        print(e)  # Includes context and suggestions
"""

from difflib import get_close_matches
from typing import List, Optional


class MjEnvironmentError(Exception):
    """Base exception for all mj_environment errors."""

    pass


class ObjectTypeNotFoundError(MjEnvironmentError):
    """Raised when an object type is not registered in the scene config."""

    def __init__(self, obj_type: str, available: List[str]):
        self.obj_type = obj_type
        self.available = available
        similar = get_close_matches(obj_type, available, n=1, cutoff=0.6)

        msg = f"Object type '{obj_type}' not found in registry.\n"
        msg += f"  Available types: {sorted(available)}\n"
        if similar:
            msg += f"  Did you mean: '{similar[0]}'?\n"
        msg += f"  Hint: Check that '{obj_type}' is defined in scene_config.yaml."

        super().__init__(msg)


class ObjectNotFoundError(MjEnvironmentError):
    """Raised when an object instance is not found."""

    def __init__(self, name: str, available: List[str]):
        self.name = name
        self.available = available
        similar = get_close_matches(name, available, n=1, cutoff=0.6)

        msg = f"Object '{name}' not found in registry.\n"
        if available:
            shown = available[:10]
            msg += f"  Available objects: {shown}"
            if len(available) > 10:
                msg += f" (and {len(available) - 10} more)"
            msg += "\n"
        if similar:
            msg += f"  Did you mean: '{similar[0]}'?\n"
        msg += "  Hint: Object names follow the pattern '{type}_{index}' (e.g., 'cup_0')."

        super().__init__(msg)


class ObjectPoolExhaustedError(MjEnvironmentError):
    """Raised when all instances of an object type are already active."""

    def __init__(self, obj_type: str, total: int, active: List[str]):
        self.obj_type = obj_type
        self.total = total
        self.active = active

        msg = f"No available '{obj_type}' instances - all {total} are active.\n"
        msg += f"  Active instances: {active}\n"
        msg += "  Hint: Either hide() an existing instance or increase the count in scene_config.yaml:\n"
        msg += "    objects:\n"
        msg += f"      {obj_type}:\n"
        msg += f"        count: {total + 2}  # increase from {total}"

        super().__init__(msg)


class ConfigurationError(MjEnvironmentError):
    """Raised when configuration files are missing or invalid."""

    def __init__(self, message: str, path: Optional[str] = None, hint: Optional[str] = None):
        self.path = path
        self.hint = hint

        msg = message
        if path:
            msg += f"\n  Path: {path}"
        if hint:
            msg += f"\n  Hint: {hint}"

        super().__init__(msg)


class StateError(MjEnvironmentError):
    """Raised when loading or saving state fails."""

    def __init__(self, message: str, hint: Optional[str] = None):
        self.hint = hint

        msg = message
        if hint:
            msg += f"\n  Hint: {hint}"

        super().__init__(msg)
