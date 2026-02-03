"""Quaternion utilities for MuJoCo."""
from typing import Union, Sequence
import numpy as np

from .constants import IDENTITY_QUATERNION, QUAT_NORM_EPSILON


def normalize_quaternion(
    quat: Union[Sequence[float], np.ndarray],
    warn_on_normalization: bool = False
) -> np.ndarray:
    """
    Normalize a quaternion to unit length.

    MuJoCo expects unit quaternions for proper physics simulation.
    This function ensures quaternions are valid, raising an error
    for invalid inputs rather than silently correcting them.

    Args:
        quat: Quaternion array [w, x, y, z]
        warn_on_normalization: If True, log warning when normalization needed
                               (reserved for future use, currently unused)

    Returns:
        Normalized quaternion as numpy array

    Raises:
        ValueError: If quaternion has near-zero magnitude (< 1e-10)

    Example:
        >>> normalize_quaternion([2, 0, 0, 0])
        array([1., 0., 0., 0.])

        >>> normalize_quaternion([0, 0, 0, 0])  # doctest: +SKIP
        ValueError: Cannot normalize near-zero quaternion [0. 0. 0. 0.]
    """
    q = np.array(quat, dtype=float)
    norm = np.linalg.norm(q)

    if norm < QUAT_NORM_EPSILON:
        raise ValueError(
            f"Cannot normalize near-zero quaternion {q}. "
            f"Magnitude {norm} is below threshold {QUAT_NORM_EPSILON}."
        )

    return q / norm
