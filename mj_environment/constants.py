"""Constants used throughout mj_environment."""
import numpy as np

# Quaternion
IDENTITY_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
QUAT_NORM_EPSILON = 1e-10

# State array dimensions
POSITION_DIM = 3
QUATERNION_DIM = 4
DOF_DIM = 6
RGBA_ALPHA_CHANNEL = 3

# Schema
STATE_IO_SCHEMA_VERSION = 1
