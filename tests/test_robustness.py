"""
Tests for robustness improvements (Issue #9).

Tests cover:
1. Quaternion normalization
2. Path handling (absolute paths)
"""

import pytest
import numpy as np
import mujoco
import os
from mj_environment.environment import Environment
from mj_environment.object_registry import _normalize_quaternion as normalize_quaternion


@pytest.fixture
def env():
    return Environment(
        base_scene_xml="data/scene.xml",
        objects_dir="data/objects",
        scene_config_yaml="data/scene_config.yaml",
    )


class TestQuaternionNormalization:
    """Tests for quaternion normalization."""

    def test_normalize_unit_quaternion(self):
        """Test that unit quaternions are unchanged."""
        quat = np.array([1, 0, 0, 0])
        result = normalize_quaternion(quat)
        assert np.allclose(result, quat)

    def test_normalize_non_unit_quaternion(self):
        """Test that non-unit quaternions are normalized."""
        quat = np.array([2, 0, 0, 0])  # Not unit length
        result = normalize_quaternion(quat)
        assert np.allclose(np.linalg.norm(result), 1.0)
        assert np.allclose(result, [1, 0, 0, 0])

    def test_normalize_arbitrary_quaternion(self):
        """Test normalization of arbitrary quaternion."""
        quat = np.array([1, 1, 1, 1])  # Norm = 2
        result = normalize_quaternion(quat)
        assert np.allclose(np.linalg.norm(result), 1.0)
        expected = np.array([0.5, 0.5, 0.5, 0.5])
        assert np.allclose(result, expected)

    def test_normalize_zero_quaternion(self):
        """Test that zero quaternion raises ValueError."""
        quat = np.array([0, 0, 0, 0])
        with pytest.raises(ValueError, match="Cannot normalize near-zero quaternion"):
            normalize_quaternion(quat)

    def test_normalize_near_zero_quaternion(self):
        """Test that near-zero quaternion raises ValueError."""
        quat = np.array([1e-15, 1e-15, 1e-15, 1e-15])
        with pytest.raises(ValueError, match="Cannot normalize near-zero quaternion"):
            normalize_quaternion(quat)

    def test_activate_normalizes_quaternion(self, env):
        """Test that activate() normalizes quaternions."""
        obj_type = next(iter(env.registry.objects))

        # Activate with non-unit quaternion
        non_unit_quat = [2, 0, 0, 0]
        name = env.registry.activate(obj_type, [0.1, 0.1, 0.3], non_unit_quat)

        # Get the quaternion from qpos
        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        joint_adr = env.model.body_jntadr[body_id]
        qpos_adr = env.model.jnt_qposadr[joint_adr]
        stored_quat = env.data.qpos[qpos_adr+3:qpos_adr+7]

        # Should be normalized
        assert np.allclose(np.linalg.norm(stored_quat), 1.0)

    def test_update_normalizes_quaternion(self, env):
        """Test that update() normalizes quaternions."""
        obj_type = next(iter(env.registry.objects))
        name = env.registry.activate(obj_type, [0.1, 0.1, 0.3])

        # Update with non-unit quaternion
        non_unit_quat = [3, 0, 0, 0]
        env.update([{"name": name, "pos": [0.2, 0.2, 0.3], "quat": non_unit_quat}])

        # Get the quaternion from qpos
        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        joint_adr = env.model.body_jntadr[body_id]
        qpos_adr = env.model.jnt_qposadr[joint_adr]
        stored_quat = env.data.qpos[qpos_adr+3:qpos_adr+7]

        # Should be normalized
        assert np.allclose(np.linalg.norm(stored_quat), 1.0)


class TestPathHandling:
    """Tests for path handling robustness."""

    def test_environment_loads_with_relative_path(self):
        """Test that environment loads correctly with relative paths."""
        # This should work regardless of current working directory
        env = Environment(
            base_scene_xml="data/scene.xml",
            objects_dir="data/objects",
            scene_config_yaml="data/scene_config.yaml",
        )
        assert env.model is not None

    def test_environment_loads_with_absolute_path(self):
        """Test that environment loads correctly with absolute paths."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = Environment(
            base_scene_xml=os.path.join(base_dir, "data/scene.xml"),
            objects_dir=os.path.join(base_dir, "data/objects"),
            scene_config_yaml=os.path.join(base_dir, "data/scene_config.yaml"),
        )
        assert env.model is not None


class TestUpdateInputValidation:
    """Tests for input validation in update()."""

    def test_update_rejects_non_list(self, env):
        """Test that update() raises TypeError for non-list input."""
        with pytest.raises(TypeError, match="updates must be a list"):
            env.registry.update("not a list")

    def test_update_rejects_non_dict_items(self, env):
        """Test that update() raises TypeError for non-dict items."""
        with pytest.raises(TypeError, match=r"updates\[0\] must be a dict"):
            env.registry.update(["not a dict"])

    def test_update_rejects_missing_name(self, env):
        """Test that update() raises ValueError for missing 'name' key."""
        with pytest.raises(ValueError, match=r"updates\[0\] missing required key 'name'"):
            env.registry.update([{"pos": [0, 0, 0]}])

    def test_update_rejects_missing_pos(self, env):
        """Test that update() raises ValueError for missing 'pos' key."""
        obj_type = next(iter(env.registry.objects))
        name = f"{obj_type}_0"
        with pytest.raises(ValueError, match=r"missing required key 'pos'"):
            env.registry.update([{"name": name}])

    def test_update_rejects_wrong_pos_length(self, env):
        """Test that update() raises ValueError for wrong pos length."""
        obj_type = next(iter(env.registry.objects))
        name = f"{obj_type}_0"
        with pytest.raises(ValueError, match=r"'pos' must have 3 elements"):
            env.registry.update([{"name": name, "pos": [0, 0]}])

    def test_update_rejects_non_sequence_pos(self, env):
        """Test that update() raises ValueError for non-sequence pos."""
        obj_type = next(iter(env.registry.objects))
        name = f"{obj_type}_0"
        with pytest.raises(ValueError, match=r"'pos' must be a sequence"):
            env.registry.update([{"name": name, "pos": 123}])

    def test_update_accepts_valid_input(self, env):
        """Test that update() accepts valid input."""
        obj_type = next(iter(env.registry.objects))
        name = f"{obj_type}_0"
        # Should not raise
        env.registry.update([{"name": name, "pos": [0.1, 0.2, 0.3]}])
        assert env.registry.is_active(name)


class TestThreadSafetyDocumentation:
    """Tests to verify thread safety documentation exists."""

    def test_registry_has_thread_safety_docs(self):
        """Test that ObjectRegistry has thread safety documentation."""
        from mj_environment.object_registry import ObjectRegistry
        docstring = ObjectRegistry.__doc__
        assert "Thread Safety" in docstring
        assert "NOT thread-safe" in docstring
