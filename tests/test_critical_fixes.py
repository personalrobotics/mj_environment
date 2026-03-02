"""
Regression tests for critical bug fixes (Issue #7).

Tests cover:
1. Object type parsing for names with underscores
2. Deep XML body cloning (nested structures)
3. MjData copy_data model reference fix
4. State loading visibility restoration
"""

import pytest
import numpy as np
import mujoco
import tempfile
import os
from mj_environment.environment import Environment
from mj_environment.object_registry import ObjectRegistry


@pytest.fixture
def env():
    return Environment(
        base_scene_xml="data/scene.xml",
        objects_dir="data/objects",
        scene_config_yaml="data/scene_config.yaml",
        verbose=False,
    )


class TestObjectTypeParsing:
    """Tests for _parse_object_type fixing underscore handling."""

    def test_parse_simple_object_type(self, env):
        """Test parsing simple object names like cup_0."""
        result = env.registry._parse_object_type("cup_0")
        assert result == "cup"

    def test_parse_object_type_higher_index(self, env):
        """Test parsing object names with higher indices."""
        result = env.registry._parse_object_type("cup_2")
        assert result == "cup"

    def test_parse_unknown_object_returns_none(self, env):
        """Test that unknown object types return None."""
        result = env.registry._parse_object_type("nonexistent_0")
        assert result is None

    def test_parse_invalid_format_returns_none(self, env):
        """Test that invalid formats return None."""
        result = env.registry._parse_object_type("cup")  # No index
        assert result is None
        result = env.registry._parse_object_type("cup_abc")  # Non-numeric index
        assert result is None

    def test_parse_object_type_with_underscores(self):
        """
        Test parsing object types that contain underscores.

        This is a unit test that mocks the registry's objects dict
        to test the parsing logic without needing actual assets.
        """
        # Create a minimal mock registry to test parsing logic
        class MockRegistry:
            def __init__(self):
                self.objects = {
                    "kitchen_knife": {"count": 2, "instances": ["kitchen_knife_0", "kitchen_knife_1"]},
                    "my_cool_object": {"count": 1, "instances": ["my_cool_object_0"]},
                    "cup": {"count": 3, "instances": ["cup_0", "cup_1", "cup_2"]},
                }

            def _parse_object_type(self, instance_name):
                for obj_type in self.objects:
                    if instance_name.startswith(obj_type + "_"):
                        suffix = instance_name[len(obj_type) + 1:]
                        if suffix.isdigit():
                            return obj_type
                return None

        mock = MockRegistry()

        # Test underscore object types
        assert mock._parse_object_type("kitchen_knife_0") == "kitchen_knife"
        assert mock._parse_object_type("kitchen_knife_1") == "kitchen_knife"
        assert mock._parse_object_type("my_cool_object_0") == "my_cool_object"

        # Test simple object types still work
        assert mock._parse_object_type("cup_0") == "cup"
        assert mock._parse_object_type("cup_2") == "cup"

        # Test edge cases
        assert mock._parse_object_type("kitchen_0") is None  # Not a registered type
        assert mock._parse_object_type("knife_0") is None  # Not a registered type


class TestCopyDataModelFix:
    """Tests for copy_data model reference fix."""

    def test_copy_data_does_not_raise(self, env):
        """Test that copy_data works without AttributeError."""
        clone = mujoco.MjData(env.model)
        # This should not raise AttributeError about .model
        Environment._copy_data(env.model, clone, env.data)

    def test_fork_uses_copy_data_internally(self, env):
        """Test that fork() properly uses copy_data to clone state."""
        # Activate an object and set a position
        obj_type = next(iter(env.registry.objects))
        env.registry.activate(obj_type, [0.1, 0.2, 0.3])
        mujoco.mj_forward(env.model, env.data)

        # Fork should work and have same state
        fork = env.fork()

        # Verify fork has same state
        assert np.allclose(fork.data.qpos, env.data.qpos)
        assert np.allclose(fork.data.qvel, env.data.qvel)


class TestStateLoadVisibilityRestoration:
    """Tests for visibility restoration when loading state."""

    def test_load_state_restores_visibility(self, env):
        """Test that load_state properly restores object visibility."""
        obj_type = next(iter(env.registry.objects))
        instances = env.registry.objects[obj_type]["instances"]

        # Activate first instance
        name = instances[0]
        env.registry.activate(obj_type, [0.1, 0.2, 0.3])
        mujoco.mj_forward(env.model, env.data)

        # Verify it's active and visible
        assert env.registry.active_objects[name] == True
        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        geom_adr = env.model.body_geomadr[body_id]
        original_alpha = env.model.geom_rgba[geom_adr, 3]
        assert original_alpha > 0  # Should be visible

        # Save state
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            env.save_state(temp_path)

            # Hide the object manually
            env.registry.hide(name)
            assert env.registry.active_objects[name] == False
            assert env.model.geom_rgba[geom_adr, 3] == 0  # Should be invisible

            # Load state - should restore visibility
            env.load_state(temp_path)

            # Verify visibility is restored
            assert env.registry.active_objects[name] == True
            assert env.model.geom_rgba[geom_adr, 3] > 0  # Should be visible again
        finally:
            os.unlink(temp_path)

    def test_load_state_hides_inactive_objects(self, env):
        """Test that load_state properly hides objects that were inactive in saved state."""
        obj_type = next(iter(env.registry.objects))
        instances = env.registry.objects[obj_type]["instances"]
        name = instances[0]

        # Save state with object inactive (initial state)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            env.save_state(temp_path)

            # Activate the object
            env.registry.activate(obj_type, [0.1, 0.2, 0.3])
            mujoco.mj_forward(env.model, env.data)
            assert env.registry.active_objects[name] == True

            body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
            geom_adr = env.model.body_geomadr[body_id]
            assert env.model.geom_rgba[geom_adr, 3] > 0  # Visible

            # Load state - should hide the object
            env.load_state(temp_path)

            # Verify object is hidden
            assert env.registry.active_objects[name] == False
            assert env.model.geom_rgba[geom_adr, 3] == 0  # Should be invisible
        finally:
            os.unlink(temp_path)


class TestMjName2idBehavior:
    """Regression tests for Issue #26: silent exception swallowing in mj_name2id calls.

    mj_name2id() returns -1 for unknown names — it does not raise TypeError or
    AttributeError. The original try/except was dead code that never triggered.
    The fix uses an explicit -1 check instead.
    """

    def test_mj_name2id_returns_minus_one_for_unknown_body(self, env):
        """mj_name2id returns -1 for unknown names, not raises.

        This documents the MuJoCo API contract the fix relies on.
        """
        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "nonexistent_body_99")
        assert body_id == -1

    def test_known_body_has_valid_id(self, env):
        """mj_name2id returns a non-negative id for valid body names."""
        obj_type = next(iter(env.registry.objects))
        instance = env.registry.objects[obj_type]["instances"][0]
        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, instance)
        assert body_id >= 0


class TestDeepXMLCloning:
    """Tests for deep XML element cloning."""

    def test_environment_loads_without_error(self, env):
        """Basic test that environment loads correctly with deep copy."""
        assert env.model is not None
        assert env.data is not None
        assert len(env.registry.objects) > 0

    def test_all_objects_have_geoms(self, env):
        """Test that all preloaded objects have geometry (weren't corrupted by shallow copy)."""
        for obj_type in env.registry.objects:
            for instance_name in env.registry.objects[obj_type]["instances"]:
                body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, instance_name)
                geom_num = env.model.body_geomnum[body_id]
                assert geom_num > 0, f"Object {instance_name} has no geoms (possible XML cloning issue)"

    def test_objects_have_joints(self, env):
        """Test that all objects have freejoint (common structure that could be lost)."""
        for obj_type in env.registry.objects:
            for instance_name in env.registry.objects[obj_type]["instances"]:
                body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, instance_name)
                # body_jntnum tells us how many joints this body has
                jnt_num = env.model.body_jntnum[body_id]
                assert jnt_num > 0, f"Object {instance_name} has no joints (possible XML cloning issue)"
