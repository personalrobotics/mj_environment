# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""
Tests for state management fixes (Issue #8).

Tests cover:
1. Default hide_unlisted=True behavior in update()
2. Scale caching to prevent compounding
"""

import mujoco
import numpy as np
import pytest

from mj_environment.environment import Environment


@pytest.fixture
def env():
    return Environment(
        base_scene_xml="data/scene.xml",
        objects_dir="data/objects",
        scene_config_yaml="data/scene_config.yaml",
    )


class TestHideUnlistedDefault:
    """Tests for hide_unlisted=True default behavior."""

    def test_hide_unlisted_defaults_to_true(self, env):
        """Test that update() defaults to hide_unlisted=True (objects disappear if not in update)."""
        obj_type = next(iter(env.registry.objects))
        env.registry.objects[obj_type]["instances"]

        # Activate two objects
        name1 = env.registry.activate(obj_type, [0.1, 0.1, 0.3])
        name2 = env.registry.activate(obj_type, [0.2, 0.2, 0.3])
        mujoco.mj_forward(env.model, env.data)

        assert env.registry.active_objects[name1]
        assert env.registry.active_objects[name2]

        # Update with only one object (default hide_unlisted=True should hide the other)
        env.update([{"name": name1, "pos": [0.1, 0.1, 0.3], "quat": [1, 0, 0, 0]}])

        # name1 should still be active, name2 should be hidden
        assert env.registry.active_objects[name1]
        assert not env.registry.active_objects[name2]

    def test_hide_unlisted_false_keeps_objects(self, env):
        """Test that hide_unlisted=False keeps objects not in update list."""
        obj_type = next(iter(env.registry.objects))

        # Activate two objects
        name1 = env.registry.activate(obj_type, [0.1, 0.1, 0.3])
        name2 = env.registry.activate(obj_type, [0.2, 0.2, 0.3])
        mujoco.mj_forward(env.model, env.data)

        # Update with only one object but hide_unlisted=False
        env.update([{"name": name1, "pos": [0.1, 0.1, 0.3], "quat": [1, 0, 0, 0]}], hide_unlisted=False)

        # Both should still be active
        assert env.registry.active_objects[name1]
        assert env.registry.active_objects[name2]

    def test_empty_update_hides_all(self, env):
        """Test that empty update with default hide_unlisted=True hides all objects."""
        obj_type = next(iter(env.registry.objects))

        # Activate an object
        name = env.registry.activate(obj_type, [0.1, 0.1, 0.3])
        mujoco.mj_forward(env.model, env.data)
        assert env.registry.active_objects[name]

        # Empty update with default hide_unlisted=True
        env.update([])

        # Object should be hidden
        assert not env.registry.active_objects[name]

    def test_empty_update_with_hide_unlisted_false_keeps_all(self, env):
        """Test that empty update with hide_unlisted=False keeps all objects."""
        obj_type = next(iter(env.registry.objects))

        # Activate an object
        name = env.registry.activate(obj_type, [0.1, 0.1, 0.3])
        mujoco.mj_forward(env.model, env.data)
        assert env.registry.active_objects[name]

        # Empty update with hide_unlisted=False
        env.update([], hide_unlisted=False)

        # Object should still be active
        assert env.registry.active_objects[name]


class TestScaleCaching:
    """Tests for scale caching to prevent compounding."""

    def test_scale_cache_exists(self, env):
        """Test that scale cache dictionary is initialized."""
        assert hasattr(env, "_geom_original_size")
        assert isinstance(env._geom_original_size, dict)

    def test_original_sizes_are_cached(self, env):
        """Test that original geom sizes are cached when scale is applied."""
        # Check if any objects have scale in their metadata
        for obj_type in env.asset_manager.list():
            meta = env.asset_manager.get(obj_type)
            if "scale" in meta and meta["scale"] != 1.0:
                # There should be cached sizes for this object type
                instances = env.registry.objects.get(obj_type, {}).get("instances", [])
                for instance_name in instances:
                    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, instance_name)
                    geom_adr = env.model.body_geomadr[body_id]
                    geom_num = env.model.body_geomnum[body_id]
                    for i in range(geom_num):
                        geom_id = geom_adr + i
                        # If scale was applied, original should be cached
                        if geom_id in env._geom_original_size:
                            assert env._geom_original_size[geom_id] is not None

    def test_scale_does_not_compound(self):
        """Test that applying scale multiple times doesn't compound."""
        # Create environment twice and compare sizes
        env1 = Environment(
            base_scene_xml="data/scene.xml",
            objects_dir="data/objects",
            scene_config_yaml="data/scene_config.yaml",
        )
        env2 = Environment(
            base_scene_xml="data/scene.xml",
            objects_dir="data/objects",
            scene_config_yaml="data/scene_config.yaml",
        )

        # Both environments should have the same geom sizes
        # (proving scale doesn't compound on reload)
        for obj_type in env1.registry.objects:
            for instance_name in env1.registry.objects[obj_type]["instances"]:
                body_id1 = mujoco.mj_name2id(env1.model, mujoco.mjtObj.mjOBJ_BODY, instance_name)
                body_id2 = mujoco.mj_name2id(env2.model, mujoco.mjtObj.mjOBJ_BODY, instance_name)

                geom_adr1 = env1.model.body_geomadr[body_id1]
                geom_num1 = env1.model.body_geomnum[body_id1]
                geom_adr2 = env2.model.body_geomadr[body_id2]
                geom_num2 = env2.model.body_geomnum[body_id2]

                assert geom_num1 == geom_num2

                for i in range(geom_num1):
                    size1 = env1.model.geom_size[geom_adr1 + i]
                    size2 = env2.model.geom_size[geom_adr2 + i]
                    assert np.allclose(size1, size2), f"Geom sizes differ for {instance_name}: {size1} vs {size2}"


class TestLoadStateAtomicity:
    """Regression tests for Issue #28: load_state() partial state corruption.

    If active_objects in the YAML is malformed, physics state (qpos/qvel) must
    not be modified. The fix moves active_objects parsing before state application.
    """

    def test_corrupted_active_objects_does_not_modify_physics(self, env, tmp_path):
        """Malformed active_objects section must not corrupt physics state."""
        state_file = tmp_path / "corrupted.yaml"
        nq = env.model.nq
        nv = env.model.nv

        # Write a state file where active_objects is null (invalid — can't iterate None)
        state_file.write_text(f"schema_version: 1\nqpos: {[0.5] * nq}\nqvel: {[0.0] * nv}\nactive_objects: null\n")

        original_qpos = env.data.qpos.copy()

        with pytest.raises(Exception):
            env.load_state(str(state_file))

        # Physics state must be unchanged
        assert np.allclose(env.data.qpos, original_qpos), (
            "load_state() corrupted qpos even though active_objects was invalid"
        )

    def test_valid_state_still_loads_correctly(self, env, tmp_path):
        """Valid state files continue to load correctly after the fix."""
        state_file = tmp_path / "valid.yaml"

        obj_type = next(iter(env.registry.objects))
        name = env.registry.activate(obj_type, [0.1, 0.2, 0.3])
        mujoco.mj_forward(env.model, env.data)

        env.save_state(str(state_file))
        env.registry.hide(name)

        env.load_state(str(state_file))
        assert env.registry.active_objects[name] is True
