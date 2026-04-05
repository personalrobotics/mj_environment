# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import mujoco
import numpy as np
import pytest
import yaml

from mj_environment.environment import Environment
from mj_environment.exceptions import ConfigurationError
from mj_environment.object_registry import HIDE_GRID_SPACING


@pytest.fixture
def env():
    return Environment(
        base_scene_xml="data/scene.xml",
        objects_dir="data/objects",
        scene_config_yaml="data/scene_config.yaml",
    )


@pytest.fixture
def custom_names_env(tmp_path):
    """Environment with custom instance names."""
    config = {
        "objects": {
            "cup": {"names": ["cup_left", "cup_right"]},
            "plate": {"count": 1},
        }
    }
    config_path = str(tmp_path / "scene_config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    return Environment(
        base_scene_xml="data/scene.xml",
        objects_dir="data/objects",
        scene_config_yaml=config_path,
    )


def test_activate_and_hide(env):
    obj_type = next(iter(env.registry.objects))
    name = env.registry.activate(obj_type, [0, 0, 0.5])
    assert name in env.registry.active_objects
    assert env.registry.active_objects[name]

    env.registry.hide(name)
    assert name in env.registry.active_objects
    assert not env.registry.active_objects[name]


def test_update_positions(env):
    obj_type = next(iter(env.registry.objects))
    name = env.registry.activate(obj_type, [0, 0, 0.5])
    new_pos = [0.1, 0.2, 0.3]
    env.update([{"name": name, "pos": new_pos, "quat": [1, 0, 0, 0]}])
    # Note: env.update() already calls mj_forward internally
    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert np.allclose(env.data.xpos[body_id], new_pos, atol=1e-3)


def test_hidden_objects_have_no_collisions(env):
    """Test that hidden objects have collisions disabled (contype/conaffinity = 0)."""
    obj_type = next(iter(env.registry.objects))
    name = f"{obj_type}_0"

    # Get geom info for this object
    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
    geom_adr = env.model.body_geomadr[body_id]
    geom_num = env.model.body_geomnum[body_id]

    # Initially hidden - collisions should be disabled
    assert not env.registry.is_active(name)
    for i in range(geom_num):
        geom_id = geom_adr + i
        assert env.model.geom_contype[geom_id] == 0, "Hidden object should have contype=0"
        assert env.model.geom_conaffinity[geom_id] == 0, "Hidden object should have conaffinity=0"

    # Activate - collisions should be enabled
    env.registry.activate(obj_type, [0, 0, 0.5])
    for i in range(geom_num):
        geom_id = geom_adr + i
        # Check that collision settings are restored (at least one should be non-zero)
        original_contype, original_conaffinity = env.registry.geom_collision[geom_id]
        assert env.model.geom_contype[geom_id] == original_contype
        assert env.model.geom_conaffinity[geom_id] == original_conaffinity

    # Hide again - collisions should be disabled
    env.registry.hide(name)
    for i in range(geom_num):
        geom_id = geom_adr + i
        assert env.model.geom_contype[geom_id] == 0, "Hidden object should have contype=0"
        assert env.model.geom_conaffinity[geom_id] == 0, "Hidden object should have conaffinity=0"


def test_hidden_objects_at_unique_positions(env):
    """Hidden objects should each have a unique parking position on a grid."""
    registry = env.registry

    # Every instance should have a hide position assigned
    all_instances = []
    for obj_type in registry.objects:
        all_instances.extend(registry.objects[obj_type]["instances"])
    assert len(registry._hide_positions) == len(all_instances)

    # All hide positions should be unique
    positions = list(registry._hide_positions.values())
    for i, p1 in enumerate(positions):
        for j, p2 in enumerate(positions):
            if i != j:
                assert not np.allclose(p1, p2), (
                    f"Instances {list(registry._hide_positions.keys())[i]} and "
                    f"{list(registry._hide_positions.keys())[j]} share position {p1}"
                )

    # Minimum spacing should be >= HIDE_GRID_SPACING
    if len(positions) > 1:
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                assert dist >= HIDE_GRID_SPACING - 1e-9, (
                    f"Distance {dist:.3f} < {HIDE_GRID_SPACING} between hide positions"
                )


def test_hide_returns_object_to_its_grid_spot(env):
    """After activate then hide, object should return to its unique grid position."""
    registry = env.registry
    obj_type = next(iter(registry.objects))
    name = f"{obj_type}_0"
    expected_pos = registry._hide_positions[name]

    # Activate somewhere, then hide
    registry.activate(obj_type, [1.0, 2.0, 3.0])
    registry.hide(name)

    # Read back position from qpos
    indices = registry._index_cache.get_body_indices(name)
    actual_pos = registry.data.qpos[indices.qpos_adr : indices.qpos_adr + 3]
    assert np.allclose(actual_pos, expected_pos), f"Expected hide position {expected_pos}, got {actual_pos}"


# ---- Custom instance names (#23) ----


class TestCustomInstanceNames:
    """Tests for optional custom instance names in scene config."""

    def test_custom_names_registered(self, custom_names_env):
        """Custom names appear as instances in the registry."""
        instances = custom_names_env.registry.objects["cup"]["instances"]
        assert instances == ["cup_left", "cup_right"]

    def test_auto_names_still_work(self, custom_names_env):
        """Types using count still get auto-generated names."""
        instances = custom_names_env.registry.objects["plate"]["instances"]
        assert instances == ["plate_0"]

    def test_activate_returns_custom_name(self, custom_names_env):
        """activate() returns the custom name."""
        name = custom_names_env.registry.activate("cup", [0, 0, 0.5])
        assert name == "cup_left"

    def test_hide_custom_name(self, custom_names_env):
        """hide() works with custom names."""
        name = custom_names_env.registry.activate("cup", [0, 0, 0.5])
        custom_names_env.registry.hide(name)
        assert not custom_names_env.registry.is_active(name)

    def test_update_custom_name(self, custom_names_env):
        """update() works with custom names."""
        name = custom_names_env.registry.activate("cup", [0, 0, 0.5])
        custom_names_env.update([{"name": name, "pos": [0.1, 0.2, 0.3]}])
        body_id = mujoco.mj_name2id(custom_names_env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert np.allclose(custom_names_env.data.xpos[body_id], [0.1, 0.2, 0.3], atol=1e-3)

    def test_get_type(self, custom_names_env):
        """get_type() returns correct type for custom and auto names."""
        assert custom_names_env.registry.get_type("cup_left") == "cup"
        assert custom_names_env.registry.get_type("cup_right") == "cup"
        assert custom_names_env.registry.get_type("plate_0") == "plate"

    def test_get_type_unknown_raises(self, custom_names_env):
        """get_type() raises ObjectNotFoundError for unknown names."""
        from mj_environment.exceptions import ObjectNotFoundError

        with pytest.raises(ObjectNotFoundError):
            custom_names_env.registry.get_type("nonexistent")

    def test_parse_object_type_custom_name(self, custom_names_env):
        """_parse_object_type() resolves custom names."""
        assert custom_names_env.registry._parse_object_type("cup_left") == "cup"
        assert custom_names_env.registry._parse_object_type("cup_right") == "cup"

    def test_names_and_count_conflict(self, tmp_path):
        """Specifying both names and count raises ConfigurationError."""
        config = {
            "objects": {
                "cup": {"names": ["a", "b"], "count": 2},
            }
        }
        config_path = str(tmp_path / "scene_config.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        with pytest.raises(ConfigurationError, match="both 'names' and 'count'"):
            Environment(
                base_scene_xml="data/scene.xml",
                objects_dir="data/objects",
                scene_config_yaml=config_path,
            )

    def test_duplicate_names_rejected(self, tmp_path):
        """Duplicate names across types raise ConfigurationError."""
        config = {
            "objects": {
                "cup": {"names": ["shared_name"]},
                "plate": {"names": ["shared_name"]},
            }
        }
        config_path = str(tmp_path / "scene_config.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        with pytest.raises(ConfigurationError, match="Duplicate instance name"):
            Environment(
                base_scene_xml="data/scene.xml",
                objects_dir="data/objects",
                scene_config_yaml=config_path,
            )
