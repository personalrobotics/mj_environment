import pytest
import numpy as np
import mujoco
from mj_environment.constants import HIDE_GRID_SPACING, POSITION_DIM
from mj_environment.environment import Environment

@pytest.fixture
def env():
    return Environment(
        base_scene_xml="data/scene.xml",
        objects_dir="data/objects",
        scene_config_yaml="data/scene_config.yaml",
        verbose=False,
    )

def test_activate_and_hide(env):
    obj_type = next(iter(env.registry.objects))
    name = env.registry.activate(obj_type, [0, 0, 0.5])
    assert name in env.registry.active_objects
    assert env.registry.active_objects[name] == True

    env.registry.hide(name)
    assert name in env.registry.active_objects
    assert env.registry.active_objects[name] == False

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
    actual_pos = registry.data.qpos[indices.qpos_adr:indices.qpos_adr + POSITION_DIM]
    assert np.allclose(actual_pos, expected_pos), (
        f"Expected hide position {expected_pos}, got {actual_pos}"
    )
