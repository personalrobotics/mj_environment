import pytest
import numpy as np
import mujoco
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
