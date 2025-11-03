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
    mujoco.mj_forward(env.model, env.data)
    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert np.allclose(env.data.xpos[body_id], new_pos, atol=1e-3)
