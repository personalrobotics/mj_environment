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

def test_clone_and_restore(env):
    obj_type = next(iter(env.registry.objects))
    name = env.registry.activate(obj_type, [0, 0, 0.5])
    mujoco.mj_forward(env.model, env.data)  # Ensure xpos is updated

    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
    pos_before = env.data.xpos[body_id].copy()
    cloned = env.clone_data()

    env.update([{"name": name, "pos": [0.3, 0.3, 0.5], "quat": [1, 0, 0, 0]}])
    pos_after_move = env.data.xpos[body_id].copy()
    assert not np.allclose(pos_before, pos_after_move)

    env.update_from_clone(cloned)
    pos_after_restore = env.data.xpos[body_id].copy()
    assert np.allclose(pos_before, pos_after_restore, atol=1e-6)
