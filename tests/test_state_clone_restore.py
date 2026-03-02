"""
Tests for state preservation via fork().

Previously tested clone_data/update_from_clone which are now replaced by fork().
"""

import pytest
import numpy as np
import mujoco
from mj_environment import Environment


@pytest.fixture
def env():
    return Environment(
        base_scene_xml="data/scene.xml",
        objects_dir="data/objects",
        scene_config_yaml="data/scene_config.yaml",
    )


def test_fork_preserves_original_state(env):
    """Test that fork() preserves original state when fork is modified."""
    obj_type = next(iter(env.registry.objects))
    name = env.registry.activate(obj_type, [0, 0, 0.5])
    mujoco.mj_forward(env.model, env.data)

    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
    pos_before = env.data.xpos[body_id].copy()

    # Fork and modify
    fork = env.fork()
    fork.update([{"name": name, "pos": [0.3, 0.3, 0.5], "quat": [1, 0, 0, 0]}])

    # Verify fork has new position
    fork_pos = fork.data.xpos[body_id].copy()
    assert not np.allclose(pos_before, fork_pos)

    # Verify original is unchanged
    original_pos = env.data.xpos[body_id].copy()
    assert np.allclose(pos_before, original_pos, atol=1e-6)
