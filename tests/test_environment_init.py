import pytest
from mj_environment.environment import Environment

@pytest.fixture
def env():
    return Environment(
        base_scene_xml="data/scene.xml",
        objects_dir="data/objects",
        scene_config_yaml="data/scene_config.yaml",
        verbose=False,
    )

def test_environment_loads(env):
    assert env.model is not None
    assert env.data is not None
    assert len(env.registry.objects) > 0, "No objects loaded"
    assert env.asset_manager is not None
