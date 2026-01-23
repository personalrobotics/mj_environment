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


def test_status_returns_expected_keys(env):
    """Test that status() returns a dict with expected keys."""
    status = env.status()
    assert "time" in status
    assert "active_count" in status
    assert "active_objects" in status
    assert "object_types" in status


def test_status_reflects_active_objects(env):
    """Test that status() correctly reflects active objects."""
    # Initially no active objects
    status = env.status()
    assert status["active_count"] == 0
    assert len(status["active_objects"]) == 0

    # Activate an object
    obj_type = next(iter(env.registry.objects))
    name = env.registry.activate(obj_type, [0.1, 0.2, 0.3])

    # Status should reflect the active object
    status = env.status()
    assert status["active_count"] == 1
    assert name in status["active_objects"]
    assert status["active_objects"][name]["active"] is True


def test_status_object_types_info(env):
    """Test that status() includes object type availability info."""
    status = env.status()

    for obj_type, info in status["object_types"].items():
        assert "total" in info
        assert "active" in info
        assert "available" in info
        assert info["total"] == info["active"] + info["available"]


def test_status_verbose_includes_all_objects(env):
    """Test that status(verbose=True) includes inactive objects."""
    # With verbose=False, only active objects are returned
    status = env.status(verbose=False)
    assert len(status["active_objects"]) == 0

    # With verbose=True, all objects (including inactive) are returned
    status = env.status(verbose=True)
    total_objects = sum(info["total"] for info in status["object_types"].values())
    assert len(status["active_objects"]) == total_objects
