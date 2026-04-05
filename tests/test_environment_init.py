# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import pytest

from mj_environment.environment import Environment


@pytest.fixture
def env():
    return Environment(
        base_scene_xml="data/scene.xml",
        objects_dir="data/objects",
        scene_config_yaml="data/scene_config.yaml",
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


def test_status_include_inactive(env):
    """Test that status(include_inactive=True) includes hidden objects."""
    # By default, only active objects are returned
    status = env.status()
    assert len(status["active_objects"]) == 0

    # With include_inactive=True, all objects (including hidden) are returned
    status = env.status(include_inactive=True)
    total_objects = sum(info["total"] for info in status["object_types"].values())
    assert len(status["active_objects"]) == total_objects


# ==============================================================================
# Robot-Only Scene Tests (no objects)
# ==============================================================================


class TestRobotOnlyScene:
    """Tests for environments without object management."""

    def test_robot_only_environment_creates(self):
        """Environment can be created with only a base scene XML."""
        env = Environment(
            base_scene_xml="data/scene.xml",
            objects_dir=None,
            scene_config_yaml=None,
        )
        assert env.model is not None
        assert env.data is not None

    def test_robot_only_has_no_registry(self):
        """Robot-only environment has registry=None and asset_manager=None."""
        env = Environment(
            base_scene_xml="data/scene.xml",
            objects_dir=None,
            scene_config_yaml=None,
        )
        assert env.registry is None
        assert env.asset_manager is None
        assert env._has_objects is False

    def test_robot_only_status(self):
        """Robot-only environment returns minimal status."""
        env = Environment(
            base_scene_xml="data/scene.xml",
            objects_dir=None,
            scene_config_yaml=None,
        )
        status = env.status()
        assert "time" in status
        assert status["active_count"] == 0
        assert status["active_objects"] == {}
        assert status["object_types"] == {}

    def test_robot_only_fork(self):
        """Robot-only environment can be forked."""
        env = Environment(
            base_scene_xml="data/scene.xml",
            objects_dir=None,
            scene_config_yaml=None,
        )
        fork = env.fork()
        assert fork.model is not None
        assert fork.data is not None
        assert fork.registry is None
        assert fork.asset_manager is None

    def test_robot_only_fork_independent(self):
        """Forked robot-only environment has independent state."""
        env = Environment(
            base_scene_xml="data/scene.xml",
            objects_dir=None,
            scene_config_yaml=None,
        )
        fork = env.fork()

        # Modify fork state
        fork.data.time = 999.0

        # Original should be unchanged
        assert env.data.time != 999.0

    def test_robot_only_save_load_state(self, tmp_path):
        """Robot-only environment can save and load state."""
        env = Environment(
            base_scene_xml="data/scene.xml",
            objects_dir=None,
            scene_config_yaml=None,
        )
        # Save state (even if qpos is empty, it should work)
        state_file = str(tmp_path / "state.yaml")
        env.save_state(state_file)

        # Load state - should work without errors
        env.load_state(state_file)

        # Verify file was created and has expected structure
        import yaml

        with open(state_file) as f:
            state = yaml.safe_load(f)
        assert "schema_version" in state
        assert "qpos" in state
        assert "qvel" in state
        assert state["active_objects"] == {}
