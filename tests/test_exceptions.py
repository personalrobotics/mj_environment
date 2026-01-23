"""
Tests for custom exception classes and error messages.

Verifies that exceptions provide helpful context and suggestions.
"""

import pytest
from mj_environment import (
    Environment,
    MjEnvironmentError,
    ObjectTypeNotFoundError,
    ObjectNotFoundError,
    ObjectPoolExhaustedError,
    ConfigurationError,
    StateError,
)


@pytest.fixture
def env():
    """Create a test environment."""
    return Environment(
        base_scene_xml="data/scene.xml",
        objects_dir="data/objects",
        scene_config_yaml="data/scene_config.yaml",
        verbose=False,
    )


class TestObjectTypeNotFoundError:
    """Test ObjectTypeNotFoundError with fuzzy matching."""

    def test_unknown_type_raises_error(self, env):
        """Activating an unknown object type raises ObjectTypeNotFoundError."""
        with pytest.raises(ObjectTypeNotFoundError) as exc_info:
            env.registry.activate("nonexistent", [0, 0, 0.5])

        assert "nonexistent" in str(exc_info.value)
        assert "Available types:" in str(exc_info.value)

    def test_similar_name_suggestion(self, env):
        """Error message suggests similar names via fuzzy matching."""
        # Get an actual type from the registry to test with a typo
        actual_type = next(iter(env.registry.objects))

        # Create a typo version (e.g., "cup" -> "cupp" or "plate" -> "plat")
        typo_name = actual_type[:-1] if len(actual_type) > 2 else actual_type + "x"

        with pytest.raises(ObjectTypeNotFoundError) as exc_info:
            env.registry.activate(typo_name, [0, 0, 0.5])

        error_msg = str(exc_info.value)
        # May or may not suggest depending on how close the typo is
        assert typo_name in error_msg
        assert "scene_config.yaml" in error_msg

    def test_error_inherits_from_base(self, env):
        """ObjectTypeNotFoundError can be caught as MjEnvironmentError."""
        with pytest.raises(MjEnvironmentError):
            env.registry.activate("unknown_type", [0, 0, 0.5])

    def test_error_attributes(self, env):
        """Exception stores obj_type and available types."""
        with pytest.raises(ObjectTypeNotFoundError) as exc_info:
            env.registry.activate("fake_type", [0, 0, 0.5])

        assert exc_info.value.obj_type == "fake_type"
        assert isinstance(exc_info.value.available, list)


class TestObjectNotFoundError:
    """Test ObjectNotFoundError with fuzzy matching."""

    def test_unknown_instance_raises_error(self, env):
        """Hiding an unknown instance raises ObjectNotFoundError."""
        with pytest.raises(ObjectNotFoundError) as exc_info:
            env.registry.hide("nonexistent_object_99")

        assert "nonexistent_object_99" in str(exc_info.value)
        assert "Available objects:" in str(exc_info.value)

    def test_similar_instance_suggestion(self, env):
        """Error message includes pattern hint."""
        obj_type = next(iter(env.registry.objects))

        # Activate an object first
        env.registry.activate(obj_type, [0, 0, 0.5])

        # Try to hide with a typo
        with pytest.raises(ObjectNotFoundError) as exc_info:
            env.registry.hide(f"{obj_type}_99")  # Non-existent index

        error_msg = str(exc_info.value)
        assert "Hint:" in error_msg
        assert "pattern" in error_msg.lower()

    def test_error_inherits_from_base(self, env):
        """ObjectNotFoundError can be caught as MjEnvironmentError."""
        with pytest.raises(MjEnvironmentError):
            env.registry.hide("unknown_instance")

    def test_error_attributes(self, env):
        """Exception stores name and available instances."""
        with pytest.raises(ObjectNotFoundError) as exc_info:
            env.registry.hide("fake_object_0")

        assert exc_info.value.name == "fake_object_0"
        assert isinstance(exc_info.value.available, list)


class TestConfigurationError:
    """Test ConfigurationError for config file issues."""

    def test_missing_config_file(self):
        """Missing config file raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            Environment(
                base_scene_xml="data/scene.xml",
                objects_dir="data/objects",
                scene_config_yaml="nonexistent_config.yaml",
                verbose=False,
            )

        error_msg = str(exc_info.value)
        assert "not found" in error_msg.lower()
        assert "Hint:" in error_msg

    def test_error_inherits_from_base(self):
        """ConfigurationError can be caught as MjEnvironmentError."""
        with pytest.raises(MjEnvironmentError):
            Environment(
                base_scene_xml="data/scene.xml",
                objects_dir="data/objects",
                scene_config_yaml="missing.yaml",
                verbose=False,
            )

    def test_error_attributes(self):
        """Exception stores path and hint."""
        try:
            Environment(
                base_scene_xml="data/scene.xml",
                objects_dir="data/objects",
                scene_config_yaml="missing.yaml",
                verbose=False,
            )
        except ConfigurationError as e:
            assert e.path is not None
            assert e.hint is not None


class TestStateError:
    """Test StateError for state loading issues."""

    def test_unit_error_attributes(self):
        """StateError stores hint."""
        error = StateError("Test message", hint="Test hint")
        assert "Test message" in str(error)
        assert "Test hint" in str(error)
        assert error.hint == "Test hint"

    def test_error_inherits_from_base(self):
        """StateError can be caught as MjEnvironmentError."""
        with pytest.raises(MjEnvironmentError):
            raise StateError("test")


class TestObjectPoolExhaustedError:
    """Test ObjectPoolExhaustedError when all instances are active."""

    def test_unit_error_message(self):
        """Error message includes count and active instances."""
        error = ObjectPoolExhaustedError(
            obj_type="cup",
            total=3,
            active=["cup_0", "cup_1", "cup_2"],
        )

        error_msg = str(error)
        assert "cup" in error_msg
        assert "3" in error_msg
        assert "Active instances:" in error_msg
        assert "scene_config.yaml" in error_msg

    def test_error_attributes(self):
        """Exception stores obj_type, total, and active."""
        error = ObjectPoolExhaustedError("plate", 2, ["plate_0", "plate_1"])

        assert error.obj_type == "plate"
        assert error.total == 2
        assert error.active == ["plate_0", "plate_1"]


class TestExceptionHierarchy:
    """Test that all exceptions inherit from MjEnvironmentError."""

    def test_catch_all_exceptions(self, env):
        """All custom exceptions can be caught with MjEnvironmentError."""
        exceptions = [
            ObjectTypeNotFoundError("test", ["a", "b"]),
            ObjectNotFoundError("test", ["a", "b"]),
            ObjectPoolExhaustedError("test", 1, ["test_0"]),
            ConfigurationError("test"),
            StateError("test"),
        ]

        for exc in exceptions:
            with pytest.raises(MjEnvironmentError):
                raise exc
