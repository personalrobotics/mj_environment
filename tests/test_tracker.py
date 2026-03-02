"""Tests for ObjectTracker detection-to-instance association."""

import numpy as np
import pytest
from mj_environment import Environment, ObjectTracker


@pytest.fixture
def env():
    return Environment(
        base_scene_xml="data/scene.xml",
        objects_dir="data/objects",
        scene_config_yaml="data/scene_config.yaml",
        verbose=False,
    )


@pytest.fixture
def tracker(env):
    return ObjectTracker(env.registry, max_distance=0.15)


class TestNewObjectAssignment:
    """First-time detections get assigned to available instances."""

    def test_single_detection(self, env, tracker):
        obj_type = next(iter(env.registry.objects))
        updates = tracker.associate([{"type": obj_type, "pos": [0.1, 0.2, 0.4]}])

        assert len(updates) == 1
        assert updates[0]["name"].startswith(f"{obj_type}_")
        assert updates[0]["pos"] == [0.1, 0.2, 0.4]
        assert updates[0]["quat"] == [1, 0, 0, 0]

    def test_multiple_types(self, env, tracker):
        types = list(env.registry.objects.keys())[:2]
        detections = [
            {"type": types[0], "pos": [0.1, 0.0, 0.4]},
            {"type": types[1], "pos": [-0.1, 0.0, 0.4]},
        ]
        updates = tracker.associate(detections)

        assert len(updates) == 2
        names = {u["name"] for u in updates}
        assert any(n.startswith(f"{types[0]}_") for n in names)
        assert any(n.startswith(f"{types[1]}_") for n in names)

    def test_custom_quat_preserved(self, env, tracker):
        obj_type = next(iter(env.registry.objects))
        quat = [0.707, 0.707, 0.0, 0.0]
        updates = tracker.associate([
            {"type": obj_type, "pos": [0.0, 0.0, 0.4], "quat": quat},
        ])
        assert updates[0]["quat"] == quat


class TestStableAssignment:
    """Same object in ~same position keeps the same instance name."""

    def test_same_position_same_name(self, env, tracker):
        obj_type = next(iter(env.registry.objects))
        det = [{"type": obj_type, "pos": [0.1, 0.2, 0.4]}]

        updates1 = tracker.associate(det)
        # Simulate env.update() making the object active
        env.update(updates1, hide_unlisted=True)

        updates2 = tracker.associate(det)
        assert updates1[0]["name"] == updates2[0]["name"]

    def test_small_movement_same_name(self, env, tracker):
        obj_type = next(iter(env.registry.objects))

        updates1 = tracker.associate([{"type": obj_type, "pos": [0.1, 0.2, 0.4]}])
        env.update(updates1, hide_unlisted=True)

        # Move within max_distance (0.15)
        updates2 = tracker.associate([{"type": obj_type, "pos": [0.12, 0.22, 0.4]}])
        assert updates1[0]["name"] == updates2[0]["name"]


class TestTypeAwareMatching:
    """Detections only match instances of the same type."""

    def test_different_type_no_match(self, env, tracker):
        types = list(env.registry.objects.keys())[:2]

        # Track a plate at (0.1, 0.2, 0.4)
        updates1 = tracker.associate([{"type": types[0], "pos": [0.1, 0.2, 0.4]}])
        env.update(updates1, hide_unlisted=True)

        # Detect a cup at the same position — should NOT match the plate
        updates2 = tracker.associate([{"type": types[1], "pos": [0.1, 0.2, 0.4]}])
        assert updates2[0]["name"] != updates1[0]["name"]
        assert updates2[0]["name"].startswith(f"{types[1]}_")


class TestMaxDistanceThreshold:
    """Detections beyond max_distance create new assignments."""

    def test_far_detection_gets_new_instance(self, env, tracker):
        obj_type = next(iter(env.registry.objects))

        updates1 = tracker.associate([{"type": obj_type, "pos": [0.0, 0.0, 0.4]}])
        env.update(updates1, hide_unlisted=True)

        # Move well beyond max_distance (0.15)
        updates2 = tracker.associate([{"type": obj_type, "pos": [1.0, 1.0, 0.4]}])

        # Old instance was dropped, new one assigned
        assert len(updates2) == 1
        assert updates2[0]["name"] != updates1[0]["name"]


class TestPoolExhaustion:
    """Excess detections are silently skipped when pool is full."""

    def test_more_detections_than_instances(self, env, tracker):
        obj_type = next(iter(env.registry.objects))
        pool_size = env.registry.objects[obj_type]["count"]

        # Detect more objects than the pool can hold
        detections = [
            {"type": obj_type, "pos": [i * 0.3, 0.0, 0.4]}
            for i in range(pool_size + 2)
        ]
        updates = tracker.associate(detections)

        # Should cap at pool_size
        assert len(updates) == pool_size


class TestUnknownType:
    """Detections with unregistered types are skipped."""

    def test_unknown_type_skipped(self, env, tracker):
        updates = tracker.associate([
            {"type": "nonexistent_object", "pos": [0.0, 0.0, 0.4]},
        ])
        assert len(updates) == 0

    def test_missing_type_key_skipped(self, env, tracker):
        updates = tracker.associate([{"pos": [0.0, 0.0, 0.4]}])
        assert len(updates) == 0


class TestReset:
    """reset() clears all tracking state."""

    def test_reset_clears_assignments(self, env, tracker):
        obj_type = next(iter(env.registry.objects))

        updates1 = tracker.associate([{"type": obj_type, "pos": [0.1, 0.2, 0.4]}])
        env.update(updates1, hide_unlisted=True)

        tracker.reset()

        # After reset, same position may get a different instance
        # (the original is still active in the registry, so tracker picks the next one)
        updates2 = tracker.associate([{"type": obj_type, "pos": [0.1, 0.2, 0.4]}])
        assert len(updates2) == 1
        assert updates2[0]["name"] != updates1[0]["name"]


class TestIntegrationWithUpdate:
    """End-to-end: tracker + env.update() cycle."""

    def test_full_cycle(self, env, tracker):
        """Tracker output feeds directly into env.update()."""
        obj_type = next(iter(env.registry.objects))
        pos = [0.1, 0.2, 0.4]

        updates = tracker.associate([{"type": obj_type, "pos": pos}])
        env.update(updates, hide_unlisted=True)

        # Object should be active at the specified position
        name = updates[0]["name"]
        assert env.registry.is_active(name)

    def test_hide_unlisted_removes_lost_objects(self, env, tracker):
        """Objects not in latest detections get hidden."""
        types = list(env.registry.objects.keys())[:2]

        # Frame 1: detect both types
        updates1 = tracker.associate([
            {"type": types[0], "pos": [0.1, 0.0, 0.4]},
            {"type": types[1], "pos": [-0.1, 0.0, 0.4]},
        ])
        env.update(updates1, hide_unlisted=True)
        assert env.registry.is_active(updates1[0]["name"])
        assert env.registry.is_active(updates1[1]["name"])

        # Frame 2: only detect first type
        updates2 = tracker.associate([
            {"type": types[0], "pos": [0.1, 0.0, 0.4]},
        ])
        env.update(updates2, hide_unlisted=True)

        # First object still active, second hidden
        assert env.registry.is_active(updates1[0]["name"])
        assert not env.registry.is_active(updates1[1]["name"])
