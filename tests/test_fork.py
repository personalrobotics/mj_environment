"""
Tests for Environment.fork() API.

Tests cover:
- Basic fork creation and independence
- Multiple forks (fork(n=N))
- Context manager support
- Full functionality of forked environments
- Parallel execution scenarios
"""

import pytest
import numpy as np
import mujoco
from concurrent.futures import ThreadPoolExecutor
from mj_environment import Environment


@pytest.fixture
def env():
    """Create a test environment."""
    return Environment(
        base_scene_xml="data/scene.xml",
        objects_dir="data/objects",
        scene_config_yaml="data/scene_config.yaml",
        verbose=False,
    )


class TestForkBasics:
    """Test basic fork creation and properties."""

    def test_fork_returns_environment(self, env):
        """fork() returns an Environment instance."""
        fork = env.fork()
        assert isinstance(fork, Environment)

    def test_fork_has_independent_data(self, env):
        """Forked environment has its own MjData instance."""
        fork = env.fork()
        assert fork.data is not env.data
        assert fork.sim.data is not env.sim.data

    def test_fork_shares_model(self, env):
        """Forked environment shares the immutable MjModel."""
        fork = env.fork()
        assert fork.model is env.model
        assert fork.sim.model is env.sim.model

    def test_fork_has_independent_registry(self, env):
        """Forked environment has its own ObjectRegistry."""
        fork = env.fork()
        assert fork.registry is not env.registry
        assert fork.registry.data is fork.data

    def test_fork_shares_asset_manager(self, env):
        """Forked environment shares the read-only AssetManager."""
        fork = env.fork()
        assert fork.asset_manager is env.asset_manager


class TestForkIndependence:
    """Test that forks are truly independent from the original."""

    def test_fork_modification_does_not_affect_original(self, env):
        """Modifying a fork does not change the original environment."""
        # Activate an object in the original
        obj_type = next(iter(env.registry.objects))
        original_name = env.registry.activate(obj_type, [0, 0, 0.5])
        mujoco.mj_forward(env.model, env.data)

        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, original_name)
        original_pos = env.data.xpos[body_id].copy()

        # Fork and modify
        fork = env.fork()
        fork.update([{"name": original_name, "pos": [0.5, 0.5, 0.5], "quat": [1, 0, 0, 0]}])

        # Original should be unchanged
        assert np.allclose(env.data.xpos[body_id], original_pos)

    def test_original_modification_does_not_affect_fork(self, env):
        """Modifying the original does not change a fork."""
        # Activate an object
        obj_type = next(iter(env.registry.objects))
        name = env.registry.activate(obj_type, [0, 0, 0.5])
        mujoco.mj_forward(env.model, env.data)

        # Fork
        fork = env.fork()
        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        fork_pos = fork.data.xpos[body_id].copy()

        # Modify original
        env.update([{"name": name, "pos": [0.8, 0.8, 0.8], "quat": [1, 0, 0, 0]}])

        # Fork should be unchanged
        assert np.allclose(fork.data.xpos[body_id], fork_pos)

    def test_fork_registry_state_independent(self, env):
        """Fork's registry active_objects is independent."""
        obj_type = next(iter(env.registry.objects))
        env.registry.activate(obj_type, [0, 0, 0.5])

        fork = env.fork()

        # Hide in fork
        instances = env.registry.objects[obj_type]["instances"]
        active_instance = [n for n in instances if fork.registry.active_objects[n]][0]
        fork.registry.hide(active_instance)

        # Original should still have it active
        assert env.registry.active_objects[active_instance] is True
        assert fork.registry.active_objects[active_instance] is False


class TestForkMultiple:
    """Test creating multiple forks."""

    def test_fork_n_returns_list(self, env):
        """fork(n=N) returns a list of N environments."""
        forks = env.fork(n=3)
        assert isinstance(forks, list)
        assert len(forks) == 3
        assert all(isinstance(f, Environment) for f in forks)

    def test_multiple_forks_are_independent(self, env):
        """Multiple forks are independent from each other."""
        forks = env.fork(n=3)

        # Activate object in first fork only
        obj_type = next(iter(env.registry.objects))
        forks[0].registry.activate(obj_type, [0, 0, 0.5])
        mujoco.mj_forward(forks[0].model, forks[0].data)

        # Other forks should not have it active
        instances = env.registry.objects[obj_type]["instances"]
        first_instance = instances[0]

        assert forks[0].registry.active_objects[first_instance] is True
        assert forks[1].registry.active_objects[first_instance] is False
        assert forks[2].registry.active_objects[first_instance] is False

    def test_fork_n_zero_returns_empty_list(self, env):
        """fork(n=0) returns an empty list."""
        forks = env.fork(n=0)
        assert forks == []


class TestForkContextManager:
    """Test context manager support for forks."""

    def test_fork_context_manager(self, env):
        """Fork can be used as a context manager."""
        obj_type = next(iter(env.registry.objects))

        with env.fork() as fork:
            assert isinstance(fork, Environment)
            fork.registry.activate(obj_type, [0, 0, 0.5])
            assert any(fork.registry.active_objects.values())

        # After context, fork is still valid (GC handles cleanup)
        # This test just verifies the context manager syntax works

    def test_original_env_context_manager(self, env):
        """Original environment also supports context manager."""
        with env as e:
            assert e is env


class TestForkFunctionality:
    """Test that forked environments have full functionality."""

    def test_fork_can_step_simulation(self, env):
        """Forked environment can step physics."""
        fork = env.fork()
        initial_time = fork.data.time

        for _ in range(10):
            fork.sim.step()

        assert fork.data.time > initial_time
        # Original should not have stepped
        assert env.data.time == initial_time

    def test_fork_can_update_objects(self, env):
        """Forked environment can update objects."""
        fork = env.fork()
        obj_type = next(iter(fork.registry.objects))

        fork.update([
            {"name": f"{obj_type}_0", "pos": [0.1, 0.2, 0.3], "quat": [1, 0, 0, 0]}
        ])

        body_id = mujoco.mj_name2id(fork.model, mujoco.mjtObj.mjOBJ_BODY, f"{obj_type}_0")
        assert np.allclose(fork.data.xpos[body_id], [0.1, 0.2, 0.3], atol=0.01)

    def test_fork_can_reset(self, env):
        """Forked environment can reset."""
        fork = env.fork()

        # Step simulation
        for _ in range(10):
            fork.sim.step()

        fork.reset()
        assert fork.data.time == 0.0

    def test_fork_get_object_metadata(self, env):
        """Forked environment can get object metadata."""
        fork = env.fork()
        obj_type = next(iter(fork.registry.objects))
        instance_name = f"{obj_type}_0"

        meta = fork.get_object_metadata(instance_name)
        assert isinstance(meta, dict)


class TestForkParallelExecution:
    """Test parallel execution with forks."""

    def test_parallel_planning_simulation(self, env):
        """Simulate parallel planning with multiple forks."""
        # Activate an object in the original
        obj_type = next(iter(env.registry.objects))
        env.registry.activate(obj_type, [0, 0, 0.5])
        mujoco.mj_forward(env.model, env.data)

        # Create forks for parallel "planning"
        forks = env.fork(n=4)

        def simulate_planner(fork, seed):
            """Simulate a planner stepping the simulation."""
            rng = np.random.default_rng(seed)
            for _ in range(50):
                fork.sim.step()
            return fork.data.time

        # Run planners in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(simulate_planner, fork, i)
                for i, fork in enumerate(forks)
            ]
            results = [f.result() for f in futures]

        # All forks should have advanced simulation time
        assert all(t > 0 for t in results)
        # Original should be unchanged
        assert env.data.time == 0.0

    def test_parallel_object_updates(self, env):
        """Test parallel object updates in different forks."""
        obj_type = next(iter(env.registry.objects))
        forks = env.fork(n=4)

        def update_fork(fork, idx):
            """Update object position in a fork."""
            pos = [0.1 * idx, 0.2 * idx, 0.5]
            fork.update([
                {"name": f"{obj_type}_0", "pos": pos, "quat": [1, 0, 0, 0]}
            ])
            body_id = mujoco.mj_name2id(fork.model, mujoco.mjtObj.mjOBJ_BODY, f"{obj_type}_0")
            return fork.data.xpos[body_id].copy()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(update_fork, fork, i)
                for i, fork in enumerate(forks)
            ]
            positions = [f.result() for f in futures]

        # Each fork should have different position
        for i, pos in enumerate(positions):
            expected = [0.1 * i, 0.2 * i, 0.5]
            assert np.allclose(pos, expected, atol=0.01)


class TestSyncFrom:
    """Test syncing state from a fork back to the original."""

    def test_sync_from_updates_physics_state(self, env):
        """sync_from copies MjData state."""
        obj_type = next(iter(env.registry.objects))
        name = env.registry.activate(obj_type, [0, 0, 0.5])
        mujoco.mj_forward(env.model, env.data)

        # Fork and modify position
        fork = env.fork()
        fork.update([{"name": name, "pos": [0.5, 0.5, 0.5], "quat": [1, 0, 0, 0]}])

        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        original_pos = env.data.xpos[body_id].copy()
        fork_pos = fork.data.xpos[body_id].copy()

        # Sync from fork
        env.sync_from(fork)

        # Original should now match fork
        assert np.allclose(env.data.xpos[body_id], fork_pos)
        assert not np.allclose(env.data.xpos[body_id], original_pos)

    def test_sync_from_updates_active_objects(self, env):
        """sync_from copies active_objects state."""
        obj_type = next(iter(env.registry.objects))
        instances = env.registry.objects[obj_type]["instances"]

        # Activate in original
        env.registry.activate(obj_type, [0, 0, 0.5])
        assert env.registry.active_objects[instances[0]] is True

        # Fork and hide
        fork = env.fork()
        fork.registry.hide(instances[0])
        assert fork.registry.active_objects[instances[0]] is False

        # Sync from fork
        env.sync_from(fork)

        # Original should now have it hidden
        assert env.registry.active_objects[instances[0]] is False

    def test_sync_from_updates_visibility(self, env):
        """sync_from updates geom visibility to match."""
        obj_type = next(iter(env.registry.objects))
        name = env.registry.activate(obj_type, [0, 0, 0.5])

        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        geom_id = env.model.body_geomadr[body_id]

        # Object should be visible
        assert env.model.geom_rgba[geom_id, 3] > 0

        # Fork and hide
        fork = env.fork()
        fork.registry.hide(name)

        # Sync from fork
        env.sync_from(fork)

        # Object should now be invisible in original
        assert env.model.geom_rgba[geom_id, 3] == 0


class TestForkEdgeCases:
    """Test edge cases and error handling."""

    def test_fork_of_fork(self, env):
        """A fork can be forked again."""
        fork1 = env.fork()
        fork2 = fork1.fork()

        assert fork2 is not fork1
        assert fork2.data is not fork1.data
        assert fork2.model is fork1.model  # Still shared

    def test_fork_preserves_active_objects(self, env):
        """Fork preserves which objects are currently active."""
        obj_type = next(iter(env.registry.objects))
        name = env.registry.activate(obj_type, [0, 0, 0.5])

        fork = env.fork()

        assert fork.registry.active_objects[name] is True

    def test_fork_preserves_object_positions(self, env):
        """Fork preserves current object positions."""
        obj_type = next(iter(env.registry.objects))
        name = env.registry.activate(obj_type, [0.3, 0.4, 0.5])
        mujoco.mj_forward(env.model, env.data)

        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        original_pos = env.data.xpos[body_id].copy()

        fork = env.fork()
        fork_pos = fork.data.xpos[body_id].copy()

        assert np.allclose(original_pos, fork_pos)
