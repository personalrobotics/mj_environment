import mujoco
import mujoco.viewer
import numpy as np
import time
from mj_environment.environment import Environment


def object_update_demo():
    """Demonstrate dynamic object updates in the refactored Environment."""
    env = Environment("data/scene.xml", "data/objects/household.xml")
    model = env.sim.model
    data = env.sim.data

    # Get table height
    table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    table_geom_id = model.body_geomadr[table_body_id]
    table_height = model.geom_size[table_geom_id, 2]

    # Use first object to estimate its half height
    sample_name = next(iter(env.registry.objects))
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, sample_name)
    object_geom_id = model.body_geomadr[object_body_id]
    object_half_height = model.geom_size[object_geom_id, 1]

    z_center = table_height + object_half_height
    update_interval = 0.5  # seconds

    # Initial objects
    object_list = [
        {"name": "cup", "pos": np.random.uniform(-0.4, 0.4, 3).tolist(), "quat": [1, 0, 0, 0]},
        {"name": "plate", "pos": np.random.uniform(-0.4, 0.4, 3).tolist(), "quat": [1, 0, 0, 0]},
    ]
    for obj in object_list:
        obj["pos"][2] = z_center

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0]
        viewer.cam.azimuth = -45
        viewer.cam.elevation = -45
        viewer.cam.distance = 2.0

        while viewer.is_running():
            env.update(object_list)

            fixed_name = "cup"
            other_objects = [n for n in env.registry.objects if n != fixed_name]
            random_name = np.random.choice(other_objects)

            object_list[:] = [
                {"name": fixed_name, "pos": [np.random.uniform(-0.4, 0.4),
                                             np.random.uniform(-0.4, 0.4),
                                             z_center], "quat": [1, 0, 0, 0]},
                {"name": random_name, "pos": [np.random.uniform(-0.4, 0.4),
                                              np.random.uniform(-0.4, 0.4),
                                              z_center], "quat": [1, 0, 0, 0]},
            ]

            viewer.sync()
            time.sleep(update_interval)


def cloning_demo():
    """Demonstrate cloning and independent state updates."""
    env = Environment("data/scene.xml", "data/objects/household.xml")
    model = env.sim.model
    data = env.sim.data

    object_list = [
        {"name": "cup", "pos": [0.1, 0.1, 0.41], "quat": [1, 0, 0, 0]},
        {"name": "plate", "pos": [-0.1, -0.1, 0.41], "quat": [1, 0, 0, 0]},
    ]
    env.update(object_list)

    # Clone environment state
    cloned_data = env.sim.clone_data()

    # Move one object in original
    env.registry.move("cup", [0.3, 0.3, 0.41], [1, 0, 0, 0])

    print("Original qpos (moved cup):", env.sim.data.qpos[:7])
    print("Cloned qpos (before move):", cloned_data.qpos[:7])

    # Save and reload via YAML
    env.state_io.save(env.sim.model, env.sim.data, env.registry.active_objects, "state.yaml")
    print("Saved state to state.yaml")

    active_after_reload = env.state_io.load(env.sim.model, env.sim.data, "state.yaml")
    print("Reloaded state with active objects:", active_after_reload)


if __name__ == "__main__":
    object_update_demo()
    cloning_demo()