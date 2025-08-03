import mujoco
import mujoco.viewer
import numpy as np
import time
from mj_environment import Environment


def object_update_demo():
    # Create environment
    env = Environment("data/scene.xml", "data/objects/household.xml")
    model = env.model
    data = env.data

    # Get table and object heights
    table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    table_geom_id = model.body_geomadr[table_body_id]  # Get the first geometry of the table body
    table_size = model.geom_size[table_geom_id]
    table_height = table_size[2]

    # Use one object to determine height
    sample_name = next(iter(env.objects))
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, sample_name)
    object_geom_id = model.body_geomadr[object_body_id]  # Get the first geometry of the object body
    object_size = model.geom_size[object_geom_id]
    object_half_height = object_size[1]

    z_center = table_height + object_half_height

    update_interval = 0.5  # seconds

    # Initial object list
    object_list = [
        {"name": "cup", "pos": [np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4), z_center], "quat": [1, 0, 0, 0]},
        {"name": "plate", "pos": [np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4), z_center], "quat": [1, 0, 0, 0]}
    ]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0]
        viewer.cam.azimuth = -45
        viewer.cam.elevation = -45
        viewer.cam.distance = 2.0

        while viewer.is_running():
            env.update(object_list)

            fixed_name = "cup"
            other_objects = [name for name in env.objects if name != fixed_name]
            random_name = np.random.choice(other_objects)

            object_list[:] = [
                {"name": fixed_name, "pos": [np.random.uniform(-0.4, 0.4),
                                             np.random.uniform(-0.4, 0.4),
                                             z_center], "quat": [1, 0, 0, 0]},
                {"name": random_name, "pos": [np.random.uniform(-0.4, 0.4),
                                              np.random.uniform(-0.4, 0.4),
                                              z_center], "quat": [1, 0, 0, 0]}
            ]

            viewer.sync()
            time.sleep(update_interval)

def cloning_demo():
    # 1. Initialize environment
    env = Environment("data/scene.xml", "data/objects/household.xml")
    model = env.model
    data = env.data

    # 2. Place two objects
    object_list = [
        {"name": "cup", "pos": [0.1, 0.1, 0.41], "quat": [1, 0, 0, 0]},
        {"name": "plate", "pos": [-0.1, -0.1, 0.41], "quat": [1, 0, 0, 0]}
    ]
    env.update(object_list)

    # 3. Clone the state
    cloned_data = env.clone()

    # 4. Move 'cup' in the original
    env.move_object("cup", [0.3, 0.3, 0.41], [1, 0, 0, 0])

    # 5. Print states for comparison
    print("Original qpos (moved cup):", env.data.qpos[:7].copy())
    print("Cloned qpos (original cup):", cloned_data.qpos[:7].copy())

if __name__ == "__main__":
    object_update_demo()
    cloning_demo()
