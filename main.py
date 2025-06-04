import mujoco
import mujoco.viewer
import numpy as np
import time
from environment import Environment

# Create environment
env = Environment("scene.xml", "objects/household.xml")
model = env.model
data = env.data
body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
              for i in range(model.nbody)]
print("Bodies in model:", body_names)

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
        mujoco.mj_forward(model, data)

        object_list = [
            {"name": "cup", "pos": [np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4), z_center], "quat": [1, 0, 0, 0]},
            {"name": "plate", "pos": [np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4), z_center], "quat": [1, 0, 0, 0]}
        ]

        viewer.sync()
        time.sleep(update_interval)
