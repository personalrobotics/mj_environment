import mujoco
import mujoco.viewer
import numpy as np
import time
from object_manager import ObjectManager, get_object_names_from_model

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

object_names = get_object_names_from_model(model, prefix="")
manager = ObjectManager(model, data, object_names)

# Get table and object heights dynamically
table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
table_size = model.geom_size[table_geom_id]
table_height = table_size[2]  # half-height of table

object_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, object_names[0])
object_size = model.geom_size[object_geom_id]
object_half_height = object_size[1]  # half-height of cylinder

z_center = table_height + object_half_height

update_interval = 0.5  # seconds between updates
last_update_time = time.time()

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
        current_time = time.time()
        if current_time - last_update_time >= update_interval:
            manager.update(object_list)
            object_list = [
                {"name": "cup", "pos": [np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4), z_center], "quat": [1, 0, 0, 0]},
                {"name": "plate", "pos": [np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4), z_center], "quat": [1, 0, 0, 0]}
            ]
            last_update_time = current_time

        mujoco.mj_step(model, data)
        viewer.sync()
