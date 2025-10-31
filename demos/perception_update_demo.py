import mujoco
import mujoco.viewer
import numpy as np
import threading
import queue
import time
from mj_environment.environment import Environment


def perception_thread(update_queue: queue.Queue, object_names: list[str], z_center: float, interval: float = 1.0) -> None:
    """
    Simulates a perception system that periodically detects random objects and sends updates.
    """
    while True:
        detected = []
        for name in np.random.choice(object_names, size=2, replace=False):
            pos = [np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4), z_center]
            quat = [1, 0, 0, 0]
            detected.append({"name": name, "pos": pos, "quat": quat})
        update_queue.put(detected)
        time.sleep(interval)


def perception_update_demo():
    env = Environment("data/scene.xml", "data/objects/household.xml")
    model = env.sim.model
    data = env.sim.data

    table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    table_geom_id = model.body_geomadr[table_body_id]
    table_size = model.geom_size[table_geom_id]
    table_height = table_size[2]

    sample_name = next(iter(env.registry.objects))
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, sample_name)
    object_geom_id = model.body_geomadr[object_body_id]
    object_size = model.geom_size[object_geom_id]
    object_half_height = object_size[1]

    z_center = table_height + object_half_height
    update_queue = queue.Queue()

    # Start the perception thread
    object_names = list(env.registry.objects.keys())
    perception = threading.Thread(
        target=perception_thread,
        args=(update_queue, object_names, z_center),
        daemon=True
    )
    perception.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0]
        viewer.cam.azimuth = -45
        viewer.cam.elevation = -45
        viewer.cam.distance = 2.0

        while viewer.is_running():
            try:
                while not update_queue.empty():
                    perception_update = update_queue.get_nowait()
                    # persist=False simulates a perception system that only maintains objects in the scene if they are continuously re-detected.
                    # persist=True simulates a perception system that maintains objects in the scene even if they are not detected.
                    env.update(perception_update, persist=False)
            except queue.Empty:
                pass
            viewer.sync()
            time.sleep(0.1)


if __name__ == "__main__":
    perception_update_demo()