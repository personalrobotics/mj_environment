"""
Dynamic Kitchen Demo
====================
Showcases runtime object activation, motion, cloning, and state restoration
using the refactored mj_environment Environment.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from mj_environment.environment import Environment


def dynamic_kitchen_demo():
    """Demonstrate dynamic activation, movement, cloning, and restoration."""
    env = Environment(
        base_scene_xml="data/scene.xml",
        objects_dir="data/objects",
        scene_config_yaml="data/scene_config.yaml",
        verbose=True,
    )

    model = env.model
    data = env.data

    # ------------------------------------------------------------------
    # Setup camera
    # ------------------------------------------------------------------
    def setup_camera(viewer):
        viewer.cam.lookat[:] = [0, 0, 0]
        viewer.cam.azimuth = -45
        viewer.cam.elevation = -45
        viewer.cam.distance = 2.0

    # ------------------------------------------------------------------
    # Helper: get table height + object offset
    # ------------------------------------------------------------------
    table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    table_geom_id = model.body_geomadr[table_body_id]
    table_height = model.geom_size[table_geom_id, 2]
    sample_obj = next(iter(env.registry.objects))
    first_obj_name = env.registry.objects[sample_obj]["instances"][0]
    first_obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, first_obj_name)
    first_geom_id = model.body_geomadr[first_obj_body_id]
    obj_half_height = model.geom_size[first_geom_id, 1]
    z_center = table_height + obj_half_height

    # ------------------------------------------------------------------
    # Demo parameters
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    update_interval = 0.5  # seconds

    object_types = list(env.registry.objects.keys())
    active_names = []

    # ------------------------------------------------------------------
    # Launch viewer
    # ------------------------------------------------------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:
        setup_camera(viewer)

        step_count = 0
        cloned_state = None

        while viewer.is_running():
            # 1️⃣ Activate 2 random object types if none active
            if len(active_names) < 2:
                selected = rng.choice(object_types, size=2, replace=False)
                for obj_type in selected:
                    obj_name = env.registry.activate(obj_type, [0, 0, z_center])
                    if obj_name:
                        active_names.append(obj_name)

            # 2️⃣ Move all active objects to random new positions
            updates = []
            for name in active_names:
                updates.append({
                    "name": name,
                    "pos": [rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), z_center],
                    "quat": [1, 0, 0, 0],
                })
            env.update(updates, persist=True)

            # 3️⃣ Every 10 steps: clone state, move objects drastically, then restore
            if step_count > 0 and step_count % 10 == 0:
                print("\n[Clone Demo] Saving state...")
                cloned_state = env.clone_data()
                env.save_state("data/state_snapshot.yaml")

                # Move objects far away temporarily
                print("[Clone Demo] Moving objects to random positions.")
                far_updates = []
                for name in active_names:
                    far_updates.append({
                        "name": name,
                        "pos": [rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), z_center],
                        "quat": [1, 0, 0, 0],
                    })
                env.update(far_updates, persist=True)

                # Wait for visualization
                for _ in range(5):
                    viewer.sync()
                    time.sleep(0.1)

                # Restore from clone
                print("[Clone Demo] Restoring from cloned state.")
                env.update_from_clone(cloned_state)
                env.load_state("data/state_snapshot.yaml")

            step_count += 1
            viewer.sync()
            time.sleep(update_interval)


if __name__ == "__main__":
    dynamic_kitchen_demo()