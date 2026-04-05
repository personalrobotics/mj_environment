# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""
Dynamic Kitchen Demo
====================
Showcases runtime object activation, motion, forking, and state serialization
using the mj_environment Environment.
"""

import logging
import time

import mujoco
import mujoco.viewer
import numpy as np

from mj_environment.environment import Environment

logging.basicConfig(level=logging.DEBUG, format="[%(name)s] %(message)s")


def dynamic_kitchen_demo():
    """Demonstrate dynamic activation, movement, forking, and state serialization."""
    env = Environment(
        base_scene_xml="data/scene.xml",
        objects_dir="data/objects",
        scene_config_yaml="data/scene_config.yaml",
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

        while viewer.is_running():
            # 1️⃣ Activate 2 random object types if none active
            if len(active_names) < 2:
                selected = rng.choice(object_types, size=2, replace=False)
                for obj_type in selected:
                    obj_name = env.registry.activate(obj_type, [0, 0, z_center])
                    active_names.append(obj_name)

            # 2️⃣ Move all active objects to random new positions
            updates = []
            for name in active_names:
                updates.append(
                    {
                        "name": name,
                        "pos": [rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), z_center],
                        "quat": [1, 0, 0, 0],
                    }
                )
            env.update(updates, hide_unlisted=False)

            # 3️⃣ Every 10 steps: demonstrate fork() for planning
            if step_count > 0 and step_count % 10 == 0:
                print("\n[Fork Demo] Creating fork for planning simulation...")

                # Fork the environment - original stays unchanged
                with env.fork() as planning_env:
                    # Simulate planning: move objects in the fork
                    print("[Fork Demo] Moving objects in forked environment...")
                    far_updates = []
                    for name in active_names:
                        far_updates.append(
                            {
                                "name": name,
                                "pos": [rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), z_center],
                                "quat": [1, 0, 0, 0],
                            }
                        )
                    planning_env.update(far_updates, hide_unlisted=False)

                    # Step the fork's physics a few times
                    for _ in range(10):
                        planning_env.step()

                    print(f"[Fork Demo] Fork sim time: {planning_env.data.time:.3f}s")
                    print(f"[Fork Demo] Original sim time: {env.data.time:.3f}s (unchanged)")

                # Fork is discarded, original env unchanged
                print("[Fork Demo] Fork discarded, original environment preserved.")

            step_count += 1
            viewer.sync()
            time.sleep(update_interval)


if __name__ == "__main__":
    dynamic_kitchen_demo()
