"""
Perception Update Demo
======================
Demonstrates alias-based object detection with fork/sync for safe updates.

Each "perception module" (YCB, COCO, custom detector) detects objects via
human-readable aliases like "red cup" or "mug". AssetManager resolves each
alias to a canonical object type, and a fork/sync cycle applies the merged
detections atomically to the main environment.

Key features:
- Alias-based detection: alias → AssetManager.resolve_alias() → object type
- Fork/sync pattern: process detections in a fork, then sync_from() to commit
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from typing import Dict, List, Any
from mj_environment import Environment


def collect_perception_aliases(env: Environment) -> Dict[str, List[str]]:
    """
    Collect all aliases from perception modules in the asset metadata.

    Returns:
        Dictionary mapping module names to lists of available aliases.
    """
    aliases_by_module: Dict[str, List[str]] = {}

    for obj_type in env.asset_manager.list():
        meta = env.asset_manager.get(obj_type)
        perception = meta.get("perception", {})

        for module_name, module_config in perception.items():
            if not isinstance(module_config, dict):
                continue
            aliases = module_config.get("aliases", [])
            if aliases:
                aliases_by_module.setdefault(module_name, []).extend(aliases)

    return aliases_by_module


def simulate_detections(
    module_name: str,
    aliases: List[str],
    asset_manager,
    registry,
    z_center: float,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """
    Simulate one round of detections from a perception module.

    Picks 1-2 aliases at random, resolves each to an object instance,
    and returns update dicts with randomized positions on the table.
    """
    n = rng.integers(1, min(3, len(aliases) + 1))
    selected = rng.choice(aliases, size=n, replace=False)

    detections = []
    for alias in selected:
        obj_type = asset_manager.resolve_alias(alias, module=module_name)
        if obj_type is None or obj_type not in registry.objects:
            continue

        instances = registry.objects[obj_type]["instances"]
        if not instances:
            continue

        instance = rng.choice(instances)
        pos = [rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), z_center]
        detections.append({"name": instance, "pos": pos, "quat": [1, 0, 0, 0]})

        print(f"  [{module_name}] '{alias}' -> {instance} at ({pos[0]:+.2f}, {pos[1]:+.2f})")

    return detections


def perception_update_demo():
    """
    Demonstrate alias-based perception with fork/sync updates.

    Each viewer tick:
    1. Each perception module "detects" objects via aliases
    2. Detections are merged (last-write-wins per instance)
    3. A fork applies the merged update, then sync_from() commits it
    """
    env = Environment(
        base_scene_xml="data/scene.xml",
        objects_dir="data/objects",
        scene_config_yaml="data/scene_config.yaml",
        verbose=False,
    )
    model = env.model
    data = env.data

    # Compute table surface height for object placement
    table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    table_geom_id = model.body_geomadr[table_body_id]
    table_height = model.geom_size[table_geom_id, 2]
    sample_type = next(iter(env.registry.objects))
    sample_name = env.registry.objects[sample_type]["instances"][0]
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, sample_name)
    obj_geom_id = model.body_geomadr[obj_body_id]
    z_center = table_height + model.geom_size[obj_geom_id, 1]

    # Discover which aliases each perception module can detect
    aliases_by_module = collect_perception_aliases(env)

    print("=" * 60)
    print("Perception Update Demo (alias resolution + fork/sync)")
    print("=" * 60)
    for module_name, aliases in aliases_by_module.items():
        print(f"  {module_name}: {', '.join(aliases)}")
    print("=" * 60)
    print("Press Esc to exit\n")

    rng = np.random.default_rng(42)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0]
        viewer.cam.azimuth = -45
        viewer.cam.elevation = -45
        viewer.cam.distance = 2.0

        while viewer.is_running():
            # 1. Simulate detections from each perception module
            merged: Dict[str, Dict[str, Any]] = {}
            print(f"\n[t={data.time:.1f}s] Detection round:")
            for module_name, aliases in aliases_by_module.items():
                for det in simulate_detections(
                    module_name, aliases, env.asset_manager, env.registry, z_center, rng
                ):
                    merged[det["name"]] = det  # last-write-wins per instance

            # 2. Apply detections via fork/sync (the core pattern)
            if merged:
                with env.fork() as perception_fork:
                    perception_fork.update(list(merged.values()), hide_unlisted=True)
                    env.sync_from(perception_fork)

            viewer.sync()
            time.sleep(0.5)


if __name__ == "__main__":
    perception_update_demo()
