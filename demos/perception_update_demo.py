"""
Perception Update Demo
======================
Demonstrates alias-based object detection with ObjectTracker for persistent
instance association, and fork/sync for safe environment updates.

Each "perception module" (YCB, COCO, custom detector) detects objects via
human-readable aliases like "red cup" or "mug". AssetManager resolves each
alias to a canonical object type, and ObjectTracker maintains consistent
instance identity across frames.

Key features:
- Alias-based detection: alias → AssetManager.resolve_alias() → object type
- ObjectTracker: nearest-neighbor association for persistent instance names
- Fork/sync pattern: process detections in a fork, then sync_from() to commit
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from typing import Dict, List, Any
from mj_environment import Environment, ObjectTracker


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
    z_center: float,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """
    Simulate one round of raw detections from a perception module.

    Returns dicts with "type" and "pos" — no instance names.
    The ObjectTracker handles instance assignment.
    """
    n = rng.integers(1, min(3, len(aliases) + 1))
    selected = rng.choice(aliases, size=n, replace=False)

    detections = []
    for alias in selected:
        obj_type = asset_manager.resolve_alias(alias, module=module_name)
        if obj_type is None:
            continue

        pos = [rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), z_center]
        detections.append({"type": obj_type, "pos": pos})
        print(f"  [{module_name}] '{alias}' -> type={obj_type} at ({pos[0]:+.2f}, {pos[1]:+.2f})")

    return detections


def perception_update_demo():
    """
    Demonstrate alias-based perception with ObjectTracker and fork/sync.

    Each viewer tick:
    1. Each perception module "detects" objects via aliases → raw {type, pos}
    2. ObjectTracker associates detections to persistent instance names
    3. A fork applies the tracked updates, then sync_from() commits them
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

    # ObjectTracker maintains persistent detection → instance mapping
    tracker = ObjectTracker(env.registry, max_distance=0.20)

    print("=" * 60)
    print("Perception Update Demo (tracker + alias resolution + fork/sync)")
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
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_BODY

        while viewer.is_running():
            # 1. Simulate raw detections from each perception module
            raw_detections: List[Dict[str, Any]] = []
            print(f"\n[t={data.time:.1f}s] Detection round:")
            for module_name, aliases in aliases_by_module.items():
                raw_detections.extend(
                    simulate_detections(module_name, aliases, env.asset_manager, z_center, rng)
                )

            # 2. Tracker assigns persistent instance names
            updates = tracker.associate(raw_detections)

            # 3. Apply via fork/sync (the core pattern)
            if updates:
                with env.fork() as perception_fork:
                    perception_fork.update(updates, hide_unlisted=True)
                    env.sync_from(perception_fork)

                for u in updates:
                    print(f"  -> {u['name']} at ({u['pos'][0]:+.2f}, {u['pos'][1]:+.2f})")

            viewer.sync()
            time.sleep(0.5)


if __name__ == "__main__":
    perception_update_demo()
