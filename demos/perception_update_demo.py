"""
Demonstrates integration with multiple simulated perception systems using aliases.

This demo shows how different perception modules (YCB, COCO, custom detectors) can
use aliases to identify objects, which are then resolved to actual object instances
via AssetManager's alias resolution system.

Key features:
- Multiple perception modules running concurrently
- Alias-based detection (e.g., "cup001", "red cup", "coffee cup")
- Automatic alias resolution to object types and instances
- Realistic simulation of different detection behaviors per module
"""

import mujoco
import mujoco.viewer
import numpy as np
import threading
import queue
import time
from typing import Dict, List, Any
from mj_environment import Environment


def perception_module_thread(
    update_queue: queue.Queue,
    module_name: str,
    asset_manager,
    registry,
    z_center: float,
    aliases_by_module: Dict[str, List[str]],
    interval: float = 1.0,
) -> None:
    """
    Simulates a perception module that detects objects using aliases.

    Args:
        update_queue: Queue to send detection updates
        module_name: Name of the perception module (e.g., "ycb", "coco", "custom_detector")
        asset_manager: AssetManager instance for alias resolution
        registry: ObjectRegistry to get available instances
        z_center: Z-coordinate for object placement
        aliases_by_module: Dict mapping module names to lists of aliases they can detect
        interval: Detection interval in seconds
    """
    available_aliases = aliases_by_module.get(module_name, [])
    if not available_aliases:
        return
    
    while True:
        detected = []
        # Simulate detecting 1-2 random objects using aliases (reduced from 1-3)
        num_detections = np.random.randint(1, min(3, len(available_aliases) + 1))
        selected_aliases = np.random.choice(available_aliases, size=num_detections, replace=False)
        
        for alias in selected_aliases:
            try:
                # Resolve alias to object type
                obj_type = asset_manager.resolve_alias(alias, module=module_name)
                
                # Get available instances for this object type
                if obj_type in registry.objects:
                    instances = registry.objects[obj_type]["instances"]
                    if instances:
                        # Select a random instance (or use the first available hidden one)
                        instance_name = np.random.choice(instances)
                        
                        pos = [
                            np.random.uniform(-0.4, 0.4),
                            np.random.uniform(-0.4, 0.4),
                            z_center,
                        ]
                        detected.append({
                            "name": instance_name,
                            "pos": pos,
                            "quat": [1, 0, 0, 0],
                            "alias": alias,  # Keep track of the alias used
                            "module": module_name,  # Keep track of the module
                        })
            except (KeyError, ValueError) as e:
                # Alias not found or resolution failed - skip
                continue
        
        if detected:
            update_queue.put({
                "module": module_name,
                "detections": detected,
            })
        time.sleep(interval)


def collect_perception_aliases(env: Environment) -> Dict[str, List[str]]:
    """
    Collect all aliases from perception modules for each object type.
    
    Returns:
        Dictionary mapping module names to lists of available aliases.
    """
    aliases_by_module: Dict[str, List[str]] = {}
    
    # Iterate through all object types
    for obj_type in env.asset_manager.list():
        try:
            meta = env.asset_manager.get(obj_type)
            perception_config = meta.get("perception", {})
            
            # For each perception module (ycb, coco, custom_detector, etc.)
            for module_name, module_config in perception_config.items():
                if not isinstance(module_config, dict):
                    continue
                
                # Get aliases for this module
                module_aliases = module_config.get("aliases", [])
                if module_aliases:
                    if module_name not in aliases_by_module:
                        aliases_by_module[module_name] = []
                    aliases_by_module[module_name].extend(module_aliases)
        except Exception:
            continue
    
    return aliases_by_module


def perception_update_demo():
    """Run comprehensive perception update demo with multiple modules and aliases."""
    env = Environment(
        base_scene_xml="data/scene.xml",
        objects_dir="data/objects",
        scene_config_yaml="data/scene_config.yaml",
        verbose=False,
    )
    model = env.model
    data = env.data

    # Calculate table height and object placement height
    table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    table_geom_id = model.body_geomadr[table_body_id]
    table_size = model.geom_size[table_geom_id]
    table_height = table_size[2]

    sample_type = next(iter(env.registry.objects))
    sample_name = env.registry.objects[sample_type]["instances"][0]
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, sample_name)
    object_geom_id = model.body_geomadr[object_body_id]
    object_size = model.geom_size[object_geom_id]
    object_half_height = object_size[1]

    z_center = table_height + object_half_height
    update_queue = queue.Queue()

    # Collect all aliases organized by perception module
    aliases_by_module = collect_perception_aliases(env)
    
    print("=" * 60)
    print("Perception Modules and Aliases:")
    print("=" * 60)
    for module_name, aliases in aliases_by_module.items():
        print(f"  {module_name}: {len(aliases)} aliases")
        print(f"    {', '.join(aliases[:5])}" + ("..." if len(aliases) > 5 else ""))
    print("=" * 60)

    # Start multiple perception module threads
    perception_threads = []
    for module_name in aliases_by_module.keys():
        thread = threading.Thread(
            target=perception_module_thread,
            args=(
                update_queue,
                module_name,
                env.asset_manager,
                env.registry,
                z_center,
                aliases_by_module,
                1.0 + np.random.uniform(-0.3, 0.3),  # Slightly different intervals
            ),
            daemon=True,
        )
        thread.start()
        perception_threads.append(thread)
        print(f"Started {module_name} perception module")

    print(f"\nRunning {len(perception_threads)} perception modules concurrently")
    print("persist=False: Objects disappear if not continuously detected")
    print("Press Esc to exit\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Configure camera
        viewer.cam.lookat[:] = [0, 0, 0]
        viewer.cam.azimuth = -45
        viewer.cam.elevation = -45
        viewer.cam.distance = 2.0

        # Track latest detections with timestamps (keyed by module name)
        # Format: {module_name: {instance_name: {"detection": {...}, "timestamp": float}}}
        detection_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        detection_timeout = 2.0  # Keep detections visible for 2 seconds
        
        while viewer.is_running():
            current_time = time.time()
            
            # Process all pending perception updates from all modules
            try:
                while not update_queue.empty():
                    update = update_queue.get_nowait()
                    module = update["module"]
                    detections = update["detections"]
                    
                    # Initialize module cache if needed
                    if module not in detection_cache:
                        detection_cache[module] = {}
                    
                    # Update cache with new detections (add timestamp)
                    for det in detections:
                        instance_name = det["name"]
                        detection_cache[module][instance_name] = {
                            "detection": det,
                            "timestamp": current_time,
                        }
                        # Print detection info
                        print(
                            f"[{module}] Detected '{det['alias']}' → {det['name']} "
                            f"at ({det['pos'][0]:.2f}, {det['pos'][1]:.2f}, {det['pos'][2]:.2f})"
                        )
            except queue.Empty:
                pass

            # Collect all valid (non-expired) detections from all modules
            # Deduplicate by instance_name, keeping the most recent detection
            valid_detections_dict: Dict[str, Dict[str, Any]] = {}
            for module_cache in detection_cache.values():
                for instance_name, cached_data in list(module_cache.items()):
                    # Remove expired detections
                    if current_time - cached_data["timestamp"] > detection_timeout:
                        del module_cache[instance_name]
                        continue
                    # Add valid detection (keep most recent if multiple modules detect same instance)
                    det = cached_data["detection"]
                    if instance_name not in valid_detections_dict or \
                       cached_data["timestamp"] > valid_detections_dict[instance_name].get("_timestamp", 0):
                        valid_detections_dict[instance_name] = {
                            "name": det["name"],
                            "pos": det["pos"],
                            "quat": det["quat"],
                            "_timestamp": cached_data["timestamp"],
                        }
            
            # Convert to list
            valid_detections = [
                {k: v for k, v in det.items() if k != "_timestamp"}
                for det in valid_detections_dict.values()
            ]
            
            # Limit total visible objects to avoid clutter (max 4 objects)
            if len(valid_detections) > 4:
                # Randomly sample 4 detections to keep
                indices = np.random.choice(len(valid_detections), size=4, replace=False)
                valid_detections = [valid_detections[i] for i in indices]

            # Update environment with all valid detections (from all modules)
            if valid_detections:
                env.update(valid_detections, persist=False)

            viewer.sync()
            time.sleep(0.1)


if __name__ == "__main__":
    perception_update_demo()