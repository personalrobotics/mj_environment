"""
Demonstration of the AssetManager and its YAML-driven asset pipeline.
"""

import os
import mujoco
import mujoco.viewer
from mj_environment.asset_manager import AssetManager


def filter_by_category(am: AssetManager, category: str):
    """Return list of asset names that belong to a given category."""
    matches = []
    for name, meta in am.assets.items():
        cats = meta.get("category", [])
        if isinstance(cats, str):
            cats = [cats]
        if category in cats:
            matches.append(name)
    return matches


def main():
    base_dir = "data/objects"  # adjust if needed
    print("[Demo] Initializing AssetManager...")
    am = AssetManager(base_dir=base_dir, verbose=True)

    # -------------------------------------------------------------
    # 1️⃣ Print summary of all loaded assets
    # -------------------------------------------------------------
    am.summary()

    # -------------------------------------------------------------
    # 2️⃣ Filter assets by category
    # -------------------------------------------------------------
    print("[Demo] Querying by category:")
    for category in ["metal", "ceramic", "drinkware"]:
        items = filter_by_category(am, category)
        print(f"  - {category:10s}: {items}")

    # -------------------------------------------------------------
    # 3️⃣ Load and inspect a MuJoCo model with overrides
    # -------------------------------------------------------------
    print("\n[Demo] Loading 'cup' model with overrides applied...")
    cup_model = am.load_model("cup")

    print(f"  nbody: {cup_model.nbody}")
    print(f"  ngeom: {cup_model.ngeom}")
    print(f"  mass[0]: {cup_model.body_mass[0]:.3f}")
    print(f"  color[0]: {cup_model.geom_rgba[0]}")
    print(f"  geom size[0]: {cup_model.geom_size[0]}")

    # -------------------------------------------------------------
    # 4️⃣ Simulate reloading after a metadata change
    # -------------------------------------------------------------
    print("\n[Demo] Simulating reload...")
    am.reload()
    am.summary()

    # -------------------------------------------------------------
    # 5️⃣ Example: using model in a viewer (optional)
    # -------------------------------------------------------------
    print("[Demo] Launching viewer for visual inspection (press ESC to quit)...")
    data = mujoco.MjData(cup_model)
    try:
        with mujoco.viewer.launch_passive(cup_model, data) as viewer:
            viewer.cam.lookat[:] = [0, 0, 0]
            viewer.cam.distance = 0.5
            while viewer.is_running():
                viewer.sync()
    except Exception as e:
        print(f"[WARN] Viewer unavailable: {e}")

    print("\n[Demo] Done!")


if __name__ == "__main__":
    main()