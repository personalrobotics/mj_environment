"""AssetManager — Unified YAML + XML loader with automatic override application.

Supports nested per-object directories of the form:
data/objects/<object_name>/{model.xml, meta.yaml}
"""

import os
import yaml
import mujoco
import numpy as np
from typing import Dict, Any, List


class AssetManager:
    """
    Loads and manages object metadata (YAML) and corresponding MuJoCo XML assets.

    Responsibilities:
      - Discover and parse object metadata recursively under base_dir
      - Track absolute XML paths relative to each meta.yaml file
      - Automatically apply YAML-defined overrides (mass, color, scale)
      - Provide verified access for ObjectRegistry and Environment

    Expected directory structure:
      base_dir/
        cup/
          model.xml
          meta.yaml
        plate/
          model.xml
          meta.yaml
    """

    def __init__(self, base_dir: str, verbose: bool = False):
        self.base_dir = os.path.abspath(base_dir)
        self.verbose = verbose
        self.assets: Dict[str, Dict[str, Any]] = {}

        if not os.path.isdir(self.base_dir):
            raise FileNotFoundError(f"Asset base directory not found: {self.base_dir}")

        self._load_all_metadata()
        self.verify()
        if self.verbose:
            self.summary()

    # ---------------------------------------------------------------------
    # Metadata discovery and loading
    # ---------------------------------------------------------------------
    def _load_all_metadata(self):
        """Recursively find and load all meta.yaml files under base_dir."""
        for root, _, files in os.walk(self.base_dir):
            for filename in files:
                if not filename.endswith(".yaml"):
                    continue
                filepath = os.path.join(root, filename)
                data = self._load_yaml_file(filepath)

                name = data.get("name")
                if not name:
                    raise ValueError(f"Missing 'name' in {filepath}")
                if name in self.assets:
                    raise ValueError(f"Duplicate asset name detected: {name}")

                self.assets[name] = data

    def _load_yaml_file(self, path: str) -> Dict[str, Any]:
        """Load and normalize one YAML file, resolving relative xml_path."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML structure in {path}: must be a dict.")

        xml_rel = data.get("xml_path", "model.xml")
        xml_abs = xml_rel if os.path.isabs(xml_rel) else os.path.join(os.path.dirname(path), xml_rel)
        data["xml_path"] = os.path.abspath(xml_abs)

        return data

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def get(self, name: str) -> Dict[str, Any]:
        """Return metadata dictionary for a given asset."""
        if name not in self.assets:
            raise KeyError(f"Asset '{name}' not found in AssetManager.")
        return self.assets[name]

    def has(self, name: str) -> bool:
        """Return True if asset is known."""
        return name in self.assets

    def list(self) -> List[str]:
        """Return list of all asset names."""
        return sorted(self.assets.keys())

    def verify(self) -> None:
        """Check all XML files exist."""
        for name, meta in self.assets.items():
            xml_path = meta["xml_path"]
            if not os.path.exists(xml_path):
                raise FileNotFoundError(f"XML file not found for asset '{name}': {xml_path}")

    def reload(self):
        """Reload all YAML metadata."""
        self.assets.clear()
        self._load_all_metadata()
        self.verify()
        if self.verbose:
            print("[AssetManager] Reloaded all assets.")

    # ---------------------------------------------------------------------
    # High-level API: automatic model loading
    # ---------------------------------------------------------------------
    def load_model(self, name: str) -> mujoco.MjModel:
        """
        Load the MuJoCo model for a given asset name.
        Automatically applies all YAML-defined overrides.
        """
        if name not in self.assets:
            raise KeyError(f"Asset '{name}' not found.")

        meta = self.assets[name]
        model = mujoco.MjModel.from_xml_path(meta["xml_path"])
        self._apply_overrides(model, meta)
        return model

    # ---------------------------------------------------------------------
    # Internal: applying overrides
    # ---------------------------------------------------------------------
    def _apply_overrides(self, model: mujoco.MjModel, meta: Dict[str, Any]) -> None:
        """Apply YAML-specified overrides to a MuJoCo model."""

        # Mass override
        if "mass" in meta:
            m = meta["mass"]
            if np.isscalar(m):
                model.body_mass[:] = m
            elif isinstance(m, (list, np.ndarray)):
                model.body_mass[: len(m)] = m
            if self.verbose:
                print(f"[AssetManager] Applied mass override to {meta['name']}")

        # Color override
        if "color" in meta:
            rgba = np.array(meta["color"], dtype=float)
            if rgba.size == 4:
                for i in range(model.ngeom):
                    model.geom_rgba[i, :] = rgba
            if self.verbose:
                print(f"[AssetManager] Applied color override to {meta['name']}")

        # Scale override
        if "scale" in meta:
            s = float(meta["scale"])
            model.geom_size[:] *= s
            if self.verbose:
                print(f"[AssetManager] Applied scale={s} to {meta['name']}")

    # ---------------------------------------------------------------------
    # Summary helper
    # ---------------------------------------------------------------------
    def summary(self):
        """Print a table of loaded assets with nicely formatted categories."""
        print(f"\n[AssetManager] Loaded {len(self.assets)} assets from {self.base_dir}")
        for name, meta in self.assets.items():
            # Normalize category to list
            cat = meta.get("category", [])
            if isinstance(cat, str):
                cat = [cat]
            cat_str = ", ".join(cat)

            xml_rel = os.path.relpath(meta["xml_path"], self.base_dir)
            print(f"  - {name:10s} | {cat_str:35s} | XML: {xml_rel}")
        print()