# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

from asset_manager import AssetManager


def test_assets_load_correctly():
    am = AssetManager(base_dir="data/objects")
    asset_list = am.list()
    assert len(asset_list) > 0, "No assets found"
    for name in asset_list:
        meta = am.get(name)
        assert "name" in meta, f"Missing name in metadata for {name}"
