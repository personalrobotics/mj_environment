from mj_environment.asset_manager import AssetManager

def test_assets_load_correctly():
    am = AssetManager(base_dir="data/objects")
    assert len(am.assets) > 0, "No assets found"
    for name, meta in am.assets.items():
        assert "xml_path" in meta, f"Missing XML path for {name}"
        assert "name" in meta, f"Missing name in metadata for {name}"
