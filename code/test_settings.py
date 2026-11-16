#!/usr/bin/env python3
"""Unit tests for settings module.

Tests configuration loading, environment variable handling, and
path resolution without requiring actual model files or GPU.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_basic_settings():
    """Test that settings module loads without errors."""
    import settings
    
    # Basic attributes should exist
    assert hasattr(settings, "PROJECT_ROOT")
    assert hasattr(settings, "MODELS_ROOT")
    assert hasattr(settings, "TAU")
    assert hasattr(settings, "API_KEY")
    assert hasattr(settings, "HOST")
    assert hasattr(settings, "PORT")
    
    # PROJECT_ROOT should be valid
    assert settings.PROJECT_ROOT.is_dir()
    assert (settings.PROJECT_ROOT / "code").is_dir()
    
    print(f"✓ PROJECT_ROOT: {settings.PROJECT_ROOT}")
    print(f"✓ MODELS_ROOT: {settings.MODELS_ROOT}")
    print(f"✓ TAU: {settings.TAU}")


def test_describe():
    """Test settings.describe() returns expected structure."""
    import settings
    
    desc = settings.describe()
    
    # Check expected keys
    expected_keys = {
        "project_root", "models_root", "models_present",
        "valhalla", "service", "pipeline_out", "characters_dir"
    }
    assert expected_keys.issubset(desc.keys()), f"Missing keys: {expected_keys - desc.keys()}"
    
    # Check structure
    assert isinstance(desc["models_present"], dict)
    assert isinstance(desc["valhalla"], dict)
    assert isinstance(desc["service"], dict)
    
    # Valhalla config should have expected fields
    assert "root" in desc["valhalla"]
    assert "exe" in desc["valhalla"]
    assert "exe_exists" in desc["valhalla"]
    
    print("✓ describe() returns expected structure")


def test_env_var_override():
    """Test that environment variables override defaults."""
    import importlib
    
    # Set env var
    os.environ["FATE_TAU"] = "0.75"
    os.environ["FATE_PORT"] = "9999"
    
    # Reload settings module to pick up env vars
    if "settings" in sys.modules:
        importlib.reload(sys.modules["settings"])
    import settings
    
    # Check overrides
    assert settings.TAU == 0.75, f"Expected TAU=0.75, got {settings.TAU}"
    assert settings.PORT == 9999, f"Expected PORT=9999, got {settings.PORT}"
    
    # Clean up
    del os.environ["FATE_TAU"]
    del os.environ["FATE_PORT"]
    importlib.reload(sys.modules["settings"])
    
    print("✓ Environment variable overrides work")


def test_valhalla_path_handling():
    """Test that Valhalla paths handle missing configuration gracefully."""
    import settings
    
    # If Valhalla is not configured, these should be None or raise clear errors
    if settings.VALHALLA_EXE is None:
        print("✓ VALHALLA_EXE is None when not configured (expected)")
        
        # require_valhalla_exe should raise RuntimeError with helpful message
        try:
            settings.require_valhalla_exe()
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "not configured" in str(e).lower() or "not found" in str(e).lower()
            print(f"✓ require_valhalla_exe() raises helpful error: {str(e)[:80]}...")
    else:
        print(f"✓ VALHALLA_EXE configured: {settings.VALHALLA_EXE}")
        
        # If configured, check if file exists
        if settings.VALHALLA_EXE.is_file():
            # Should not raise
            result = settings.require_valhalla_exe()
            assert result == settings.VALHALLA_EXE
            print("✓ require_valhalla_exe() returns path when file exists")
        else:
            # Should raise FileNotFoundError
            try:
                settings.require_valhalla_exe()
                assert False, "Should have raised FileNotFoundError"
            except FileNotFoundError as e:
                assert "not found" in str(e).lower()
                print(f"✓ require_valhalla_exe() raises FileNotFoundError: {str(e)[:80]}...")


def test_backward_compatibility_fate_config():
    """Test that old fate_config module still works."""
    import fate_config
    
    # Should have same attributes as before
    assert hasattr(fate_config, "MODELS_ROOT")
    assert hasattr(fate_config, "TAU")
    assert hasattr(fate_config, "API_KEY")
    assert hasattr(fate_config, "HOST")
    assert hasattr(fate_config, "PORT")
    assert hasattr(fate_config, "PIPELINE_OUT")
    assert hasattr(fate_config, "JUGGERNAUT_DIR")
    assert hasattr(fate_config, "I2V_DIR")
    assert hasattr(fate_config, "COGVIDEO_DIR")
    assert hasattr(fate_config, "describe")
    
    # describe() should work
    desc = fate_config.describe()
    assert isinstance(desc, dict)
    
    print("✓ fate_config backward compatibility maintained")


def main():
    print("Testing settings module...")
    print()
    
    try:
        test_basic_settings()
        print()
        
        test_describe()
        print()
        
        test_env_var_override()
        print()
        
        test_valhalla_path_handling()
        print()
        
        test_backward_compatibility_fate_config()
        print()
        
        print("=" * 60)
        print("All settings tests PASSED")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
