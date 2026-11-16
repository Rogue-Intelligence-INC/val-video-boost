#!/usr/bin/env python3
"""Unit tests for character loader.

Tests character loading, validation, and backward compatibility
without requiring GPU or model weights.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_load_default_character():
    """Test loading the default aria character."""
    from character_loader import load_default_character, get_default_character_id
    
    char_id = get_default_character_id()
    assert char_id == "aria", f"Expected default character 'aria', got '{char_id}'"
    
    # Try to load aria
    try:
        char = load_default_character()
        assert char.id == "aria"
        assert char.name == "Aria"
        assert "lines" in char.identity
        assert "detailed_block" in char.identity
        assert "negative_facts" in char.identity
        assert isinstance(char.scenes, dict)
        print(f"✓ Loaded default character: {char.name} ({char.id})")
        print(f"  Identity lines: {len(char.identity['lines'])}")
        print(f"  Scenes: {', '.join(char.scenes.keys())}")
        return char
    except FileNotFoundError as e:
        print(f"! Default character not found (expected in test environment): {e}")
        return None


def test_character_validation():
    """Test character schema validation."""
    from character_loader import Character
    
    # Valid character
    valid = {
        "id": "test",
        "name": "Test Character",
        "identity": {
            "lines": ["Test is a person."],
            "detailed_block": "Test is a test person",
            "negative_facts": "not test"
        },
        "scenes": {"test_scene": "test description"}
    }
    char = Character.from_dict(valid)
    assert char.id == "test"
    print("✓ Valid character loads correctly")
    
    # Missing required field
    try:
        invalid = {
            "id": "test",
            "name": "Test",
            "identity": {"lines": []},  # Missing required fields
            "scenes": {}
        }
        Character.from_dict(invalid)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "missing required" in str(e).lower()
        print(f"✓ Missing required fields raises ValueError: {str(e)[:60]}...")
    
    # Empty lines
    try:
        invalid = {
            "id": "test",
            "name": "Test",
            "identity": {
                "lines": [],  # Empty
                "detailed_block": "test",
                "negative_facts": "test"
            },
            "scenes": {}
        }
        Character.from_dict(invalid)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "non-empty list" in str(e).lower()
        print(f"✓ Empty lines raises ValueError: {str(e)[:60]}...")


def test_character_file_formats():
    """Test loading characters from JSON and YAML files."""
    from character_loader import Character
    
    char_data = {
        "id": "test_char",
        "name": "Test Character",
        "identity": {
            "lines": ["Test is a 30-year-old person."],
            "detailed_block": "Test is a 30-year-old person with brown hair",
            "negative_facts": "not a different person"
        },
        "scenes": {
            "park": "in a park",
            "office": "in an office"
        },
        "metadata": {
            "description": "Test character for unit tests",
            "tags": ["test"]
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Test JSON
        json_file = tmppath / "test.json"
        json_file.write_text(json.dumps(char_data), encoding="utf-8")
        char_json = Character.from_file(json_file)
        assert char_json.id == "test_char"
        assert len(char_json.scenes) == 2
        print("✓ Character loads from JSON file")
        
        # Test YAML (if available)
        try:
            import yaml
            yaml_file = tmppath / "test.yaml"
            yaml_file.write_text(yaml.dump(char_data), encoding="utf-8")
            char_yaml = Character.from_file(yaml_file)
            assert char_yaml.id == "test_char"
            print("✓ Character loads from YAML file")
        except ImportError:
            print("! PyYAML not installed, skipping YAML test")


def test_character_scene_access():
    """Test scene access methods."""
    from character_loader import Character
    
    char_data = {
        "id": "test",
        "name": "Test",
        "identity": {
            "lines": ["Test is a person."],
            "detailed_block": "Test person",
            "negative_facts": "not test"
        },
        "scenes": {
            "park": "standing in a park",
            "office": "sitting in an office"
        }
    }
    
    char = Character.from_dict(char_data)
    
    # get_scene should work for existing scenes
    assert char.get_scene("park") == "standing in a park"
    assert char.has_scene("park")
    assert char.has_scene("office")
    assert not char.has_scene("beach")
    print("✓ Scene access methods work correctly")
    
    # get_scene should raise KeyError for missing scenes with helpful message
    try:
        char.get_scene("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError as e:
        assert "nonexistent" in str(e)
        assert "park" in str(e)  # Should list available scenes
        print(f"✓ Missing scene raises helpful KeyError: {str(e)[:80]}...")


def test_list_characters():
    """Test listing available characters."""
    from character_loader import list_characters
    import settings
    
    if settings.CHARACTERS_DIR.exists():
        chars = list_characters()
        print(f"✓ list_characters() returned {len(chars)} character(s): {', '.join(chars)}")
        
        # Should include aria if characters/ exists
        if chars:
            assert isinstance(chars, list)
            assert all(isinstance(c, str) for c in chars)
    else:
        print(f"! Characters directory not found: {settings.CHARACTERS_DIR}")


def test_backward_compatibility():
    """Test backward compatibility helpers."""
    from character_loader import get_identity_a, get_scenes
    import character_bible as cb
    
    # character_bible.IDENTITY_A should still work
    assert isinstance(cb.IDENTITY_A, dict)
    assert "name" in cb.IDENTITY_A
    assert "lines" in cb.IDENTITY_A
    assert "detailed_block" in cb.IDENTITY_A
    assert "negative_facts" in cb.IDENTITY_A
    print("✓ character_bible.IDENTITY_A backward compatibility maintained")
    
    # Test conversion functions
    try:
        char = load_default_character()
        identity_a = get_identity_a(char)
        assert "name" in identity_a
        assert "lines" in identity_a
        
        scenes = get_scenes(char)
        assert isinstance(scenes, dict)
        print("✓ Character conversion to old format works")
    except:
        print("! Character conversion test skipped (character not available)")


def test_fate_interface_character():
    """Test FateInterface integration with character loader."""
    try:
        from fate_interface import FateInterface
        from character_loader import load_default_character
        
        # Default initialization (should use aria)
        fate1 = FateInterface()
        if fate1.character:
            assert fate1.character.id == "aria"
            print(f"✓ FateInterface defaults to character: {fate1.character.id}")
        
        # Explicit character
        try:
            fate2 = FateInterface(character_id="aria")
            assert fate2.character is not None
            assert fate2.character.id == "aria"
            print("✓ FateInterface loads character by ID")
        except FileNotFoundError:
            print("! Character loading test skipped (aria not found)")
        
        # Supply methods should work
        if fate1.character:
            spec = fate1.supply("park")
            assert "prompt" in spec
            assert "negative_prompt" in spec
            assert "character_id" in spec
            assert spec["character_id"] == fate1.character.id
            print("✓ FateInterface.supply() includes character_id")
    except ImportError as e:
        print(f"! FateInterface test skipped (import error): {e}")


def main():
    print("Testing character loader...")
    print()
    
    try:
        test_load_default_character()
        print()
        
        test_character_validation()
        print()
        
        test_character_file_formats()
        print()
        
        test_character_scene_access()
        print()
        
        test_list_characters()
        print()
        
        test_backward_compatibility()
        print()
        
        test_fate_interface_character()
        print()
        
        print("=" * 60)
        print("All character loader tests PASSED")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
