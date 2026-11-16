#!/usr/bin/env python3
"""Character data loader with schema validation.

Characters are defined in JSON/YAML files in the characters/ directory.
Each character file contains identity information, scenes, and metadata.

Character File Schema:
{
    "id": "aria",  // unique identifier
    "name": "Aria",  // display name
    "identity": {
        "lines": [  // one fact per line, exported to identity patch
            "Aria is a 28-year-old woman with shoulder-length chestnut brown hair.",
            ...
        ],
        "prompt_fragment": "the same 28-year-old woman Aria...",  // short form
        "detailed_block": "the same single woman Aria, 28 years old...",  // full pinned block
        "negative_facts": "blonde hair, black hair, ..."  // explicit negatives
    },
    "scenes": {  // named scene descriptions
        "studio": "close-up portrait, studio light, plain grey background, facing camera",
        ...
    },
    "metadata": {  // optional metadata
        "description": "Female protagonist, 28 years old",
        "tags": ["protagonist", "modern"]
    }
}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Allow importing settings from same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings


class Character:
    """A loaded character with identity information and scenes."""

    def __init__(
        self,
        id: str,
        name: str,
        identity: dict[str, Any],
        scenes: dict[str, str],
        metadata: dict[str, Any] | None = None,
    ):
        self.id = id
        self.name = name
        self.identity = identity
        self.scenes = scenes
        self.metadata = metadata or {}

        # Validate required identity fields
        required = ["lines", "detailed_block", "negative_facts"]
        missing = [f for f in required if f not in self.identity]
        if missing:
            raise ValueError(f"Character {id}: missing required identity fields: {missing}")

        # Validate lines is a non-empty list
        if not isinstance(self.identity["lines"], list) or not self.identity["lines"]:
            raise ValueError(f"Character {id}: identity.lines must be a non-empty list")

        # Set defaults for optional fields
        if "prompt_fragment" not in self.identity:
            # Generate from detailed_block if not provided
            self.identity["prompt_fragment"] = self.identity["detailed_block"]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Character":
        """Load character from a dictionary."""
        required_fields = ["id", "name", "identity", "scenes"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        return cls(
            id=data["id"],
            name=data["name"],
            identity=data["identity"],
            scenes=data["scenes"],
            metadata=data.get("metadata"),
        )

    @classmethod
    def from_file(cls, path: Path) -> "Character":
        """Load character from a JSON or YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Character file not found: {path}")

        content = path.read_text(encoding="utf-8")

        # Try JSON first
        try:
            data = json.loads(content)
            return cls.from_dict(data)
        except json.JSONDecodeError:
            pass

        # Try YAML if JSON fails
        try:
            import yaml
            data = yaml.safe_load(content)
            return cls.from_dict(data)
        except ImportError:
            raise RuntimeError(
                f"Failed to parse {path} as JSON, and PyYAML is not installed. "
                "Install PyYAML to use YAML character files."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to parse character file {path}: {e}")

    def to_dict(self) -> dict[str, Any]:
        """Export character to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "identity": self.identity,
            "scenes": self.scenes,
            "metadata": self.metadata,
        }

    def get_scene(self, scene_id: str) -> str:
        """Get scene description by ID, with helpful error if not found."""
        if scene_id not in self.scenes:
            available = ", ".join(self.scenes.keys())
            raise KeyError(
                f"Scene '{scene_id}' not found for character '{self.id}'. "
                f"Available scenes: {available}"
            )
        return self.scenes[scene_id]

    def has_scene(self, scene_id: str) -> bool:
        """Check if character has a scene with the given ID."""
        return scene_id in self.scenes


def list_characters(characters_dir: Path | None = None) -> list[str]:
    """List all available character IDs in the characters directory."""
    if characters_dir is None:
        characters_dir = settings.CHARACTERS_DIR

    if not characters_dir.exists():
        return []

    character_ids = []
    for path in characters_dir.iterdir():
        if path.suffix in (".json", ".yaml", ".yml") and not path.name.startswith("."):
            # Character ID is filename without extension
            character_ids.append(path.stem)

    return sorted(character_ids)


def load_character(
    character_id: str,
    characters_dir: Path | None = None,
) -> Character:
    """Load a character by ID from the characters directory.

    Args:
        character_id: Character ID (filename without extension)
        characters_dir: Directory containing character files (default: settings.CHARACTERS_DIR)

    Returns:
        Loaded Character object

    Raises:
        FileNotFoundError: If character file not found
        ValueError: If character data is invalid
    """
    if characters_dir is None:
        characters_dir = settings.CHARACTERS_DIR

    if not characters_dir.exists():
        raise FileNotFoundError(
            f"Characters directory not found: {characters_dir}\n"
            f"Set VAL_CHARACTERS_DIR to the correct path."
        )

    # Try common extensions
    for ext in (".json", ".yaml", ".yml"):
        path = characters_dir / f"{character_id}{ext}"
        if path.exists():
            return Character.from_file(path)

    # Not found
    available = list_characters(characters_dir)
    raise FileNotFoundError(
        f"Character '{character_id}' not found in {characters_dir}\n"
        f"Available characters: {', '.join(available) or 'none'}"
    )


def get_default_character_id() -> str:
    """Get the ID of the default character (for backward compatibility)."""
    return "aria"


def load_default_character() -> Character:
    """Load the default character (aria, for backward compatibility)."""
    return load_character(get_default_character_id())


# Backward compatibility: export character data in the old format
def get_identity_a(character: Character) -> dict[str, Any]:
    """Convert Character to old IDENTITY_A format."""
    return {
        "name": character.name,
        "lines": character.identity["lines"],
        "prompt_fragment": character.identity.get("prompt_fragment", ""),
        "detailed_block": character.identity["detailed_block"],
        "negative_facts": character.identity["negative_facts"],
    }


def get_scenes(character: Character) -> dict[str, str]:
    """Get scenes dictionary from character."""
    return character.scenes
