# Character File Format

## Overview

Characters are defined in JSON or YAML files in the `characters/` directory. Each character file contains:
- Identity information (lines, detailed block, negative facts)
- Scene descriptions
- Optional metadata

## File Structure

### JSON Format (recommended)

```json
{
  "id": "character_id",
  "name": "Character Name",
  "identity": {
    "lines": [
      "Fact 1 about the character (one line per fact).",
      "Fact 2 about appearance, age, features.",
      "Fact 3 reinforcing key identity traits."
    ],
    "prompt_fragment": "Short identity description (legacy, optional)",
    "detailed_block": "Complete, pinned identity description. This MUST be byte-identical across all segments. Never paraphrase or modify during generation.",
    "negative_facts": "Explicit negative traits to exclude (e.g., wrong hair color, different person)"
  },
  "scenes": {
    "scene_id": "Scene description for this character",
    "another_scene": "Another scene description"
  },
  "metadata": {
    "description": "Human-readable description",
    "tags": ["tag1", "tag2"],
    "created": "2026-11-16",
    "version": "1.0"
  }
}
```

### Required Fields

- `id` (string): Unique identifier (used in CLI args and API requests)
- `name` (string): Display name
- `identity` (object):
  - `lines` (array of strings): One fact per line. **Must be non-empty.**
  - `detailed_block` (string): Pinned identity block. **Critical: never modify during generation.**
  - `negative_facts` (string): Explicit negatives for the negative prompt
- `scenes` (object): Named scene descriptions (keys are scene IDs)

### Optional Fields

- `identity.prompt_fragment` (string): Short form for legacy compatibility (generated from detailed_block if not provided)
- `metadata` (object): Arbitrary metadata for documentation

## Identity Guidelines

### The Pinned Identity Block

The `detailed_block` is the **single most important field**. It must be:

1. **Byte-identical** across all segments (never paraphrase)
2. **Detailed** with specific physical traits (hair color, age, facial features, body type)
3. **Consistent** with the facts in `lines`
4. **Measured** - paraphrasing has been measured to reduce identity consistency by -0.083 ArcFace

Example (from aria.json):
```json
"detailed_block": "the same single woman Aria, 28 years old, shoulder-length chestnut brown hair with a center parting, small mole on her left cheek, hazel eyes, fair skin, slim oval face, high cheekbones, straight nose, dark brown eyebrows, natural pink lips, small chin"
```

### Identity Lines

Lines are used for:
- Valhalla identity patch export (when using Valhalla engine features)
- Bible graph edges for retrieval
- Auditable evidence trail

Each line should be a single, clear fact:
```json
"lines": [
  "Aria is a 28-year-old woman with shoulder-length chestnut brown hair.",
  "Aria has a small mole on her left cheek and hazel eyes and fair skin.",
  "Aria has a slim oval face with a straight nose and natural makeup.",
  "Aria's hair is always chestnut brown, never blonde and never black.",
  "Aria is the same single person in every scene and every segment."
]
```

### Negative Facts

Explicit negatives help exclude unwanted traits:
```json
"negative_facts": "blonde hair, black hair, grey hair, different person, another woman, changing face, beard, masculine features"
```

## Scene Descriptions

Scenes are free-form descriptions used in prompts. Common scenes:

```json
"scenes": {
  "studio": "close-up portrait, studio light, plain grey background, facing camera",
  "park": "standing outdoors in a sunlit park, natural daylight, waist-up, facing camera",
  "harbour": "at a quiet harbour at dusk, sea breeze, waist-up, facing camera",
  "library": "in an old library between tall shelves, soft window light, waist-up, facing camera",
  "office": "in a modern office, neutral expression, fluorescent light, waist-up, facing camera"
}
```

The final prompt is constructed as:
```
photorealistic photograph of {detailed_block}, {scene_description}
```

## Creating a New Character

### Step 1: Create the file

Create `characters/your_character.json`:

```json
{
  "id": "your_character",
  "name": "Your Character",
  "identity": {
    "lines": [
      "Your character is a [age]-year-old [gender] with [key physical trait].",
      "[Another specific physical detail].",
      "[Reinforcing key identity point]."
    ],
    "detailed_block": "the same single [gender] Your Character, [age] years old, [detailed physical description with specific colors, shapes, and features]",
    "negative_facts": "[traits to explicitly exclude]"
  },
  "scenes": {
    "studio": "close-up portrait, studio light, plain grey background, facing camera",
    "park": "standing outdoors in a sunlit park, natural daylight, waist-up, facing camera"
  },
  "metadata": {
    "description": "[Brief character description]",
    "tags": ["[category]"],
    "created": "2026-11-16",
    "version": "1.0"
  }
}
```

### Step 2: Validate

Run the character loader test to validate your file:

```bash
python test_character_loader.py
```

Or list characters:

```bash
python run_pipeline.py --list-characters
```

### Step 3: Test

Run a short pipeline with your character:

```bash
python run_pipeline.py --character your_character --scenes park,studio --seeds 7 --k 2
```

### Step 4: Measure

Review the output:
- Check `manifest.json` for identity fingerprints (should be identical across segments)
- Check ArcFace scores (`chosen_cos` values)
- Look for ABSTAINED segments (may indicate identity issues)

## Usage

### CLI

```bash
# List available characters
python run_pipeline.py --list-characters

# Use specific character
python run_pipeline.py --character your_character --scenes park,harbour

# Default (aria) if --character not specified
python run_pipeline.py --scenes park,harbour
```

### API

```bash
# Supply with character
curl -X POST http://localhost:8790/v1/supply \
  -H "Content-Type: application/json" \
  -d '{"scene": "park", "character_id": "your_character"}'

# Run pipeline with character
curl -X POST http://localhost:8790/v1/run \
  -H "Content-Type: application/json" \
  -d '{
    "character_id": "your_character",
    "scenes": ["park", "harbour"],
    "seeds": [7],
    "k": 4
  }'

# Health check shows available characters
curl http://localhost:8790/health
```

### Python

```python
from character_loader import load_character
from fate_interface import FateInterface

# Load character
char = load_character("your_character")
print(f"Loaded: {char.name}")
print(f"Scenes: {list(char.scenes.keys())}")

# Use with FateInterface
fate = FateInterface(character_id="your_character")
spec = fate.supply("park")
print(spec["prompt"])
```

## Migration from Hardcoded Characters

The original `character_bible.py` hardcoded `IDENTITY_A` (Aria). This has been migrated to:

1. `characters/aria.json` - the data file
2. `character_bible.IDENTITY_A` - backward compatibility wrapper

**Backward compatibility is maintained:**
- Old code importing `character_bible.IDENTITY_A` still works
- Old code using `FateInterface()` without args defaults to Aria
- Old manifests/reports remain valid

**New code should:**
- Use `character_loader.load_character(id)` to load characters
- Pass `character_id` to `FateInterface()` or `run_pipeline.py --character`
- Store character ID in manifests for traceability

## Best Practices

1. **Specificity**: Be specific about physical traits (hair color & style, eye color, facial features, age, body type)
2. **Consistency**: Ensure `detailed_block` and `lines` agree on all facts
3. **Negatives**: Include explicit negatives for common failure modes (wrong hair color, different person, gender confusion)
4. **Testing**: Always test new characters on multiple seeds and scenes before production use
5. **Measurement**: Track ArcFace scores to validate identity consistency
6. **Documentation**: Use metadata to document character purpose, version, and creation date

## Troubleshooting

### Character not found

```
FileNotFoundError: Character 'xxx' not found in /workspace/characters
```

- Check file exists: `ls characters/xxx.json`
- Check character ID in file matches filename (without extension)
- Run `python run_pipeline.py --list-characters` to see available characters

### Invalid character file

```
ValueError: Character xxx: missing required identity fields: ['detailed_block']
```

- Ensure all required fields are present
- Check JSON is valid: `python -m json.tool characters/xxx.json`
- Run `python test_character_loader.py` for detailed validation

### Low identity scores

If `chosen_cos` values are consistently < 0.5:

1. Check `detailed_block` is sufficiently detailed (compare to aria.json)
2. Add more specific physical traits (facial features, hair style, age markers)
3. Verify `negative_facts` exclude common failure modes
4. Try different seeds to check if issue is character or seed-specific

### Scene not found

```
KeyError: Scene 'xxx' not found for character 'yyy'. Available scenes: park, harbour
```

- Add the scene to your character's `scenes` object
- Or pass a full scene description instead of a scene ID:
  ```bash
  python run_pipeline.py --character yyy --scenes "standing on a beach at sunset"
  ```

## See Also

- [LAUNCH.md](./LAUNCH.md) - Setup and deployment guide
- [DESIGN.md](./DESIGN.md) - Architecture and mechanisms
- [code/character_loader.py](../code/character_loader.py) - Implementation reference
- [code/test_character_loader.py](../code/test_character_loader.py) - Validation tests
