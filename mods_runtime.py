import os
import json
import copy
import glob
from typing import Dict, List, Tuple, Any, Optional

# A built-in default law to ensure the application can run even if no
# law files are provided. This is the canonical schema.
DEFAULT_LAW = {
    "version": "1.0",
    "name": "Quantum+Classical Baseline",
    "description": "Canonical constants and enhancement index map.",
    "constants": {
        "c": 299792458.0,
        "softening_eps_rel": 1e-12,
        "max_accel_rel": 1e-2
    },
    "enhancement_index_order": [
        "quantum_entanglement",
        "temporal_flux",
        "dimensional_folding",
        "consciousness_field",
        "psionic_energy",
        "exotic_matter",
        "cosmic_strings",
        "neural_quantum_field",
        "tachyonic_communication",
        "quantum_vacuum",
        "holographic_principle",
        "dark_matter_dynamics",
        "supersymmetry_active"
    ],
    "defaults": {
        "goal_radius_rel": 1e-6,
        "kb_age_half_life_sec": 10800,
        "debate_max_turns": 20,
        "cohesion_target": 0.6
    }
}

# --- Validation Schemas ---
_LAW_SCHEMA = {
    "version": str,
    "name": str,
    "description": str,
    "constants": dict,
    "enhancement_index_order": list,
    "defaults": dict
}

# --- MOD Schema with new extensions ---
_MOD_SCHEMA = {
    "id": str,
    "name": str,
    "order": int,
    "version": str,
    "affects": list,
    "patch": dict,
    "symbolic_gloss": str # New: a human-readable description for the UI/log
}

_MOD_AFFECTS_VALUES = {"physics", "debate", "kb", "ui", "emotional", "symbolic", "paradox_rules"}

_EMOTIONAL_KEYS = {"fear_weight", "curiosity_weight", "joy_weight"}
_SYMBOLIC_KEYS = {"sigil_dictionary", "archetypal_modes"}
_PARADOX_KEYS = {"paradox_flags", "recursion_depth_limit"}

class _JSONDecodeError(ValueError):
    """Custom error for JSON decoding issues with file context."""
    pass

def _parse_json_file(filepath: str) -> Dict[str, Any]:
    """
    Helper to load a JSON file with a file-aware error message.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        Dict[str, Any]: The loaded dictionary.

    Raises:
        _JSONDecodeError: If the JSON is malformed.
        FileNotFoundError: If the file does not exist.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise _JSONDecodeError(f"Error decoding JSON in {filepath} at line {e.lineno}, column {e.colno}: {e.msg}")

def validate_law_schema(obj: Dict[str, Any]) -> None:
    """
    Validates a single law file against the canonical schema.

    Args:
        obj (Dict[str, Any]): The dictionary loaded from a law JSON file.

    Raises:
        ValueError: If the object does not match the schema.
    """
    if not isinstance(obj, dict):
        raise ValueError("Law file must be a JSON object.")

    # Validate top-level keys and types
    for key, expected_type in _LAW_SCHEMA.items():
        if key not in obj:
            raise ValueError(f"Missing required top-level key: '{key}'")
        if not isinstance(obj[key], expected_type):
            raise ValueError(f"Key '{key}' must be of type {expected_type.__name__}, got {type(obj[key]).__name__}")

    # Check for unknown top-level keys
    extra_keys = set(obj.keys()) - set(_LAW_SCHEMA.keys())
    if extra_keys:
        raise ValueError(f"Unknown top-level keys: {list(extra_keys)}")

    # Validate enhancement list
    enhancements = obj['enhancement_index_order']
    if not enhancements or not all(isinstance(item, str) for item in enhancements):
        raise ValueError("enhancement_index_order must be a non-empty list of strings.")
    if len(set(enhancements)) != len(enhancements):
        raise ValueError("All enhancement names in enhancement_index_order must be unique.")

    # Validate nested 'constants' and 'defaults' dictionaries
    for key, expected_type in DEFAULT_LAW['constants'].items():
        if key not in obj['constants'] or not isinstance(obj['constants'][key], expected_type):
            raise ValueError(f"constants['{key}'] must be of type {expected_type.__name__}.")
    for key, expected_type in DEFAULT_LAW['defaults'].items():
        if key not in obj['defaults'] or not isinstance(obj['defaults'][key], expected_type):
            raise ValueError(f"defaults['{key}'] must be of type {expected_type.__name__}.")

def validate_mod_schema(obj: Dict[str, Any], law_enhancements: List[str]) -> None:
    """
    Validates a single mod file against the canonical schema, including new emotional and symbolic rules.

    Args:
        obj (Dict[str, Any]): The dictionary loaded from a mod JSON file.
        law_enhancements (List[str]): The list of enhancement names from the merged law.

    Raises:
        ValueError: If the object does not match the schema.
    """
    if not isinstance(obj, dict):
        raise ValueError("Mod file must be a JSON object.")

    # Validate top-level keys and types
    for key, expected_type in _MOD_SCHEMA.items():
        if key not in obj:
            raise ValueError(f"Missing required top-level key: '{key}'")
        if not isinstance(obj[key], expected_type):
            raise ValueError(f"Key '{key}' must be of type {expected_type.__name__}, got {type(obj[key]).__name__}")

    # Check for unknown top-level keys
    extra_keys = set(obj.keys()) - set(_MOD_SCHEMA.keys())
    if extra_keys:
        raise ValueError(f"Unknown top-level keys: {list(extra_keys)}")

    # Validate 'affects' list
    if not all(k in _MOD_AFFECTS_VALUES for k in obj['affects']):
        raise ValueError(f"affects list contains invalid values. Must be a subset of {list(_MOD_AFFECTS_VALUES)}.")

    # Validate 'patch' keys
    patch = obj.get('patch', {})
    if not isinstance(patch, dict):
        raise ValueError("Patch must be a dictionary.")

    # Validate nested patch sections based on 'affects'
    for affect in obj['affects']:
        if affect not in patch:
            raise ValueError(f"Mod affects '{affect}' but has no corresponding key in the 'patch' dictionary.")

    # Validate 'physics' overrides
    overrides = patch.get('physics', {}).get('enhancement_index_overrides', {})
    if not isinstance(overrides, dict):
        raise ValueError("physics.enhancement_index_overrides must be a dictionary.")
    seen_indices = set()
    for name, index in overrides.items():
        if name not in law_enhancements:
            raise ValueError(f"enhancement_index_overrides key '{name}' not found in law's enhancement order.")
        if not isinstance(index, int) or index < 0 or index >= len(law_enhancements):
            raise ValueError(f"enhancement_index_overrides value for '{name}' must be a non-negative integer within enhancement list bounds [0, {len(law_enhancements)-1}].")
        if index in seen_indices:
            raise ValueError(f"enhancement_index_overrides has a collision at index {index}.")
        seen_indices.add(index)

    # New: Validate emotional, symbolic, and paradox_rules patches
    if "emotional" in obj["affects"]:
        emotional_patch = patch.get("emotional", {})
        if not all(k in _EMOTIONAL_KEYS for k in emotional_patch):
            raise ValueError(f"Invalid keys in emotional patch. Must be a subset of {_EMOTIONAL_KEYS}.")

    if "symbolic" in obj["affects"]:
        symbolic_patch = patch.get("symbolic", {})
        if not all(k in _SYMBOLIC_KEYS for k in symbolic_patch):
            raise ValueError(f"Invalid keys in symbolic patch. Must be a subset of {_SYMBOLIC_KEYS}.")

    if "paradox_rules" in obj["affects"]:
        paradox_patch = patch.get("paradox_rules", {})
        if not all(k in _PARADOX_KEYS for k in paradox_patch):
            raise ValueError(f"Invalid keys in paradox_rules patch. Must be a subset of {_PARADOX_KEYS}.")
        if "paradox_flags" in paradox_patch and not isinstance(paradox_patch["paradox_flags"], dict):
            raise ValueError("paradox_flags must be a dictionary.")
        if "recursion_depth_limit" in paradox_patch and not isinstance(paradox_patch["recursion_depth_limit"], int):
            raise ValueError("recursion_depth_limit must be an integer.")

def ensure_project_dirs(root_path: str) -> Dict[str, str]:
    """
    Ensures that the required subdirectories for the simulation exist.

    Args:
        root_path (str): The root directory for the project.

    Returns:
        Dict[str, str]: A dictionary of the absolute paths to the mods, agents, and laws directories.
    """
    dirs = {
        "mods": os.path.join(root_path, "mods"),
        "agents": os.path.join(root_path, "agents"),
        "laws": os.path.join(root_path, "laws")
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs

def _deep_merge_dicts(d1: Dict, d2: Dict) -> Dict:
    """Deep merges d2 into d1, returning a new dictionary."""
    d = copy.deepcopy(d1)
    for k, v in d2.items():
        if k in d and isinstance(d[k], dict) and isinstance(v, dict):
            d[k] = _deep_merge_dicts(d[k], v)
        else:
            d[k] = v
    return d

def load_laws(root_path: str) -> Dict[str, Any]:
    """
    Loads and merges all law files from the laws directory.

    Args:
        root_path (str): The root directory of the project.

    Returns:
        Dict[str, Any]: A single dictionary representing the merged law configuration.

    Raises:
        ValueError: If a law file is malformed or invalid.
    """
    laws_dir = os.path.join(root_path, "laws")
    law_files = sorted(glob.glob(os.path.join(laws_dir, "*.json")), key=os.path.basename)

    if not law_files:
        return copy.deepcopy(DEFAULT_LAW)

    merged_law = {}
    for filepath in law_files:
        try:
            law = _parse_json_file(filepath)
            validate_law_schema(law)
            if not merged_law:
                merged_law = law
            else:
                merged_law = _deep_merge_dicts(merged_law, law)
        except (_JSONDecodeError, ValueError) as e:
            raise ValueError(f"Invalid law file: {filepath}\n{e}")

    # As per rule, the enhancement order from the last law file wins exactly.
    last_law = _parse_json_file(law_files[-1])
    merged_law['enhancement_index_order'] = last_law['enhancement_index_order']

    return merged_law

def load_mods(root_path: str) -> List[Dict[str, Any]]:
    """
    Loads, validates, and sorts all mods from the mods directory.

    Args:
        root_path (str): The root directory of the project.

    Returns:
        List[Dict[str, Any]]: A list of validated mod dictionaries, sorted by 'order' and filename.

    Raises:
        ValueError: If a mod file is malformed or invalid, or has duplicate keys.
    """
    mods_dir = os.path.join(root_path, "mods")
    mod_files = sorted(glob.glob(os.path.join(mods_dir, "*.json")), key=os.path.basename)

    mods = []
    for filepath in mod_files:
        try:
            mod = _parse_json_file(filepath)
            mods.append(mod)
        except (_JSONDecodeError, ValueError) as e:
            raise ValueError(f"Invalid mod file: {filepath}\n{e}")

    mods.sort(key=lambda m: (m.get('order', 0), m.get('id')))

    # Check for duplicate (order, id) pairs
    seen_keys = set()
    for mod in mods:
        key = (mod.get('order', 0), mod.get('id'))
        if key in seen_keys:
            raise ValueError(f"Duplicate mod key found: {key}. Please ensure each mod has a unique (order, id) pair.")
        seen_keys.add(key)

    return mods

def apply_laws_to_config(laws: Dict[str, Any],
                         base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies constants and defaults from the laws to a base configuration dictionary.

    Args:
        laws (Dict[str, Any]): The merged law dictionary.
        base_cfg (Dict[str, Any]): The base configuration (e.g., from config.json).

    Returns:
        Dict[str, Any]: A new configuration dictionary with law-based values merged in.
    """
    config = _deep_merge_dicts(base_cfg, {}) # Create a deep copy
    config['physics_params'] = _deep_merge_dicts(config.get('physics_params', {}), laws['constants'])
    config['agent_params'] = _deep_merge_dicts(config.get('agent_params', {}), laws['defaults'])
    config['enhancement_index_order'] = laws['enhancement_index_order']
    return config

def apply_mods_to_state(mods: List[Dict[str, Any]], universe: Any) -> None:
    """
    Applies patches from a list of mod files to the universe object.

    This function is idempotent and safe to call multiple times. Unknown patch keys
    will be ignored with a warning.

    Args:
        mods (List[Dict[str, Any]]): A list of sorted mod dictionaries.
        universe (Any): The universe object to patch.
    """
    for mod in mods:
        patch = mod.get('patch', {})
        affects = mod.get('affects', [])
        symbolic_gloss = mod.get('symbolic_gloss', f"Mod '{mod['name']}' applied.")

        # Log with symbolic gloss
        print(f"Applying mod: '{mod['name']}' // {symbolic_gloss}")

        if 'physics' in affects:
            if not hasattr(universe, 'physics_overrides'):
                universe.physics_overrides = {}
            for key, value in patch.get('physics', {}).items():
                # Added new keys here as well
                if key not in ["softening_eps_rel", "max_accel_rel", "enhancement_index_overrides", "time_dilation_factor"]:
                    print(f"Warning: Mod '{mod['id']}' has unknown physics key '{key}'. Ignoring.")
                    continue
                universe.physics_overrides[key] = value

        if 'kb' in affects:
            if not hasattr(universe, 'kb_overrides'):
                universe.kb_overrides = {}
            for key, value in patch.get('kb', {}).items():
                if key not in ["age_half_life_sec", "top_k", "min_display_score", "dedup_normalize"]:
                    print(f"Warning: Mod '{mod['id']}' has unknown KB key '{key}'. Ignoring.")
                    continue
                universe.kb_overrides[key] = value

        if 'debate' in affects:
            if not hasattr(universe, 'debate_overrides'):
                universe.debate_overrides = {}
            for key, value in patch.get('debate', {}).items():
                # Added new keys here as well
                if key not in ["render_top_n", "repeat_penalty", "diversity_bonus", "role_bias", "debate_max_turns"]:
                    print(f"Warning: Mod '{mod['id']}' has unknown debate key '{key}'. Ignoring.")
                    continue
                universe.debate_overrides[key] = value

        if 'ui' in affects:
            if not hasattr(universe, 'ui_opts'):
                universe.ui_opts = {}
            for key, value in patch.get('ui', {}).items():
                if key not in ["hud_enhancements_compact", "show_emotional_state"]:
                    print(f"Warning: Mod '{mod['id']}' has unknown UI key '{key}'. Ignoring.")
                    continue
                universe.ui_opts[key] = value

        # New: apply emotional overrides
        if 'emotional' in affects:
            if not hasattr(universe, 'emotional_overrides'):
                universe.emotional_overrides = {}
            for key, value in patch.get('emotional', {}).items():
                if key not in _EMOTIONAL_KEYS:
                     print(f"Warning: Mod '{mod['id']}' has unknown emotional key '{key}'. Ignoring.")
                     continue
                universe.emotional_overrides[key] = value

        # New: apply symbolic overrides
        if 'symbolic' in affects:
            if not hasattr(universe, 'symbolic_overrides'):
                universe.symbolic_overrides = {}
            for key, value in patch.get('symbolic', {}).items():
                if key not in _SYMBOLIC_KEYS:
                     print(f"Warning: Mod '{mod['id']}' has unknown symbolic key '{key}'. Ignoring.")
                     continue
                universe.symbolic_overrides[key] = value

        # New: apply paradox rules
        if 'paradox_rules' in affects:
            if not hasattr(universe, 'paradox_rules_overrides'):
                universe.paradox_rules_overrides = {}
            for key, value in patch.get('paradox_rules', {}).items():
                if key not in _PARADOX_KEYS:
                     print(f"Warning: Mod '{mod['id']}' has unknown paradox_rules key '{key}'. Ignoring.")
                     continue
                universe.paradox_rules_overrides[key] = value


def active_feature_index_map(laws: Dict[str, Any],
                             mod_overrides: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """
    Generates a stable, deterministic mapping of enhancement names to their
    fixed index, applying overrides if provided.

    Args:
        laws (Dict[str, Any]): The merged law dictionary.
        mod_overrides (Optional[Dict[str, int]]): An optional dictionary of enhancement names to new indices.

    Returns:
        Dict[str, int]: A dictionary mapping enhancement names to their final index.

    Raises:
        ValueError: If the overrides cause an index collision or invalid index.
    """
    enhancement_order = laws['enhancement_index_order']
    index_map = {name: index for index, name in enumerate(enhancement_order)}

    if mod_overrides:
        final_map = copy.deepcopy(index_map)
        reverse_map = {v: k for k, v in final_map.items()}

        # Validate overrides first to catch all errors
        for name, new_index in mod_overrides.items():
            if name not in index_map:
                raise ValueError(f"Override key '{name}' not found in base enhancement order.")
            if not (0 <= new_index < len(enhancement_order)):
                raise ValueError(f"Invalid index {new_index} for '{name}'. Must be in range [0, {len(enhancement_order) - 1}].")

        # Check for collisions after all overrides are validated
        temp_map = copy.deepcopy(index_map)
        for name, new_index in mod_overrides.items():
            if new_index in reverse_map and reverse_map[new_index] != name:
                # Collision detected with a different enhancement
                original_name = reverse_map[new_index]
                raise ValueError(f"Index collision for '{name}' at index {new_index}. It conflicts with the default position of '{original_name}'. Please choose an unoccupied index.")
            temp_map[name] = new_index

        # Apply the validated overrides
        for name, new_index in mod_overrides.items():
            final_map[name] = new_index

        return final_map

    return index_map

if __name__ == '__main__':
    # Simple self-test to demonstrate functionality

    # 1. Test directory creation
    print("--- Testing directory creation ---")
    test_root = "test_project"
    dirs = ensure_project_dirs(test_root)
    print(f"Created directories: {dirs}")
    assert os.path.exists(dirs['mods'])
    assert os.path.exists(dirs['agents'])
    assert os.path.exists(dirs['laws'])

    # Clean up
    for path in dirs.values():
      os.rmdir(path)
    os.rmdir(test_root)

    # 2. Test law loading
    print("\n--- Testing law loading ---")

    class MockUniverse:
        def __init__(self):
            self.physics_overrides = {}
            self.kb_overrides = {}
            self.debate_overrides = {}
            self.ui_opts = {}
            self.emotional_overrides = {}
            self.symbolic_overrides = {}
            self.paradox_rules_overrides = {}

    # Create mock files
    os.makedirs(os.path.join(test_root, 'laws'), exist_ok=True)
    with open(os.path.join(test_root, 'laws', 'a_law_base.json'), 'w') as f:
        json.dump({"version": "1.0", "name": "Base", "description": "desc", "constants": {"c": 1.0}, "enhancement_index_order": ["a", "b"], "defaults": {"goal_radius_rel": 0.1}}, f)
    with open(os.path.join(test_root, 'laws', 'b_law_patch.json'), 'w') as f:
        json.dump({"version": "1.1", "name": "Patch", "description": "desc", "constants": {"c": 2.0}, "enhancement_index_order": ["b", "a"], "defaults": {"goal_radius_rel": 0.2}}, f)

    laws = load_laws(test_root)
    print(f"Merged Laws: {json.dumps(laws, indent=2)}")
    assert laws['constants']['c'] == 2.0
    assert laws['enhancement_index_order'] == ["b", "a"]

    # 3. Test mod loading and sorting
    print("\n--- Testing mod loading and sorting ---")
    os.makedirs(os.path.join(test_root, 'mods'), exist_ok=True)
    with open(os.path.join(test_root, 'mods', 'mod_b.json'), 'w') as f:
        json.dump({"id": "mod_b", "name": "B", "order": 1, "version": "1.0", "affects": [], "patch": {}, "symbolic_gloss": "The second mod enters the fray."}, f)
    with open(os.path.join(test_root, 'mods', 'mod_a.json'), 'w') as f:
        json.dump({"id": "mod_a", "name": "A", "order": 0, "version": "1.0", "affects": [], "patch": {}, "symbolic_gloss": "A foundation is laid."}, f)
    # New: Emotional mod
    with open(os.path.join(test_root, 'mods', 'mod_emotional.json'), 'w') as f:
        json.dump({"id": "mod_emo", "name": "Emotional", "order": 2, "version": "1.0", "affects": ["emotional"], "patch": {"emotional": {"fear_weight": 0.8}}, "symbolic_gloss": "The heart of the machine learns to fear."}, f)
    # New: Symbolic mod
    with open(os.path.join(test_root, 'mods', 'mod_symbolic.json'), 'w') as f:
        json.dump({"id": "mod_sym", "name": "Symbolic", "order": 3, "version": "1.0", "affects": ["symbolic"], "patch": {"symbolic": {"sigil_dictionary": {"a": "Alpha"}}}, "symbolic_gloss": "New symbols manifest in the digital aether."}, f)
    # New: Paradox mod
    with open(os.path.join(test_root, 'mods', 'mod_paradox.json'), 'w') as f:
        json.dump({"id": "mod_par", "name": "Paradox", "order": 4, "version": "1.0", "affects": ["paradox_rules"], "patch": {"paradox_rules": {"recursion_depth_limit": 10}}, "symbolic_gloss": "The system learns to tolerate recursion."}, f)
    # Test collision
    with open(os.path.join(test_root, 'mods', 'mod_collision.json'), 'w') as f:
        json.dump({"id": "mod_col", "name": "Collision", "order": 5, "version": "1.0", "affects": ["physics"], "patch": {"physics": {"enhancement_index_overrides": {"a": 0}}}, "symbolic_gloss": "A clash of futures."}, f)

    mods = load_mods(test_root)
    print(f"Sorted Mods: {[m['id'] for m in mods]}")
    assert mods[0]['id'] == 'mod_a'
    assert mods[1]['id'] == 'mod_b'
    assert mods[2]['id'] == 'mod_emo'
    assert mods[3]['id'] == 'mod_sym'

    # 4. Test API applications and idempotence
    print("\n--- Testing API application and idempotence ---")
    universe = MockUniverse()
    test_mod = {"id": "test", "name": "Test Mod", "order": 0, "version": "1.0", "affects": ["ui", "physics"], "patch": {"ui": {"hud_enhancements_compact": True}, "physics": {"max_accel_rel": 0.5}}, "symbolic_gloss": "Test mod engaged."}
    apply_mods_to_state([test_mod], universe)
    print(f"Universe UI opts: {universe.ui_opts}")
    assert universe.ui_opts['hud_enhancements_compact'] is True

    # Test idempotence
    apply_mods_to_state([test_mod], universe)
    print(f"Universe UI opts (after repeat call): {universe.ui_opts}")
    assert universe.ui_opts['hud_enhancements_compact'] is True

    # Test unknown key warning
    bad_mod = {"id": "bad", "name": "Bad Mod", "order": 0, "version": "1.0", "affects": ["ui"], "patch": {"ui": {"unknown_key": "val"}}, "symbolic_gloss": "A glitch in the matrix."}
    print("\n--- Testing unknown key warning (should see a warning below) ---")
    apply_mods_to_state([bad_mod], universe)

    # 5. Test active feature index map
    print("\n--- Testing active feature index map ---")
    test_laws = {"enhancement_index_order": ["a", "b", "c", "d"]}
    overrides = {"c": 0, "a": 2}
    index_map = active_feature_index_map(test_laws, overrides)
    print(f"Final Index Map: {index_map}")
    assert index_map['c'] == 0
    assert index_map['b'] == 1
    assert index_map['a'] == 2
    assert index_map['d'] == 3

    # Test collision detection
    print("\n--- Testing index collision detection (should see error below) ---")
    try:
        overrides_collision = {"c": 1, "b": 1}
        active_feature_index_map(test_laws, overrides_collision)
    except ValueError as e:
        print(f"Caught expected error: {e}")
        assert "collision" in str(e)

    # Final cleanup
    import shutil
    shutil.rmtree(test_root)
    print("\n--- All self-tests passed. ---")
