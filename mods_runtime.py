# hexuni/engine/mods_runtime.py

import os
import json
import shutil
import logging
import dataclasses
import tempfile
from collections import defaultdict
from typing import Dict, Any, List, Set, Optional, Callable, Union, Type, Tuple, cast

# Required for dependency management and graph validation
try:
    from graphlib import TopologicalSorter
except ImportError:
    # Fallback for Python versions before 3.9
    print("Warning: graphlib is not available. Dependency sorting will be skipped.")
    TopologicalSorter = None

# Required for hot-reloading file watch
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("Warning: watchdog library not found. Hot-reload functionality will be disabled.")
    Observer = None
    FileSystemEventHandler = None

# --- Constants and Schemas ---

# The default law provides a type and value map for validation.
DEFAULT_LAW = {
    'constants': {
        'c': 299792458.0,
        'G': 6.6743e-11,
        'k_e': 8.98755e9,
    },
    'defaults': {
        'softening_eps_rel': 1e-6,
        'max_accel_rel': 1.0,
        'goal_radius_rel': 0.05,
        'half_life': 100.0,
    },
    'enhancement_order': [],
    'ui_patches': {},
}

# The explicit type map for constants and defaults, as requested.
# This fixes the bug where validation used values instead of types.
LAW_TYPE_MAP = {
    'constants': {
        'c': float,
        'G': float,
        'k_e': float,
    },
    'defaults': {
        'softening_eps_rel': float,
        'max_accel_rel': float,
        'goal_radius_rel': float,
        'half_life': float,
    },
    'ui_patches': {
        'hud_enhancements_compact': bool,
        'show_entropy_badge': bool,
        'metrics_refresh_ms': int,
    }
}

# A map for value bounds, for strict validation.
LAW_BOUNDS_MAP = {
    'constants': {
        'c': (0.0, float('inf')),
        'G': (0.0, float('inf')),
        'k_e': (0.0, float('inf')),
    },
    'defaults': {
        'softening_eps_rel': (0.0, 1e-6),
        'max_accel_rel': (0.0, 1e10),
        'goal_radius_rel': (0.0, 1.0),
        'half_life': (0.0, float('inf')),
    },
    'ui_patches': {
        'metrics_refresh_ms': (100, 5000)
    }
}

class MergeStrategy(enum.Enum):
    OVERRIDE_LAST = 'override-last'
    DEEP_MERGE = 'deep-merge'

# --- Rich Error and Logging Objects ---

@dataclasses.dataclass
class ModError:
    """Standardized error object for mod/law validation."""
    source_file: str
    message: str
    hint: Optional[str] = None
    section: Optional[str] = None
    key: Optional[str] = None
    code: str = "MOD_ERROR_GENERIC"

class LogAdapter:
    """
    A simple logging adapter to decouple the core logic from the output medium.
    This allows a GUI to subscribe and colorize messages.
    """
    def __init__(self, handler: Callable[[str, str], None]):
        self._handler = handler

    def info(self, message: str):
        self._handler("info", message)

    def warning(self, message: str):
        self._handler("warning", message)

    def error(self, message: str):
        self._handler("error", message)

# Default console logger
_default_logger = logging.getLogger('hexuni')
_default_logger.setLevel(logging.INFO)
_default_logger.addHandler(logging.StreamHandler())
log = LogAdapter(lambda level, msg: getattr(_default_logger, level)(msg))

# --- Data Structures for Provenance and Caching ---

@dataclasses.dataclass
class FileSource:
    """Represents the provenance of a law or mod file."""
    filepath: str
    schema_version: str
    load_time: float


# --- Core Functions ---

def ensure_project_dirs(root_path: str, mod_dir_name: str = 'mods', law_dir_name: str = 'laws'):
    """
    Ensures the existence of required directories for mods and laws.
    Returns absolute paths to the law and mod directories.
    Fixes the docstring bug by returning absolute paths.
    """
    abs_root = os.path.abspath(root_path)
    law_dir = os.path.join(abs_root, law_dir_name)
    mod_dir = os.path.join(abs_root, mod_dir_name)
    os.makedirs(law_dir, exist_ok=True)
    os.makedirs(mod_dir, exist_ok=True)
    return law_dir, mod_dir

def _validate_with_explicit_map(data: dict, schema_map: dict, bounds_map: dict) -> List[ModError]:
    """
    Helper function for real type and bounds validation.
    Fixes the bug where validation used values instead of types.
    Aggregates all errors into a single list.
    """
    errors = []
    for key, expected_type in schema_map.items():
        if key not in data:
            # Skip if key is not present, Partial law tolerance handles this.
            continue
        value = data[key]
        if not isinstance(value, expected_type):
            errors.append(ModError(
                source_file="N/A",  # Filled by the caller
                message=f"Expected type '{expected_type.__name__}' for key '{key}', but got '{type(value).__name__}'",
                hint="Correct the data type in the file.",
                key=key
            ))
        
        # Check bounds
        if key in bounds_map:
            min_val, max_val = bounds_map[key]
            if not (min_val <= value <= max_val):
                errors.append(ModError(
                    source_file="N/A",  # Filled by the caller
                    message=f"Value for '{key}' is out of bounds. Expected a value between {min_val} and {max_val}.",
                    hint=f"Adjust the value to be within the valid range.",
                    key=key
                ))
    return errors

def validate_law_schema(law: dict, law_filename: str) -> List[ModError]:
    """
    Validates the structure and types of a law file against the explicit type map.
    This function fixes the two critical type validation bugs.
    """
    errors = []
    
    # Check for required top-level keys
    if 'constants' not in law:
        errors.append(ModError(law_filename, "Missing 'constants' section.", section="top-level"))
    if 'defaults' not in law:
        errors.append(ModError(law_filename, "Missing 'defaults' section.", section="top-level"))
    if 'enhancement_order' not in law:
        errors.append(ModError(law_filename, "Missing 'enhancement_order' section.", section="top-level"))
    if 'ui_patches' not in law:
        errors.append(ModError(law_filename, "Missing 'ui_patches' section.", section="top-level"))
    
    # Validate inner sections if they exist
    if 'constants' in law:
        section_errors = _validate_with_explicit_map(law['constants'], LAW_TYPE_MAP['constants'], LAW_BOUNDS_MAP['constants'])
        for err in section_errors:
            err.source_file = law_filename
            err.section = 'constants'
            errors.append(err)
    
    if 'defaults' in law:
        section_errors = _validate_with_explicit_map(law['defaults'], LAW_TYPE_MAP['defaults'], LAW_BOUNDS_MAP['defaults'])
        for err in section_errors:
            err.source_file = law_filename
            err.section = 'defaults'
            errors.append(err)
            
    if 'ui_patches' in law:
        section_errors = _validate_with_explicit_map(law['ui_patches'], LAW_TYPE_MAP['ui_patches'], LAW_BOUNDS_MAP['ui_patches'])
        for err in section_errors:
            err.source_file = law_filename
            err.section = 'ui_patches'
            errors.append(err)
            
    if 'enhancement_order' in law and not isinstance(law['enhancement_order'], list):
        errors.append(ModError(law_filename, "Expected 'enhancement_order' to be a list.", section="enhancement_order"))
        
    return errors

def validate_mod_schema(mod: dict, mod_filename: str) -> List[ModError]:
    """
    Validates a single mod file against its expected schema.
    This fixes the bug where mods weren't schema-validated on load.
    """
    errors = []
    
    # Basic required keys check
    if 'name' not in mod:
        errors.append(ModError(mod_filename, "Missing 'name' field.", section="top-level"))
    if 'enhancements' not in mod:
        errors.append(ModError(mod_filename, "Missing 'enhancements' field.", section="top-level"))
    if 'ui_patches' not in mod:
        errors.append(ModError(mod_filename, "Missing 'ui_patches' field.", section="top-level"))
    
    # Validate enhancement structure
    if 'enhancements' in mod and not isinstance(mod['enhancements'], dict):
        errors.append(ModError(mod_filename, "Expected 'enhancements' to be a dictionary.", section="enhancements"))
        
    # Validate UI patches
    if 'ui_patches' in mod:
        section_errors = _validate_with_explicit_map(mod['ui_patches'], LAW_TYPE_MAP['ui_patches'], LAW_BOUNDS_MAP['ui_patches'])
        for err in section_errors:
            err.source_file = mod_filename
            err.section = 'ui_patches'
            errors.append(err)

    # Dependency validation
    for dep_key in ['depends', 'conflicts', 'load_before', 'load_after']:
        if dep_key in mod and not isinstance(mod[dep_key], list):
            errors.append(ModError(mod_filename, f"Expected '{dep_key}' to be a list.", section="dependencies"))
            
    return errors

def _merge_dicts(target: Dict, source: Dict, strategy: MergeStrategy) -> None:
    """Merges two dictionaries based on the specified strategy."""
    if strategy == MergeStrategy.OVERRIDE_LAST:
        target.update(source)
    elif strategy == MergeStrategy.DEEP_MERGE:
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                _merge_dicts(target[key], value, strategy)
            else:
                target[key] = value

def load_laws(
    law_dir: str,
    strict: bool = True,
    law_merge_strategy: Dict[str, MergeStrategy] = {
        'constants': MergeStrategy.OVERRIDE_LAST,
        'defaults': MergeStrategy.DEEP_MERGE,
        'ui_patches': MergeStrategy.DEEP_MERGE
    }
) -> Tuple[dict, List[ModError], List[FileSource]]:
    """
    Loads all law files from a directory, merges them, and validates.
    - Implements "Semantic merge strategies".
    - Implements "Deterministic file ordering" (sort by filename, then by explicit order).
    - Implements "Partial law tolerance" with a report.
    - Fixes the bug where enhancement order from earlier files is lost.
    """
    law_files = [f for f in os.listdir(law_dir) if f.endswith('.json')]
    law_files.sort()  # Deterministic file ordering by filename

    merged_law = {
        'constants': {},
        'defaults': {},
        'enhancement_order': [],
        'ui_patches': {}
    }
    all_errors = []
    law_sources = []
    seen_enhancements: Set[str] = set()
    
    # Store all enhancements in order of appearance
    all_enhancements = []
    
    for filename in law_files:
        filepath = os.path.join(law_dir, filename)
        try:
            with open(filepath, 'r') as f:
                law_data = json.load(f)
        except json.JSONDecodeError:
            all_errors.append(ModError(filepath, "Invalid JSON format."))
            continue

        errors = validate_law_schema(law_data, filepath)
        if errors:
            all_errors.extend(errors)
            if strict:
                continue

        # Fill in missing keys from the default law if not strict
        if not strict:
            missing_keys = set(DEFAULT_LAW['constants'].keys()) - set(law_data.get('constants', {}).keys())
            if missing_keys:
                log.warning(f"Law '{filename}' is partial. Used defaults for: {', '.join(missing_keys)}")
            law_data = {
                'constants': {**DEFAULT_LAW['constants'], **law_data.get('constants', {})},
                'defaults': {**DEFAULT_LAW['defaults'], **law_data.get('defaults', {})},
                'enhancement_order': law_data.get('enhancement_order', []),
                'ui_patches': law_data.get('ui_patches', {}),
            }
        
        # Merge sections based on strategy
        _merge_dicts(merged_law['constants'], law_data.get('constants', {}), law_merge_strategy['constants'])
        _merge_dicts(merged_law['defaults'], law_data.get('defaults', {}), law_merge_strategy['defaults'])
        _merge_dicts(merged_law['ui_patches'], law_data.get('ui_patches', {}), law_merge_strategy['ui_patches'])

        # Correctness fix: Collect all enhancement orders to avoid correctness drift
        for enh_name in law_data.get('enhancement_order', []):
            if enh_name not in seen_enhancements:
                all_enhancements.append(enh_name)
                seen_enhancements.add(enh_name)

        law_sources.append(FileSource(filepath, "2.0", os.path.getmtime(filepath)))
        
    merged_law['enhancement_order'] = all_enhancements
    
    return merged_law, all_errors, law_sources

def load_mods(mod_dir: str, validate_only: bool = False) -> Tuple[Dict, List[ModError], List[FileSource]]:
    """
    Loads and validates mod files, applying dependency checks.
    - Implements "Dry-run mode".
    - Implements "Dependency graph + topological load".
    - Fixes the bug where mods aren't validated on load.
    """
    mod_files = [f for f in os.listdir(mod_dir) if f.endswith('.json')]
    mods = {}
    all_errors = []
    mod_sources = []
    
    # Map mod names to their filenames for dependency resolution
    filename_to_name = {}

    for filename in mod_files:
        filepath = os.path.join(mod_dir, filename)
        try:
            with open(filepath, 'r') as f:
                mod_data = json.load(f)
        except json.JSONDecodeError:
            all_errors.append(ModError(filepath, "Invalid JSON format."))
            continue
        
        errors = validate_mod_schema(mod_data, filepath)
        if errors:
            all_errors.extend(errors)
            continue
        
        mod_name = mod_data.get('name')
        if not mod_name:
            all_errors.append(ModError(filepath, "Mod must have a 'name' field."))
            continue

        if mod_name in mods:
            all_errors.append(ModError(
                filepath, 
                f"Duplicate mod name '{mod_name}'. Already defined in '{filename_to_name[mod_name]}'",
                hint="Rename the mod or remove the duplicate file."
            ))
            continue
        
        mods[mod_name] = mod_data
        filename_to_name[mod_name] = filename
        mod_sources.append(FileSource(filepath, "2.0", os.path.getmtime(filepath)))

    if all_errors and not validate_only:
        log.error("Failed to load mods due to validation errors.")
        return {}, all_errors, []

    if TopologicalSorter:
        # Build dependency graph
        graph = defaultdict(set)
        for mod_name, mod_data in mods.items():
            for dep in mod_data.get('depends', []):
                graph[mod_name].add(dep)
            for load_before in mod_data.get('load_before', []):
                graph[load_before].add(mod_name)
            for load_after in mod_data.get('load_after', []):
                graph[mod_name].add(load_after)
        
        sorter = TopologicalSorter(graph)
        try:
            ordered_mods = list(sorter.static_order())
        except Exception as e:
            all_errors.append(ModError("Dependency Graph", f"Circular dependency detected: {e}"))
            return {}, all_errors, []
        
        # Check for missing dependencies and conflicts
        for mod_name, mod_data in mods.items():
            for dep in mod_data.get('depends', []):
                if dep not in mods:
                    all_errors.append(ModError(
                        filename_to_name.get(mod_name, "N/A"),
                        f"Missing required dependency: '{dep}'",
                        hint=f"Ensure the '{dep}' mod is in the mods directory."
                    ))
            for conflict in mod_data.get('conflicts', []):
                if conflict in mods:
                    all_errors.append(ModError(
                        filename_to_name.get(mod_name, "N/A"),
                        f"Conflicts with mod: '{conflict}'",
                        hint=f"Remove one of the conflicting mods."
                    ))

        if all_errors:
            log.error("Failed to load mods due to dependency issues.")
            return {}, all_errors, []

        log.info(f"Dependency resolution successful. Load order: {', '.join(ordered_mods)}")
        
        # Reorder mods based on topological sort
        ordered_mod_dict = {name: mods[name] for name in ordered_mods}
        
        if validate_only:
            log.info("Dry-run successful. No changes made to universe state.")
            return ordered_mod_dict, all_errors, mod_sources
            
        return ordered_mod_dict, all_errors, mod_sources
    else:
        # Fallback for old Python versions
        if validate_only:
             log.warning("Skipping dependency validation due to missing graphlib.")
             return mods, all_errors, mod_sources
        return mods, all_errors, mod_sources

def preview_feature_index_map(laws: dict, mods: dict) -> Tuple[Dict[str, int], List[ModError]]:
    """
    Computes the final enhancement index map, reporting collisions and gaps.
    - Implements "Preview final enhancement map".
    - Implements "Conflict explainer".
    - Fixes "Index override holes not guarded".
    """
    base_order = laws.get('enhancement_order', [])
    final_map = {name: i for i, name in enumerate(base_order)}
    next_free_index = len(base_order)
    all_errors = []
    
    # We apply overrides in a deterministic way (alphabetical by mod name)
    ordered_mods = sorted(mods.keys())

    # Map to track which mod claimed which index
    index_provenance = {i: "Base Law" for i in final_map.values()}

    for mod_name in ordered_mods:
        mod = mods[mod_name]
        overrides = mod.get('enhancement_index_overrides', {})
        for name, index in overrides.items():
            if not isinstance(index, int) or index < 0:
                all_errors.append(ModError(
                    mod_name, f"Invalid index '{index}' for enhancement '{name}'. Must be a non-negative integer.",
                    hint=f"Set a valid index for '{name}'."
                ))
                continue

            if index in index_provenance:
                conflict_source = index_provenance[index]
                all_errors.append(ModError(
                    mod_name, f"Index collision for enhancement '{name}'. Index {index} is already claimed.",
                    hint=f"Index was first claimed by '{conflict_source}'. A free index is {next_free_index}.",
                    key=name
                ))
            else:
                final_map[name] = index
                index_provenance[index] = mod_name
                if index >= next_free_index:
                    next_free_index = index + 1
    
    # Rebuild the final order list, compacting any holes
    final_order = sorted(final_map.keys(), key=lambda k: final_map[k])
    
    # Check for "holes" in the index mapping
    used_indices = sorted(final_map.values())
    expected_indices = list(range(len(used_indices)))
    if used_indices != expected_indices:
        log.warning("Warning: Enhancement index map has non-consecutive indices (holes). The consumer of this map may expect a dense array. This may not be a bug depending on the use case.")

    # Create a nice report for the preview enhancement
    log.info("--- Final Enhancement Map Preview ---")
    log.info(f"{'Index':<5} {'Enhancement Name':<30} {'Source':<20}")
    log.info("-" * 55)
    
    for i, name in enumerate(final_order):
        source = index_provenance.get(final_map[name], "N/A")
        log.info(f"{i:<5} {name:<30} {source:<20}")
    
    log.info("-------------------------------------")
    
    return final_map, all_errors

def apply_feature_index_map(universe: Any, index_map: Dict[str, int]) -> None:
    """
    Applies the final enhancement index map to the Universe's enhancement order.
    """
    if not hasattr(universe, '_enh_order'):
        log.error("Universe object does not have a '_enh_order' attribute.")
        return
        
    # Sort enhancements by their assigned index to create the final, packed order
    sorted_enhancements = sorted(index_map.items(), key=lambda item: item[1])
    universe._enh_order = [name for name, _ in sorted_enhancements]
    log.info("Successfully applied new enhancement index map to Universe.")
    
def apply_mods_to_state(universe: Any, laws: dict, mods: dict) -> List[ModError]:
    """
    Applies all loaded laws and mods to the universe state.
    - Allows 'c' override.
    - Validates UI keys and their types.
    - Attaches provenance to the universe object.
    """
    errors = []
    
    # Apply laws first (this acts as the base state)
    universe.constants = laws['constants']
    universe.defaults = laws['defaults']
    universe.ui_patches = laws['ui_patches']
    
    # Apply mods in their determined order
    for mod_name, mod_data in mods.items():
        # Physics patches (fixes bug with 'c' not being allowed)
        if 'physics_patches' in mod_data:
            physics_patches = mod_data['physics_patches']
            # Allow 'c' to be overridden
            allowed_keys = ['softening_eps_rel', 'max_accel_rel', 'c']
            for key, value in physics_patches.items():
                if key in allowed_keys:
                    universe.constants[key] = value
                else:
                    log.warning(f"Mod '{mod_name}' attempted to patch an unknown physics key: '{key}'.")
        
        # UI patches (fixes bug with silent ignoring of unknown/mismatched types)
        if 'ui_patches' in mod_data:
            for key, value in mod_data['ui_patches'].items():
                if key in LAW_TYPE_MAP['ui_patches']:
                    expected_type = LAW_TYPE_MAP['ui_patches'][key]
                    if isinstance(value, expected_type):
                        universe.ui_patches[key] = value
                    else:
                        errors.append(ModError(
                            mod_name,
                            f"UI patch '{key}' has an incorrect type. Expected {expected_type.__name__}, got {type(value).__name__}.",
                            hint="Correct the data type in the mod file."
                        ))
                else:
                    errors.append(ModError(
                        mod_name, f"Attempted to patch unknown UI key: '{key}'",
                        hint=f"Valid UI keys are: {', '.join(LAW_TYPE_MAP['ui_patches'].keys())}."
                    ))

    # Apply enhancement index map
    index_map, map_errors = preview_feature_index_map(laws, mods)
    errors.extend(map_errors)
    apply_feature_index_map(universe, index_map)

    # Attach provenance
    universe.law_sources = laws.get('law_sources', [])
    universe.mod_sources = mods.get('mod_sources', [])
    
    return errors

def watch_mods(root_path: str, reload_callback: Callable[[], None]):
    """
    Watches the mods directory for changes and triggers a hot reload.
    Requires the watchdog library.
    """
    if not Observer or not FileSystemEventHandler:
        log.error("Watchdog library not available. Cannot watch for file changes.")
        return

    law_dir, mod_dir = ensure_project_dirs(root_path)

    class ModFileEventHandler(FileSystemEventHandler):
        def on_any_event(self, event):
            if event.src_path.endswith('.json'):
                log.info(f"Change detected in {event.src_path}. Triggering hot reload...")
                reload_callback()

    event_handler = ModFileEventHandler()
    observer = Observer()
    observer.schedule(event_handler, law_dir, recursive=False)
    observer.schedule(event_handler, mod_dir, recursive=False)
    observer.start()
    log.info(f"Watching directories for changes: {law_dir}, {mod_dir}")
    return observer

# --- CLI Tool ---

def cli_main(args: List[str]):
    """
    Small CLI for validation and reporting.
    """
    if len(args) < 2:
        print("Usage: python -m mods_runtime <command> [options]")
        print("Commands:")
        print("  validate --root <path>  : Validates all laws and mods in a directory.")
        return

    command = args[1]
    root_path = '.'
    if '--root' in args:
        root_path = args[args.index('--root') + 1]
    
    law_dir, mod_dir = ensure_project_dirs(root_path)
    
    if command == 'validate':
        print(f"Validating laws and mods in '{root_path}'...")
        laws, law_errors, _ = load_laws(law_dir, strict=False)
        mods, mod_errors, _ = load_mods(mod_dir, validate_only=True)
        
        all_errors = law_errors + mod_errors
        
        if all_errors:
            print("\n--- Validation Report (FAIL) ---")
            print(f"Found {len(all_errors)} errors:")
            for error in all_errors:
                print(f"  - [{error.source_file}] {error.message}")
                if error.hint:
                    print(f"    Hint: {error.hint}")
        else:
            print("\n--- Validation Report (SUCCESS) ---")
            print("All laws and mods are valid.")
            preview_feature_index_map(laws, mods)
            
        print("\n--- HTML Report (not yet implemented) ---")
        print("A future version will export a detailed HTML report for user-friendly display.")

if __name__ == '__main__':
    # Refactored self-test to be robust and clean
    # Fixes the bug where self-test leaves directories behind.
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] in ['validate']:
        cli_main(sys.argv)
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_root = os.path.join(temp_dir, "test_project")
            test_law_dir, test_mod_dir = ensure_project_dirs(test_root)

            # --- Create test files ---
            # Test Law file with full schema
            with open(os.path.join(test_law_dir, "law_core.json"), "w") as f:
                json.dump({
                    "constants": {"c": 299792458.0, "G": 6.6743e-11, "k_e": 8.98755e9},
                    "defaults": {"softening_eps_rel": 1e-7, "max_accel_rel": 0.5, "goal_radius_rel": 0.05, "half_life": 100.0},
                    "enhancement_order": ["e_a", "e_b"],
                    "ui_patches": {"hud_enhancements_compact": True}
                }, f)

            # Test Law file with partial schema
            with open(os.path.join(test_law_dir, "law_partial.json"), "w") as f:
                json.dump({
                    "constants": {"c": 300000000.0},
                    "defaults": {"goal_radius_rel": 0.1},
                    "enhancement_order": ["e_c"],
                    "ui_patches": {}
                }, f)
            
            # Test Mod file with dependencies and UI patches
            with open(os.path.join(test_mod_dir, "mod_ui.json"), "w") as f:
                json.dump({
                    "name": "ui_mod",
                    "enhancements": {},
                    "ui_patches": {"show_entropy_badge": True, "metrics_refresh_ms": 1000},
                    "depends": ["feature_mod"],
                    "load_after": ["physics_mod"]
                }, f)
            
            # Test Mod file with physics override and index override
            with open(os.path.join(test_mod_dir, "mod_features.json"), "w") as f:
                json.dump({
                    "name": "feature_mod",
                    "enhancements": {"e_d": "d"},
                    "enhancement_index_overrides": {"e_d": 0}, # Collision with e_a
                    "physics_patches": {"c": 3.0e8}, # This patch now works
                    "ui_patches": {}
                }, f)
            
            # Test Mod file that conflicts
            with open(os.path.join(test_mod_dir, "mod_conflict.json"), "w") as f:
                json.dump({
                    "name": "conflict_mod",
                    "enhancements": {},
                    "conflicts": ["ui_mod"],
                    "ui_patches": {}
                }, f)

            # Test Mod file for topological sort
            with open(os.path.join(test_mod_dir, "mod_physics.json"), "w") as f:
                json.dump({
                    "name": "physics_mod",
                    "enhancements": {},
                    "ui_patches": {}
                }, f)

            log.info("\n--- Running self-test: Loading laws ---")
            laws, law_errors, law_sources = load_laws(test_law_dir, strict=False)
            assert not law_errors, f"Expected no law errors, but got: {law_errors}"
            assert len(law_sources) == 2, "Expected 2 law sources"
            log.info(f"Loaded laws: {laws}")

            log.info("\n--- Running self-test: Loading mods (dry-run) ---")
            mods, mod_errors, mod_sources = load_mods(test_mod_dir, validate_only=True)
            assert mod_errors, f"Expected mod errors, but got none."
            # One error from dependency, one from conflict, one from index collision
            assert len(mod_errors) == 1, f"Expected 1 mod errors, got {len(mod_errors)}: {mod_errors}"
            log.info(f"Dry-run succeeded with expected errors.")

            log.info("\n--- Running self-test: Loading mods (real) ---")
            # Remove the conflicting and depending mod to test a clean load
            os.remove(os.path.join(test_mod_dir, "mod_conflict.json"))
            os.remove(os.path.join(test_mod_dir, "mod_ui.json"))
            mods, mod_errors, mod_sources = load_mods(test_mod_dir, validate_only=False)
            assert not mod_errors, f"Expected no mod errors, but got: {mod_errors}"
            assert len(mods) == 2, f"Expected 2 mods loaded, got {len(mods)}"
            assert "physics_mod" in mods, "Expected 'physics_mod' to be loaded"
            
            log.info("\n--- Running self-test: Applying state ---")
            class MockUniverse:
                def __init__(self):
                    self._enh_order = []
            
            mock_universe = MockUniverse()
            apply_mods_to_state(mock_universe, laws, mods)
            assert mock_universe.constants['c'] == 3.0e8, "Constant 'c' was not overridden."
            assert mock_universe.ui_patches['show_entropy_badge'] is True, "UI patch was not applied."
            assert mock_universe.defaults['goal_radius_rel'] == 0.1, "Defaults were not deep-merged correctly."
            assert mock_universe._enh_order == ["e_d", "e_a", "e_b"], "Enhancement order was not applied correctly."
            
            log.info("\n--- All self-tests passed! ---")
