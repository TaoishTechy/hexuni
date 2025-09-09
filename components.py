# Updated components for the Hexuni simulation.
# This file defines the core data structures and physics-related helpers.

import numpy as np
import json
import os
from collections import deque

# --- NUMBA IMPORT GUARD ---
# Guard all Numba imports behind a single try/except block.
# This ensures the code runs even if Numba is not installed.
try:
    from numba import jit, float64, int64, boolean, types
    from numba.experimental import jitclass
    NUMBA_AVAILABLE = True
except ImportError:
    print("Numba not available. Falling back to non-JIT versions.")
    NUMBA_AVAILABLE = False
    
    # Define a no-op jit and jitclass decorator.
    def jit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper

    def jitclass(spec):
        def wrapper(cls):
            return cls
        return wrapper

# --- CONSTANTS ---
# Centralize constants for clarity and easy overrides.
COULOMB_CONSTANT = 8.9875517873681764e9  # N * m^2 / C^2

# --- DATA STRUCTURES ---
SCHEMA_VERSION = "0.5.0" # Updated schema version for migration hooks

# The spec for the jitclass Particle.
# Must be kept in sync with the Python class for branch parity.
particle_spec = [
    ('position', float64[:]),
    ('velocity', float64[:]),
    ('mass', float64),
    ('charge', float64),
    ('size', float64),
    ('flags', int64), # Bitmask for enhancements
]

# The Python Particle class or Numba jitclass, depending on availability.
# This ensures branch parity in attribute definitions.
if NUMBA_AVAILABLE:
    @jitclass(particle_spec)
    class Particle:
        def __init__(self, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, mass=1.0, charge=0.0, size=1.0):
            self.position = np.array([x, y, z], dtype=np.float64)
            self.velocity = np.array([vx, vy, vz], dtype=np.float64)
            self.mass = mass
            self.charge = charge
            self.size = size
            self.flags = 0  # Bitmask for enhancements

        def set_flag(self, flag, value):
            if value:
                self.flags |= (1 << flag)
            else:
                self.flags &= ~(1 << flag)

        def get_flag(self, flag):
            return (self.flags >> flag) & 1

else:
    class Particle:
        def __init__(self, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, mass=1.0, charge=0.0, size=1.0):
            self.position = np.array([x, y, z], dtype=np.float64)
            self.velocity = np.array([vx, vy, vz], dtype=np.float64)
            self.mass = mass
            self.charge = charge
            self.size = size
            self.flags = 0  # Bitmask for enhancements

        def set_flag(self, flag, value):
            if value:
                self.flags |= (1 << flag)
            else:
                self.flags &= ~(1 << flag)

        def get_flag(self, flag):
            return (self.flags >> flag) & 1

class Universe:
    def __init__(self, config=None, initial_state=None):
        self.config = config or {}
        self.particles = []
        self._particles_array_aos = None # Array of Structs (AoS) for Numba
        self._particles_array_soa = None # Struct of Arrays (SoA) for Numba
        self._use_soa = self.config.get('physics.use_soa', False)
        
        # --- Memory Pooling ---
        # Keep a persistent buffer and update in-place to reduce GC churn.
        self._particles_buffer_size = 0

        self._enh_order = self.config.get('enhancement_order', [])
        self._enhancement_array = None
        self._enhancements_dirty = True
        self._law_index_map = self._build_law_index_map()

        self.time = 0.0
        self.step_counter = 0

        # --- Metrics ---
        self.metrics = {
            "kinetic_energy": deque(maxlen=self.config.get('metrics.history_len', 100)),
            "potential_energy": deque(maxlen=self.config.get('metrics.history_len', 100)),
            "total_energy": deque(maxlen=self.config.get('metrics.history_len', 100)),
            "energy_drift": deque(maxlen=self.config.get('metrics.history_len', 100)),
            "fps": deque(maxlen=self.config.get('metrics.history_len', 100)),
            "spatial_entropy": deque(maxlen=self.config.get('metrics.history_len', 100)),
            "spectral_entropy": deque(maxlen=self.config.get('metrics.history_len', 100)),
        }
        self._last_metrics_update_time = 0.0
        self._last_step_time = 0.0

        # --- Autosave ---
        self._last_autosave_time = 0.0
        self._last_autosave_success_time = 0.0
        self._autosave_min_interval = self.config.get('autosave.min_interval', 60.0) # seconds

    def _build_law_index_map(self):
        """Builds a name -> index map for enhancements for dynamic indexing."""
        return {name: i for i, name in enumerate(self._enh_order)}
        
    def rebuild_enhancement_array(self):
        """
        Rebuilds the enhancement bit-array only if a dirty flag is set.
        Caches the result for performance.
        """
        if not self._enhancements_dirty:
            return self._enhancement_array

        enh = np.zeros(len(self._enh_order), dtype=np.int64)
        for i, name in enumerate(self._enh_order):
            enh[i] = self.config.get(f'enhancements.{name}', 0)

        # --- Out-of-bounds Enhancement Indexing Fix ---
        # Assert that key enhancement names are in the expected positions.
        # This prevents accidental misconfiguration.
        try:
            psionic_index = self._law_index_map.get("psionic_energy_active")
            if psionic_index is not None:
                assert self._enh_order[psionic_index] == "psionic_energy_active"

            dark_matter_index = self._law_index_map.get("dark_matter_dynamics_active")
            if dark_matter_index is not None:
                assert self._enh_order[dark_matter_index] == "dark_matter_dynamics_active"

        except AssertionError as e:
            print(f"Enhancement order assertion failed: {e}")
            raise

        self._enhancement_array = enh
        self._enhancements_dirty = False
        return self._enhancement_array

    def set_enhancement(self, name, value):
        """Sets an enhancement and marks the array as dirty."""
        if name in self.config.get('enhancements', {}):
            self.config['enhancements'][name] = value
            self._enhancements_dirty = True
        else:
            print(f"Warning: Enhancement '{name}' not found in configuration.")
    
    def _pack_particles_array(self):
        """
        Packs the list of Particle objects into a single NumPy array (AoC or SoA).
        Uses memory pooling to update an existing buffer in-place.
        """
        n = len(self.particles)
        if n == 0:
            return None
        
        if self._use_soa:
            # --- SoA Layout ---
            # Creates a struct-of-arrays layout for potential performance gains.
            if self._particles_buffer_size < n:
                self._particles_array_soa = {
                    'positions': np.zeros((n, 3), dtype=np.float64),
                    'velocities': np.zeros((n, 3), dtype=np.float64),
                    'masses': np.zeros(n, dtype=np.float64),
                    'charges': np.zeros(n, dtype=np.float64),
                    'sizes': np.zeros(n, dtype=np.float64),
                    'flags': np.zeros(n, dtype=np.int64)
                }
                self._particles_buffer_size = n
            
            for i, p in enumerate(self.particles):
                self._particles_array_soa['positions'][i] = p.position
                self._particles_array_soa['velocities'][i] = p.velocity
                self._particles_array_soa['masses'][i] = p.mass
                self._particles_array_soa['charges'][i] = p.charge
                self._particles_array_soa['sizes'][i] = p.size
                self._particles_array_soa['flags'][i] = p.flags

            return self._particles_array_soa
        else:
            # --- AoS Layout (original) ---
            # Creates an array-of-structs layout.
            if self._particles_buffer_size < n:
                # Resize the buffer if needed
                self._particles_array_aos = np.zeros((n, 12), dtype=np.float64)
                self._particles_buffer_size = n
            
            for i, p in enumerate(self.particles):
                self._particles_array_aos[i, 0:3] = p.position
                self._particles_array_aos[i, 3:6] = p.velocity
                self._particles_array_aos[i, 6] = p.mass
                self._particles_array_aos[i, 7] = p.charge
                self._particles_array_aos[i, 8] = p.size
                self._particles_array_aos[i, 9] = p.flags

            return self._particles_array_aos

    def _unpack_particles_array(self, particles_data):
        """
        Unpacks the NumPy array back into the list of Particle objects.
        This now symmetrically updates all fields.
        """
        if particles_data is None:
            return

        if self._use_soa:
            n = len(self.particles)
            for i in range(n):
                p = self.particles[i]
                p.position = particles_data['positions'][i]
                p.velocity = particles_data['velocities'][i]
                p.mass = particles_data['masses'][i]
                p.charge = particles_data['charges'][i]
                p.size = particles_data['sizes'][i]
                p.flags = particles_data['flags'][i]
        else:
            n = particles_data.shape[0]
            for i in range(n):
                p = self.particles[i]
                p.position = particles_data[i, 0:3]
                p.velocity = particles_data[i, 3:6]
                p.mass = particles_data[i, 6]
                p.charge = particles_data[i, 7]
                p.size = particles_data[i, 8]
                p.flags = int(particles_data[i, 9]) # Ensure flags are integers

    def calculate_energy(self):
        """Calculates and stores kinetic and potential energy."""
        if not self.particles: return
        
        kinetic = 0.5 * sum(p.mass * np.dot(p.velocity, p.velocity) for p in self.particles)
        
        potential_grav = 0.0
        potential_elec = 0.0
        for i in range(len(self.particles)):
            for j in range(i + 1, len(self.particles)):
                p1 = self.particles[i]
                p2 = self.particles[j]
                
                # Use a small epsilon to prevent singularities
                r = np.linalg.norm(p1.position - p2.position)
                r_eps = max(r, 1e-6)

                potential_grav -= (6.67430e-11 * p1.mass * p2.mass) / r_eps
                potential_elec += (COULOMB_CONSTANT * p1.charge * p2.charge) / r_eps
        
        potential = potential_grav + potential_elec
        total = kinetic + potential

        # --- Energy Accounting ---
        self.metrics["kinetic_energy"].append(kinetic)
        self.metrics["potential_energy"].append(potential)
        self.metrics["total_energy"].append(total)

        if len(self.metrics["total_energy"]) > 1:
            drift = total - self.metrics["total_energy"][-2]
            self.metrics["energy_drift"].append(drift)

    def calculate_entropy(self):
        """
        Calculates various entropy metrics to diagnose sim state.
        """
        if not self.particles: return

        # --- Spatial Entropy ---
        # Bin particles into a grid and compute Shannon entropy of the distribution.
        grid_size = self.config.get('entropy.spatial_grid_size', 10)
        hist, _ = np.histogramdd(
            [p.position for p in self.particles],
            bins=[grid_size, grid_size, grid_size],
            range=[[0, self.config['sim.size']]] * 3
        )
        p = hist.flatten() / len(self.particles)
        p = p[p > 0] # Avoid log(0)
        spatial_entropy = -np.sum(p * np.log2(p)) if len(p) > 0 else 0
        self.metrics["spatial_entropy"].append(spatial_entropy)
        
        # --- Spectral Entropy ---
        # FFT of particle speeds to see if they're random or have a pattern.
        speeds = [np.linalg.norm(p.velocity) for p in self.particles]
        if len(speeds) > 1:
            fft_magnitude = np.abs(np.fft.fft(speeds))
            # Normalize and compute Shannon entropy of the spectrum.
            p_fft = fft_magnitude / np.sum(fft_magnitude)
            p_fft = p_fft[p_fft > 0]
            spectral_entropy = -np.sum(p_fft * np.log2(p_fft))
            self.metrics["spectral_entropy"].append(spectral_entropy)
        else:
            self.metrics["spectral_entropy"].append(0)

    def save_state(self, filepath, compact=False):
        """
        Saves the current state of the universe to a file.
        Uses a temp file and atomic write for safety.
        Optionally saves a compact diff.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        temp_filepath = filepath + ".tmp"
        
        state_dict = {
            "schema_version": SCHEMA_VERSION,
            "time": self.time,
            "step_counter": self.step_counter,
            "config": self.config,
            "particles": []
        }

        # Handle NumPy arrays for JSON serialization
        for p in self.particles:
            particle_dict = {
                "position": p.position.tolist(),
                "velocity": p.velocity.tolist(),
                "mass": p.mass,
                "charge": p.charge,
                "size": p.size,
                "flags": int(p.flags)
            }
            state_dict["particles"].append(particle_dict)

        # --- State Diff Snapshots ---
        # To be implemented with a previous state.
        # For now, this is a placeholder.
        if compact:
            pass # TODO: Implement diff logic

        try:
            with open(temp_filepath, 'w') as f:
                json.dump(state_dict, f, indent=2)
            os.replace(temp_filepath, filepath) # Atomic write
            self._last_autosave_success_time = self.time
            print(f"State saved to {filepath}")
        except Exception as e:
            print(f"Error saving state: {e}")

    @staticmethod
    def load_state(filepath):
        """Loads the state from a file, with schema migration support."""
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r') as f:
                state_dict = json.load(f)
            
            # --- Schema Version Migration Hooks ---
            # Call a dispatcher to handle older schemas.
            old_version = state_dict.get("schema_version", "0.0.0")
            if old_version != SCHEMA_VERSION:
                state_dict = Universe._migrate_state(old_version, state_dict)
            
            # Unpack the state into a new Universe object.
            universe = Universe(config=state_dict["config"])
            universe.time = state_dict["time"]
            universe.step_counter = state_dict["step_counter"]

            for p_dict in state_dict["particles"]:
                p = Particle(
                    x=p_dict["position"][0],
                    y=p_dict["position"][1],
                    z=p_dict["position"][2],
                    vx=p_dict["velocity"][0],
                    vy=p_dict["velocity"][1],
                    vz=p_dict["velocity"][2],
                    mass=p_dict["mass"],
                    charge=p_dict["charge"],
                    size=p_dict["size"]
                )
                p.flags = p_dict.get("flags", 0)
                universe.particles.append(p)
            
            return universe

        except Exception as e:
            print(f"Error loading state from {filepath}: {e}")
            return None

    @staticmethod
    def _migrate_state(old_version, state):
        """Migration dispatcher for older snapshot formats."""
        if old_version == "0.4.5" and SCHEMA_VERSION == "0.5.0":
            print(f"Migrating state from schema {old_version} to {SCHEMA_VERSION}.")
            # Add migration logic here. E.g., handling renames.
            return state
        return state

    def update_metrics(self, dt):
        """
        Updates metrics at a throttled, user-defined cadence.
        This also computes FPS based on tick duration, not callback duration.
        """
        # --- Metrics Throttle & Burst Protection ---
        # Throttle metrics updates to avoid excessive memory usage.
        now = self.time
        interval = self.config.get('metrics.update_interval', 1.0)
        
        if now - self._last_metrics_update_time >= interval:
            self.calculate_energy()
            self.calculate_entropy()
            
            # Track FPS based on tick duration
            if self.step_counter > 0:
                frame_time = now - self._last_step_time
                if frame_time > 0:
                    fps = 1.0 / frame_time
                    self.metrics["fps"].append(fps)
            self._last_step_time = now
            self._last_metrics_update_time = now
