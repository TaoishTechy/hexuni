"""
engine/sim_core.py - Universe Simulation v0.4.5 Physics and World Core.

This module encapsulates the core physics, particle management, and
universe state for the simulation. It is designed for robustness and
extensibility, with new metrics and hooks for symbolic, quantum, and
recursive phenomena.

Acceptance Tests:
- goal_alignment returns ~1.0 when all goals parallel; does not shrink as agent count increases.
- get_sociometric_cohesion does not divide by zero with zero coalitions.
- Holographic refresh does not allocate huge temporaries per frame.
- Enhancement array respects laws order and physics_overrides index overrides.
- All new symbolic/emotional fields are properly serialized and restored.
"""
import numpy as np
import numba
from numba import float64, int64
from numba.experimental import jitclass
import json
import os
import time
import psutil
import warnings
import copy
from typing import List, Dict, Any, Optional, NamedTuple, Tuple
from collections import deque

# To integrate with main.py, we define core components here.
# This ensures this module is a single, self-contained unit.

NUMBA_AVAILABLE = False
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    warnings.warn("Numba not found. Physics simulation will run in pure NumPy, which may be slower.")
    def jit(signature=None, nopython=False):
        def wrapper(func):
            return func
        return wrapper

# --- Constants and Settings from mods_runtime.py and components.py ---
G = 6.67430e-11  # Gravitational constant
YEAR_IN_SECONDS = 365.25 * 24 * 60 * 60
BOLTZMANN_CONSTANT = 1.380649e-23
MAX_METRICS_HISTORY = 10_000
HOLOGRAM_REFRESH_RATE = 100
AUTOSAVE_REFRESH_RATE = 100
SCHEMA_VERSION = "0.4.5"

# Fallback values from mods_runtime.DEFAULT_LAW if it's not available
DEFAULT_PHYSICS_CONSTANTS = {
    "c": 299792458.0,
    "softening_eps_rel": 1e-12,
    "max_accel_rel": 1e-2
}
DEFAULT_DEFAULTS = {
    "goal_radius_rel": 1e-6
}

# --- Shared JIT-compiled classes and functions ---

if NUMBA_AVAILABLE:
    spec_particle = [
        ('position', float64[:]),
        ('velocity', float64[:]),
        ('mass', float64),
        ('charge', float64),
        ('force', float64[:]),
        ('is_exotic', int64),
        ('is_dark', int64),
        ('is_entangled', int64),
        ('entangled_pair', int64),
        ('psionic_field', float64),
        ('is_supersymmetric', int64),
        ('super_partner_id', int64)
    ]

    @jitclass(spec_particle)
    class Particle:
        """A jit-compiled class for a single particle."""
        def __init__(self, pos, vel, mass, charge):
            self.position = pos
            self.velocity = vel
            self.mass = mass
            self.charge = charge
            self.force = np.zeros(2, dtype=np.float64)
            self.is_exotic = 0
            self.is_dark = 0
            self.is_entangled = 0
            self.entangled_pair = -1
            self.psionic_field = 0.0
            self.is_supersymmetric = 0
            self.super_partner_id = -1
else:
    class Particle:
        def __init__(self, pos, vel, mass, charge):
            self.position = np.asarray(pos, dtype=np.float64)
            self.velocity = np.asarray(vel, dtype=np.float64)
            self.mass = float(mass)
            self.charge = float(charge)
            self.force = np.zeros(2, dtype=np.float64)
            self.is_exotic = 0
            self.is_dark = 0
            self.is_entangled = 0
            self.entangled_pair = -1
            self.psionic_field = 0.0
            self.is_supersymmetric = 0
            self.super_partner_id = -1

# Numba-friendly physics kernel
# psionic_energy [4], dimensional_folding [2], dark_matter_dynamics [11]
# as per prompt. The 'enhancements' array is guaranteed to be in the correct
# order by the `rebuild_enhancement_array` method.
@jit(nopython=True)
def _compute_forces_and_effects_numba(particle_data, psionic_field, size, active_enhancements, physics_overrides):
    """
    Computes forces and applies physical effects to particles using Numba.
    This is a vectorized, high-performance kernel.
    """
    n = particle_data.shape[0]
    forces = np.zeros((n, 2))

    # Safely get physics constants
    softening_eps_rel = physics_overrides.get('softening_eps_rel', DEFAULT_PHYSICS_CONSTANTS['softening_eps_rel'])
    max_accel_rel = physics_overrides.get('max_accel_rel', DEFAULT_PHYSICS_CONSTANTS['max_accel_rel'])
    c = physics_overrides.get('c', DEFAULT_PHYSICS_CONSTANTS['c'])

    eps2 = (softening_eps_rel * size)**2
    max_accel = max_accel_rel * c

    # Enhancement Flag Mapping
    psionic_energy_active = active_enhancements[4]
    dark_matter_dynamics_active = active_enhancements[11]
    dimensional_folding_active = active_enhancements[2]

    if psionic_energy_active:
        for i in range(n):
            psionic_force = psionic_field[i] * 1e15
            forces[i, 0] += psionic_force

    for i in range(n):
        for j in range(i + 1, n):
            r = particle_data[j, :2] - particle_data[i, :2]
            r_mag_sq = np.sum(r**2)
            # Add a tiny epsilon to prevent r_mag from becoming exactly zero
            r_mag = np.sqrt(r_mag_sq) + 1e-18

            is_exotic_i = particle_data[i, 6] == 1
            is_exotic_j = particle_data[j, 6] == 1
            mass_i = particle_data[i, 2]
            mass_j = particle_data[j, 2]

            # Gravitational Force with Softening
            f_grav_mag = G * mass_i * mass_j / (r_mag_sq + eps2)
            f_grav_vec = f_grav_mag * r / r_mag

            forces[i] += f_grav_vec
            forces[j] -= f_grav_vec

            # Exotic Matter Force (placeholder, should be integrated more carefully)
            if is_exotic_i or is_exotic_j:
                f_exotic_mag = -G * mass_i * mass_j / (r_mag_sq + eps2) * np.exp(-r_mag / 1e-10)
                f_exotic_vec = f_exotic_mag * r / r_mag
                forces[i] += f_exotic_vec
                forces[j] -= f_exotic_vec

            # Electrostatic Force
            charge_i = particle_data[i, 3]
            charge_j = particle_data[j, 3]
            if charge_i != 0 and charge_j != 0:
                f_elec_mag = 8.9875517873681764e9 * charge_i * charge_j / (r_mag_sq + eps2)
                f_elec_vec = f_elec_mag * r / r_mag
                forces[i] += f_elec_vec
                forces[j] -= f_elec_vec

            # Extra-dimensional Force with singularity guard
            if dimensional_folding_active:
                if r_mag > 1e-15: # Avoid divide by zero
                    extra_dim_force = (1 / (r_mag_sq + eps2)**1.5) * np.sin(2 * np.pi / r_mag)
                    force_vec = extra_dim_force * r / r_mag
                    forces[i] += force_vec
                    forces[j] -= force_vec

    # Dark Matter Force
    if dark_matter_dynamics_active:
        dark_matter_indices = np.where(particle_data[:, 7] == 1)[0]
        if len(dark_matter_indices) > 0:
            for i in range(n):
                if particle_data[i, 7] == 0: # not dark matter
                    for j in dark_matter_indices:
                        r = particle_data[j, :2] - particle_data[i, :2]
                        r_mag_sq = np.sum(r**2)
                        r_mag = np.sqrt(r_mag_sq) + 1e-10
                        f_dark_mag = G * particle_data[i, 2] * particle_data[j, 2] / (r_mag_sq + eps2)
                        f_dark_vec = f_dark_mag * r / r_mag
                        forces[i] += f_dark_vec

    # Clamp acceleration
    for i in range(n):
        mass = particle_data[i, 2]
        if mass > 1e-20:
            accel_mag = np.sqrt(np.sum(forces[i]**2)) / mass
            if accel_mag > max_accel:
                forces[i] = (forces[i] / accel_mag) * max_accel * mass

    return forces

def _compute_forces_and_effects_numpy(particle_data, psionic_field, size, active_enhancements, physics_overrides):
    """
    Pure NumPy fallback for _compute_forces_and_effects if Numba is not available.
    """
    n = particle_data.shape[0]
    forces = np.zeros((n, 2))

    # Safely get physics constants
    softening_eps_rel = physics_overrides.get('softening_eps_rel', DEFAULT_PHYSICS_CONSTANTS['softening_eps_rel'])
    max_accel_rel = physics_overrides.get('max_accel_rel', DEFAULT_PHYSICS_CONSTANTS['max_accel_rel'])
    c = physics_overrides.get('c', DEFAULT_PHYSICS_CONSTANTS['c'])

    eps2 = (softening_eps_rel * size)**2
    max_accel = max_accel_rel * c

    # This is a brute-force approach, less performant than Numba.
    # The logic is a direct translation of the Numba function.

    psionic_energy_active = active_enhancements[4]
    dark_matter_dynamics_active = active_enhancements[11]
    dimensional_folding_active = active_enhancements[2]

    if psionic_energy_active:
        for i in range(n):
            psionic_force = psionic_field[i] * 1e15
            forces[i, 0] += psionic_force

    for i in range(n):
        for j in range(i + 1, n):
            r = particle_data[j, :2] - particle_data[i, :2]
            r_mag_sq = np.sum(r**2)
            r_mag = np.sqrt(r_mag_sq) + 1e-18

            is_exotic_i = particle_data[i, 6] == 1
            is_exotic_j = particle_data[j, 6] == 1
            mass_i = particle_data[i, 2]
            mass_j = particle_data[j, 2]

            f_grav_mag = G * mass_i * mass_j / (r_mag_sq + eps2)
            f_grav_vec = f_grav_mag * r / r_mag

            forces[i] += f_grav_vec
            forces[j] -= f_grav_vec

            if is_exotic_i or is_exotic_j:
                f_exotic_mag = -G * mass_i * mass_j / (r_mag_sq + eps2) * np.exp(-r_mag / 1e-10)
                f_exotic_vec = f_exotic_mag * r / r_mag
                forces[i] += f_exotic_vec
                forces[j] -= f_exotic_vec

            charge_i = particle_data[i, 3]
            charge_j = particle_data[j, 3]
            if charge_i != 0 and charge_j != 0:
                f_elec_mag = 8.9875517873681764e9 * charge_i * charge_j / (r_mag_sq + eps2)
                f_elec_vec = f_elec_mag * r / r_mag
                forces[i] += f_elec_vec
                forces[j] -= f_elec_vec

            if dimensional_folding_active:
                if r_mag > 1e-15:
                    extra_dim_force = (1 / (r_mag_sq + eps2)**1.5) * np.sin(2 * np.pi / r_mag)
                    force_vec = extra_dim_force * r / r_mag
                    forces[i] += force_vec
                    forces[j] -= force_vec

    if dark_matter_dynamics_active:
        dark_matter_indices = np.where(particle_data[:, 7] == 1)[0]
        if len(dark_matter_indices) > 0:
            for i in range(n):
                if particle_data[i, 7] == 0:
                    for j in dark_matter_indices:
                        r = particle_data[j, :2] - particle_data[i, :2]
                        r_mag_sq = np.sum(r**2)
                        r_mag = np.sqrt(r_mag_sq) + 1e-10
                        f_dark_mag = G * particle_data[i, 2] * particle_data[j, 2] / (r_mag_sq + eps2)
                        f_dark_vec = f_dark_mag * r / r_mag
                        forces[i] += f_dark_vec

    for i in range(n):
        mass = particle_data[i, 2]
        if mass > 1e-20:
            accel_mag = np.sqrt(np.sum(forces[i]**2)) / mass
            if accel_mag > max_accel:
                forces[i] = (forces[i] / accel_mag) * max_accel * mass

    return forces

# Assign the correct function based on Numba availability
_compute_forces_and_effects = _compute_forces_and_effects_numba if NUMBA_AVAILABLE else _compute_forces_and_effects_numpy

@jit(nopython=True)
def _bell_state_check(p1_pos, p2_pos, threshold):
    """Checks if particles are within the entanglement threshold."""
    dist = np.sqrt(np.sum((p1_pos - p2_pos)**2))
    return dist < threshold

@jit(nopython=True)
def _apply_quantum_decoherence(particle_data, time_step):
    """Simulates wavefunction collapse due to observation."""
    for i in range(len(particle_data)):
        if np.random.rand() < 0.05:
            particle_data[i, 4] += np.random.normal(0, 1e-15)
            particle_data[i, 5] += np.random.normal(0, 1e-15)

# --- MetricsCollector Class ---
class MetricsCollector:
    """
    Collects and manages simulation metrics in real-time.
    Uses a rolling buffer to prevent memory leaks and includes new symbolic metrics.
    """
    def __init__(self, window: int = MAX_METRICS_HISTORY):
        # Basic Metrics
        self.metrics = {
            'time': deque(maxlen=window),
            'fps': deque(maxlen=window),
            'memory_usage': deque(maxlen=window),
            'avg_reward': deque(maxlen=window),
            'total_energy': deque(maxlen=window),
            'entropy': deque(maxlen=window),
            'avg_learning_rate': deque(maxlen=window),
            'active_enhancements': deque(maxlen=window),
            'sociometric_cohesion': deque(maxlen=window),
            'goal_alignment': deque(maxlen=window),
        }
        # New Metrics
        self.metrics['emotional_resonance'] = deque(maxlen=window)
        self.metrics['symbolic_entropy'] = deque(maxlen=window)
        self.metrics['paradox_frequency'] = deque(maxlen=window)
        self.metrics['goal_alignment_delta'] = deque(maxlen=window)
        self.metrics['coalition_cohesion'] = deque(maxlen=window)

        self.start_time = time.time()
        self.frame_count = 0
        self.anomalies_detected = deque(maxlen=20)
        self.is_bootstrapped = False

    def get_metric(self, name: str, default: Any = "N/A") -> Any:
        """Helper to get a metric safely, avoiding KeyErrors."""
        try:
            return self.metrics.get(name, [default])[-1]
        except IndexError:
            return default

    def update(self, universe: "Universe") -> None:
        if not self.is_bootstrapped:
            self.is_bootstrapped = True
            self.start_time = time.time()

        current_time = time.time()
        frame_time = current_time - self.start_time
        fps = 1.0 / frame_time if frame_time > 0 else 0
        self.metrics['fps'].append(fps)
        self.start_time = current_time
        self.metrics['time'].append(universe.time)

        # Agent-related Metrics
        if universe.agents:
            rewards = [a.total_reward for a in universe.agents]
            self.metrics['avg_reward'].append(np.mean(rewards) if rewards else 0.0)
            avg_lr = np.mean([a.neurotransmitters.get_modulation().get('learning_rate', 0.0) for a in universe.agents])
            self.metrics['avg_learning_rate'].append(avg_lr)
            self.metrics['sociometric_cohesion'].append(universe.get_sociometric_cohesion())
            current_alignment = universe.goal_alignment()
            self.metrics['goal_alignment'].append(current_alignment)

            # New metric: goal alignment delta
            if len(self.metrics['goal_alignment']) > 1:
                delta = current_alignment - self.metrics['goal_alignment'][-2]
                self.metrics['goal_alignment_delta'].append(delta)
            else:
                self.metrics['goal_alignment_delta'].append(0.0)

            # New metric: emotional resonance
            emotional_states = [np.mean(list(a.neurotransmitters.levels.values())) for a in universe.agents]
            self.metrics['emotional_resonance'].append(np.mean(emotional_states))

            # New metric: coalition cohesion
            coalition_cohesion_scores = [c.last_cohesion for c in universe.coalitions.values()]
            self.metrics['coalition_cohesion'].append(np.mean(coalition_cohesion_scores) if coalition_cohesion_scores else 0.0)
        else:
            self.metrics['avg_reward'].append(0.0)
            self.metrics['avg_learning_rate'].append(0.0)
            self.metrics['sociometric_cohesion'].append(0.0)
            self.metrics['goal_alignment'].append(0.0)
            self.metrics['goal_alignment_delta'].append(0.0)
            self.metrics['emotional_resonance'].append(0.0)
            self.metrics['coalition_cohesion'].append(0.0)

        # Physics Metrics (robust against empty lists)
        if universe.particles:
            total_energy = np.sum([0.5 * p.mass * np.sum(p.velocity**2) for p in universe.particles])
            self.metrics['total_energy'].append(total_energy)
        else:
            self.metrics['total_energy'].append(0.0)

        if len(universe.particles) > 1:
            velocities = np.array([np.linalg.norm(p.velocity) for p in universe.particles])
            hist, _ = np.histogram(velocities, bins=10, range=(0, np.max(velocities) + 1e-10))
            total = np.sum(hist)
            hist = hist / max(total, 1)
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            self.metrics['entropy'].append(entropy)
        else:
            self.metrics['entropy'].append(0.0)

        # New symbolic metrics (placeholders for now)
        self.metrics['symbolic_entropy'].append(
            np.sum([hash(e) for e in universe.active_enhancements.keys()]) % 1000
        )
        self.metrics['paradox_frequency'].append(
            np.random.poisson(0.01) # A small chance of a paradox per step
        )

        self.metrics['memory_usage'].append(psutil.virtual_memory().percent)
        self.metrics['active_enhancements'].append(list(universe.active_enhancements.keys()))
        self.frame_count += 1

    def export_metrics(self, filename="metrics.json"):
        with open(filename, 'w') as f:
            serializable_metrics = self.metrics.copy()
            for key, value in serializable_metrics.items():
                if isinstance(value, deque):
                    serializable_metrics[key] = list(value)
            json.dump(serializable_metrics, f, indent=4)
        print(f"Metrics exported to {filename}")

# --- Universe Core Class ---
class Universe:
    """
    Main simulation class managing particles, agents, and physics.
    """
    def __init__(self, num_particles: int, size: float, time_step: float):
        self.size = size
        self.time_step = time_step
        self.time = 0.0
        self.step_counter = 0

        self.particles: List[Particle] = [
            Particle(
                pos=np.random.rand(2) * size,
                vel=np.random.randn(2) * 1e3,
                mass=1e-12 + 1e-10 * np.random.rand(),
                charge=0.0
            ) for _ in range(num_particles)
        ]
        # Ensure particle array is 2D, even for 0 or 1 particle
        if len(self.particles) > 0:
            self.particles_array = self._build_particle_array()
        else:
            self.particles_array = np.empty((0, 12), dtype=np.float64)


        # Public APIs as per prompt
        self.agents: List[Any] = [] # AGIAgent objects attached later
        self.active_enhancements: Dict[str, bool] = {}
        self._enh_order: List[str] = []
        self.autosave_enabled: bool = False
        self.holographic_data: Dict[str, np.ndarray] = {}
        self.metrics_collector: MetricsCollector = MetricsCollector()
        self.kb_overrides: Dict[str, Any] = {}
        self.physics_overrides: Dict[str, Any] = {}
        self.debate_overrides: Dict[str, Any] = {}
        self.goal_radius = DEFAULT_DEFAULTS['goal_radius_rel'] * self.size
        self.coalitions = {} # New: stores Coalition objects

        # Private/internal attributes and new symbolic hooks
        self._hologram_counter = 0
        self.last_hologram_time = 0.0
        self.recursive_sim_log: List[str] = []
        self.enhancement_sigils: Dict[str, str] = {
            "quantum_entanglement": "🔗",
            "temporal_flux": "⏳",
            "dimensional_folding": "🌀",
            "consciousness_field": "💡",
            "psionic_energy": "🧠",
            "exotic_matter": "👽",
            "cosmic_strings": "〰️",
            "neural_quantum_field": "🧠🔗",
            "tachyonic_communication": "⚡",
            "quantum_vacuum": "🌌",
            "holographic_principle": "🔮",
            "dark_matter_dynamics": "🌑",
            "supersymmetry_active": "✨"
        }
        self.emotional_propagation_factor: float = 0.0

        # Initialize enhancement state from a sane default.
        default_enhancements = [
            "quantum_entanglement", "temporal_flux", "dimensional_folding",
            "consciousness_field", "psionic_energy", "exotic_matter",
            "cosmic_strings", "neural_quantum_field", "tachyonic_communication",
            "quantum_vacuum", "holographic_principle", "dark_matter_dynamics",
            "supersymmetry_active"
        ]
        self.active_enhancements = {name: False for name in default_enhancements}
        self._enh_order = default_enhancements

    def _log_message(self, message: str) -> None:
        """A friendly, safe logging helper."""
        try:
            print(f"[Universe Core]: {message}")
        except Exception:
            pass

    def _build_particle_array(self):
        """Builds a single NumPy array for Numba processing."""
        if not self.particles:
            return np.empty((0, 12), dtype=np.float64)

        n = len(self.particles)
        # Columns: pos_x, pos_y, mass, charge, vel_x, vel_y,
        # is_exotic, is_dark, is_entangled, ent_pair, is_susy, susy_id
        particles_array = np.zeros((n, 12), dtype=np.float64)
        for i, p in enumerate(self.particles):
            particles_array[i, 0:2] = p.position
            particles_array[i, 2] = p.mass
            particles_array[i, 3] = p.charge
            particles_array[i, 4:6] = p.velocity
            particles_array[i, 6] = p.is_exotic
            particles_array[i, 7] = p.is_dark
            particles_array[i, 8] = p.is_entangled
            particles_array[i, 9] = p.entangled_pair
            particles_array[i, 10] = p.is_supersymmetric
            particles_array[i, 11] = p.super_partner_id
        return particles_array

    def _unpack_particle_array(self):
        """Unpacks the NumPy array back into the Particle objects."""
        if not self.particles:
            return

        for i, p in enumerate(self.particles):
            p.position = self.particles_array[i, 0:2]
            p.velocity = self.particles_array[i, 4:6]
            p.is_exotic = self.particles_array[i, 6]
            p.is_dark = self.particles_array[i, 7]
            p.is_entangled = self.particles_array[i, 8]
            p.entangled_pair = self.particles_array[i, 9]
            p.is_supersymmetric = self.particles_array[i, 10]
            p.super_partner_id = self.particles_array[i, 11]

    def rebuild_enhancement_array(self) -> np.ndarray:
        """
        Packs enhancement flags into a stable, ordered NumPy array.
        This array is used by the Numba kernel for fast lookups.
        """
        enhancements_array = np.zeros(len(self._enh_order), dtype=np.int64)
        for i, name in enumerate(self._enh_order):
            if self.active_enhancements.get(name, False):
                enhancements_array[i] = 1
        return enhancements_array

    def toggle_enhancement(self, name: str, value: bool) -> None:
        """Toggles a specific enhancement flag."""
        if name in self.active_enhancements:
            self.active_enhancements[name] = value
        else:
            self._log_message(f"Warning: Enhancement '{name}' not found.")

    def goal_alignment(self) -> float:
        """
        Calculates the mean magnitude of goal vectors for all agents.
        Returns a value from 0.0 to 1.0.
        """
        if not self.agents:
            return 0.0

        goal_vectors = np.array([a.goal - a.position for a in self.agents])

        if len(goal_vectors) == 0:
            return 0.0

        goal_magnitudes = np.linalg.norm(goal_vectors, axis=1)
        mean_magnitude = np.mean(goal_magnitudes)

        # Normalize by universe size to get a meaningful 0-1 scale.
        return np.clip(1.0 - mean_magnitude / self.size, 0.0, 1.0)

    def get_sociometric_cohesion(self) -> float:
        """
        Calculates the average relationship score between all agents.
        Returns 0.0 if there are no agents or relationships to avoid div-by-zero.
        """
        if len(self.agents) < 2:
            return 0.0

        total_score = 0.0
        count = 0
        for i, agent1 in enumerate(self.agents):
            if hasattr(agent1, 'relationships'):
                for j, agent2 in enumerate(self.agents):
                    if i != j and agent2.id in agent1.relationships:
                        total_score += agent1.relationships[agent2.id]
                        count += 1

        return total_score / count if count > 0 else 0.0

    def save_state(self, path: str) -> None:
        """
        Saves the current simulation state to a JSON file.
        Includes full particle and agent state, and new symbolic fields.
        """
        state = {
            "schema_version": SCHEMA_VERSION,
            "sim_params": {
                "size": self.size,
                "time_step": self.time_step,
                "time": self.time,
                "step_counter": self.step_counter,
                "autosave_enabled": self.autosave_enabled,
                "active_enhancements": self.active_enhancements,
                "enhancement_order": self._enh_order,
                "goal_radius": self.goal_radius,
                "emotional_propagation_factor": self.emotional_propagation_factor
            },
            "particles": [],
            "agents": [],
            "coalitions": {name: {"name": c.name, "members": list(c.members), "goal": c.goal.tolist(), "cohesion_target": c.cohesion_target} for name, c in self.coalitions.items()},
            "metrics": {k: list(v) for k, v in self.metrics_collector.metrics.items() if isinstance(v, deque)},
            "recursive_sim_log": self.recursive_sim_log,
        }

        # Serialize Particles
        for p in self.particles:
            state["particles"].append({
                "position": p.position.tolist(),
                "velocity": p.velocity.tolist(),
                "mass": p.mass,
                "charge": p.charge,
                "is_exotic": int(p.is_exotic),
                "is_dark": int(p.is_dark),
                "is_entangled": int(p.is_entangled),
                "entangled_pair": int(p.entangled_pair),
                "psionic_field": p.psionic_field,
                "is_supersymmetric": int(p.is_supersymmetric),
                "super_partner_id": int(p.super_partner_id)
            })

        # Serialize Agents (key fields only)
        for a in self.agents:
            agent_data = {
                "id": a.id,
                "position": a.position.tolist(),
                "velocity": a.velocity.tolist(),
                "goal": a.goal.tolist(),
                "role": a.role,
                "neurotransmitter_levels": a.neurotransmitters.levels,
                "total_reward": a.total_reward,
                "relationships": a.relationships,
                "self_awareness_score": a.self_awareness_score,
                "predictive_accuracy": a.predictive_accuracy,
                "avatar": a.avatar
            }
            state["agents"].append(agent_data)

        try:
            with open(path, 'w') as f:
                json.dump(state, f, indent=4)
            self._log_message(f"State saved to {path}")
        except Exception as e:
            self._log_message(f"Error saving state: {e}")

    def load_state(self, path: str) -> None:
        """
        Loads and restores a simulation state from a JSON file.
        Robustly handles missing or corrupt attributes.
        """
        try:
            with open(path, 'r') as f:
                state = json.load(f)
        except Exception as e:
            raise IOError(f"Failed to load state from {path}: {e}")

        schema_version = state.get("schema_version", "0.0.0")
        self._log_message(f"Loading state with schema version: {schema_version}")

        # Restore simulation parameters
        sim_params = state.get("sim_params", {})
        self.size = sim_params.get("size", self.size)
        self.time_step = sim_params.get("time_step", self.time_step)
        self.time = sim_params.get("time", 0.0)
        self.step_counter = sim_params.get("step_counter", 0)
        self.autosave_enabled = sim_params.get("autosave_enabled", False)
        self.active_enhancements = sim_params.get("active_enhancements", {})
        self._enh_order = sim_params.get("enhancement_order", self._enh_order)
        self.goal_radius = sim_params.get("goal_radius", self.goal_radius)
        self.emotional_propagation_factor = sim_params.get("emotional_propagation_factor", 0.0)

        # Restore Metrics
        metrics_data = state.get("metrics", {})
        for key, value in metrics_data.items():
            if key in self.metrics_collector.metrics:
                self.metrics_collector.metrics[key].extend(value)

        # Restore Logs
        self.recursive_sim_log = state.get("recursive_sim_log", [])

        # Restore Particles
        particles_data = state.get("particles", [])
        self.particles = []
        for p_data in particles_data:
            p = Particle(
                pos=np.array(p_data.get("position", [0, 0])),
                vel=np.array(p_data.get("velocity", [0, 0])),
                mass=p_data.get("mass", 1.0),
                charge=p_data.get("charge", 0.0)
            )
            p.is_exotic = p_data.get("is_exotic", 0)
            p.is_dark = p_data.get("is_dark", 0)
            p.is_entangled = p_data.get("is_entangled", 0)
            p.entangled_pair = p_data.get("entangled_pair", -1)
            p.psionic_field = p_data.get("psionic_field", 0.0)
            p.is_supersymmetric = p_data.get("is_supersymmetric", 0)
            p.super_partner_id = p_data.get("super_partner_id", -1)
            self.particles.append(p)

        # Restore Agents (placeholder)
        agents_data = state.get("agents", [])
        if agents_data:
            self._log_message(f"Found {len(agents_data)} agents in state file. GUI should handle their restoration.")

        self.particles_array = self._build_particle_array()

    def step(self) -> None:
        """
        Performs one simulation step: physics, agent updates, and metrics.
        """
        # Ensure particle array is up-to-date with particle object state
        self.particles_array = self._build_particle_array()

        # Get enhancement flags in a stable order
        enhancements = self.rebuild_enhancement_array()

        # Generate psionic field from agent emotional states if enabled
        psionic_field = np.zeros(len(self.particles))
        if self.active_enhancements.get("psionic_energy", False) and self.agents:
            # Propagate emotional state to a psionic field
            for agent in self.agents:
                # Simple metric: sum of Dopamine and Serotonin
                emotional_charge = agent.neurotransmitters.levels.get('dopamine', 0.0) + \
                                   agent.neurotransmitters.levels.get('serotonin', 0.0)

                # Apply a simple emotional force to nearby particles
                if emotional_charge > 0.5:
                    distances = np.linalg.norm(self.particles_array[:, :2] - agent.position, axis=1)
                    # Create a decaying field
                    psionic_field += (emotional_charge * self.emotional_propagation_factor) / (distances**2 + 1e-10)

        # Compute forces and update particle positions/velocities
        forces = _compute_forces_and_effects(
            self.particles_array[:, :8], # Pass only needed data to Numba
            psionic_field,
            self.size,
            enhancements,
            self.physics_overrides
        )

        # Update positions and velocities
        if len(self.particles_array) > 0:
            masses = self.particles_array[:, 2:3]
            masses[masses < 1e-20] = 1e-20 # Prevent division by zero
            self.particles_array[:, 4:6] += forces / masses * self.time_step
            self.particles_array[:, 0:2] += self.particles_array[:, 4:6] * self.time_step
            # Unpack the changes back into the Particle objects
            self._unpack_particle_array()

        # Update agents' positions based on a simple movement model
        for agent in self.agents:
            if not hasattr(agent, 'position') or not hasattr(agent, 'goal'): continue
            # Move towards goal
            direction = agent.goal - agent.position
            dist = np.linalg.norm(direction)
            if dist > self.goal_radius:
                move_speed = 1e18 # Arbitrary speed
                direction /= dist # Normalize
                agent.position += direction * move_speed * self.time_step

        # Agent-specific updates
        for agent in self.agents:
            if hasattr(agent, 'step'):
                agent.step()

        # New hook for recursive self-simulation
        if self.step_counter % 500 == 0 and len(self.agents) > 0:
            self.recursive_self_simulation_hook()

        # Update state and metrics
        self.time += self.time_step
        self.step_counter += 1

        self.metrics_collector.update(self)

        # Holographic projection
        if self.active_enhancements.get("holographic_principle", False) and self.step_counter % HOLOGRAM_REFRESH_RATE == 0:
            if len(self.particles) > 0:
                self.holographic_data['boundary_projection'] = self._holographic_encoding(
                    self.particles_array[:, :2], self.size
                )
            else:
                self.holographic_data['boundary_projection'] = np.zeros((0, 2))

        # Idempotent Autosave
        if self.autosave_enabled and self.step_counter % AUTOSAVE_REFRESH_RATE == 0:
            autosave_path = "timeline_checkpoint.json"
            try:
                self.save_state(autosave_path)
                self._log_message(f"Autosave checkpoint created at step {self.step_counter}")
            except Exception as e:
                self._log_message(f"Autosave failed: {e}")

    def recursive_self_simulation_hook(self) -> None:
        """
        A placeholder hook for a recursive self-simulation event.
        Simulates an agent predicting a future state of the universe.
        """
        if not self.agents:
            return

        agent = self.agents[np.random.randint(len(self.agents))]
        log_message = f"Agent {agent.id} initiated a recursive self-simulation at time {self.time:.2e}."
        self.recursive_sim_log.append(log_message)
        self._log_message(log_message)
        # In a real implementation, this would spawn a new simulation instance
        # to run a few steps ahead based on the agent's current state and beliefs.

    def _holographic_encoding(self, positions, size):
        """
        Conceptual model of holographic encoding.
        This simplified version takes a random subspace projection of positions.
        """
        n = positions.shape[0]
        if n == 0:
            return np.zeros((0, 2))

        try:
            # Use SVD for a stable, low-rank approximation
            U, s, V = np.linalg.svd(positions - positions.mean(axis=0))
            projected = U @ np.diag(s)
            return projected[:, :2]
        except np.linalg.LinAlgError:
            # Fallback for degenerate cases (e.g., all particles in a line)
            return positions * np.random.rand(1, 2)

    def _get_random_particle(self) -> Optional[Particle]:
        if not self.particles:
            return None
        return self.particles[np.random.randint(len(self.particles))]

    def _get_entangled_pairs(self) -> List[Tuple[Particle, Particle]]:
        entangled_pairs = []
        for i, p1 in enumerate(self.particles):
            if p1.is_entangled:
                pair_id = p1.entangled_pair
                if pair_id != -1 and pair_id > i and pair_id < len(self.particles):
                    p2 = self.particles[int(pair_id)]
                    entangled_pairs.append((p1, p2))
        return entangled_pairs

    def _get_supersymmetric_pairs(self) -> List[Tuple[Particle, Particle]]:
        susy_pairs = []
        for i, p1 in enumerate(self.particles):
            if p1.is_supersymmetric:
                pair_id = p1.super_partner_id
                if pair_id != -1 and pair_id > i and pair_id < len(self.particles):
                    p2 = self.particles[int(pair_id)]
                    susy_pairs.append((p1, p2))
        return susy_pairs
