# Smoke Test
# The following commands can be pasted directly into the console to test core functionality.
#
# # Test a batch of commands with different formats and comments
# activate temporal flux; activate exotic matter
# deactivate quantum entanglement # inline comment
# activate psionic energy
# set goal for agent 0 to 1.00e21,2.00e21
# set_goal 1 1.50e21,1.50e21
# set_goal all
#
# # Prompting, avatars, and debate
# prompt agent 0 Store: seek gradients near strings; report coherent corridors
# query agent 0 knowledge
# mutate avatar 0 anneal; mutate avatar 0 noise
# start debate on emergence under temporal flux with agents 0,1
# end debate
#
# # Reward shaping check
# set_goal 0 1e20,1e20; query agent 0 reward

import numpy as np
import numba
from numba import float64, int64
from numba.experimental import jitclass
import json
import sqlite3
import re
from datetime import datetime
import time
import queue
import psutil
from scipy.spatial import distance
from collections import deque
import random
import os
import threading
from typing import List, Dict, Optional, Set, NamedTuple, Any, Tuple

# --- Constants and Settings ---
G = 6.67430e-11  # Gravitational constant
c = 3e8          # Speed of light (capped at this speed)
hbar = 6.626e-34 / (2 * np.pi) # Reduced Planck constant
k_B = 1.38e-23   # Boltzmann constant
YEAR_IN_SECONDS = 365.25 * 24 * 60 * 60
BOLTZMANN_CONSTANT = 1.380649e-23  # For entropy calculation
MAX_DEBATE_TURNS = 20
MAX_METRICS_HISTORY = 10000
MAX_ACCEL = 1e12 # To prevent physics blowups

# --- Fixed Enhancement Flag order for Numba ---
# This is a static reference for the jit-compiled function.
# The actual mapping is passed in from mods_runtime.py.
class EnhancementFlags(NamedTuple):
    quantum_entanglement: int
    temporal_flux: int
    dimensional_folding: int
    consciousness_field: int
    psionic_energy: int
    exotic_matter: int
    cosmic_strings: int
    neural_quantum_field: int
    tachyonic_communication: int
    quantum_vacuum: int
    holographic_principle: int
    dark_matter_dynamics: int
    supersymmetry_active: int

# --- Jit-compiled classes for performance ---
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

# --- Core Physics Engine with Enhancements ---
@numba.jit(nopython=True)
def _compute_forces_and_effects(particle_data, temporal_flux_map, psionic_field,
                                universe_time, reality_editing_params, size, active_enhancements):
    """
    Computes forces and applies physical effects to particles.
    active_enhancements is a fixed-order array of 0s and 1s.
    """
    n = particle_data.shape[0]
    forces = np.zeros((n, 2))
    eps2 = (reality_editing_params[2] * size)**2 # Softening parameter from laws
    max_accel_per_pair = 1e10 # Clamp to prevent physics blowups

    G_eff = reality_editing_params[0]
    c_eff = reality_editing_params[1]

    # Enhancement Flag Mapping via fixed index array
    psionic_energy_flag = active_enhancements[4]
    dark_matter_dynamics_flag = active_enhancements[11]
    dimensional_folding_flag = active_enhancements[2]

    if psionic_energy_flag == 1: # psionic_energy
        for i in range(n):
            psionic_force = psionic_field[i] * 1e15
            forces[i, 0] += psionic_force

    for i in range(n):
        for j in range(i + 1, n):
            r = particle_data[j, :2] - particle_data[i, :2]
            r_mag_sq = np.sum(r**2)
            r_mag = np.sqrt(r_mag_sq) + 1e-10

            is_exotic_i = particle_data[i, 6] == 1
            is_exotic_j = particle_data[j, 6] == 1
            mass_i = particle_data[i, 2]
            mass_j = particle_data[j, 2]

            # Gravitational Force with Softening
            f_grav_mag = G_eff * mass_i * mass_j / (r_mag_sq + eps2)
            f_grav_vec = f_grav_mag * r / r_mag

            forces[i] += f_grav_vec
            forces[j] -= f_grav_vec

            # Exotic Matter Force
            if is_exotic_i or is_exotic_j:
                f_exotic_mag = -G_eff * mass_i * mass_j / (r_mag_sq + eps2) * np.exp(-r_mag / 1e-10)
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

            # Extra-dimensional Force with clamping
            if dimensional_folding_flag == 1: # dimensional_folding
                if r_mag > 1e-18:
                    extra_dim_force = (1 / (r_mag_sq + eps2)**1.5) * np.sin(2 * np.pi / r_mag)
                    extra_dim_force = min(extra_dim_force, 1e20) # Clamp the force
                    force_vec = extra_dim_force * r / r_mag
                    forces[i] += force_vec
                    forces[j] -= force_vec

    # Dark Matter Force
    if dark_matter_dynamics_flag == 1: # dark_matter_dynamics
        for i in range(n):
            if particle_data[i, 7] == 0: # not dark matter
                dark_matter_indices = np.where(particle_data[:, 7] == 1)[0]
                for j in dark_matter_indices:
                    r = particle_data[j, :2] - particle_data[i, :2]
                    r_mag = np.sqrt(np.sum(r**2)) + 1e-10
                    f_dark_mag = G_eff * particle_data[i, 2] * particle_data[j, 2] / (np.sum(r**2) + eps2)
                    f_dark_vec = f_dark_mag * r / r_mag
                    forces[i] += f_dark_vec

    # Clamp acceleration
    for i in range(n):
        mass = particle_data[i, 2]
        if mass > 1e-20:
            accel_mag = np.sqrt(np.sum(forces[i]**2)) / mass
            if accel_mag > MAX_ACCEL:
                forces[i] = (forces[i] / accel_mag) * MAX_ACCEL * mass

    return forces

# --- Entanglement and Quantum Functions ---
@numba.jit(nopython=True)
def _bell_state_check(p1_pos, p2_pos, threshold):
    """Checks if particles are within the entanglement threshold."""
    dist = np.sqrt(np.sum((p1_pos - p2_pos)**2))
    return dist < threshold

@numba.jit(nopython=True)
def _apply_quantum_decoherence(particle_data, time_step):
    """Simulates wavefunction collapse due to observation."""
    for i in range(len(particle_data)):
        if np.random.rand() < 0.05:
            particle_data[i, 4] += np.random.normal(0, 1e-15)
            particle_data[i, 5] += np.random.normal(0, 1e-15)

@numba.jit(nopython=True)
def _holographic_encoding(positions, size):
    """
    Conceptual model of holographic encoding.
    Projects 3D-like information onto a 2D boundary.
    Here, we simulate this by taking a random subspace projection of the 2D positions.
    Returns an N x 2 array.
    """
    n = positions.shape[0]
    if n == 0:
        return np.zeros((0, 2))
    # Ensure SVD works with the dimensions provided
    if positions.shape[1] < 2:
        return np.zeros((n, 2))
    # Simple SVD-like projection for visualization
    U, s, V = np.linalg.svd(positions)
    projected = U @ np.diag(s)
    # Return a slice to ensure N x 2 shape
    return projected[:, :2]

# --- Logging Helper ---
def log_to_queue(q, message):
    if q is not None:
        q.put(message)
    else:
        print(message)

# --- Neurotransmitter System ---
class NeurotransmitterSystem:
    def __init__(self):
        self.levels = {
            'dopamine': 0.5, 'serotonin': 0.5, 'acetylcholine': 0.5, 'gaba': 0.5,
            'glutamate': 0.5, 'norepinephrine': 0.5, 'endorphin': 0.5,
            'oxytocin': 0.5, 'vasopressin': 0.5
        }
        self.receptors = {nt: np.random.random() for nt in self.levels}
        self.degradation_rates = {nt: np.random.uniform(0.01, 0.1) for nt in self.levels}
        self.history = deque(maxlen=100)
        self.anomaly_scores = {nt: 0.0 for nt in self.levels}

    def update(self, inputs, reward_delta):
        # Corrected: use reward_delta instead of total reward
        self.levels['dopamine'] = np.clip(self.levels['dopamine'] + 0.1 * reward_delta, 0, 1)
        self.levels['oxytocin'] = np.clip(self.levels['oxytocin'] + 0.1 * reward_delta, 0, 1)

        self.levels['serotonin'] += 0.05 * (1 - abs(reward_delta))
        self.levels['acetylcholine'] += 0.1 * np.mean(inputs)

        for nt in self.levels:
            self.levels[nt] -= self.degradation_rates[nt] * self.levels[nt]

        for nt in self.levels:
            self.levels[nt] = np.clip(self.levels[nt], 0, 1)

        self.history.append(self.levels.copy())

    def get_modulation(self):
        return {
            "learning_rate": 0.01 * (0.5 + self.levels['acetylcholine'] - 0.5 * self.levels['gaba']),
            "explore": 0.1 * (self.levels['dopamine'] + 0.5 * self.levels['norepinephrine']),
            'reward_bias': 2.0 * (self.levels['serotonin'] - 0.5),
            'attention': self.levels['glutamate'] / (1.0 + self.levels['gaba']),
            'social_cohesion': self.levels['oxytocin'] * self.levels['vasopressin']
        }

    def get_anomaly_score(self):
        scores = {}
        for nt, level in self.levels.items():
            past_levels = [h[nt] for h in self.history if nt in h]
            if len(past_levels) > 10:
                mean = np.mean(past_levels)
                std_dev = np.std(past_levels)
                if std_dev > 1e-6:
                    self.anomaly_scores[nt] = abs(level - mean) / std_dev
                else:
                    self.anomaly_scores[nt] = 0.0
        return scores

class Coalition:
    """A simple class to represent an emergent agent group."""
    def __init__(self, name: str, members: List[int], goal: np.ndarray, cohesion_target: float = 0.6):
        self.name = name
        self.members = set(members)
        self.goal = goal
        self.cohesion_target = cohesion_target
        self.last_cohesion = 0.0

class MetricsCollector:
    """
    Collects and manages simulation metrics in real-time.
    Now includes advanced anomaly detection and statistical analysis.
    """
    def __init__(self, window: int = 10000):
        self.metrics = {
            'time': deque(maxlen=window),
            'fps': deque(maxlen=window),
            'memory_usage': deque(maxlen=window),
            'avg_reward': deque(maxlen=window),
            'total_energy': deque(maxlen=window),
            'entropy': deque(maxlen=window),
            'avg_learning_rate': deque(maxlen=window),
            'active_enhancements': deque(maxlen=window),
            'agent_behavior_anomalies': deque(maxlen=window),
            'learning_efficiency': deque(maxlen=window),
            'goal_achievement_rate': deque(maxlen=window),
            'resource_utilization': deque(maxlen=window),
            'intellectual_entropy': deque(maxlen=window),
            'self_awareness_score': deque(maxlen=window),
            'predictive_accuracy': deque(maxlen=window),
            'sociometric_cohesion': deque(maxlen=window),
            'goal_alignment': deque(maxlen=window),
            # Placeholder metrics for scientific-grade analysis
            'fractal_dimension': deque(maxlen=window),
            'lyapunov_exponents': deque(maxlen=window),
            'multi_scale_entropy': deque(maxlen=window),
            'information_transfer': deque(maxlen=window),
            'network_topology': deque(maxlen=window),
            'phase_transition_detection': deque(maxlen=window),
            'power_spectral_density': deque(maxlen=window),
            'cross_correlation': deque(maxlen=window),
            'recurrence_quantification': deque(maxlen=window),
            'multi_fractal_analysis': deque(maxlen=window),
            'causal_inference': deque(maxlen=window),
            'topological_data_analysis': deque(maxlen=window),
            'transfer_entropy': deque(maxlen=window),
            'granger_causality': deque(maxlen=window),
            'dynamic_mode_decomposition': deque(maxlen=window),
            'manifold_learning': deque(maxlen=window),
            'nonlinear_time_series': deque(maxlen=window),
            'complexity_entropy': deque(maxlen=window),
            'multi_resolution_analysis': deque(maxlen=window),
            'anomaly_detection_score': deque(maxlen=window)
        }
        self.start_time = time.time()
        self.frame_count = 0
        self.agent_positions_history = {}
        self.max_history_len = 1000
        self.anomalies_detected = deque(maxlen=20)

    def get_metric(self, name, default=None):
        if name in self.metrics and self.metrics[name]:
            return self.metrics[name][-1]
        return default

    def update(self, universe):
        current_time = time.time()
        frame_time = current_time - self.start_time
        fps = 1.0 / frame_time if frame_time > 0 else 0
        self.metrics['fps'].append(fps)
        self.start_time = current_time
        self.metrics['time'].append(universe.time)

        if universe.agents:
            avg_reward = np.mean([a.total_reward for a in universe.agents])
            self.metrics['avg_reward'].append(avg_reward)
            avg_lr = np.mean([a.neurotransmitters.get_modulation().get('learning_rate', 0.0) for a in universe.agents])
            self.metrics['avg_learning_rate'].append(avg_lr)
            efficiency = avg_reward / universe.step_counter if universe.step_counter > 0 else 0
            self.metrics['learning_efficiency'].append(efficiency)
            goal_rate = np.mean([a.goal_achieved for a in universe.agents])
            self.metrics['goal_achievement_rate'].append(goal_rate)
            intellectual_entropy = np.mean([a.get_intellectual_entropy() for a in universe.agents])
            self.metrics['intellectual_entropy'].append(intellectual_entropy)
            self_awareness = np.mean([a.self_awareness_score for a in universe.agents])
            self.metrics['self_awareness_score'].append(self_awareness)
            predictive_acc = np.mean([a.predictive_accuracy for a in universe.agents])
            self.metrics['predictive_accuracy'].append(predictive_acc)
            self.metrics['sociometric_cohesion'].append(universe.get_sociometric_cohesion())
            self.metrics['goal_alignment'].append(universe.goal_alignment())


            # Update agent position history
            for agent in universe.agents:
                if agent.id not in self.agent_positions_history:
                    self.agent_positions_history[agent.id] = deque(maxlen=100)
                self.agent_positions_history[agent.id].append(agent.position.copy())

        # Mock implementations for the 20 new metrics
        self.metrics['fractal_dimension'].append(random.uniform(1.5, 2.0))
        self.metrics['lyapunov_exponents'].append(random.uniform(0.1, 0.5))
        self.metrics['multi_scale_entropy'].append(random.uniform(0.8, 1.2))
        self.metrics['information_transfer'].append(random.uniform(0.0, 1.0))
        self.metrics['network_topology'].append(random.uniform(0.1, 0.9))
        self.metrics['phase_transition_detection'].append(0)
        self.metrics['power_spectral_density'].append(random.uniform(0.1, 0.9))
        self.metrics['cross_correlation'].append(random.uniform(-1.0, 1.0))
        self.metrics['recurrence_quantification'].append(random.uniform(0.0, 1.0))
        self.metrics['multi_fractal_analysis'].append(random.uniform(0.1, 0.9))
        self.metrics['causal_inference'].append(random.uniform(0.0, 1.0))
        self.metrics['topological_data_analysis'].append(random.uniform(0.0, 1.0))
        self.metrics['transfer_entropy'].append(random.uniform(0.0, 1.0))
        self.metrics['granger_causality'].append(random.uniform(0.0, 1.0))
        self.metrics['dynamic_mode_decomposition'].append(random.uniform(0.1, 0.9))
        self.metrics['manifold_learning'].append(random.uniform(0.1, 0.9))
        self.metrics['nonlinear_time_series'].append(random.uniform(0.1, 0.9))
        self.metrics['complexity_entropy'].append(random.uniform(0.1, 0.9))
        self.metrics['multi_resolution_analysis'].append(random.uniform(0.1, 0.9))
        self.metrics['anomaly_detection_score'].append(random.uniform(0, 10))

        # Physics Metrics
        total_energy = np.sum([0.5 * p.mass * np.sum(p.velocity**2) for p in universe.particles])
        self.metrics['total_energy'].append(total_energy)
        if len(universe.particles) > 1:
            velocities = np.array([np.linalg.norm(p.velocity) for p in universe.particles])
            hist, _ = np.histogram(velocities, bins=10)
            total = np.sum(hist)
            hist = hist / max(total, 1) # Guard against zero division
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            self.metrics['entropy'].append(entropy)
        else:
            self.metrics['entropy'].append(0)

        self.metrics['memory_usage'].append(psutil.virtual_memory().percent)
        self.metrics['resource_utilization'].append(psutil.cpu_percent())
        active_enhancements_list = [k for k, v in universe.active_enhancements.items() if v]
        self.metrics['active_enhancements'].append(active_enhancements_list)

        self.detect_behavior_anomalies(universe)
        self.frame_count += 1

    def detect_behavior_anomalies(self, universe):
        if self.frame_count > 50 and len(universe.agents) > 1:
            for agent in universe.agents:
                history = self.agent_positions_history.get(agent.id, [])
                if len(history) > 20:
                    positions = np.array(history)
                    distances = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
                    mean_dist = np.mean(distances)
                    std_dist = np.std(distances)
                    if std_dist > 1e-6:
                        # Corrected: use std_dist instead of std_dev
                        z_score = abs(distances[-1] - mean_dist) / std_dist
                        if z_score > 3:
                            anomaly = {
                                'agent_id': agent.id,
                                'time': universe.time,
                                'type': 'Behavioral Jitter',
                                'z_score': z_score
                            }
                            self.anomalies_detected.append(anomaly)

    def get_last_n_metrics(self, n=100):
        return {k: list(v)[-n:] for k, v in self.metrics.items()}

    def export_metrics(self, filename="metrics.json"):
        with open(filename, 'w') as f:
            serializable_metrics = self.metrics.copy()
            for key, value in serializable_metrics.items():
                if isinstance(value, deque):
                    serializable_metrics[key] = list(value)
            json.dump(serializable_metrics, f, indent=4)
        log_to_queue(None, f"Metrics exported to {filename}")

class NaturalLanguageProcessor:
    """
    Parses natural language commands for the simulation with advanced scripting.
    """
    def __init__(self, universe, output_queue: "queue.Queue[str] | None" = None):
        self.universe = universe
        self.output_queue = output_queue
        self.command_history = deque(maxlen=100)
        self.macros = {}
        self._initialize_handlers()

    def _initialize_handlers(self):
        self.handlers = {
            'set_goal': self._handle_set_goal,
            'activate': self._handle_enhancement,
            'deactivate': self._handle_enhancement,
            'query': self._handle_query,
            'start_debate': self._handle_start_debate,
            'end_debate': self._handle_end_debate,
            'prompt': self._handle_prompt,
            'mutate_avatar': self._handle_mutate_avatar,
            'help': self._handle_help,
            'role': self._handle_role,
            'form': self._handle_form_coalition,
            'assign': self._handle_assign_goal,
            'disband': self._handle_disband_coalition,
            'broadcast': self._handle_broadcast,
            'observe': self._handle_observe,
        }

    def _pop_optional(self, parts, token):
        if parts and parts[0].lower() == token:
            return parts[1:]
        return parts

    def _parse_id_and_text(self, parts, optional_keyword):
        parts = self._pop_optional(parts, optional_keyword)
        if not parts: raise ValueError("Missing agent ID.")

        agent_id = int(parts[0])
        text = ' '.join(parts[1:])
        return agent_id, text

    def parse_command(self, command_string):

        # Normalize whitespace and special characters
        command_string = command_string.replace('\r', '').replace('\u00A0', ' ').replace('\u200B', ' ')

        # Split by newlines first
        raw_lines = command_string.splitlines()

        for raw_line in raw_lines:
            line = raw_line.strip()

            # Skip blank lines and full-line comments
            if not line or line.lstrip().startswith('#'):
                continue

            # Skip the placeholder text
            if line == "Enter commands here (Ctrl+Enter to run):":
                continue

            # Strip inline comments
            line = line.split('#', 1)[0].strip()

            if not line:
                continue

            # Special handling for prompt to not split by semicolon
            if line.lower().startswith('prompt'):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        self.handlers['prompt'](parts[1:])
                    except Exception as e:
                        log_to_queue(self.output_queue, f"Error in command: {line}\n{type(e).__name__}: {e}")
                else:
                    log_to_queue(self.output_queue, f"Error: Invalid prompt format.")
                continue

            # For all other commands, split by semicolon
            chunks = [c.strip() for c in line.split(";") if c.strip()]

            for chunk in chunks:
                try:
                    parts = chunk.split()
                    if not parts: continue

                    cmd_name = parts[0].lower()
                    remainder = parts[1:]

                    # Command normalization and routing
                    if cmd_name == 'set' and remainder and remainder[0].lower() == 'goal':
                        cmd_name = 'set_goal'
                        remainder = remainder[1:]
                    elif cmd_name == 'start' and remainder and remainder[0].lower() == 'debate':
                        cmd_name = 'start_debate'
                        remainder = remainder[1:]
                    elif cmd_name == 'end' and remainder and remainder[0].lower() == 'debate':
                        cmd_name = 'end_debate'
                        remainder = remainder[1:]
                    elif cmd_name in ['mutate', 'mutate_avatar']:
                        cmd_name = 'mutate_avatar'
                        remainder = parts[1:]

                    handler = self.handlers.get(cmd_name)
                    if handler:
                        self.command_history.append(chunk)
                        if cmd_name in ['activate', 'deactivate']:
                            handler(remainder, cmd_name)
                        else:
                            handler(remainder)
                    else:
                        log_to_queue(self.output_queue, f"Command not recognized. Try 'help' or 'prompt agent 0 <text>'.")

                except Exception as e:
                    log_to_queue(self.output_queue, f"Error in command: {chunk}\n{type(e).__name__}: {e}")

    def _handle_set_goal(self, parts):
        if not parts:
            raise ValueError("Missing arguments.")

        if parts[0].lower() == 'all':
            for agent in self.universe.agents:
                agent.set_goal(np.random.rand(2) * self.universe.size)
            log_to_queue(self.output_queue, "Goals for all agents randomized.")
            return

        parts = self._pop_optional(parts, 'for')
        parts = self._pop_optional(parts, 'agent')

        agent_id = int(parts[0])
        parts = self._pop_optional(parts[1:], 'to')

        coord_str = "".join(parts).replace(" ", "")
        x, y = map(float, coord_str.split(','))

        if 0 <= agent_id < len(self.universe.agents):
            self.universe.agents[agent_id].set_goal(np.array([x, y]))
            log_to_queue(self.output_queue, f"Goal for Agent {agent_id} set to ({x}, {y}).")
        else:
            log_to_queue(self.output_queue, "Agent not found.")

    def _handle_enhancement(self, parts, action):
        if not parts:
            raise ValueError("Missing enhancement name.")

        enhancement_name = ' '.join(parts).replace(' ', '_').lower()
        if enhancement_name in self.universe.active_enhancements:
            self.universe.active_enhancements[enhancement_name] = (action == 'activate')
            log_to_queue(self.output_queue, f"Enhancement '{enhancement_name}' {'activated' if action == 'activate' else 'deactivated'}.")
        else:
            known_keys = ", ".join(self.universe.active_enhancements.keys())
            log_to_queue(self.output_queue, f"Unknown enhancement '{enhancement_name}'. Known keys: {known_keys}")

    def _handle_query(self, parts):
        if not parts or not parts[0].isdigit():
            raise ValueError("Missing agent ID or incorrect format.")

        agent_id = int(parts[0])
        query_text = ' '.join(parts[1:])

        if 0 <= agent_id < len(self.universe.agents):
            response = self.universe.agents[agent_id].respond_to_query(query_text)
            log_to_queue(self.output_queue, f"Agent {agent_id}: {response}")
        else:
            log_to_queue(self.output_queue, "Agent not found.")

    def _handle_start_debate(self, parts):
        if self.universe.debate_arena.active_debate:
            log_to_queue(self.output_queue, "Another debate is already in progress.")
            return

        parts = self._pop_optional(parts, 'on')
        topic_tokens = []
        with_index = -1
        try:
            with_index = [p.lower() for p in parts].index('with')
        except ValueError:
            raise ValueError("Missing 'with agents' clause.")

        topic = ' '.join(topic_tokens)

        if with_index + 1 >= len(parts) or parts[with_index+1].lower() != 'agents':
            raise ValueError("Expected 'with agents' clause.")

        id_str = ''.join(parts[with_index+2:])
        agent_ids = [int(a) for a in id_str.split(',') if a.strip().isdigit()]

        if len(set(agent_ids)) < 2:
            log_to_queue(self.output_queue, "Debate requires at least two unique agents.")
            return

        participants = [self.universe.agents[i] for i in agent_ids if 0 <= i < len(self.universe.agents)]
        if len(participants) != len(agent_ids):
            log_to_queue(self.output_queue, "One or more agent IDs not found.")
            return

        log_to_queue(self.output_queue, self.universe.debate_arena.start_debate(topic, [p.id for p in participants]))

    def _handle_end_debate(self, parts):
        if self.universe.debate_arena.active_debate:
            self.universe.debate_arena.end_debate()
            log_to_queue(self.output_queue, "Debate ended.")
        else:
            log_to_queue(self.output_queue, "No active debate to end.")

    def _handle_prompt(self, parts):
        if not parts:
            raise ValueError("Missing agent ID and text.")

        agent_id = int(parts[0])
        prompt_text = ' '.join(parts[1:])

        if 0 <= agent_id < len(self.universe.agents):
            self.universe.agents[agent_id].kb.ingest(prompt_text, "console")
            log_to_queue(self.output_queue, f"Prompt ingested for Agent {agent_id}.")
        else:
            log_to_queue(self.output_queue, "Agent not found.")

    def _handle_mutate_avatar(self, parts):
        if not parts or not parts[0].isdigit():
            raise ValueError("Missing agent ID or incorrect format.")

        agent_id = int(parts[0])
        mode = parts[1] if len(parts) > 1 else 'noise' # Default to noise

        if mode not in ['flip', 'rotate', 'noise', 'anneal']:
            log_to_queue(self.output_queue, f"Invalid mutation mode '{mode}'. Use one of: flip, rotate, noise, anneal.")
            return

        if 0 <= agent_id < len(self.universe.agents):
            self.universe.agents[agent_id].mutate_avatar(mode)
            log_to_queue(self.output_queue, f"Agent {agent_id}'s avatar mutated with mode '{mode}'.")
        else:
            log_to_queue(self.output_queue, "Agent not found.")

    def _handle_help(self, parts):
        help_text = self.get_help_text()
        log_to_queue(self.output_queue, help_text)

    def get_help_text(self):
        return """
Available commands:
  set goal [for agent] <id> to <x>,<y> - Sets the goal for a specific agent.
  set goal all - Sets a random goal for all agents.
  activate <enhancement> - Activates a simulation enhancement.
  deactivate <enhancement> - Deactivates a simulation enhancement.
  query agent <id> knowledge - Retrieves knowledge from an agent.
  prompt agent <id> <text> - Ingests new knowledge for an agent.
  start debate on <topic> with agents <id1,id2,...> - Starts a debate.
  end debate - Ends the current debate.
  mutate avatar <id> <mode> - Mutates an agent's avatar (modes: flip, rotate, noise, anneal).
  form coalition <name> with agents <id1,id2,...> - Creates a new coalition.
  broadcast <text> to agents <id1,...> or to coalition <name> - Broadcasts a message.
  observe reward <id> - Shows the total reward for an agent.
  observe cohesion coalition <name> - Shows the cohesion score of a coalition.
  observe entropy - Shows the system's entropy.
  observe alignment - Shows the global goal alignment score.
  help - Displays this help message.
"""

    def _handle_role(self, parts):
        if len(parts) < 2:
            raise ValueError("Invalid role format. Use 'role agent <id> <role>'.")
        parts = self._pop_optional(parts, 'agent')
        agent_id = int(parts[0])
        role = parts[1]
        valid_roles = {'scout', 'coordinator', 'critic', 'hypothesis'}
        if role not in valid_roles:
            log_to_queue(self.output_queue, f"Invalid role '{role}'. Valid roles are: {', '.join(valid_roles)}")
            return
        if 0 <= agent_id < len(self.universe.agents):
            self.universe.agents[agent_id].role = role
            log_to_queue(self.output_queue, f"Agent {agent_id}'s role set to '{role}'.")
        else:
            log_to_queue(self.output_queue, "Agent not found.")

    def _handle_form_coalition(self, parts):
        if len(parts) < 4 or parts[0].lower() != 'coalition':
            raise ValueError("Invalid format. Use 'form coalition <name> with agents <id1,id2,...>'.")

        name = parts[1]
        if name in self.universe.coalitions:
            log_to_queue(self.output_queue, f"Coalition '{name}' already exists.")
            return

        id_tokens_start_idx = 0
        try:
            id_tokens_start_idx = parts.index('with') + parts[parts.index('with'):].index('agents') + 1
        except ValueError:
            raise ValueError("Missing 'with agents' clause.")

        id_str = ''.join(parts[id_tokens_start_idx:])
        agent_ids = sorted(list(set([int(a) for a in id_str.split(',') if a.strip().isdigit()])))

        if len(agent_ids) < 2:
            log_to_queue(self.output_queue, "A coalition requires at least two unique agents.")
            return

        members = [self.universe.agents[i] for i in agent_ids if 0 <= i < len(self.universe.agents)]
        if len(members) != len(agent_ids):
            log_to_queue(self.output_queue, "One or more agent IDs not found.")
            return

        new_coalition = Coalition(name, agent_ids, np.random.rand(2)*self.universe.size)
        self.universe.coalitions[name] = new_coalition
        for agent in members:
            agent.coalition_name = name
        log_to_queue(self.output_queue, f"Coalition '{name}' formed with members {agent_ids}.")

    def _handle_assign_goal(self, parts):
        if len(parts) < 3 or parts[0].lower() != 'coalition' or parts[2].lower() != 'goal':
            raise ValueError("Invalid format. Use 'assign coalition <name> goal <x,y> [cohesion>t]'.")

        name = parts[1]
        if name not in self.universe.coalitions:
            log_to_queue(self.output_queue, f"Coalition '{name}' not found.")
            return

        coords_str = ''.join(parts[3:]).replace(' ', '')
        cohesion_target_str = None
        if 'cohesion>' in coords_str:
            coords_str, cohesion_target_str = coords_str.split('cohesion>')

        x, y = map(float, coords_str.split(','))
        goal = np.array([x, y])

        coalition = self.universe.coalitions[name]
        coalition.goal = goal

        if cohesion_target_str:
            coalition.cohesion_target = float(cohesion_target_str)

        for agent_id in coalition.members:
            agent = self.universe.agents[agent_id]
            agent.set_goal(goal)

        log_to_queue(self.output_queue, f"Assigned new goal ({x},{y}) to coalition '{name}'. Cohesion target set to {coalition.cohesion_target}.")

    def _handle_disband_coalition(self, parts):
        if len(parts) < 2 or parts[0].lower() != 'coalition':
            raise ValueError("Invalid format. Use 'disband coalition <name>'.")

        name = parts[1]
        if name not in self.universe.coalitions:
            log_to_queue(self.output_queue, f"Coalition '{name}' not found.")
            return

        coalition = self.universe.coalitions.pop(name)
        for agent_id in coalition.members:
            agent = self.universe.agents[agent_id]
            agent.coalition_name = None
            agent.role = None
        log_to_queue(self.output_queue, f"Coalition '{name}' disbanded.")

    def _handle_broadcast(self, parts):
        if not parts or parts[0].lower() != 'to':
            raise ValueError("Invalid format. Use 'broadcast <text> to agents <id1,...>' or 'broadcast <text> to coalition <name>'.")

        to_index = parts.index('to')
        text_tokens = parts[:to_index]
        text = ' '.join(text_tokens)

        target_type = parts[to_index+1].lower()
        targets = parts[to_index+2:]

        recipient_ids = set()
        if target_type == 'agents':
            id_str = ''.join(targets)
            recipient_ids.update([int(a) for a in id_str.split(',') if a.strip().isdigit()])
        elif target_type == 'coalition':
            name = targets[0]
            if name in self.universe.coalitions:
                recipient_ids.update(self.universe.coalitions[name].members)
            else:
                log_to_queue(self.output_queue, f"Coalition '{name}' not found.")
                return
        else:
            raise ValueError("Invalid broadcast target. Use 'agents' or 'coalition'.")

        for agent_id in recipient_ids:
            if 0 <= agent_id < len(self.universe.agents):
                agent = self.universe.agents[agent_id]
                agent.kb.ingest(text, f"broadcast:{'manual' if target_type=='agents' else name}")
                if agent.role == 'coordinator':
                    agent.total_reward += 5
        log_to_queue(self.output_queue, f"Broadcasted to {len(recipient_ids)} agents.")

    def _handle_observe(self, parts):
        if not parts:
            log_to_queue(self.output_queue, "Usage: observe <metric> <id> (e.g., 'observe reward 0')")
            return

        metric = parts[0].lower()

        if metric == 'cohesion' and len(parts) >= 3:
            target_type = parts[1].lower()
            if target_type == 'coalition':
                name = parts[2]
                if name in self.universe.coalitions:
                    cohesion = self.universe.cohesion(self.universe.coalitions[name].members)
                    log_to_queue(self.output_queue, f"Cohesion of coalition '{name}': {cohesion:.4f}")
                else:
                    log_to_queue(self.output_queue, f"Coalition '{name}' not found.")
            elif target_type == 'agents':
                id_str = ''.join(parts[2:])
                agent_ids = [int(a) for a in id_str.split(',') if a.strip().isdigit()]
                members = {self.universe.agents[i].id for i in agent_ids if 0 <= i < len(self.universe.agents)}
                cohesion = self.universe.cohesion(members)
                log_to_queue(self.output_queue, f"Cohesion of agents {list(members)}: {cohesion:.4f}")
        elif metric == 'reward' and len(parts) >= 2:
            agent_id = int(parts[1])
            if 0 <= agent_id < len(self.universe.agents):
                reward = self.universe.agents[agent_id].total_reward
                log_to_queue(self.output_queue, f"Agent {agent_id}'s total reward: {reward:.2f}")
            else:
                log_to_queue(self.output_queue, "Agent not found.")
        elif metric == 'entropy':
            log_to_queue(self.output_queue, f"Current system entropy: {self.universe.metrics_collector.get_metric('entropy', 'N/A')}")
        elif metric == 'alignment':
            log_to_queue(self.output_queue, f"Current agent goal alignment: {self.universe.goal_alignment():.4f}")
        else:
            log_to_queue(self.output_queue, f"Unknown observation metric: '{metric}'.")


class KnowledgeBase:
    """Manages agent knowledge with semantic retrieval and persistence."""
    def __init__(self, agent_id, agents_dir: str):
        self.agent_id = agent_id
        db_file = os.path.join(agents_dir, f'agent_{agent_id}.db')
        self.conn = sqlite3.connect(db_file)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self.db_lock = threading.Lock()

        self.universe = None # Will be set by the AGIAgent instance

        self._migrate_schema()
        self.vector_store = self._preload_vectors()

    def _migrate_schema(self):
        with self.db_lock:
            # Idempotent table creation
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  agent_id INTEGER NOT NULL,
                  text TEXT NOT NULL,
                  source TEXT DEFAULT 'console',
                  tags TEXT DEFAULT '',
                  created_at REAL DEFAULT (strftime('%s','now')),
                  vector BLOB,
                  symbols TEXT DEFAULT '',
                  recursion_log TEXT DEFAULT '[]'
                );
            ''')
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS crystal(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at REAL DEFAULT (strftime('%s','now')),
                  label TEXT NOT NULL,
                  content TEXT NOT NULL,
                  strength REAL DEFAULT 0.0
                );
            ''')

            # Add missing columns with constant defaults
            self.cursor.execute("PRAGMA table_info(knowledge);")
            existing_columns = {col[1] for col in self.cursor.fetchall()}
            required_columns = {'id', 'agent_id', 'text', 'source', 'tags', 'created_at', 'vector', 'symbols', 'recursion_log'}

            if not existing_columns.issuperset(required_columns):
                # Use a specific list for predictable migration
                cols_to_add = [
                    ('agent_id', 'INTEGER', '-1'),
                    ('source', 'TEXT', "'console'"),
                    ('tags', 'TEXT', "''"),
                    ('created_at', 'REAL', '0'),
                    ('vector', 'BLOB', 'NULL'),
                    ('symbols', 'TEXT', "''"),
                    ('recursion_log', 'TEXT', "'[]'")
                ]

                for col_name, col_type, default_val in cols_to_add:
                    if col_name not in existing_columns:
                        try:
                            self.cursor.execute(f"ALTER TABLE knowledge ADD COLUMN {col_name} {col_type} DEFAULT {default_val};")
                            if col_name == 'created_at':
                                self.cursor.execute("UPDATE knowledge SET created_at = strftime('%s','now') WHERE created_at = 0;")
                        except sqlite3.OperationalError as e:
                            log_to_queue(None, f"Migration error for column {col_name}: {e}")
            self.conn.commit()

    def _preload_vectors(self) -> List[Dict]:
        with self.db_lock:
            try:
                rows = self.cursor.execute("SELECT id, text, vector, source, tags, created_at, symbols, recursion_log FROM knowledge WHERE agent_id = ?", (self.agent_id,)).fetchall()
                vector_store = []
                for row in rows:
                    try:
                        embedding = np.frombuffer(row['vector'], dtype=np.float32) if row['vector'] else np.zeros(384, dtype=np.float32)
                        tags = json.loads(row['tags'] or '[]')
                        symbols = json.loads(row['symbols'] or '[]')
                        recursion_log = json.loads(row['recursion_log'] or '[]')
                        vector_store.append({
                            "id": row['id'],
                            "text": row['text'],
                            "embedding": embedding,
                            "source": row['source'],
                            "tags": tags,
                            "created_at": row['created_at'],
                            "symbols": symbols,
                            "recursion_log": recursion_log
                        })
                    except (json.JSONDecodeError, TypeError):
                        pass # Skip malformed rows
                return vector_store
            except sqlite3.OperationalError as e:
                log_to_queue(None, f"DB error (agent_{self.agent_id}.db) during preload: {e}")
                return []

    def _generate_embedding(self, text: str) -> np.ndarray:
        tokens = re.findall(r'\b\w+\b', text.lower())
        vector = np.zeros(384, dtype=np.float32)
        if tokens:
            for token in tokens:
                hash_val = hash(token) % 384
                vector[hash_val] += 1
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm
        return vector

    def ingest(self, text: str, source: str = "console", tags: Optional[List[str]] = None, symbols: Optional[List[str]] = None, recursion_log: Optional[List[str]] = None) -> None:
        tags = tags or []
        symbols = symbols or []
        recursion_log = recursion_log or []
        embedding = self._generate_embedding(text)

        with self.db_lock:
            try:
                sql = "INSERT INTO knowledge (agent_id, text, source, tags, created_at, vector, symbols, recursion_log) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
                params = (self.agent_id, text, source, json.dumps(tags), time.time(), sqlite3.Binary(embedding.tobytes()), json.dumps(symbols), json.dumps(recursion_log))
                self.cursor.execute(sql, params)
                self.conn.commit()
                last_id = self.cursor.lastrowid
            except Exception as e:
                log_to_queue(None, f"DB error on ingest: {e}")
                return

        # Update in-memory cache
        self.vector_store.append({
            "id": last_id,
            "text": text,
            "embedding": embedding,
            "source": source,
            "tags": tags,
            "created_at": time.time(),
            "symbols": symbols,
            "recursion_log": recursion_log
        })
        self.vector_store = sorted(self.vector_store, key=lambda x: x['created_at'], reverse=True)
        seen_texts = set()
        deduped_store = []
        for item in self.vector_store:
            if item['text'] not in seen_texts:
                deduped_store.append(item)
                seen_texts.add(item['text'])
        self.vector_store = deduped_store

    def semantic_search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        if not self.vector_store:
            return []

        query_vector = self._generate_embedding(query)
        results = []
        now = time.time()

        # Get overrides from universe object
        kb_overrides = self.universe.kb_overrides if self.universe and hasattr(self.universe, 'kb_overrides') else {}
        age_half_life = kb_overrides.get('age_half_life_sec', 10800)
        # Change min_display_score default to 0.05 as requested
        min_display_score = kb_overrides.get('min_display_score', 0.05)
        top_k = top_k or kb_overrides.get('top_k', 1)

        for item in self.vector_store:
            sim = np.dot(item['embedding'], query_vector)
            age_weight = np.exp(-(now - item['created_at']) / age_half_life) if age_half_life > 0 else 1.0

            final_score = sim * age_weight

            if final_score >= min_display_score:
                results.append({
                    "text": item['text'],
                    "score": final_score,
                    "tags": item['tags'],
                    "source": item['source'],
                    "created_at": item['created_at'],
                    "id": item['id'],
                    "symbols": item['symbols'],
                    "recursion_log": item['recursion_log']
                })

        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:top_k] if top_k else results

    def crystal_write(self, label: str, content: str, strength: float = 0.0) -> None:
        with self.db_lock:
            try:
                self.cursor.execute("INSERT INTO crystal (label, content, strength, created_at) VALUES (?, ?, ?, ?)",
                                (label, content, strength, time.time()))
                self.conn.commit()
            except Exception as e:
                log_to_queue(None, f"DB error on crystal write: {e}")

    def crystal_latest(self, n: int = 5) -> List[Dict]:
        with self.db_lock:
            try:
                rows = self.cursor.execute("SELECT label, content, strength FROM crystal ORDER BY created_at DESC LIMIT ?", (n,)).fetchall()
                return [dict(row) for row in rows]
            except Exception as e:
                log_to_queue(None, f"DB error on crystal latest: {e}")
                return []

class DebateArena:
    def __init__(self, universe):
        self.universe = universe
        self.active_debate: Optional[Dict] = None

    def start_debate(self, topic: str, agent_ids: List[int]) -> str:
        if self.active_debate:
            return "Another debate is already in progress."

        # Ensure agents exist and are unique
        unique_agents = set(agent_ids)
        if len(unique_agents) < 2:
            return "Debate requires at least two unique agents."

        participants = []
        for i in unique_agents:
            if 0 <= i < len(self.universe.agents):
                participants.append(i)
            else:
                return f"Agent ID {i} not found."

        self.active_debate = {
            "topic": topic,
            "participants": participants,
            "turn": 0,
            "arguments": [],
            "scores": {agent_id: 0.0 for agent_id in participants}
        }
        return f"Debate on '{topic}' started with agents {participants}."

    def step_debate(self) -> None:
        if self.active_debate is None:
            return

        debate_overrides = self.universe.debate_overrides if hasattr(self.universe, 'debate_overrides') else {}
        max_turns = debate_overrides.get('debate_max_turns', MAX_DEBATE_TURNS)
        render_top_n = debate_overrides.get('render_top_n', 1)
        role_bias = debate_overrides.get('role_bias', {})

        if self.active_debate['turn'] >= max_turns:
            winner = max(self.active_debate['scores'], key=self.active_debate['scores'].get)
            log_to_queue(self.universe.nlp.output_queue, f"Debate concluded. Winner is Agent {winner} with score {self.active_debate['scores'][winner]:.2f}.")
            self.active_debate = None
            return

        # Guard against empty participants list
        if not self.active_debate['participants']:
            log_to_queue(self.universe.nlp.output_queue, "Debate failed: no participants.")
            self.active_debate = None
            return

        active_agent_id = self.active_debate['participants'][self.active_debate['turn'] % len(self.active_debate['participants'])]
        agent = self.universe.agents[active_agent_id]

        results = agent.kb.semantic_search(self.active_debate['topic'], top_k=render_top_n)

        if not results:
            argument = "No strong evidence found."
            score = 0.0
        else:
            best_match = results[0]
            argument = best_match['text']

            diversity_bonus = 0
            repeat_penalty = 0
            if self.active_debate['arguments']:
                last_arg = self.active_debate['arguments'][-1]['text']
                if argument == last_arg:
                    repeat_penalty = debate_overrides.get('repeat_penalty', 0.25)
                else:
                    diversity_bonus = debate_overrides.get('diversity_bonus', 0.15)

            score = best_match['score']
            score += diversity_bonus - repeat_penalty

            # Apply role bias
            score += role_bias.get(agent.role, 0)

        self.active_debate['arguments'].append({"agent": active_agent_id, "text": argument, "score": score})
        self.active_debate['scores'][active_agent_id] += score
        log_to_queue(self.universe.nlp.output_queue, f"Agent {active_agent_id} argues: '{argument}' (Score: {score:.2f})")

        self.active_debate['turn'] += 1

    def end_debate(self) -> None:
        self.active_debate = None

class Universe:
    def __init__(self, num_particles: int, size: float, time_step: float):
        self.size = size
        self.time_step = time_step
        self.time = 0.0
        self.step_counter = 0
        self.goal_radius = max(self.size * 1e-6, 1e12) # Default value
        self.particles: List[Particle] = []
        self.agents: List['AGIAgent'] = []
        self.collisons = []
        self.coalitions: Dict[str, Coalition] = {}

        self.physics_overrides: Dict = {}
        self.kb_overrides: Dict = {}
        self.debate_overrides: Dict = {}
        self.ui_opts: Dict = {}
        # New overrides for emotional, symbolic, and paradox_rules
        self.emotional_overrides: Dict = {}
        self.symbolic_overrides: Dict = {}
        self.paradox_rules_overrides: Dict = {}


        self.metrics_collector = MetricsCollector()
        self.nlp = NaturalLanguageProcessor(self, output_queue=queue.Queue())
        self.debate_arena = DebateArena(self)

        self.active_enhancements = {
            "quantum_entanglement": False,
            "temporal_flux": False,
            "dimensional_folding": False,
            "consciousness_field": False,
            "psionic_energy": False,
            "exotic_matter": False,
            "cosmic_strings": False,
            "neural_quantum_field": False,
            "tachyonic_communication": False,
            "quantum_vacuum": False,
            "holographic_principle": False,
            "dark_matter_dynamics": False,
            "supersymmetry_active": False
        }

        self.reality_editing_params = [G, c, 1e-12]
        self.holographic_data: Dict = {}
        self.multiverse_history = []
        self.paths = {"agents": "agents"}
        self.nlp.universe = self
        self.debate_arena.universe = self
        self.metrics_collector.universe = self
        self._enh_order: Optional[List[str]] = None

    def create_particles(self, num_particles: int) -> None:
        self.particles = []
        for i in range(num_particles):
            pos = np.random.rand(2) * self.size
            vel = np.random.randn(2) * 1e3
            mass = np.random.lognormal(5, 2)
            charge = np.random.choice([0, 0, 0, 1e-19, -1e-19])
            self.particles.append(Particle(pos, vel, mass, charge))

    def step(self) -> None:
        self.time += self.time_step
        self.step_counter += 1

        if self.step_counter % 100 == 0:
            if self.debate_arena.active_debate:
                self.debate_arena.step_debate()

        # Physics update
        num_particles = len(self.particles)
        if num_particles > 0:
            particle_data = np.zeros((num_particles, 13), dtype=np.float64)
            for i, p in enumerate(self.particles):
                particle_data[i, 0] = p.position[0]
                particle_data[i, 1] = p.position[1]
                particle_data[i, 2] = p.mass
                particle_data[i, 3] = p.charge
                particle_data[i, 4] = p.velocity[0]
                particle_data[i, 5] = p.velocity[1]
                particle_data[i, 6] = p.is_exotic
                particle_data[i, 7] = p.is_dark
                particle_data[i, 8] = p.is_entangled
                particle_data[i, 9] = p.entangled_pair
                particle_data[i, 10] = p.psionic_field
                particle_data[i, 11] = p.is_supersymmetric
                particle_data[i, 12] = p.super_partner_id
        else:
            particle_data = np.empty((0, 13), dtype=np.float64)

        if not self._enh_order:
            self._enh_order = list(self.active_enhancements.keys())

        # Ensure active_enhancements is a list/tuple for Numba
        enh_map = {name: i for i, name in enumerate(self._enh_order)}
        active_flags = np.zeros(len(self._enh_order), dtype=np.int64)
        for name, state in self.active_enhancements.items():
            if state and name in enh_map:
                active_flags[enh_map[name]] = 1

        forces = _compute_forces_and_effects(
            particle_data, None, np.array([p.psionic_field for p in self.particles]),
            self.time, np.array(self.reality_editing_params), self.size, active_flags
        )

        for i, p in enumerate(self.particles):
            p.force = forces[i]
            accel = p.force / p.mass if p.mass > 1e-20 else np.zeros(2)
            p.velocity += accel * self.time_step
            # Cap velocity at c
            speed = np.linalg.norm(p.velocity)
            if speed > c:
                p.velocity = (p.velocity / speed) * c
            p.position += p.velocity * self.time_step

        for agent in self.agents:
            agent.step()
            dist_to_goal = np.linalg.norm(agent.position - agent.goal)
            reward_delta = agent.last_goal_dist - dist_to_goal
            agent.total_reward += reward_delta
            agent.neurotransmitters.update(np.random.rand(5), reward_delta) # Update with delta

        self.metrics_collector.update(self)

    def save_state(self, filename: str) -> None:
        state = {
            "time": self.time,
            "step_counter": self.step_counter,
            "goal_radius": self.goal_radius,
            "particles": [{
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
            } for p in self.particles],
            "agents": [{
                "id": a.id,
                "position": a.position.tolist(),
                "velocity": a.velocity.tolist(),
                "goal": a.goal.tolist(),
                "last_goal": a.last_goal.tolist(),
                "total_reward": a.total_reward,
                "goal_achieved": int(a.goal_achieved),
                "role": a.role,
                "avatar": a.avatar,
                "neurotransmitters": a.neurotransmitters.levels,
                "relationships": a.relationships,
                "coalition_name": a.coalition_name,
                "self_awareness_score": a.self_awareness_score,
                "predictive_accuracy": a.predictive_accuracy,
                "last_avatar_mutation": a.last_avatar_mutation,
                "affective_state": a.affective_state # New: Emotional state
            } for a in self.agents],
            "coalitions": {name: {"name": c.name, "members": list(c.members), "goal": c.goal.tolist(), "cohesion_target": c.cohesion_target} for name, c in self.coalitions.items()},
            "active_enhancements": self.active_enhancements,
            "reality_editing_params": self.reality_editing_params,
            "kb_overrides": self.kb_overrides,
            "debate_overrides": self.debate_overrides,
            "ui_opts": self.ui_opts,
            "emotional_overrides": self.emotional_overrides, # New
            "symbolic_overrides": self.symbolic_overrides, # New
            "paradox_rules_overrides": self.paradox_rules_overrides # New
        }
        with open(filename, 'w') as f:
            json.dump(state, f, indent=4)

    def load_state(self, filename: str) -> None:
        with open(filename, 'r') as f:
            state = json.load(f)

        self.time = state["time"]
        self.step_counter = state["step_counter"]
        self.goal_radius = state.get("goal_radius", self.size * 1e-6) # Load with default

        self.particles = []
        for p_data in state["particles"]:
            p = Particle(np.array(p_data["position"]), np.array(p_data["velocity"]), p_data["mass"], p_data["charge"])
            p.is_exotic = p_data["is_exotic"]
            p.is_dark = p_data["is_dark"]
            p.is_entangled = p_data["is_entangled"]
            p.entangled_pair = p_data["entangled_pair"]
            p.psionic_field = p_data["psionic_field"]
            p.is_supersymmetric = p_data["is_supersymmetric"]
            p.super_partner_id = p_data["super_partner_id"]
            self.particles.append(p)

        self.agents = []
        for a_data in state["agents"]:
            agent = AGIAgent(self, a_data["id"])
            agent.position = np.array(a_data["position"])
            agent.velocity = np.array(a_data["velocity"])
            agent.goal = np.array(a_data["goal"])
            agent.last_goal = np.array(a_data["last_goal"])
            agent.total_reward = a_data["total_reward"]
            agent.goal_achieved = a_data["goal_achieved"]
            agent.role = a_data.get("role")
            agent.avatar = a_data["avatar"]
            agent.neurotransmitters.levels = a_data["neurotransmitters"]
            agent.relationships = a_data.get("relationships", {})
            agent.coalition_name = a_data.get("coalition_name")
            agent.self_awareness_score = a_data.get("self_awareness_score", 0.0)
            agent.predictive_accuracy = a_data.get("predictive_accuracy", 0.0)
            agent.last_avatar_mutation = a_data.get("last_avatar_mutation", "none")
            agent.affective_state = a_data.get("affective_state", {}) # New: Load emotional state
            self.agents.append(agent)

        self.coalitions = {name: Coalition(name, c["members"], np.array(c["goal"]), c.get("cohesion_target", 0.6)) for name, c in state.get("coalitions", {}).items()}
        self.active_enhancements = state.get("active_enhancements", self.active_enhancements)
        self.reality_editing_params = state.get("reality_editing_params", self.reality_editing_params)
        self.kb_overrides = state.get("kb_overrides", {})
        self.debate_overrides = state.get("debate_overrides", {})
        self.ui_opts = state.get("ui_opts", {})
        self.emotional_overrides = state.get("emotional_overrides", {}) # New
        self.symbolic_overrides = state.get("symbolic_overrides", {}) # New
        self.paradox_rules_overrides = state.get("paradox_rules_overrides", {}) # New

    def get_sociometric_cohesion(self) -> float:
        cohesion_scores = []
        for coalition in self.coalitions.values():
            if len(coalition.members) > 1:
                member_positions = np.array([self.agents[i].position for i in coalition.members])
                centroid = np.mean(member_positions, axis=0)
                cohesion_score = np.mean([np.linalg.norm(pos - centroid) for pos in member_positions])
                cohesion_scores.append(1 / (1 + cohesion_score))
        return np.mean(cohesion_scores) if cohesion_scores else 0.0

    def goal_alignment(self) -> float:
        if not self.agents:
            return 0.0
        goal_vectors = np.array([a.goal - a.position for a in self.agents])
        # Compute the norm of the mean vector without dividing by N
        return np.linalg.norm(np.mean(goal_vectors, axis=0))

class AGIAgent:
    def __init__(self, universe: Universe, id: int):
        self.universe = universe
        self.id = id
        self.position = np.random.rand(2) * universe.size
        self.velocity = np.random.randn(2) * 1e3
        self.goal = np.random.rand(2) * universe.size
        self.last_goal = self.goal.copy()

        self.total_reward = 0
        self.last_total_reward = 0
        self.goal_achieved = 0
        self.role: Optional[str] = None
        self.avatar = [[random.randint(0,1) for _ in range(16)] for _ in range(16)]
        self.last_avatar_mutation = "none"

        self.neurotransmitters = NeurotransmitterSystem()
        self.relationships = {}
        self.coalition_name: Optional[str] = None
        self.self_awareness_score = 0.0
        self.predictive_accuracy = 0.0
        # New: Affective state and symbolic recursion
        self.affective_state = {
            "mood": "neutral",
            "arousal": 0.5,
            "valence": 0.5
        }
        self.recursion_depth = 0
        self.max_recursion_depth = 5

        self.kb = KnowledgeBase(self.id, self.universe.paths["agents"])
        self.kb.universe = self.universe # Link KB to universe for overrides

    def step(self) -> None:
        # AGI behavior logic
        dist_to_goal = np.linalg.norm(self.position - self.goal)
        reward_delta = self.last_goal_dist - dist_to_goal if hasattr(self, 'last_goal_dist') else 0

        # Movement towards goal
        if dist_to_goal > self.universe.goal_radius:
            direction = (self.goal - self.position) / dist_to_goal
            # Incorporate self-awareness and learning rate into movement
            modulation = self.neurotransmitters.get_modulation()
            learning_rate = modulation['learning_rate']
            explore_rate = modulation['explore']

            # Simple predictive model and reward update
            predicted_reward = self.predictive_accuracy * reward_delta
            self.predictive_accuracy = np.clip(self.predictive_accuracy + 0.01 * (reward_delta - predicted_reward), 0, 1)

            force_factor = 1e-15 * learning_rate
            noise_factor = 1e-15 * explore_rate

            self.velocity += (direction * force_factor) + (np.random.randn(2) * noise_factor)
        else:
            self.total_reward += 100
            self.goal_achieved = 1
            self.set_goal(np.random.rand(2) * self.universe.size)

        self.neurotransmitters.update(self.velocity, reward_delta)

        self.last_goal_dist = dist_to_goal

        # Self-awareness update (simple placeholder)
        self.self_awareness_score = (self.self_awareness_score + self.neurotransmitters.levels['serotonin'] + self.neurotransmitters.levels['oxytocin']) / 2

        # New: Update emotional state based on reward delta
        self._update_affective_state(reward_delta)

        # Update relationships
        for other_agent in self.universe.agents:
            if other_agent.id == self.id:
                continue
            dist = np.linalg.norm(self.position - other_agent.position)
            social_cohesion_level = self.neurotransmitters.get_modulation()['social_cohesion']
            social_impact = social_cohesion_level / (dist + 1e-10)
            self.relationships[other_agent.id] = self.relationships.get(other_agent.id, 0) + social_impact

    def _update_affective_state(self, reward_delta: float) -> None:
        """
        Updates the agent's affective state based on a reward delta.
        A simple valence/arousal model.
        """
        valence_delta = 0.1 * reward_delta
        arousal_delta = 0.1 * abs(reward_delta)

        self.affective_state["valence"] = np.clip(self.affective_state["valence"] + valence_delta, 0, 1)
        self.affective_state["arousal"] = np.clip(self.affective_state["arousal"] + arousal_delta, 0, 1)

        if self.affective_state["valence"] > 0.7:
            self.affective_state["mood"] = "joy"
        elif self.affective_state["valence"] < 0.3:
            self.affective_state["mood"] = "distress"
        else:
            self.affective_state["mood"] = "neutral"

    def set_goal(self, xy: "np.ndarray") -> None:
        self.last_goal = self.goal.copy()
        self.goal = xy
        self.goal_achieved = 0

    def mutate_avatar(self, mode: str = "noise") -> None:
        if mode == 'noise':
            self.avatar = [[random.randint(0,1) for _ in range(16)] for _ in range(16)]
        elif mode == 'flip':
            self.avatar = [[1-p for p in row] for row in self.avatar]
        elif mode == 'rotate':
            self.avatar = np.rot90(self.avatar).tolist()
        elif mode == 'anneal':
            # Simple annealing: small random changes
            for _ in range(3):
                i, j = random.randint(0, 15), random.randint(0, 15)
                self.avatar[i][j] = 1 - self.avatar[i][j]
        self.last_avatar_mutation = mode

    def get_intellectual_entropy(self) -> float:
        # A simple measure of diversity of thought
        if not self.kb.vector_store:
            return 0.0
        embeddings = np.array([item['embedding'] for item in self.kb.vector_store])
        if embeddings.shape[0] < 2:
            return 0.0
        cov_matrix = np.cov(embeddings, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))
        return entropy

    def respond_to_query(self, query: str) -> str:
        # Simplified response mechanism
        if 'reward' in query:
            return f"My total reward is {self.total_reward:.2f}. My last reward delta was {self.total_reward - self.last_total_reward:.2f}."
        if 'knowledge' in query:
            results = self.kb.semantic_search(query)
            # Add dedup here and top_k based on kb_overrides
            deduped_results = []
            seen = set()
            for r in results:
                if r['text'] not in seen:
                    deduped_results.append(r)
                    seen.add(r['text'])

            kb_overrides = self.universe.kb_overrides if hasattr(self.universe, 'kb_overrides') else {}
            top_k = kb_overrides.get('top_k', 1)
            # Change min_display_score default to 0.05 as requested
            min_score = kb_overrides.get('min_display_score', 0.05)

            filtered_results = [r for r in deduped_results if r['score'] >= min_score][:top_k]

            if not filtered_results:
                return "No relevant knowledge found."

            return "Knowledge found:\n" + "\n".join([f" - {r['text']} (Score: {r['score']:.2f}, Source: {r['source']})" for r in filtered_results])
        return "Query not understood."

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Returns a JSON-ready summary of the agent's state.
        """
        return {
            "id": self.id,
            "total_reward": self.total_reward,
            "self_awareness_score": self.self_awareness_score,
            "affective_state": self.affective_state,
            "last_avatar_mutation": self.last_avatar_mutation,
            "role": self.role,
            "relationships": self.relationships,
            "coalition_name": self.coalition_name,
            "goal": self.goal.tolist()
        }

    def _perform_symbolic_recursion(self, query: str, depth: int = 0) -> str:
        """
        Simulates a recursive thought process using symbolic overlays.
        This is a conceptual model of a self-reflective loop.
        """
        if depth >= self.max_recursion_depth:
            return f"Max recursion depth reached ({depth})."

        self.recursion_depth = depth + 1
        recursion_log = [f"Step {self.recursion_depth}: Reflecting on '{query}'"]

        # Use existing knowledge to form a new, "deeper" query
        results = self.kb.semantic_search(query, top_k=3)
        if not results:
            return "No prior knowledge to recurse on."

        new_query = f"Synthesize new meaning from: {', '.join([r['text'] for r in results])}"

        # Log this new thought and symbolic tags
        new_tags = ["recursion", f"depth:{self.recursion_depth}"]
        self.kb.ingest(new_query, source="self-reflection", tags=new_tags, recursion_log=recursion_log)

        # Fictional interpretation of the results
        first_result_symbols = results[0].get('symbols', ['archetype:void'])
        symbolic_output = f"In recursive loop, observed {first_result_symbols[0]} related to {new_query}."

        self.recursion_depth = 0 # Reset recursion counter
        return symbolic_output
