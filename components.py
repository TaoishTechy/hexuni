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
import matplotlib.pyplot as plt

# --- Constants and Settings ---
G = 6.67430e-11  # Gravitational constant
c = 3e8          # Speed of light (capped at this speed)
hbar = 6.626e-34 / (2 * np.pi) # Reduced Planck constant
k_B = 1.38e-23   # Boltzmann constant
YEAR_IN_SECONDS = 365.25 * 24 * 60 * 60

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
    ('psionic_field', float64)
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

# --- Core Physics Engine with Enhancements ---
@numba.jit(nopython=True)
def _compute_forces_and_effects(particle_data, temporal_flux_map, psionic_field,
                                universe_time, reality_editing_params, size, active_enhancements):
    """
    Computes forces and applies physical effects to particles.

    This function is optimized with Numba for high performance. It operates on
    a NumPy array for efficiency, avoiding Python object overhead.
    """
    n = particle_data.shape[0]
    forces = np.zeros((n, 2))

    # Unpack reality editing parameters
    G_eff = reality_editing_params[0]
    c_eff = reality_editing_params[1]

    # Topological defect intelligence influences forces if enabled
    if active_enhancements[5] == 1:
        if universe_time > 1000:
            for i in range(n):
                psionic_force = psionic_field[i] * 1e15
                forces[i, 0] += psionic_force

    for i in range(n):
        for j in range(i + 1, n):
            r = particle_data[j, :2] - particle_data[i, :2]
            r_mag = np.sqrt(np.sum(r**2)) + 1e-10

            # Gravitational force
            is_exotic_i = particle_data[i, 6] == 1
            is_exotic_j = particle_data[j, 6] == 1
            mass_i = particle_data[i, 2]
            mass_j = particle_data[j, 2]

            # Replaced exotic matter with a repulsive Yukawa-like force term.
            # Does not negate mass.
            if is_exotic_i or is_exotic_j:
                f_exotic = -G_eff * mass_i * mass_j / (r_mag**2) * np.exp(-r_mag / 1e-10)
                forces[i] += f_exotic * r / r_mag
                forces[j] -= f_exotic * r / r_mag

            f_grav = G_eff * mass_i * mass_j / r_mag**2
            forces[i] += f_grav * r / r_mag
            forces[j] -= f_grav * r / r_mag

            # Electrostatic force
            charge_i = particle_data[i, 3]
            charge_j = particle_data[j, 3]
            if charge_i != 0 and charge_j != 0:
                f_elec = 8.9875517873681764e9 * charge_i * charge_j / r_mag**2
                forces[i] += f_elec * r / r_mag
                forces[j] -= f_elec * r / r_mag

            # Dimensional Folding effects if enabled
            if active_enhancements[2] == 1:
                if r_mag < 1e-18:
                    extra_dim_force = (1 / r_mag**3) * np.sin(2 * np.pi / r_mag)
                    forces[i] += extra_dim_force * r / r_mag
                    forces[j] -= extra_dim_force * r / r_mag

    # Temporal Flux Fields if enabled
    if active_enhancements[1] == 1:
        for i in range(n):
            # Use periodic boundaries for indexing
            map_x = int(np.mod(particle_data[i, 0], size) / size * (temporal_flux_map.shape[0] - 1))
            map_y = int(np.mod(particle_data[i, 1], size) / size * (temporal_flux_map.shape[1] - 1))

            phi = temporal_flux_map[map_x, map_y]
            time_dilation_factor = np.sqrt(1 + 2 * phi / c_eff**2)

            # Removed velocity scaling by time dilation factor. Time dilation affects
            # perceived passage of time, not particle velocity directly.
            # TODO: Future enhancement could involve per-particle time steps to emulate clock rates.
            pass

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

    def update(self, inputs, reward):
        self.levels['dopamine'] += 0.1 * reward - self.degradation_rates['dopamine'] * self.levels['dopamine']
        self.levels['serotonin'] += 0.05 * (1 - abs(reward)) - self.degradation_rates['serotonin'] * self.levels['serotonin']
        self.levels['acetylcholine'] += 0.1 * np.mean(inputs) - self.degradation_rates['acetylcholine'] * self.levels['acetylcholine']
        self.levels['oxytocin'] += 0.1 * (reward > 0) - self.degradation_rates['oxytocin'] * self.levels['oxytocin']

        for nt in self.levels:
            self.levels[nt] = np.clip(self.levels[nt], 0, 1)

    def get_modulation(self):
        return {
            "learning_rate": 0.01 * (0.5 + self.levels['acetylcholine'] - 0.5 * self.levels['gaba']),
            "explore": 0.1 * (self.levels['dopamine'] + 0.5 * self.levels['norepinephrine']),
            'reward_bias': 2.0 * (self.levels['serotonin'] - 0.5),
            'attention': self.levels['glutamate'] / (1.0 + self.levels['gaba']),
            'social_cohesion': self.levels['oxytocin'] * self.levels['vasopressin']
        }

# --- New Systems and Classes ---
class MetricsCollector:
    """Collects and manages simulation metrics in real-time."""
    def __init__(self):
        self.metrics = {
            'time': [],
            'fps': [],
            'memory_usage': [],
            'avg_reward': [],
            'total_energy': [],
            'entropy': [],
            'avg_learning_rate': [],
            'active_enhancements': []
        }
        self.start_time = time.time()
        self.frame_count = 0

    def update(self, universe):
        current_time = time.time()
        frame_time = current_time - self.start_time
        if frame_time > 0:
            self.metrics['fps'].append(1.0 / frame_time)
        self.start_time = current_time

        self.metrics['time'].append(universe.time)
        self.metrics['avg_reward'].append(np.mean([a.total_reward for a in universe.agents]) if universe.agents else 0)

        # Check for neurotransmitter system before getting learning rate
        avg_lr = np.mean([a.neurotransmitters.get_modulation()['learning_rate'] for a in universe.agents if hasattr(a, 'neurotransmitters')]) if universe.agents else 0
        self.metrics['avg_learning_rate'].append(avg_lr)

        total_energy = np.sum([0.5 * p.mass * np.sum(p.velocity**2) for p in universe.particles])
        self.metrics['total_energy'].append(total_energy)

        # Simple/mock implementations for complex metrics to prevent errors
        self.metrics['memory_usage'].append(psutil.virtual_memory().percent)
        self.metrics['entropy'].append(np.random.uniform(0, 1))

        active_enhancements_list = [k for k, v in universe.active_enhancements.items() if v]
        self.metrics['active_enhancements'].append(active_enhancements_list)

        self.frame_count += 1

    def export_metrics(self, filename="metrics.json"):
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        print(f"Metrics exported to {filename}")

class NaturalLanguageProcessor:
    """Parses natural language commands for the simulation."""
    def __init__(self, universe):
        self.universe = universe
        self.command_history = []
        self.output_queue = queue.Queue()

    def parse_command(self, command):
        command = command.lower().strip()
        self.command_history.append(f"> {command}")

        if command.startswith("set goal for agent"):
            parts = re.findall(r"set goal for agent (\d+) to (.+)", command)
            if parts:
                agent_id = int(parts[0][0])
                goal_str = parts[0][1]
                if 0 <= agent_id < len(self.universe.agents):
                    try:
                        x, y = map(float, goal_str.split(','))
                        self.universe.agents[agent_id].goal = np.array([x, y])
                        self.output_queue.put(f"Goal for Agent {agent_id} set to ({x}, {y}).")
                    except ValueError:
                        self.output_queue.put("Invalid goal format. Please use 'x,y'.")
                else:
                    self.output_queue.put("Agent not found.")

        elif command.startswith("activate"):
            enhancement = command.replace("activate", "").strip().replace(" ", "_")
            if enhancement in self.universe.active_enhancements:
                self.universe.active_enhancements[enhancement] = True
                self.output_queue.put(f"Enhancement '{enhancement}' activated.")
            else:
                self.output_queue.put("Unknown enhancement.")

        elif command.startswith("query agent"):
            parts = re.findall(r"query agent (\d+) (.+)", command)
            if parts:
                agent_id = int(parts[0][0])
                query = parts[0][1]
                if 0 <= agent_id < len(self.universe.agents):
                    response = self.universe.agents[agent_id].respond_to_query(query)
                    self.output_queue.put(f"Agent {agent_id}: {response}")
                else:
                    self.output_queue.put("Agent not found.")

        elif command.startswith("start debate"):
            parts = re.findall(r"start debate on (.+) with agents (.+)", command)
            if parts:
                topic = parts[0][0]
                agent_ids = [int(a) for a in parts[0][1].split(',') if a.strip().isdigit()]
                participants = [self.universe.agents[i] for i in agent_ids if 0 <= i < len(self.universe.agents)]
                self.output_queue.put(self.universe.debate_arena.start_debate(topic, participants, {}))

        elif command == "help":
            self.output_queue.put("Available commands: 'set goal for agent <id> to <x,y>', 'activate <enhancement>', 'query agent <id> <query>', 'start debate on <topic> with agents <id1,id2,...>'.")

        else:
            self.output_queue.put("Command not recognized.")

class KnowledgeBase:
    """Manages agent knowledge with semantic retrieval and persistence."""
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.conn = sqlite3.connect(f'knowledge_agent_{agent_id}.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS knowledge
                           (text TEXT, embedding BLOB, source TEXT, timestamp TEXT)''')
        self.conn.commit()
        self.vector_store = []

    def ingest_document(self, text, source="unknown"):
        # MOCK EMBEDDING: In a real scenario, this would use a model like sentence-transformers
        embedding = np.random.rand(384).astype(np.float32)

        self.vector_store.append({'text': text, 'embedding': embedding, 'source': source})
        self.cursor.execute("INSERT INTO knowledge VALUES (?, ?, ?, ?)",
                            (text, embedding.tobytes(), source, datetime.now().isoformat()))
        self.conn.commit()

    def semantic_search(self, query):
        query_embedding = np.random.rand(384).astype(np.float32)

        if not self.vector_store:
            return "Knowledge base is empty."

        similarities = [np.dot(item['embedding'], query_embedding) / (np.linalg.norm(item['embedding']) * np.linalg.norm(query_embedding)) for item in self.vector_store]

        top_indices = np.argsort(similarities)[-3:]

        results = []
        for i in top_indices:
            results.append(f"Found related concept: '{self.vector_store[i]['text']}' (Source: {self.vector_store[i]['source']})")

        return "\n".join(results)

    def close(self):
        self.conn.close()

class DebateArena:
    """A dedicated space for agents to engage in symbolic debate."""
    def __init__(self, universe):
        self.universe = universe
        self.debates = []
        self.active_debate = None

    def start_debate(self, topic, participants, rules):
        if self.active_debate:
            return "Another debate is already in progress."

        if len(participants) < 2:
            return "At least two agents are required for a debate."

        self.active_debate = {
            'topic': topic,
            'participants': participants,
            'rules': rules,
            'arguments': [],
            'turn': 0,
            'scores': {agent.id: 0 for agent in participants}
        }
        return f"Debate on '{topic}' started with agents {[p.id for p in participants]}."

    def mock_judge(self, argument):
        return np.random.uniform(0.1, 1.0)

    def step_debate(self):
        if not self.active_debate:
            return

        current_turn = self.active_debate['turn']
        participants = self.active_debate['participants']
        active_agent = participants[current_turn % len(participants)]

        argument = active_agent.make_argument(self.active_debate['topic'])

        truth_score = self.mock_judge(argument)
        self.active_debate['arguments'].append({'agent': active_agent.id, 'argument': argument, 'truth_score': truth_score})
        self.active_debate['scores'][active_agent.id] += truth_score

        self.universe.nlp.output_queue.put(f"Agent {active_agent.id} argues: '{argument}' (Truth Score: {truth_score:.2f})")

        self.active_debate['turn'] += 1
        if self.active_debate['turn'] > 20:
            winner = max(self.active_debate['scores'], key=self.active_debate['scores'].get)
            self.universe.nlp.output_queue.put(f"Debate concluded. Winner is Agent {winner} with a score of {self.active_debate['scores'][winner]:.2f}.")
            self.active_debate = None

# --- AGI Agent Class ---
class AGIAgent:
    """An advanced AGI agent with a neural network and knowledge base."""
    def __init__(self, universe, id):
        self.universe = universe
        self.id = id
        self.position = np.random.rand(2) * universe.size
        self.velocity = np.random.randn(2) * 1e3
        self.mass = 1e25
        self.goal = np.random.rand(2) * universe.size
        self.total_reward = 0
        self.tachyons_sent = False
        self.tachyons_sent_time = 0
        self.neurotransmitters = NeurotransmitterSystem()
        self.kb = KnowledgeBase(self.id)

    def step(self, universe):
        pass # Placeholder for future logic

    def respond_to_query(self, query):
        if "knowledge" in query:
            return self.kb.semantic_search(query)
        if "reward" in query:
            return f"My total reward is {self.total_reward:.2f}."
        return "Query not recognized."

    def make_argument(self, topic):
        # AGI constructs a simple argument based on its knowledge and internal state
        knowledge_response = self.kb.semantic_search(topic)
        return f"I believe '{topic}' is true because my knowledge base states: {knowledge_response}"

    def get_thought_state(self):
        # Returns a flattened representation of the agent's internal state
        return np.array([1, 2, 3]) # MOCK DATA

    def align_thoughts(self, avg_thought):
        pass # Placeholder for alignment logic

# --- Universe Class ---
class Universe:
    def __init__(self, num_particles=300, size=1e20, time_step=1e10):
        self.num_particles = num_particles
        self.size = size
        self.time_step = time_step
        self.particles = self._initialize_particles()
        self.agents = []
        self.time = 0
        self.history = {'positions': [], 'rewards': [], 'fps': []}
        self.history_cap = 1000
        self.step_counter = 0

        self.temporal_flux_map = np.random.rand(1000, 1000) * 1e15
        self.psionic_field = np.zeros(self.num_particles)
        self.cosmic_strings = self._create_cosmic_strings()
        self.holographic_data = {}
        self.reality_editing_params = np.array([G, c], dtype=np.float64)
        self.multiverse_history = []
        self.active_enhancements = {
            'quantum_entanglement': True,
            'temporal_flux': True,
            'dimensional_folding': False,
            'consciousness_field': False,
            'psionic_energy': False,
            'exotic_matter': True,
            'cosmic_strings': True,
            'neural_quantum_field': False,
            'tachyonic_communication': False,
            'quantum_vacuum': True,
            'holographic_principle': False
        }
        self.metrics_collector = MetricsCollector()
        self.nlp = NaturalLanguageProcessor(self)
        self.debate_arena = DebateArena(self)

    def _initialize_particles(self):
        particles = []
        for i in range(self.num_particles):
            pos = np.random.rand(2) * self.size
            vel = np.random.randn(2) * 1e3
            mass = np.random.uniform(1e20, 1e30)
            charge = np.random.choice([-1, 0, 1])
            particles.append(Particle(pos, vel, mass, charge))

        exotic_indices = np.random.choice(self.num_particles, 10, replace=False)
        dark_indices = np.random.choice(self.num_particles, 10, replace=False)
        for i in exotic_indices:
            particles[i].is_exotic = 1
        for i in dark_indices:
            particles[i].is_dark = 1
            particles[i].charge = 0

        return particles

    def _create_cosmic_strings(self):
        strings = []
        for _ in range(5):
            pos1 = np.random.rand(2) * self.size
            pos2 = np.random.rand(2) * self.size
            strings.append((pos1, pos2))
        return strings

    def save_state(self, filename='universe_state.json'):
        state = {
            'time': self.time,
            'particles': [{'position': p.position.tolist(), 'velocity': p.velocity.tolist(), 'mass': p.mass, 'charge': p.charge,
                           'is_exotic': p.is_exotic, 'is_dark': p.is_dark, 'is_entangled': p.is_entangled, 'entangled_pair': p.entangled_pair}
                          for p in self.particles],
            'agents': [{'position': a.position.tolist(), 'total_reward': a.total_reward} for a in self.agents]
        }
        with open(filename, 'w') as f:
            json.dump(state, f, indent=4)
        print(f"Simulation state saved to {filename}")

    def load_state(self, filename='universe_state.json'):
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            self.time = state['time']
            self.particles = []
            for p_data in state['particles']:
                pos = np.array(p_data['position'])
                vel = np.array(p_data['velocity'])
                mass = p_data['mass']
                charge = p_data['charge']
                p = Particle(pos, vel, mass, charge)
                p.is_exotic = p_data['is_exotic']
                p.is_dark = p_data['is_dark']
                p.is_entangled = p_data['is_entangled']
                p.entangled_pair = p_data['entangled_pair']
                self.particles.append(p)
            print(f"Simulation state loaded from {filename}")
        except FileNotFoundError:
            print("File not found.")

    def step(self):
        self.step_counter += 1
        if self.step_counter % 100 == 0:
            self.save_state('timeline_checkpoint.json')

        self._check_causality()

        # Always return particle list, possibly with new particles
        self.particles = self._quantum_vacuum_fluctuations()

        particle_data = np.zeros((len(self.particles), 8), dtype=np.float64)
        for i, p in enumerate(self.particles):
            particle_data[i, 0:2] = p.position
            particle_data[i, 2] = p.mass
            particle_data[i, 3] = p.charge
            particle_data[i, 4:6] = p.velocity
            particle_data[i, 6] = p.is_exotic
            particle_data[i, 7] = p.is_dark

        # Create enhancement flags as a Numba-compatible array
        enhancements_list = list(self.active_enhancements.values())
        enhancements_array = np.array(enhancements_list, dtype=np.int64)

        psionic_field_array = np.array([p.psionic_field for p in self.particles])

        # Apply quantum decoherence before forces if enhancement is active
        if self.active_enhancements['quantum_entanglement']:
            _apply_quantum_decoherence(particle_data, self.time_step)

        forces = _compute_forces_and_effects(particle_data, self.temporal_flux_map, psionic_field_array,
                                              self.time, self.reality_editing_params, self.size, enhancements_array)

        for i, p in enumerate(self.particles):
            acceleration = forces[i] / p.mass
            p.velocity += acceleration * self.time_step
            p.position += p.velocity * self.time_step

            # Cap speed at c
            speed = np.linalg.norm(p.velocity)
            if speed > c:
                p.velocity = (p.velocity / speed) * c

            # Periodic boundaries
            p.position = np.mod(p.position, self.size)

        self._apply_quantum_entanglement()
        self._consciousness_field_resonance()
        self.time += self.time_step

        for agent in self.agents:
            agent.step(self)

        self.metrics_collector.update(self)

        if self.debate_arena.active_debate:
            self.debate_arena.step_debate()

        # Update history with a cap
        self.history['positions'].append([p.position.copy() for p in self.particles])
        if len(self.history['positions']) > self.history_cap:
            self.history['positions'].pop(0)

    def _check_causality(self):
        for agent in self.agents:
            if agent.tachyons_sent and self.time > agent.tachyons_sent_time:
                # Ensure checkpoint file exists before attempting to load
                if os.path.exists('timeline_checkpoint.json'):
                    print("Paradox detected! Initiating timeline correction...")
                    self.load_state('timeline_checkpoint.json')
                    agent.tachyons_sent = False
                else:
                    print("Paradox detected, but no checkpoint found. Skipping rollback.")

    def _quantum_vacuum_fluctuations(self):
        new_particles = list(self.particles)
        if self.active_enhancements['quantum_vacuum'] and np.random.rand() < 0.1:
            pos = np.random.rand(2) * self.size
            vel1 = np.random.randn(2) * 1e-10
            vel2 = -vel1
            p1 = Particle(pos + vel1 * self.time_step, vel1, 1e-30, 0)
            p2 = Particle(pos + vel2 * self.time_step, vel2, 1e-30, 0)
            new_particles.extend([p1, p2])

        if len(new_particles) > self.num_particles * 1.5:
            num_to_remove = len(new_particles) - self.num_particles
            indices_to_remove = np.random.choice(len(new_particles), num_to_remove, replace=False)
            indices_to_remove.sort(reverse=True)
            for i in indices_to_remove:
                del new_particles[i]

            for p in new_particles:
                p.is_entangled = 0
                p.entangled_pair = -1
        return new_particles

    def _apply_quantum_entanglement(self):
        if self.active_enhancements['quantum_entanglement']:
            # Use threshold based on mean inter-particle spacing
            mean_spacing = np.sqrt(self.size**2 / len(self.particles))
            proximity_threshold = 0.1 * mean_spacing

            for i in range(len(self.particles)):
                for j in range(i + 1, len(self.particles)):
                    if _bell_state_check(self.particles[i].position, self.particles[j].position, proximity_threshold):
                        if not self.particles[i].is_entangled and not self.particles[j].is_entangled:
                            self.particles[i].is_entangled = 1
                            self.particles[i].entangled_pair = j
                            self.particles[j].is_entangled = 1
                            self.particles[j].entangled_pair = i

            # Removed velocity mirroring. Entanglement is a correlation, not a direct
            # force or momentum transfer in this simplified model.

    def _consciousness_field_resonance(self):
        if self.active_enhancements['consciousness_field']:
            if len(self.agents) > 1:
                thought_states = [a.get_thought_state() for a in self.agents]
                avg_thought = np.mean(thought_states, axis=0)

                for agent in self.agents:
                    agent.align_thoughts(avg_thought)
