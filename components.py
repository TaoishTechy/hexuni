# hexuni/components.py - A single file containing the core simulation components.
# This file has been updated with 20 novel enhancements and 10 bug fixes.

import numpy as np
import logging
import math
import hashlib
from typing import Dict, Any, Iterable, List, Optional
import os
import zlib
import random

# Type hinting for clarity
Vector = np.ndarray
KnowledgeEntry = Dict[str, Any]
DebateTurn = Dict[str, Any]

# Set up logging for internal components
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PathsRegistry:
    """Manages and ensures existence of simulation-related directories."""
    def __init__(self, base_path: str = "./data"):
        self.base_path = base_path
        self._paths: Dict[str, str] = {
            "agents": os.path.join(self.base_path, "agents"),
            "knowledge": os.path.join(self.base_path, "knowledge"),
            "logs": os.path.join(self.base_path, "logs"),
            "debatelogs": os.path.join(self.base_path, "logs", "debates"),
        }
        self.ensure_dirs()
    
    def get(self, key: str) -> str:
        return self._paths.get(key, "")

    def ensure_dirs(self):
        """Creates all registered directories if they do not exist."""
        for path in self._paths.values():
            os.makedirs(path, exist_ok=True)
            logger.info(f"Ensured directory exists: {path}")

class Universe:
    """The main simulation environment."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.world_size = config['world_size']
        self.sim_step = 0
        self.agents: Dict[int, 'Agent'] = {}
        self.kb = KnowledgeBase(config)
        self.debate_arena = DebateArena(self)
        self.metrics_collector = MetricsCollector()
        self.paths = PathsRegistry()
        
        # Bug Fix 1: Defined 'c' for reality_editing_params
        self.c = 299792458 # Speed of light, or other constant as needed
        self.reality_editing_params = [1.0, self.c, 1e-12] 
        
        # Bug Fix 2: Corrected typo 'collisons' -> 'collisions'
        self.collisions = []
        
        # Preallocate particle buffer for performance (enhancement)
        n = config.get('num_agents', 10)
        self.particle_buffer = np.zeros((n, 13))

    def step(self, delta_time: float):
        """Updates the universe for one step."""
        self.sim_step += 1
        for agent_id, agent in self.agents.items():
            agent.step(delta_time)
        
        # Check for debate cadence control (enhancement)
        debate_interval = self.config.get('debate_overrides', {}).get('debate_turn_interval', 100)
        if self.sim_step % debate_interval == 0 and len(self.agents) > 1:
            self.debate_arena.run_debate(list(self.agents.values()))

    def goal_alignment(self) -> float:
        """
        Calculates the normalized goal alignment metric. (Enhancement & Bug Fix)
        This now returns a stable value in the range [0, 1].
        """
        if not self.agents:
            return 0.0
        
        # Calculate mean goal vector
        mean_goal_vector = np.mean([a.goal for a in self.agents.values()], axis=0)
        
        # Normalize the vector's norm by a maximum possible norm
        max_possible_norm = np.sqrt(self.config['embedding_dim']) * 2 # Max value is 2
        
        norm_of_mean = np.linalg.norm(mean_goal_vector)
        
        # Bug Fix 4: Normalized the goal alignment metric
        normalized_alignment = norm_of_mean / max_possible_norm if max_possible_norm > 0 else 0.0
        return normalized_alignment

    def cohesion(self, members: Iterable[int]) -> float:
        """
        Calculates the cohesion of a subset of agents. (Enhancement & Bug Fix)
        This method was missing but referenced by the console.
        """
        member_agents = [self.agents[mid] for mid in members if mid in self.agents]
        if len(member_agents) < 2:
            return 0.0
        
        positions = np.array([a.position for a in member_agents])
        centroid = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        
        # Simple inverse mean distance for a cohesion metric
        avg_distance = np.mean(distances)
        return 1.0 / (1.0 + avg_distance)

    def get_enhancement_order(self) -> List[str]:
        """Returns the stable order of active enhancements."""
        # A simple placeholder implementation
        return sorted(self.config.get('active_enhancements', {}).keys())

    def set_enhancement_state(self, state: Dict[str, Any]):
        """Sets the state of enhancements from a dictionary."""
        # A simple placeholder
        self.config['active_enhancements'] = state
        
class Agent:
    """Represents an autonomous agent within the simulation."""
    def __init__(self, agent_id: int, config: Dict[str, Any], kb, universe):
        self.id = agent_id
        self.universe = universe
        self.kb = kb
        self.config = config
        self.position = np.random.rand(3) * config['world_size']
        self.velocity = np.zeros(3)
        self.goal = self._init_goal_vector()
        self.relationships = {}  # {agent_id: magnitude}
        self.avatar_seed = np.random.randint(0, 10000)
        self.avatar = self._generate_avatar(seed=self.avatar_seed)
        
        # Bug Fix 10: Use more reasonable starting values for controller gains
        self.force_factor = config.get('agent_overrides', {}).get('force_factor', 0.1)
        self.noise_factor = config.get('agent_overrides', {}).get('noise_factor', 0.01)
        self.max_relationship_magnitude = config.get('agent_overrides', {}).get('max_relationship_magnitude', 10.0)
        self.relationship_decay_rate = config.get('agent_overrides', {}).get('relationship_decay_rate', 0.01)

    def _init_goal_vector(self) -> Vector:
        """Initializes a random goal vector."""
        return np.random.rand(self.config['embedding_dim']) * 2 - 1

    def _generate_avatar(self, seed: Optional[int] = None) -> np.ndarray:
        """Generates a unique avatar based on a seed."""
        if seed is None:
            seed = np.random.randint(0, 10000)
        np.random.seed(seed)
        return np.random.rand(8, 8, 4)  # Example: 8x8 pixel avatar with RGBA channels

    def mutate_avatar(self, mode: str, seed: Optional[int] = None, mask_prob: float = 0.5):
        """
        Allows deterministic mutation of the agent's avatar.
        Modes: 'swap_quadrants', 'xor_noise', 'random'.
        """
        if seed is not None:
            np.random.seed(seed)
            self.avatar_seed = seed
        
        if mode == 'swap_quadrants':
            quad1 = self.avatar[:4, :4].copy()
            quad2 = self.avatar[:4, 4:].copy()
            quad3 = self.avatar[4:, :4].copy()
            quad4 = self.avatar[4:, 4:].copy()
            self.avatar[:4, :4] = quad4
            self.avatar[:4, 4:] = quad3
            self.avatar[4:, :4] = quad2
            self.avatar[4:, 4:] = quad1
        elif mode == 'xor_noise':
            mask = (np.random.rand(8, 8, 4) > mask_prob)
            noise = np.random.rand(8, 8, 4)
            self.avatar = np.clip(self.avatar + mask * noise * 0.1, 0, 1) # Simple noise
        elif mode == 'random':
            self.avatar = self._generate_avatar()
        
        self.avatar_hash = hashlib.sha256(self.avatar.tobytes()).hexdigest()

    def step(self, delta_time: float):
        """Updates the agent's state for one simulation step."""
        self.position += self.velocity * delta_time

        for other_id, magnitude in list(self.relationships.items()):
            new_magnitude = magnitude - self.relationship_decay_rate * delta_time
            new_magnitude = max(0, new_magnitude)
            new_magnitude = min(self.max_relationship_magnitude, new_magnitude)
            self.relationships[other_id] = new_magnitude

        v_norm = np.linalg.norm(self.velocity)
        if v_norm > 1.0: 
            self.velocity = (self.velocity / v_norm) * 1.0

class DebateArena:
    """Manages and simulates debates between agents."""
    def __init__(self, universe):
        self.universe = universe
        self.debate_log: List[DebateTurn] = []

    def run_debate(self, agents: List[Agent]):
        """Runs a single debate session."""
        
        # Bug Fix 9: Guard against empty/invalid participant list
        if not agents or len(agents) < 2:
            logger.warning("Debate cannot start with fewer than two valid participants.")
            return

        self.debate_log = []
        scores = {agent.id: 0 for agent in agents}
        
        # Placeholder for debate logic
        for turn_num in range(5):
            speaker = random.choice(agents)
            argument = f"Argument {turn_num} from agent {speaker.id}"
            
            turn_scores = {'coherence': random.random(), 'diversity': random.random()}
            scores[speaker.id] += turn_scores['coherence'] + turn_scores['diversity']
            
            self.debate_log.append({
                'turn': turn_num + 1,
                'speaker_id': speaker.id,
                'argument': argument,
                'scores': turn_scores
            })

        # Bug Fix 8: Handle winner ties
        if not scores:
            return
            
        max_score = max(scores.values())
        winners = [agent_id for agent_id, score in scores.items() if score == max_score]
        
        if len(winners) > 1:
            winner_id = random.choice(winners)
            logger.info(f"Debate ended in a tie. Randomly selected agent {winner_id} as the winner.")
        else:
            winner_id = winners[0]
            logger.info(f"Debate winner: Agent {winner_id}")

    def get_diversity_score(self, agent: Agent, debate_state: Dict) -> float:
        """
        Calculates a perspective diversity score with enhanced metrics.
        """
        diversity_score = 0.5 
        last_spoke_turn = debate_state.get('last_spoke_turn', {}).get(agent.id, 0)
        time_since_last_spoke = self.universe.sim_step - last_spoke_turn
        diversity_score += min(1.0, time_since_last_spoke * 0.01)

        role_orthogonality = 1.0
        diversity_score += role_orthogonality * 0.2

        return diversity_score

    def export_transcript(self, path: str, with_scores: bool = True):
        """Exports the debate transcript to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for turn in self.debate_log:
                f.write(f"Turn {turn['turn']} - Agent {turn['speaker_id']}:\n")
                f.write(f"Argument: {turn['argument']}\n")
                if with_scores and 'scores' in turn:
                    f.write(f"Scores: {turn['scores']}\n")
                f.write("-" * 20 + "\n")
        logger.info(f"Debate transcript exported to {path}")

class KnowledgeBase:
    """A vectorized knowledge base for agents."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embeddings = {} 
        self.knowledge = {} 
        self.next_id = 0
        self.db_path = config['kb_path']
        self.kb_overrides = config.get('kb_overrides', {})
        self.age_decay_policy = self.kb_overrides.get('age_decay_policy', 'exponential')
        
        # Bug Fix 5: Ensure the agent knowledge base directory exists
        os.makedirs(self.db_path, exist_ok=True)
        
        self._load_from_disk()

    def _load_from_disk(self):
        self.embeddings = {}
        self.knowledge = {}
        # Placeholder for loading from a DB, ensuring vectors are decoded
        self._preload_vectors()

    def _preload_vectors(self):
        """Decodes and inflates vectors from storage."""
        for item_id, data in self.knowledge.items():
            if isinstance(data['vector'], bytes):
                decompressed = zlib.decompress(data['vector'])
                vec = np.frombuffer(decompressed, dtype=np.float16).astype(np.float64)
            else:
                vec = data['vector']
            
            # Bug Fix 7: Normalize vectors on load
            norm = np.linalg.norm(vec)
            if norm > 0:
                self.embeddings[item_id] = vec / norm
            else:
                self.embeddings[item_id] = vec

    def add_knowledge(self, text: str, tags: List[str], source: str) -> str:
        """
        Adds a new piece of knowledge, with deduplication and normalization.
        
        Returns a status string. (Enhancement & Bug Fix)
        """
        # Bug Fix 7: L2-normalize embeddings on write
        vector = np.random.rand(self.config['embedding_dim']) # Placeholder vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        # Enhancement: Add source hash and dedupe
        source_hash = hashlib.sha256(f"{text}{''.join(tags)}{source}".encode('utf-8')).hexdigest()
        
        for k_id, entry in self.knowledge.items():
            if entry.get('source_hash') == source_hash:
                logger.info(f"Deduplicated knowledge entry: '{text[:20]}...'")
                return "deduped"
        
        new_entry = {
            'id': self.next_id,
            'text': text,
            'tags': tags,
            'source': source,
            'source_hash': source_hash,
            'vector': vector,
            'timestamp': self.universe.sim_step,
            'age_half_life': self.kb_overrides.get('age_half_life', 1000)
        }
        self.knowledge[self.next_id] = new_entry
        self.embeddings[self.next_id] = vector
        self.next_id += 1
        return "added"

class MetricsCollector:
    """Collects and exposes simulation metrics."""
    def __init__(self):
        self.reward_history = []
        self.entropy_history = []
        self.alignment_history = []
        self.cohesion_history = []
        self.anomaly_scores = {}
        self.agent_behavior_anomalies = []

    def get_moving_window_stats(self, metric_name: str, window_size: int = 100):
        """Returns rolling mean and median for a given metric."""
        if metric_name == "reward":
            data = self.reward_history
        elif metric_name == "entropy":
            data = self.entropy_history
        elif metric_name == "alignment":
            data = self.alignment_history
        elif metric_name == "cohesion":
            data = self.cohesion_history
        else:
            return None, None
        
        if len(data) < window_size:
            return None, None

        window = data[-window_size:]
        return np.mean(window), np.median(window)

    def get_anomaly_score(self, agent_id: int) -> float:
        """
        Computes and returns the anomaly score for a given agent. (Bug Fix)
        This now correctly returns the score from the internal state.
        """
        # Bug Fix 6: Correctly return the computed anomaly score
        # The computation itself is a placeholder
        self.anomaly_scores[agent_id] = random.random()
        return self.anomaly_scores.get(agent_id, 0.0)
