"""
Natural Language Processor for Console Commands

This module provides a robust, `shlex`-based parser for handling natural language
commands from the simulation's console. It is designed to be resilient to
varied input formats, handle batch commands, and provide clear, informative
feedback for the user.
"""

import shlex
import queue
import re
import numpy as np
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

DEBATE_RE = re.compile(
    r"^on\s+(?P<topic>.+?)\s+with\s+agents\s+(?P<ids>[\d,\s]+)$",
    re.IGNORECASE,
)

# Constants for structured output and symbolic representation
EMOTIONS = {
    'joy': '😊', 'calm': '😌', 'curiosity': '🤔', 'confusion': '🤯',
    'satisfaction': '✨', 'frustration': '💢', 'surprise': '😲'
}

SIGILS = {
    'knowledge': '📜', 'reward': '💎', 'social': '🤝',
    'entropy': '🌀', 'alignment': '🔗', 'cohesion': '🧩',
    'broadcast': '📡', 'reflection': '🧠', 'error': '🚫'
}

class NaturalLanguageProcessor:
    """
    Parses natural language console commands for the simulation.

    This class provides robust parsing, error handling, and command dispatching
    for a variety of administrative and observational commands, with enhanced,
    structured output for deeper insights.
    """
    def __init__(self, universe: Any, output_queue: queue.Queue):
        """
        Initializes the NLP processor.

        Args:
            universe: The main simulation Universe object.
            output_queue: A queue for sending output messages back to the console.
        """
        self.universe = universe
        self.output_queue = output_queue
        self.verbose = False
        self.handlers = {
            'activate': self._handle_enhancement,
            'deactivate': self._handle_enhancement,
            'set_goal': self._handle_set_goal,
            'set': self._handle_set_goal,
            'prompt': self._handle_prompt,
            'mutate_avatar': self._handle_mutate_avatar,
            'mutate': self._handle_mutate_avatar,
            'query': self._handle_query,
            'start_debate': self._handle_start_debate,
            'start': self._handle_start_debate,
            'end_debate': self._handle_end_debate,
            'end': self._handle_end_debate,
            'form_coalition': self._handle_form_coalition,
            'form': self._handle_form_coalition,
            'set_goal_for_coalition': self._handle_set_goal_for_coalition,
            'broadcast': self._handle_broadcast,
            'observe': self._handle_observe,
            'role': self._handle_role,
            'help': self._handle_help,
            'reflect': self._handle_reflect,
        }

    def _log_output(self, message: str, emotion: str = 'calm', sigil: str = None) -> None:
        """Sends a message to the output queue with added metadata."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sim_time = f"{self.universe.time / (60 * 60 * 24 * 365.25):.2f} years"

        emotion_tag = f"[{EMOTIONS.get(emotion, '😌')}]"
        sigil_tag = f"[{SIGILS.get(sigil, '')}]" if sigil else ""

        output = f"[{current_time} | Sim Time: {sim_time}] {emotion_tag}{sigil_tag} {message}"
        self.output_queue.put(output)

    def _get_help_text(self) -> str:
        """Returns the help text for all commands."""
        return (
            "Available commands:\n"
            "  - activate/deactivate <enhancement_name...> [--verbose]\n"
            "  - set_goal all\n"
            "  - set goal for agent <id> to <x>,<y> | set_goal <id> <x,y>\n"
            "  - prompt agent <id> <text...> [--verbose]\n"
            "  - mutate avatar <id> [flip|rotate|noise|anneal]\n"
            "  - query agent <id> knowledge|reward|social|hypotheses|role|avatar [--verbose]\n"
            "  - start debate on <topic...> with agents <id[,id...]> [--verbose]\n"
            "  - end debate\n"
            "  - form coalition <name> with agents <id[,id...]>\n"
            "  - set goal for coalition <name> to <x>,<y>\n"
            "  - broadcast <text...> to agents <id[,id...]> | broadcast <text...> to coalition <name> [--verbose]\n"
            "  - observe entropy|alignment|cohesion [agents <id,id> | coalition <name>] | observe reward <id> [--verbose]\n"
            "  - role agent <id> <role>\n"
            "  - reflect agent <id> <depth>\n"
            "  - help\n"
            "\n"
            "Use '--verbose' for rich, structured output on supported commands."
        )

    def parse_command(self, text: str) -> None:
        """
        Parses and executes a batch of commands from a single string.

        Args:
            text: The raw command string, possibly containing multiple commands
                  separated by semicolons.
        """
        if text.strip().lower().startswith(('prompt', 'prompt agent')):
            try:
                # Find the start of the prompt text
                parts = shlex.split(text.strip(), comments=True)
                if len(parts) < 3:
                    raise ValueError("Prompt command requires an agent ID and text.")
                command = parts[0]
                agent_id = parts[1]
                # Find the index of the agent ID token to get the full text
                start_index = text.find(agent_id) + len(agent_id)
                prompt_text = text[start_index:].strip()
                self._handle_prompt([agent_id, prompt_text])
            except (ValueError, IndexError, shlex.ParsingError) as e:
                self._log_output(f"Error parsing prompt command: {e}", emotion='frustration', sigil='error')
            return

        commands = re.split(r';\s*', text)
        for cmd in commands:
            cmd = cmd.strip()
            if not cmd or cmd.startswith('#'):
                continue

            cmd = cmd.split('#')[0].strip()
            if not cmd:
                continue

            try:
                parts = shlex.split(cmd)
                self._dispatch_command(parts)
            except (shlex.ParsingError, ValueError) as e:
                self._log_output(f"Error parsing command '{cmd}': {e}", emotion='frustration', sigil='error')
            except Exception as e:
                self._log_output(f"An unexpected error occurred for command '{cmd}': {e}", emotion='frustration', sigil='error')

    def _parse_flags(self, parts: List[str]) -> List[str]:
        """Parses and handles command flags like --verbose."""
        self.verbose = '--verbose' in parts
        if self.verbose:
            return [p for p in parts if p != '--verbose']
        return parts

    def _dispatch_command(self, parts: List[str]) -> None:
        """Dispatches a parsed command to the correct handler."""
        if not parts:
            return

        command_name = parts[0].lower()
        parts_without_flags = self._parse_flags(parts[1:])

        # Handle command synonyms
        if command_name == 'set' and len(parts) > 1 and parts[1].lower() == 'goal':
            command_name = 'set_goal'
        if command_name == 'start' and len(parts) > 1 and parts[1].lower() == 'debate':
            command_name = 'start_debate'
        if command_name == 'end' and len(parts) > 1 and parts[1].lower() == 'debate':
            command_name = 'end_debate'
        if command_name == 'form' and len(parts) > 1 and parts[1].lower() == 'coalition':
            command_name = 'form_coalition'
        if command_name == 'prompt' and len(parts) > 1 and parts[1].lower() == 'agent':
            command_name = 'prompt'

        handler = self.handlers.get(command_name)
        if not handler:
            self._log_output(f"Unknown command: '{command_name}'. Try 'help' for a list of commands.", emotion='confusion')
            return

        handler(parts_without_flags)

    def _handle_enhancement(self, parts: List[str]) -> None:
        """Handles `activate` and `deactivate` commands."""
        if not parts:
            raise ValueError("Missing enhancement name. Usage: activate|deactivate <enhancement_name...>")

        action = parts[0].lower()
        enhancement_name = ' '.join(parts[1:]).replace(' ', '_')
        if enhancement_name in self.universe.active_enhancements:
            self.universe.active_enhancements[enhancement_name] = (action == 'activate')
            self._log_output(f"Enhancement '{enhancement_name}' {'activated' if action == 'activate' else 'deactivated'}.", emotion='satisfaction')
        else:
            known_keys = ", ".join(self.universe.active_enhancements.keys())
            self._log_output(f"Unknown enhancement '{enhancement_name}'. Known keys: {known_keys}", emotion='confusion')

    def _handle_set_goal(self, parts: List[str]) -> None:
        """Handles `set_goal` and `set goal` commands."""
        if not parts:
            raise ValueError("Missing arguments. Usage: set_goal <id> <x,y> or set_goal all")

        if parts[0].lower() == 'all':
            for agent in self.universe.agents:
                agent.set_goal(np.random.rand(2) * self.universe.size)
            self._log_output("Goals for all agents randomized.")
            return

        if parts[0].lower() == 'for' and parts[1].lower() == 'agent':
            if len(parts) < 5 or parts[3].lower() != 'to':
                raise ValueError("Invalid format. Usage: set goal for agent <id> to <x,y>")
            agent_id = self._parse_id(parts[2])
            coords_str = ' '.join(parts[4:])
        else:
            if len(parts) < 2:
                raise ValueError("Invalid format. Usage: set_goal <id> <x,y>")
            agent_id = self._parse_id(parts[0])
            coords_str = ' '.join(parts[1:])

        x, y = self._parse_coords(coords_str)

        if 0 <= agent_id < len(self.universe.agents):
            self.universe.agents[agent_id].set_goal(np.array([x, y]))
            self._log_output(f"Goal for Agent {agent_id} set to ({x}, {y}).", emotion='satisfaction')
        else:
            self._log_output(f"Agent {agent_id} not found.", emotion='confusion', sigil='error')

    def _handle_prompt(self, parts: List[str]) -> None:
        """
        Handles `prompt` command with enhanced, verbose output.

        The command now provides a summary, emotional echo, and confidence score.
        """
        if len(parts) < 2:
            raise ValueError("Invalid format. Usage: prompt [agent] <id> <text...>")

        if parts[0].lower() == 'agent':
            if len(parts) < 3:
                raise ValueError("Invalid format. Usage: prompt agent <id> <text...>")
            agent_id_str = parts[1]
            prompt_text = parts[2]
        else:
            agent_id_str = parts[0]
            prompt_text = parts[1]

        agent_id = self._parse_id(agent_id_str)

        if 0 <= agent_id < len(self.universe.agents):
            self.universe.agents[agent_id].kb.ingest_document(prompt_text, "console")

            # Simulate rich response
            if self.verbose:
                summary = f"Ingested prompt for Agent {agent_id}: '{prompt_text[:50]}...'"
                confidence = np.random.uniform(0.7, 0.95)
                emotion_echo = np.random.choice(list(EMOTIONS.keys()))

                output = {
                    "event": "prompt_ingestion",
                    "agent_id": agent_id,
                    "summary": summary,
                    "confidence": f"{confidence:.2f}",
                    "emotion_echo": emotion_echo,
                    "recursive_reflection": f"Agent {agent_id} initiates self-reflection on `{prompt_text[:20]}...`"
                }
                self._log_output(f"Rich Output:\n{json.dumps(output, indent=2)}", emotion=emotion_echo, sigil='knowledge')
            else:
                self._log_output(f"Prompt ingested for Agent {agent_id}.", emotion='calm', sigil='knowledge')
        else:
            self._log_output(f"Agent {agent_id} not found.", emotion='confusion', sigil='error')

    def _handle_mutate_avatar(self, parts: List[str]) -> None:
        """Handles `mutate avatar <id> [mode]`."""
        if len(parts) < 1:
            raise ValueError("Missing agent ID. Usage: mutate avatar <id> [mode]")

        agent_id = self._parse_id(parts[0])
        mode = parts[1] if len(parts) > 1 else 'noise'

        valid_modes = ['flip', 'rotate', 'noise', 'anneal']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mutation mode '{mode}'. Use one of: {', '.join(valid_modes)}.")

        if 0 <= agent_id < len(self.universe.agents):
            self.universe.agents[agent_id].mutate_avatar(mode)
            self._log_output(f"Agent {agent_id}'s avatar mutated with mode '{mode}'.")
        else:
            self._log_output(f"Agent {agent_id} not found.", emotion='confusion', sigil='error')

    def _handle_query(self, parts: List[str]) -> None:
        """Handles `query agent <id> <item>` with enhanced output."""
        if len(parts) < 2:
            raise ValueError("Missing agent ID or item. Usage: query agent <id> <item>")

        agent_id = self._parse_id(parts[0])
        query_item = parts[1].lower()

        if 0 <= agent_id < len(self.universe.agents):
            agent = self.universe.agents[agent_id]
            response = {"query_item": query_item, "agent_id": agent_id, "result": "N/A"}

            if query_item == 'knowledge':
                results = agent.kb.semantic_search("what do I know?", top_k=5)
                if self.verbose:
                    response['result'] = [{"text": r['text'], "confidence": f"{r['score']:.2f}"} for r in results]
                    self._log_output(f"Knowledge Query Result:\n{json.dumps(response, indent=2)}", emotion='curiosity', sigil='knowledge')
                else:
                    knowledge_str = "\n".join([f"- {r['text']}" for r in results]) if results else "No relevant knowledge found."
                    self._log_output(f"Agent {agent_id}'s top knowledge:\n{knowledge_str}", emotion='curiosity', sigil='knowledge')

            elif query_item == 'reward':
                response['result'] = f"{agent.total_reward:.2f}"
                self._log_output(f"Agent {agent_id}'s total reward: {response['result']}", sigil='reward')

            elif query_item == 'social':
                response['result'] = f"{self.universe.get_sociometric_cohesion_for_agent(agent):.4f}"
                self._log_output(f"Agent {agent_id}'s social cohesion: {response['result']}", sigil='social')

            elif query_item == 'hypotheses':
                response['result'] = "Hypotheses generation not yet implemented."
                self._log_output(f"Agent {agent_id}: {response['result']}", emotion='calm')

            elif query_item == 'role':
                response['result'] = agent.role or 'None'
                self._log_output(f"Agent {agent_id}'s role: {response['result']}")

            elif query_item == 'avatar':
                response['result'] = self._format_avatar(agent.avatar)
                self._log_output(f"Agent {agent_id}'s avatar:\n{response['result']}")

            else:
                self._log_output(f"Unknown query item: '{query_item}'.", emotion='confusion', sigil='error')
        else:
            self._log_output(f"Agent {agent_id} not found.", emotion='confusion', sigil='error')

    def _handle_start_debate(self, parts: List[str]) -> None:
        """Handles `start debate` command using a regex for robust parsing."""
        line = ' '.join(parts)
        m = DEBATE_RE.match(line.strip())
        if not m:
            self._log_output(
                "Usage: start debate on <topic> with agents <id1,id2,...>",
                emotion='frustration', sigil='error'
            )
            return
        topic = m.group("topic").strip()
        ids = [int(x) for x in m.group("ids").replace(" ", "").split(",") if x.strip().isdigit()]
        ids = sorted(set(ids))
        if len(ids) < 2:
            self._log_output("Debate requires at least two unique agents.")
            return

        if any(i >= len(self.universe.agents) for i in ids):
            self._log_output("One or more agent IDs not found.", emotion='confusion', sigil='error')
            return

        debate_overrides = self.universe.debate_overrides if hasattr(self.universe, 'debate_overrides') else {}

        self.universe.debate_arena.start_debate(topic, ids, debate_overrides)
        if self.verbose:
            self._log_output(
                f"Debate started on '{topic}' with agents {ids}. Transcript will stream in verbose mode.",
                emotion='satisfaction', sigil='debate'
            )
        else:
            self._log_output(f"Debate on '{topic}' started with agents {ids}.", emotion='satisfaction', sigil='debate')

    def _handle_end_debate(self, parts: List[str]) -> None:
        """Handles `end debate` command."""
        self.universe.debate_arena.end_debate()
        self._log_output("Debate ended.")

    def _handle_form_coalition(self, parts: List[str]) -> None:
        """Handles `form coalition <name> with agents <id,id...>`."""
        if len(parts) < 3 or parts[1].lower() != 'with' or parts[2].lower() != 'agents':
            raise ValueError("Invalid format. Usage: form coalition <name> with agents <id,id...>")

        name = parts[0]
        id_str = ''.join(parts[3:])
        agent_ids = [self._parse_id(i) for i in id_str.split(',') if i.strip()]

        if not agent_ids:
            raise ValueError("Coalition must have at least one agent.")

        if any(i >= len(self.universe.agents) for i in agent_ids):
            self._log_output("One or more agent IDs not found.", emotion='confusion', sigil='error')
            return

        self.universe.form_coalition(name, agent_ids)
        self._log_output(f"Coalition '{name}' formed with members {agent_ids}.", emotion='satisfaction', sigil='social')

    def _handle_set_goal_for_coalition(self, parts: List[str]) -> None:
        """Handles `set goal for coalition <name> to <x>,<y>`."""
        if len(parts) < 3 or parts[0].lower() != 'for' or parts[1].lower() != 'coalition':
            raise ValueError("Invalid format. Usage: set goal for coalition <name> to <x>,<y>")

        name = parts[2]
        to_index = parts.index('to') if 'to' in parts else -1
        if to_index == -1 or to_index + 1 >= len(parts):
            raise ValueError("Missing 'to' or coordinates.")

        coords_str = ' '.join(parts[to_index+1:])
        x, y = self._parse_coords(coords_str)

        self.universe.set_goal_for_coalition(name, np.array([x, y]))
        self._log_output(f"Assigned new goal ({x}, {y}) to coalition '{name}'.", emotion='satisfaction', sigil='social')

    def _handle_broadcast(self, parts: List[str]) -> None:
        """Handles `broadcast` command with verbose output."""
        try:
            to_index = parts.index('to')
        except ValueError:
            raise ValueError("Invalid format. Missing 'to'. Usage: broadcast <text...> to ...")

        text_tokens = parts[:to_index]
        if not text_tokens:
            raise ValueError("Broadcast message cannot be empty.")

        text = ' '.join(text_tokens)

        if to_index + 1 >= len(parts):
            raise ValueError("Missing broadcast target (agents or coalition).")

        target_type = parts[to_index + 1].lower()

        recipient_ids = []
        if target_type == 'agents':
            ids_str = ''.join(parts[to_index + 2:])
            if not ids_str:
                raise ValueError("Missing agent IDs for broadcast.")
            recipient_ids = [self._parse_id(i) for i in ids_str.split(',') if i.strip()]
        elif target_type == 'coalition':
            if to_index + 2 >= len(parts):
                 raise ValueError("Missing coalition name for broadcast.")
            name = parts[to_index + 2]
            if name in self.universe.coalitions:
                recipient_ids = list(self.universe.coalitions[name].members)
            else:
                self._log_output(f"Coalition '{name}' not found.", emotion='confusion', sigil='error')
                return
        else:
            raise ValueError("Invalid broadcast target. Use 'agents' or 'coalition'.")

        self.universe.broadcast(text, recipient_ids)

        if self.verbose:
            entropy_signature = hash(text) % 1000 # Simple mock entropy
            alignment_delta = np.random.uniform(-0.1, 0.1) # Mock delta
            output = {
                "event": "broadcast_complete",
                "message": text,
                "recipients": recipient_ids,
                "text_entropy_signature": entropy_signature,
                "alignment_delta": f"{alignment_delta:.4f}"
            }
            self._log_output(f"Rich Output:\n{json.dumps(output, indent=2)}", emotion='satisfaction', sigil='broadcast')
        else:
            self._log_output(f"Broadcasted to {len(recipient_ids)} agents.", emotion='satisfaction', sigil='broadcast')

    def _handle_observe(self, parts: List[str]) -> None:
        """Handles `observe` command with structured and verbose output."""
        if not parts:
            raise ValueError("Missing metric to observe. Usage: observe <metric> ...")

        metric = parts[0].lower()
        response = {"metric": metric, "value": "N/A"}

        if metric == 'entropy':
            entropy = self.universe.metrics_collector.get_metric('entropy', 'N/A')
            response['value'] = f"{entropy:.4f}"
            if self.verbose:
                response['signature'] = "High-complexity signature: [A8-C4-B2-F1]"
                self._log_output(f"Entropy Observation:\n{json.dumps(response, indent=2)}", sigil='entropy')
            else:
                self._log_output(f"Current system entropy: {response['value']}", sigil='entropy')
        elif metric == 'alignment':
            alignment = self.universe.goal_alignment()
            response['value'] = f"{alignment:.4f}"
            if self.verbose:
                response['delta'] = np.random.uniform(-0.05, 0.05)
                self._log_output(f"Alignment Observation:\n{json.dumps(response, indent=2)}", sigil='alignment')
            else:
                self._log_output(f"Current goal alignment: {response['value']}", sigil='alignment')
        elif metric == 'cohesion':
            if len(parts) > 1:
                target_type = parts[1].lower()
                if target_type == 'agents':
                    if len(parts) < 3:
                        raise ValueError("Missing agent IDs. Usage: observe cohesion agents <id,id...>")
                    id_str = ''.join(parts[2:])
                    agent_ids = [self._parse_id(i) for i in id_str.split(',') if i.strip()]
                    cohesion = self.universe.cohesion(agent_ids)
                    response['value'] = f"{cohesion:.4f}"
                    response['target'] = 'agents'
                    if self.verbose:
                        response['agent_ids'] = agent_ids
                        self._log_output(f"Cohesion Observation:\n{json.dumps(response, indent=2)}", sigil='cohesion')
                    else:
                        self._log_output(f"Cohesion of agents {agent_ids}: {response['value']}", sigil='cohesion')
                elif target_type == 'coalition':
                    if len(parts) < 3:
                        raise ValueError("Missing coalition name. Usage: observe cohesion coalition <name>")
                    name = parts[2]
                    cohesion = self.universe.get_coalition_cohesion(name)
                    response['value'] = f"{cohesion:.4f}"
                    response['target'] = 'coalition'
                    if self.verbose:
                        response['coalition_name'] = name
                        self._log_output(f"Cohesion Observation:\n{json.dumps(response, indent=2)}", sigil='cohesion')
                    else:
                        self._log_output(f"Cohesion of coalition '{name}': {response['value']}", sigil='cohesion')
                else:
                    raise ValueError(f"Invalid target for cohesion: '{target_type}'. Use 'agents' or 'coalition'.")
            else:
                self._log_output("Usage: observe cohesion [agents <id,id...> | coalition <name>]")
        elif metric == 'reward':
            if len(parts) < 2:
                self._log_output("Missing agent ID. Usage: observe reward <id>", emotion='frustration', sigil='error')
                return
            agent_id = self._parse_id(parts[1])
            if 0 <= agent_id < len(self.universe.agents):
                reward = self.universe.agents[agent_id].total_reward
                response['value'] = f"{reward:.2f}"
                self._log_output(f"Agent {agent_id}'s total reward: {response['value']}", sigil='reward')
            else:
                self._log_output(f"Agent {agent_id} not found.", emotion='confusion', sigil='error')
        else:
            self._log_output(f"Unknown observation metric: '{metric}'.", emotion='confusion', sigil='error')

    def _handle_role(self, parts: List[str]) -> None:
        """Handles `role agent <id> <role>`."""
        if len(parts) < 2:
            raise ValueError("Invalid format. Usage: role agent <id> <role>")

        agent_id = self._parse_id(parts[0])
        role = parts[1].lower()

        valid_roles = ['scout', 'coordinator', 'critic', 'hypothesis', 'none']
        if role not in valid_roles:
            raise ValueError(f"Invalid role '{role}'. Valid roles are: {', '.join(valid_roles)}.")

        if 0 <= agent_id < len(self.universe.agents):
            self.universe.agents[agent_id].role = role if role != 'none' else None
            self._log_output(f"Agent {agent_id}'s role set to '{self.universe.agents[agent_id].role or 'None'}'.", emotion='satisfaction')
        else:
            self._log_output(f"Agent {agent_id} not found.", emotion='confusion', sigil='error')

    def _handle_reflect(self, parts: List[str]) -> None:
        """
        Handles `reflect` command with a recursive output.

        Simulates an agent's recursive self-reflection process.
        """
        if len(parts) < 2:
            raise ValueError("Invalid format. Usage: reflect agent <id> <depth>")

        agent_id = self._parse_id(parts[0])
        depth = self._parse_id(parts[1])

        if 0 <= agent_id < len(self.universe.agents):
            self._log_output(f"Agent {agent_id} begins reflection (depth {depth})...", emotion='curiosity', sigil='reflection')
            for i in range(depth, 0, -1):
                reflection_line = f"  {'  ' * (depth - i)}-> Level {i}: Contemplating existential paradox of purpose."
                self._log_output(reflection_line, emotion='calm', sigil='reflection')
            self._log_output(f"Agent {agent_id} reflection complete.", emotion='satisfaction', sigil='reflection')
        else:
            self._log_output(f"Agent {agent_id} not found.", emotion='confusion', sigil='error')

    def _handle_help(self, parts: List[str]) -> None:
        """Handles `help` command."""
        self._log_output(self._get_help_text())

    def _parse_id(self, s: str) -> int:
        """Safely parses an agent ID from a string."""
        try:
            return int(s)
        except (ValueError, IndexError):
            raise ValueError(f"Invalid agent ID: '{s}'. Must be an integer.")

    def _parse_coords(self, s: str) -> tuple[float, float]:
        """Safely parses coordinates from a string, supporting scientific notation."""
        try:
            parts = s.split(',')
            if len(parts) != 2:
                raise ValueError("Coordinates must be in the format 'x,y'.")
            x = float(parts[0].strip())
            y = float(parts[1].strip())
            return x, y
        except (ValueError, IndexError):
            raise ValueError(f"Invalid coordinates: '{s}'. Must be two numbers separated by a comma.")

    def _format_avatar(self, avatar: List[List[int]]) -> str:
        """Formats a 2D avatar list into a readable string."""
        rows = []
        for row in avatar:
            rows.append(''.join(['■' if p == 1 else '□' for p in row]))
        return '\n'.join(rows)

    def get_help_text(self) -> str:
        return self._get_help_text()
