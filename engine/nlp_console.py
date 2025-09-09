# New, refactored NLPConsole parser and handlers
# Includes robust tokenization, error handling, and a command registry.

import re
import json
import os
from collections import defaultdict
from itertools import chain

class UsageError(Exception):
    """Custom exception to signal a usage error with a help message."""
    def __init__(self, message, command_name=None):
        super().__init__(message)
        self.command_name = command_name

class NLPConsole:
    """
    A robust natural language processor for the hexuni console.

    Handles command parsing, dispatch, validation, and user feedback.
    """
    def __init__(self, output_queue, config, core, handlers):
        self.output_queue = output_queue
        self.config = config
        self.core = core
        self.handlers = handlers
        self.aliases = {}
        self.command_registry = {}  # Stores handler metadata for help text
        self.load_aliases()
        self.build_registry()

    def load_aliases(self):
        """Loads command aliases from a JSON file."""
        aliases_path = self.config.get('aliases_file', 'aliases.json')
        if os.path.exists(aliases_path):
            with open(aliases_path, 'r') as f:
                self.aliases = json.load(f)

    def save_aliases(self):
        """Saves command aliases to a JSON file."""
        aliases_path = self.config.get('aliases_file', 'aliases.json')
        with open(aliases_path, 'w') as f:
                json.dump(self.aliases, f, indent=4)

    def build_registry(self):
        """
        Builds a registry of commands and their metadata from the handlers.
        """
        for attr_name in dir(self.handlers):
            handler = getattr(self.handlers, attr_name)
            if hasattr(handler, '__command_meta__'):
                meta = handler.__command_meta__
                self.command_registry[meta['name']] = meta
                for alias in meta['aliases']:
                    self.command_registry[alias] = meta

    def emit(self, level, message):
        """
        Sends a message to the output queue with a severity level.
        Levels: 'info', 'success', 'warn', 'error'
        """
        self.output_queue.put(f"[{level}] {message}")

    def normalize(self, s):
        """Normalizes whitespace and common unicode characters."""
        if not isinstance(s, str):
            s = str(s)
        for bad_char in ("\r", "\u00A0", "\u200B", "\t"):
            s = s.replace(bad_char, " ")
        return " ".join(s.strip().split())

    def tokenize(self, s):
        """
        Splits a string by spaces, respecting quotes, escapes, and commas.
        Example: 'set_goal 1 "a goal" --force' -> ['set_goal', '1', 'a goal', '--force']
        """
        s = self.normalize(s)
        s = re.sub(r'\\(.)', r'__ESCAPED_CHAR__\1', s)  # Simple escape placeholder
        
        in_quotes = False
        tokens = []
        current_token = []
        
        for char in s:
            if char == '"':
                in_quotes = not in_quotes
                if not in_quotes and current_token:
                    tokens.append("".join(current_token))
                    current_token = []
                elif not in_quotes and not current_token:
                     tokens.append("") # Allow empty quoted strings ""
            elif in_quotes:
                current_token.append(char)
            elif char.isspace() and not in_quotes:
                if current_token:
                    tokens.append("".join(current_token))
                    current_token = []
            else:
                current_token.append(char)
        if current_token:
            tokens.append("".join(current_token))

        # Revert escape characters
        final_tokens = []
        for token in tokens:
            final_tokens.append(token.replace("__ESCAPED_CHAR__", ""))
        return final_tokens

    def parse_ids(self, id_str):
        """
        Parses a string of comma-separated IDs and ranges into a list of integers.
        Example: "1,2,5-8" -> [1, 2, 5, 6, 7, 8]
        Raises ValueError on malformed input.
        """
        id_list = []
        parts = id_str.split(',')
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            if '-' in part:
                try:
                    start, end = map(int, part.split('-'))
                    if start > end:
                        raise ValueError("Range start cannot be greater than end.")
                    id_list.extend(range(start, end + 1))
                except ValueError as e:
                    raise ValueError(f"Invalid range format '{part}'. Expected 'start-end'. {e}")
            else:
                try:
                    id_list.append(int(part))
                except ValueError:
                    raise ValueError(f"Invalid ID '{part}'. Expected a number or range.")
        return id_list

    def parse_xy(self, xy_str):
        """
        Parses a string of comma-separated coordinates into a tuple of floats.
        Example: "10.2, -3" -> (10.2, -3.0)
        Raises ValueError on malformed input.
        """
        parts = xy_str.split(',')
        if len(parts) != 2:
            raise ValueError(f"Invalid coordinate format '{xy_str}'. Expected 'x,y'.")
        try:
            x = float(parts[0].strip())
            y = float(parts[1].strip())
            return (x, y)
        except ValueError:
            raise ValueError(f"Invalid coordinate values '{xy_str}'. Expected numbers.")

    def fuzzy_match(self, input_cmd):
        """
        Finds the closest matching command using Levenshtein distance.
        Returns a list of potential matches.
        """
        # Levenshtein distance implementation
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1
