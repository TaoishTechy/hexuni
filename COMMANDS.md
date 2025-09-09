
# HexUni Command Reference (COMMANDS.md)

_Last updated: 2025-09-09T14:40:24Z_

This document describes the interactive console commands supported by HexUni.
It is **extensive** and written to be copy‑paste friendly, with clear syntax,
options, and examples. It aligns with the canonical command handlers exposed by
the console parser (see `engine/nlp_console.py` in this repo).

> Tip: Anything inside `<angle brackets>` is a placeholder you replace.
> Items in `[square brackets]` are optional.

---

## Table of Contents

- [Getting Started](#getting-started)
- [General Syntax Rules](#general-syntax-rules)
- [Global Options](#global-options)
- [Command Catalogue](#command-catalogue)
  - [help](#help)
  - [activate / deactivate](#activate--deactivate)
  - [set\_goal (and 'set goal')](#set_goal-and-set-goal)
  - [prompt](#prompt)
  - [mutate avatar](#mutate-avatar)
  - [query](#query)
  - [start debate](#start-debate)
  - [end debate](#end-debate)
  - [form coalition](#form-coalition)
  - [set goal for coalition](#set-goal-for-coalition)
  - [broadcast](#broadcast)
  - [observe](#observe)
  - [role](#role)
- [Console Output Modes](#console-output-modes)
  - [Plain mode (default)](#plain-mode-default)
  - [JSON mode (optional extension)](#json-mode-optional-extension)
  - [Verbose mode (optional extension)](#verbose-mode-optional-extension)
- [Error Messages & Troubleshooting](#error-messages--troubleshooting)
- [Reserved & Future Extensions](#reserved--future-extensions)
- [Changelog](#changelog)

---

## Getting Started

Open HexUni and focus the **Console** tab (or press the bound hotkey if your
build offers one). Type a command and press **Enter**.

- Commands are **case-insensitive**.
- You can add comments after a `#` (everything to the right is ignored).
- You can chain multiple commands by separating them with semicolons `;`.
  (Note: the **`prompt`** command is handled specially to keep its text intact.)

Examples:

```
help
activate psionic_energy
set goal for agent 3 to 10.2, 4.7
start debate on 'Should we colonize the void?' with agents 0,1,2
```

---

## General Syntax Rules

- **Agents and coalitions** are identified by numeric IDs (agents) or names (coalitions).
- Floating‑point numbers accept either `.` or `,` as decimal separators inside the pair `x,y`.
- Spacing: multiple spaces are tolerated; trailing comments are allowed.
- If a command accepts **text**, it consumes everything after its required arguments.

### Coordinates

Coordinates are written as `x,y` (no parentheses needed), e.g. `10.5, -3.2`.

### Agent IDs

Agent IDs are **zero‑based**: the first agent is `0`.

---

## Global Options

Some builds include **optional** output controls:

- `--json` — Print JSON objects instead of human text (when supported by the command).
- `--verbose` — Include additional details (timings, deltas, internal notes).

> If your build does not enable these, they are silently ignored.

---

## Command Catalogue

### help

**Description:** Print a brief list of available commands and their usage.

**Usage:**
```
help
```

**Notes:** This command never fails. Useful as a quick sanity check.

---

### activate / deactivate

**Description:** Toggle simulation **enhancements** by name.

**Usage:**
```
activate <enhancement_name...>
deactivate <enhancement_name...>
```

**Examples:**
```
activate psionic_energy
deactivate dark_matter_dynamics
```

**Notes:**
- Enhancement names are case‑insensitive; spaces are converted to underscores.
- If the name is unknown, the console prints the list of known keys.

---

### set_goal (and 'set goal')

**Description:** Set **goal** positions for agents.

**Usage (single agent, short form):**
```
set_goal <agent_id> <x,y>
```

**Usage (single agent, long form):**
```
set goal for agent <agent_id> to <x,y>
```

**Usage (all agents randomized):**
```
set_goal all
```

**Examples:**
```
set_goal 2 15.0, -4.0
set goal for agent 7 to 3.14, 2.72
set_goal all
```

**Behavior:**
- On success, prints: `Goal for Agent <id> set to (<x>, <y>).`
- For `all`, randomizes goals within the world bounds and acknowledges.

---

### prompt

**Description:** Ingest arbitrary **text** into an agent’s knowledge base.

**Usage (explicit):**
```
prompt agent <agent_id> <text...>
```

**Usage (short):**
```
prompt <agent_id> <text...>
```

**Examples:**
```
prompt 0 The valley beyond the ridge has rich helium-3.
prompt agent 3 "Remember to trade with Agent 5 after sunset."
```

**Behavior:**
- The full text after the ID is ingested unmodified.
- On success, prints: `Prompt ingested for Agent <id>.`

---

### mutate avatar

**Description:** Apply a visual mutation to an agent’s avatar.

**Usage:**
```
mutate avatar <agent_id> [flip|rotate|noise|anneal]
```

**Examples:**
```
mutate avatar 1
mutate avatar 4 rotate
```

**Notes:**
- Default mode is `noise` if omitted.
- On success, prints the chosen mode.

---

### query

**Description:** Inspect agent state and data.

**Usage:**
```
query <agent_id> knowledge|reward|social|hypotheses|role|avatar
```

**Examples:**
```
query 2 knowledge
query 0 reward
query 5 avatar
```

**Behavior:**
- `knowledge` — prints top KB items (up to 5).
- `reward` — prints accumulated reward (two decimals).
- `social` — prints computed social cohesion for the agent.
- `hypotheses` — placeholder message if not implemented.
- `role` — prints the current role or "None".
- `avatar` — prints a text preview of the avatar grid.

---

### start debate

**Description:** Begin a structured debate among multiple agents.

**Usage:**
```
start debate on <topic...> with agents <id1,id2,...>
```

**Examples:**
```
start debate on "The ethics of replication" with agents 0,1
start debate on Energy policy with agents 2,3,4
```

**Behavior:**
- Requires at least two **distinct** agents.
- If any agent ID is out of range, prints an error.
- On success, prints: `Debate on '<topic>' started with agents [ids].`

---

### end debate

**Description:** End the current debate session.

**Usage:**
```
end debate
```

**Behavior:** Silently ends the debate and prints `Debate ended.`

---

### form coalition

**Description:** Create a named coalition from a set of agents.

**Usage:**
```
form coalition <name> with agents <id1,id2,...>
```

**Example:**
```
form coalition North with agents 0,3,5
```

**Behavior:**
- Fails if any ID is invalid or list is empty.
- On success, prints: `Coalition '<name>' formed with members [ids].`

---

### set goal for coalition

**Description:** Assign a goal coordinate to a coalition.

**Usage:**
```
set goal for coalition <name> to <x,y>
```

**Example:**
```
set goal for coalition North to 42.0, -7.0
```

**Behavior:** On success, acknowledges with the chosen coordinates.

---

### broadcast

**Description:** Send a message to multiple agents or to a coalition.

**Usage (agents):**
```
broadcast <text...> to agents <id1,id2,...>
```

**Usage (coalition):**
```
broadcast <text...> to coalition <name>
```

**Examples:**
```
broadcast Retreat to agents 1,2,3
broadcast "Hold position" to coalition North
```

**Behavior:** Prints an acknowledgment and target list.

---

### observe

**Description:** Report global or targeted simulation metrics.

**Usage (global):**
```
observe entropy
observe alignment
```

**Usage (cohesion for subset):**
```
observe cohesion agents <id1,id2,...>
observe cohesion coalition <name>
```

**Examples:**
```
observe entropy
observe cohesion agents 0,2,3
observe cohesion coalition North
```

**Behavior:**
- Returns scalar values formatted to 4 decimals where applicable.
- Unknown subkeys produce a helpful usage hint.

---

### role

**Description:** Assign or change an agent’s role.

**Usage:**
```
role agent <id> <role_name>
```

**Example:**
```
role agent 2 diplomat
```

**Behavior:** Prints the updated role.

---

## Console Output Modes

### Plain mode (default)
Human‑readable strings, designed for quick inspection in the Console tab.

### JSON mode (optional extension)
If your build enables JSON output, append `--json` to supported commands to
receive a machine‑readable object. Example:

```
query 0 knowledge --json
```

Possible shape:
```json
{
  "command": "query",
  "agent_id": 0,
  "item": "knowledge",
  "results": [
    { "text": "sample", "score": 0.82, "source": "console" }
  ],
  "time": { "sim": 123.4, "wall": "2025-09-09T12:34:56Z" }
}
```

### Verbose mode (optional extension)
Append `--verbose` to include timing, entropy deltas, or additional diagnostics.

---

## Error Messages & Troubleshooting

- **"Unknown command: 'xyz'."**  
  Use `help` to see the full list.

- **"Agent <id> not found."**  
  The ID is out of range or no agents exist yet. Initialize agents or load a state.

- **"Usage: start debate on <topic> with agents <id1,id2,...>"**  
  Check the commas and spacing. At least two unique IDs are required.

- **"Invalid mutation mode 'foo'."**  
  Valid modes: `flip`, `rotate`, `noise`, `anneal`.

- **"Invalid format. Usage: ..."**  
  The parser prints exact expected syntax. Copy from the Usage block.

---

## Reserved & Future Extensions

The following commands/flags may be present in some builds or planned:

- **alias / unalias / source** — command aliases and script execution.
- **pipe `|`** — pipe output of one command into the next.
- **find** — query entities by attribute expressions.
- **--seed** — set random seed for deterministic runs.
- **--headless** — run without a GUI.

> If your build does not include these, they are ignored or reported as unknown.

---

## Changelog

- **v1.1** — Expanded reference with concrete examples, cohesive structure, optional JSON/verbose notes, and exhaustive error messages.
- **v1.0** — Initial command list (activate/deactivate, set_goal, prompt, mutate, query, debate, coalition, broadcast, observe, role, help).

---
