# COMMANDS.md — HEXUNI Command Console

This document lists **all commands currently implemented** in the HEXUNI command console and how to use them. The console is **case-insensitive**, trims whitespace automatically, and returns feedback in the output panel.

---

## Quick View

| Command | Purpose | Minimal Example |
|---|---|---|
| `set goal for agent <id> to <x,y>` | Set an agent’s goal position | `set goal for agent 0 to 1.2e21,3.4e21` |
| `activate <enhancement_name>` | Enable a simulation enhancement flag | `activate temporal flux` |
| `query agent <id> <free text>` | Ask an agent a question (via its KB) | `query agent 2 what do you know about flux` |
| `start debate on <topic> with agents <id1,id2,...>` | Start a multi-agent debate on a topic | `start debate on emergence with agents 0,1,3` |
| `help` | Show the short list of available commands | `help` |

> Tip: `<id>` is zero-based; `x,y` are floats. Enhancements are written with spaces in the command, but internally use underscores (e.g., `temporal_flux`).

---

## 1) Set Goal

**Syntax**  
```
set goal for agent <id> to <x,y>
```

**Description**  
Points the specified agent toward a new target coordinate in world units.

**Arguments**  
- `<id>` — integer (0-based agent index)  
- `<x,y>` — two floats separated by a comma (no spaces required)

**Examples**  
```
set goal for agent 0 to 1.0e21,2.0e21
set goal for agent 3 to 5.5e20,4.1e20
```

**Responses**  
- `Goal for Agent <id> set to (<x>, <y>).`  
- Error if agent id is invalid or if `<x,y>` cannot be parsed.

---

## 2) Activate Enhancement

**Syntax**  
```
activate <enhancement_name>
```

**Description**  
Turns **on** a specific enhancement toggle in the simulation. (There is **no built-in deactivate command** yet; to turn something off, you must do it in code or add a command.)

**Available enhancement names**  
Write them **with spaces** (the console converts to underscores). Current keys include:

- `quantum entanglement`
- `temporal flux`
- `dimensional folding`
- `consciousness field`
- `psionic energy`
- `exotic matter`
- `cosmic strings`
- `neural quantum field`
- `tachyonic communication`
- `quantum vacuum`
- `holographic principle`

**Examples**  
```
activate temporal flux
activate exotic matter
activate quantum entanglement
```

**Responses**  
- `Enhancement '<key>' activated.` (where `<key>` is the underscore version)  
- `Unknown enhancement.` if the name isn’t one of the keys above.

---

## 3) Query Agent

**Syntax**  
```
query agent <id> <free text>
```

**Description**  
Sends a natural-language query to the agent’s knowledge base (KB) and returns a brief retrieval result. The KB stores text with mock embeddings and cosine-like similarity for now.

**Arguments**  
- `<id>` — integer (0-based)  
- `<free text>` — arbitrary string

**Examples**  
```
query agent 2 what is in your knowledge
query agent 1 show me related concepts about alignment
```

**Responses**  
- `Agent <id>: <answer text>` or a message if the KB is empty.

---

## 4) Start Debate

**Syntax**  
```
start debate on <topic> with agents <id1,id2,...>
```

**Description**  
Starts a multi-agent debate on `<topic>` with the specified participants. Requires **at least two** valid agent IDs and **no other active debate**.

**Arguments**  
- `<topic>` — free text topic/title for the debate  
- `<id1,id2,...>` — comma-separated list of integer agent ids

**Examples**  
```
start debate on the nature of consciousness with agents 0,1
start debate on emergent coordination with agents 0,2,3
```

**Responses**  
- A status string from the DebateArena, e.g., “Another debate is already in progress.” or a confirmation that the debate has started.

---

## 5) Help

**Syntax**  
```
help
```

**Description**  
Prints a short, single-line summary of available commands.

**Example**  
```
help
```

---

## Notes & Behaviors

- **Case-insensitive:** Commands are lowercased internally; `SET GOAL FOR AGENT 0 TO ...` works the same as lowercase.
- **Whitespace tolerant:** Leading/trailing spaces are stripped.
- **Parsing:** Commands are parsed with simple regular expressions. Be precise with the phrases shown in **Syntax**.
- **Feedback loop:** Every command pushes a response to the console output panel; errors are human-readable.
- **IDs and ranges:** Agent IDs must exist in the current simulation. Coordinates must be floats.
- **Enhancement toggles:** Only `activate` exists in-console; there is no `deactivate` command in this build.

---

## Ideas for Future Commands (not yet implemented)

- `deactivate <enhancement_name>` — turn features off during runtime
- `save state [<filename>]` / `load state <filename>`
- `export metrics [<filename>]`
- `set camera follow <agent_id>` / `set camera full`
- `ingest doc for agent <id> <text>` — add to agent KB
- `set time step <seconds>` — adjust simulation speed
- `set num particles <n>` — re-seed particle field
- `branch multiverse` — force a snapshot/branch now

---

## Examples Cheat Sheet

```
# Goals
set goal for agent 0 to 1e21,8e20

# Enhancements
activate temporal flux
activate quantum entanglement
activate exotic matter

# Queries
query agent 1 what did you store last
query agent 0 related concepts for emergence

# Debates
start debate on alignment under noise with agents 0,1,3

# Help
help
```
