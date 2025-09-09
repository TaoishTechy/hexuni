
# HEXUNI — AGI Emergence & Quantum Toy Universe

A hackable, instrumented sandbox for **emergent AGI behaviors** in a **quantum‑flavored 2D toy universe**.  
It blends a fast particle core (Numba), simple agents with per‑agent knowledge bases, a Tkinter + Matplotlib UI, voxel avatars, a natural‑language console (v0.3 parser with **multiline & batch**), and a lightweight debate arena that auto‑progresses and can be force‑ended.

> **Intent:** explore coordination, learning signals, and symbolic play under exotic environmental effects — without pretending to be physically exact.

---

## ✨ Highlights

- **2D particle simulator** (Numba) with gravity, charge, optional short‑range repulsion for “exotic” particles, speed‑cap at `c`, **periodic boundaries**, and a temporal‑flux map (no velocity scaling).
- **Agents (AGIAgent)** with per‑agent **SQLite KB**, prompting (`prompt agent <id> ...`), and a tiny reward‑shaped policy.
- **Voxel avatars** (16×16, grayscale) per agent with deterministic **mutations** (`flip`, `rotate`, `noise`, `anneal`) and an **auto‑mutate** toggle.
- **Debate Arena**: `start debate on <topic> with agents <ids>` auto‑steps each frame, concludes in ~20 turns, winner announced; `end debate` to force stop.
- **Console (v0.3)**: natural‑language commands, **multiline input**, **semicolon batching**, **comments** (`# ...`) allowed, robust whitespace handling, and helpful errors.
- **Metrics dashboard**: FPS, memory, energy, entropy, learning‑rate proxy, sociometric cohesion, intellectual entropy.
- **Snapshots**: timeline checkpointing, optional multiverse saves.

---

## 🔧 Requirements

- Python 3.9+ (3.10 recommended)  
- Packages: `numpy`, `numba`, `matplotlib`, `psutil`, `sqlite3` (stdlib), `tkinter` (system package on Linux)  
  - Linux: `sudo apt-get install python3-tk`
- (Optional) Tk theme `azure.tcl` — app runs fine without it.

---

## 🚀 Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install numpy numba matplotlib psutil

python main.py
```

If you see an `Axes3D` warning or theme load warning, it’s harmless. Use a clean venv for best results.

---

## ⚙️ Configuration (`config.json`)

```json
{
  "num_particles": 750,
  "universe_size": 2.5e22,
  "time_step": 5e9,
  "num_agents": 10
}
```
- **num_particles**: total particle count (O(N²) forces; keep ≤ a few thousand).  
- **universe_size**: world width/height (square).  
- **time_step**: seconds per simulation step (**HUD shows years**).  
- **num_agents**: number of agents to spawn at start.

---

## 🧠 Command Console (v0.3, multiline + batching)

- **Batching:** Split on newlines **and** `;` — executed in order.  
- **Comments:** Lines starting with `#` are ignored; inline `# ...` stripped.  
- **Aliases:** Some commands have concise forms (see below).  
- **Placeholder:** The input’s placeholder, if present, is ignored.

### Core commands

**Set goal**
```
set goal for agent <id> to <x,y>
set_goal <id> <x,y>
set_goal all
```

**Toggle enhancements**
```
activate <enhancement>
deactivate <enhancement>
```
Available keys:  
`quantum_entanglement, temporal_flux, dimensional_folding, consciousness_field, psionic_energy, exotic_matter, cosmic_strings, neural_quantum_field, tachyonic_communication, quantum_vacuum, holographic_principle, dark_matter_dynamics, supersymmetry_active`

**Query / Prompt**
```
query agent <id> <text>
prompt agent <id> <text>
```

**Debate**
```
start debate on <topic> with agents <id1,id2,...>
end debate
```

**Avatar**
```
mutate avatar <id> [flip|rotate|noise|anneal]
```

**Help**
```
help
```

**Unknown command →** `Command not recognized. Try 'help' or 'prompt agent 0 <text>'.`

---

## 🧪 Copy‑Paste Demos

### A) Feature showcase
```
# toggles
activate temporal flux; activate exotic matter
deactivate quantum entanglement   # inline ok
activate psionic energy

# goals
set goal for agent 0 to 1.00e21,2.00e21
set_goal 1 1.50e21,1.50e21
set_goal all

# KB + avatars + debate
prompt agent 0 Store: seek gradients near strings and cosmic strings
query agent 0 knowledge
mutate avatar 0 anneal; mutate avatar 0 noise; mutate avatar 0 flip; mutate avatar 0 rotate
start debate on emergence under temporal flux with agents 0,1
```

### B) Team play (second debate after the first finishes or after `end debate`)
```
activate dimensional folding
activate quantum vacuum
deactivate psionic energy

set_goal 2 8.0e20,7.0e20
set_goal 3 2.2e21,8.5e20
set_goal 4 1.1e21,1.1e21

prompt agent 2 Align with gradients near strings; prompt agent 2 store tactic: vortex-follow
prompt agent 3 Form coalition with agent 4 to bracket the target
prompt agent 4 Mirror agent 3 velocity at half magnitude

mutate avatar 2 rotate
mutate avatar 3 noise
mutate avatar 4 anneal

query agent 2 knowledge
query agent 3 knowledge
query agent 4 knowledge

start debate on coordination vs exploration under flux with agents 2,3,4
```

---

## 🏗️ Architecture

- **`components.py`**
  - `Particle` (Numba jitclass), force kernel `_compute_forces_and_effects(...)`
  - Quantum stubs: `_bell_state_check`, `_apply_quantum_decoherence`
  - `AGIAgent` (policy + KB + voxel avatar + reward shaping)
  - `KnowledgeBase` (SQLite + mock embeddings)
  - `DebateArena` (auto‑turns, winner announcement)
  - `NaturalLanguageProcessor` (v0.3 parser with batching/comments/aliases)
  - `MetricsCollector`
  - `Universe` (state, step loop, enhancements, snapshots)

- **`main.py`**
  - Tkinter **Notebook** with tabs: Universe View, Metrics Dashboard, Command Console, Entity Manager
  - **FuncAnimation** render loop; **HUD** shows years, agents, avg reward, active enhancements
  - Entity Manager with **voxel avatar** view + **Auto‑mutate** toggle
  - Debate progress bar + label (turns/20)

---

## 🛡️ Physics Notes (toy realism)

- Speeds are **capped** at `c`; positions **wrap** at world edges.  
- Temporal flux **does not** scale velocities; it’s a field for potential/structure.  
- Exotic matter is modeled as a **short‑range repulsion**, not negative mass.  
- Decoherence adds small stochastic noise.  
- This is a **toy** environment; use it to probe emergent behavior, not physical accuracy.

---

## 🧯 Troubleshooting

- **Theme errors (`azure.tcl`)** — harmless. Theme switching is guarded; UI falls back gracefully.  
- **Matplotlib 3D/Axes warnings** — safe to ignore if you’re not using 3D.  
- **Low FPS** — lower `num_particles` and/or map resolution; O(N²) forces dominate.  
- **Debate stuck** — it should auto‑advance; if needed, run `end debate` then start a new one.  
- **Zero rewards** — reward shaping shows progress; larger step counts help.

---

## 📜 License

MIT (recommended). Add a `LICENSE` file if distributing.

---

## 🙌 Credits

You — explorer of emergent AGI and playful physics.
