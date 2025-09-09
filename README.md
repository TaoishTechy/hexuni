# HEXUNI — AGI Emergence & Quantum Toy Universe

A downloadable, hackable sandbox for **emergent AGI behaviors** in a **quantum-flavored 2D toy universe**.  
It combines a fast N-body core (Numba), agent cognition stubs with a persistent knowledge base, a Tkinter + Matplotlib UI, a live metrics dashboard, and a lightweight command console for steering agents and toggling “enhancements.”

> **Goal:** Provide a playful-but-instrumented arena to probe coordination, debate, and learning signals under exotic environmental effects (entanglement tags, temporal flux fields, “exotic matter” repulsion, cosmic strings) — without pretending to be physically exact.

---

## ✨ Features

- **2D N-body physics engine** (Numba-accelerated `Particle` jitclass) with gravity, charge, and optional short-range repulsive term for “exotic” particles.  
- **Temporal flux field** (map-driven potential) and **entanglement tagging** (visual-only correlation markers).  
- **Agents (AGIAgent)** with per-agent **SQLite knowledge bases** and a simple **NLP command console** to query/set goals or start debates.  
- **Tkinter GUI** with **Matplotlib animation**, multi-tab layout (Universe View, Metrics Dashboard, Command Console).  
- **Metrics Collector** tracking FPS, average reward, energy, entropy, learning-rate proxy, and active enhancements.  
- **Multiverse checkpoints**: periodic snapshots; optional causality rollback.  

---

## 🚀 Quickstart

### 1) Create a fresh virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install numpy numba matplotlib psutil
# On Linux you may need Tk (system package): sudo apt-get install python3-tk
```

### 3) Run the sim
```bash
python main.py
```

If you see a warning about `azure.tcl`, that’s just the optional theme; the UI will fall back to the default Tk theme.

---

## ⚙️ Configuration

Edit `config.json` to define the initial universe scale and counts:
```json
{
  "num_particles": 500,
  "universe_size": 1e22,
  "time_step": 1e9,
  "num_agents": 5
}
```
- **num_particles**: initial particle count  
- **universe_size**: square world size (positions wrap around)  
- **time_step**: seconds per simulation step (HUD converts to years for display)  
- **num_agents**: number of AGI agents to spawn

---

## 🕹️ Controls (Command Console)

Type commands in the **Command Console** tab (press **Enter** to submit):

- Set a goal for an agent:
  ```
  set goal for agent 0 to 1.0e21,2.0e21
  ```

- Activate an enhancement (toggle on):
  ```
  activate temporal flux
  activate exotic matter
  activate quantum entanglement
  ```

- Query an agent:
  ```
  query agent 2 what is in your knowledge
  query agent 1 reward status
  ```

- Start a debate between agents:
  ```
  start debate on the nature of consciousness with agents 0,1,3
  ```

- Help:
  ```
  help
  ```

> **Tip:** Enhancements are also listed in the HUD when enabled.

---

## 🧩 Enhancements (Toggles)

Default flags (can be activated via console):
- `quantum_entanglement` — tags nearby pairs; visual lines in the Universe View  
- `temporal_flux` — reads from a flux map (visual/clock-rate concept; no velocity scaling)  
- `dimensional_folding` — optional short-range extra force (off by default)  
- `consciousness_field` — allows thought-state alignment between agents (off by default)  
- `psionic_energy` — psionic force influences (off by default)  
- `exotic_matter` — replaces negative mass with a short-range repulsive term  
- `cosmic_strings` — draws topological defects for effect  
- `neural_quantum_field` — reserved for future work  
- `tachyonic_communication` — paradox/rollback experiments (off by default)  
- `quantum_vacuum` — occasional pair creation with pruning  
- `holographic_principle` — reserved for future work

---

## 🏗️ Architecture at a Glance

- **`components.py`**
  - `Particle` (Numba jitclass): position, velocity, mass, charge, flags
  - Physics kernel: `_compute_forces_and_effects(...)`
  - Quantum stubs: `_bell_state_check(...)`, `_apply_quantum_decoherence(...)`
  - `NeurotransmitterSystem`, `MetricsCollector`, `NaturalLanguageProcessor`
  - `KnowledgeBase` (per-agent SQLite), `DebateArena`, `AGIAgent`
  - `Universe`: particles, agents, enhancements, save/load, step loop

- **`main.py`**
  - Loads `config.json`
  - Builds `Universe`, spawns agents
  - **UniverseVisualizer** (Tkinter): tabs, plots, HUD, and `FuncAnimation` loop

- **`config.json`**
  - Project defaults: particles, size, time step, agent count

---

## 💾 Save/Load & Multiverse

- The sim periodically writes `timeline_checkpoint.json` for timeline safety.  
- Manual snapshots are written as `universe_state.json` or `multiverse_<time>.json`.  
- If you experiment with tachyons/causality, the sim can roll back to the last checkpoint.  

---

## 📈 Metrics & HUD

- **HUD** (Universe View): sim time (years), agent count, avg reward, active enhancements.  
- **Metrics Dashboard**: FPS, total energy, learning-rate proxy, and more.  

Export metrics anytime with:
```python
# Example (inside code): universe.metrics_collector.export_metrics("metrics.json")
```

---

## 🧪 Notes on “Quantum”

This is a **toy** environment. “Quantum” features are modeled as **stochastic or structural** effects (tags, maps, repulsive pockets), not as full quantum dynamics. They’re designed to shape agent behavior and coordination patterns without heavy physics machinery.

---

## 🛠️ Troubleshooting

- **Matplotlib 3D warning** about `Axes3D`: harmless if you’re not using 3D; usually caused by mixed system/pip installs. Use a clean venv.  
- **Tk not found**: install `python3-tk` via your OS package manager.  
- **Numba first-run latency**: the first step compiles the kernels; subsequent steps are fast.  
- **Headless servers**: use a virtual display (Xvfb) or switch to a non-interactive backend and disable Tk UI.

---

## 🙌 Credits

You, the brave explorer. Have fun pushing the edges of emergence and symbolic play.
