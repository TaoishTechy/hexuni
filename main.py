import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import json
import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.animation as animation
import os
import sys
import psutil

# --- Imports from other files ---
from components import (
    AGIAgent,
    Universe,
    NeurotransmitterSystem,
    Particle,
    _compute_forces_and_effects,
    _bell_state_check,
    _apply_quantum_decoherence,
    MetricsCollector,
    NaturalLanguageProcessor,
    KnowledgeBase,
    DebateArena
)

# --- Visualization and Controls (Tkinter-based) ---
class UniverseVisualizer:
    def __init__(self, universe):
        self.universe = universe
        self.root = tk.Tk()
        self.root.title("Universe Simulation V2")
        self.root.geometry("1400x900")
        try:
            self.root.tk.call('source', 'azure.tcl')
            self.root.tk.call('set_theme', 'dark')
        except tk.TclError:
            print("Warning: azure.tcl not found. Using default Tkinter theme.")

        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        # Create frames for each tab
        self.universe_frame = ttk.Frame(self.notebook)
        self.metrics_frame = ttk.Frame(self.notebook)
        self.console_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.universe_frame, text="Universe View")
        self.notebook.add(self.metrics_frame, text="Metrics Dashboard")
        self.notebook.add(self.console_frame, text="Command Console")

        self._setup_universe_view()
        self._setup_metrics_dashboard()
        self._setup_command_console()

        self.ani = None
        self.active_agent_idx = 0
        self.camera_mode = 'follow_agent'

    def _setup_universe_view(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.universe_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.layers = {
            'particles': self.ax.scatter([], [], s=1, c='blue', alpha=0.6),
            'agents': self.ax.scatter([], [], s=50, c='red'),
            'goals': self.ax.scatter([], [], s=30, c='green', marker='x'),
            'entanglement': self.ax.plot([], [], ':', c='cyan')[0],
            'exotic': self.ax.scatter([], [], s=5, c='magenta', alpha=0.8, marker='s'),
            'cosmic_strings': self.ax.plot([], [], 'w--', alpha=0.5)[0]
        }
        self.info_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, fontsize=10)

    def _setup_metrics_dashboard(self):
        self.metrics_fig, self.metrics_ax = plt.subplots(figsize=(10, 8))
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, master=self.metrics_frame)
        self.metrics_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.line_fps, = self.metrics_ax.plot([], [], label='FPS')
        self.line_energy, = self.metrics_ax.plot([], [], label='Total Energy')
        self.metrics_ax.set_title("Real-time Metrics")
        self.metrics_ax.legend()
        self.metrics_ax.set_xlabel("Time Step")
        self.metrics_ax.set_ylabel("Value")

    def _setup_command_console(self):
        self.console_label = ttk.Label(self.console_frame, text="Command Console", font=("TkDefaultFont", 12, "bold"))
        self.console_label.pack(pady=5)
        self.console_output = scrolledtext.ScrolledText(self.console_frame, wrap=tk.WORD, height=20, bg="#2d2d2d", fg="#ffffff")
        self.console_output.pack(padx=10, pady=5, fill=tk.BOTH, expand=1)
        self.console_output.insert(tk.END, "Welcome to the Universe Simulation!\n")
        self.console_output.config(state=tk.DISABLED)

        self.input_frame = ttk.Frame(self.console_frame)
        self.input_frame.pack(fill=tk.X, padx=10, pady=5)
        self.input_label = ttk.Label(self.input_frame, text="Command:")
        self.input_label.pack(side=tk.LEFT)
        self.input_entry = ttk.Entry(self.input_frame, style="TEntry")
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=1, padx=5)
        self.input_entry.bind("<Return>", self.on_submit)

    def on_submit(self, event=None):
        command = self.input_entry.get()
        self.input_entry.delete(0, tk.END)
        self.universe.nlp.parse_command(command)
        self.check_queue()

    def check_queue(self):
        try:
            while True:
                output = self.universe.nlp.output_queue.get_nowait()
                self.console_output.config(state=tk.NORMAL)
                self.console_output.insert(tk.END, f"{output}\n")
                self.console_output.config(state=tk.DISABLED)
                self.console_output.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self.check_queue)

    def update_metrics_charts(self):
        metrics = self.universe.metrics_collector.metrics
        self.line_fps.set_data(range(len(metrics['fps'])), metrics['fps'])
        self.line_energy.set_data(range(len(metrics['total_energy'])), metrics['total_energy'])

        self.metrics_ax.relim()
        self.metrics_ax.autoscale_view()
        self.metrics_canvas.draw()

    def update_universe_view(self, frame):
        # The simulation step is now driven by FuncAnimation, not a separate thread
        self.universe.step()

        if np.random.rand() < 0.001:
            self._multiverse_branch()

        particle_pos = np.array([p.position for p in self.universe.particles])
        self.layers['particles'].set_offsets(particle_pos)

        exotic_pos = np.array([p.position for p in self.universe.particles if p.is_exotic])
        self.layers['exotic'].set_offsets(exotic_pos)

        agent_pos = np.array([a.position for a in self.universe.agents])
        self.layers['agents'].set_offsets(agent_pos)

        goal_pos = np.array([a.goal for a in self.universe.agents])
        self.layers['goals'].set_offsets(goal_pos)

        # Use np.nan to separate segments for blitting
        entangled_x = []
        entangled_y = []
        for i in range(len(self.universe.particles)):
            if self.universe.particles[i].is_entangled:
                j = self.universe.particles[i].entangled_pair
                if j != -1 and j > i and j < len(self.universe.particles):
                    p1 = self.universe.particles[i].position
                    p2 = self.universe.particles[j].position
                    entangled_x.extend([p1[0], p2[0], np.nan])
                    entangled_y.extend([p1[1], p2[1], np.nan])
        self.layers['entanglement'].set_data(entangled_x, entangled_y)

        strings_x = []
        strings_y = []
        for s in self.universe.cosmic_strings:
            strings_x.extend([s[0][0], s[1][0], np.nan])
            strings_y.extend([s[0][1], s[1][1], np.nan])
        self.layers['cosmic_strings'].set_data(strings_x, strings_y)

        # Convert simulation time to years for HUD
        years = self.universe.time / (60 * 60 * 24 * 365.25)
        hud_text = f"Time: {years:.2f} years\n"
        hud_text += f"Agents: {len(self.universe.agents)}\n"
        hud_text += f"Avg Reward: {np.mean([a.total_reward for a in self.universe.agents]) if self.universe.agents else 0:.4f}\n"
        hud_text += "\nActive Enhancements:\n"
        for k, v in self.universe.active_enhancements.items():
            if v: hud_text += f" - {k.replace('_', ' ').title()}\n"
        self.info_text.set_text(hud_text)

        if self.camera_mode == 'follow_agent' and self.universe.agents:
            self.active_agent_idx = (self.active_agent_idx + 1) % len(self.universe.agents)
            agent_pos = self.universe.agents[self.active_agent_idx].position
            self.ax.set_xlim(agent_pos[0] - self.universe.size/10, agent_pos[0] + self.universe.size/10)
            self.ax.set_ylim(agent_pos[1] - self.universe.size/10, agent_pos[1] + self.universe.size/10)
        else:
            self.ax.set_xlim(0, self.universe.size)
            self.ax.set_ylim(0, self.universe.size)

        self.update_metrics_charts()

        return list(self.layers.values()) + [self.info_text]

    def _multiverse_branch(self):
        print("Reality branching into new timeline...")
        probability = np.random.uniform(0.1, 0.9)
        filename = f'multiverse_{self.universe.time:.0f}.json'
        self.universe.save_state(filename)
        self.universe.multiverse_history.append({'time': self.universe.time, 'state_file': filename, 'probability': probability})

    def run(self):
        self.ani = animation.FuncAnimation(self.fig, self.update_universe_view, interval=50, blit=False)
        self.root.mainloop()

# --- Main Execution ---
if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)

    universe = Universe(
        num_particles=config['num_particles'],
        size=config['universe_size'],
        time_step=config['time_step']
    )

    for i in range(config['num_agents']):
        agent = AGIAgent(universe, id=i)
        universe.agents.append(agent)

    # Removed the background threading. FuncAnimation will now drive the
    # simulation step directly, simplifying the architecture and
    # preventing race conditions.
    visualizer = UniverseVisualizer(universe)
    visualizer.run()
