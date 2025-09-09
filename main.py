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
from collections import deque
import re
from functools import partial
import queue

# --- Imports from other files ---
# Only import classes that are defined in components.py to avoid ImportError.
# Other components will need to be imported from their respective files.
from components import (
    AGIAgent,
    Universe,
    Particle
)
from engine.nlp_console import NLPConsole

# --- Visualization and Controls (Tkinter-based) ---
class UniverseVisualizer:

    def _draw_particle_view(self):
        particle_pos = np.array([p.position for p in self.universe.particles])
        self.layers['particles'].set_offsets(particle_pos)
        exotic_pos = np.array([p.position for p in self.universe.particles if p.is_exotic])
        self.layers['exotic'].set_offsets(exotic_pos)
        dark_matter_pos = np.array([p.position for p in self.universe.particles if p.is_dark])
        self.layers['dark_matter'].set_offsets(dark_matter_pos)
        self.layers['entanglement'].set_alpha(1.0)
        self.layers['cosmic_strings'].set_alpha(1.0)
        self.layers['supersymmetry_partners'].set_alpha(1.0)
        self.layers['holographic_projection'].set_alpha(0.0)
        self.ax.set_facecolor('#000000')

    def _draw_energy_field(self):
        energies = np.array([0.5 * p.mass * np.sum(p.velocity**2) for p in self.universe.particles])
        norm_energies = (energies - energies.min()) / (energies.max() - energies.min() + 1e-10)
        self.layers['particles'].set_color(plt.cm.jet(norm_energies))
        self.layers['exotic'].set_color(plt.cm.jet(norm_energies[np.array([p.is_exotic for p in self.universe.particles], dtype=bool)]))
        self.layers['dark_matter'].set_color(plt.cm.jet(norm_energies[np.array([p.is_dark for p in self.universe.particles], dtype=bool)]))
        self.layers['entanglement'].set_alpha(0.0)
        self.layers['cosmic_strings'].set_alpha(0.0)
        self.layers['supersymmetry_partners'].set_alpha(0.0)
        self.layers['holographic_projection'].set_alpha(0.0)

    def _draw_info_density(self):
        info_density = np.random.uniform(1, 10, len(self.universe.particles))
        self.layers['particles'].set_sizes(info_density * 10)
        self.layers['exotic'].set_sizes(info_density[np.array([p.is_exotic for p in self.universe.particles], dtype=bool)] * 10)
        self.layers['dark_matter'].set_sizes(info_density[np.array([p.is_dark for p in self.universe.particles], dtype=bool)] * 10)
        self.layers['entanglement'].set_alpha(0.0)
        self.layers['cosmic_strings'].set_alpha(0.0)
        self.layers['supersymmetry_partners'].set_alpha(0.0)
        self.layers['holographic_projection'].set_alpha(0.0)

    def _draw_enhancement_overlay(self):
        self.ax.set_facecolor('#000033')
        self.layers['particles'].set_alpha(0.2)
        self.layers['exotic'].set_alpha(0.2)
        self.layers['dark_matter'].set_alpha(0.2)
        self.layers['entanglement'].set_alpha(1.0)
        self.layers['cosmic_strings'].set_alpha(1.0)
        self.layers['supersymmetry_partners'].set_alpha(1.0)
        self.layers['holographic_projection'].set_alpha(0.0)

    def _draw_agent_perception(self):
        self.ax.set_facecolor('#111111')
        for patch in self.ax.patches:
            patch.remove()
        for agent in self.universe.agents:
            circle = plt.Circle(agent.position, 1e16, color='white', fill=False, alpha=0.1)
            self.ax.add_patch(circle)
        self.layers['entanglement'].set_alpha(0.0)
        self.layers['cosmic_strings'].set_alpha(0.0)
        self.layers['supersymmetry_partners'].set_alpha(0.0)
        self.layers['holographic_projection'].set_alpha(0.0)

    def _draw_holographic_view(self):
        self.ax.set_facecolor('#001122')
        self.layers['particles'].set_alpha(0.0)
        self.layers['exotic'].set_alpha(0.0)
        self.layers['dark_matter'].set_alpha(0.0)
        self.layers['entanglement'].set_alpha(0.0)
        self.layers['cosmic_strings'].set_alpha(0.0)
        self.layers['supersymmetry_partners'].set_alpha(0.0)
        self.layers['holographic_projection'].set_alpha(1.0)

        if 'boundary_projection' in self.universe.holographic_data:
            proj = np.array(self.universe.holographic_data['boundary_projection'])
            x, y = proj[:, 0], proj[:, 1]
            self.layers['holographic_projection'].set_data(x, y)
        else:
            self.layers['holographic_projection'].set_data([], [])

    def _draw_multiverse_view(self):
        self.ax.set_facecolor('#000000')
        self.layers['particles'].set_alpha(0.0)
        self.layers['exotic'].set_alpha(0.0)
        self.layers['dark_matter'].set_alpha(0.0)
        self.layers['entanglement'].set_alpha(0.0)
        self.layers['cosmic_strings'].set_alpha(0.0)
        self.layers['supersymmetry_partners'].set_alpha(0.0)
        self.layers['holographic_projection'].set_alpha(0.0)

        if self.universe.multiverse_history:
            for branch in self.universe.multiverse_history:
                branch_start = (self.universe.time / 2, self.universe.size / 2)
                branch_end = (branch['time'], self.universe.size * branch['probability'])
                self.ax.plot([branch_start[0], branch_end[0]], [branch_start[1], branch_end[1]], '-', color='white', alpha=branch['probability'], zorder=-1)

    def __init__(self, universe):
        self.universe = universe
        self.root = tk.Tk()
        self.root.title("Universe Simulation v0.4")
        self.root.geometry("1400x900")
        try:
            self.root.tk.call('source', 'azure.tcl')
            self.root.tk.call('set_theme', 'dark')
        except tk.TclError:
            print("Warning: azure.tcl not found. Using default Tkinter theme.")

        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.visualization_mode = 'particle_view'
        self.view_modes = {
            'particle_view': self._draw_particle_view,
            'energy_field': self._draw_energy_field,
            'info_density': self._draw_info_density,
            'enhancement_overlay': self._draw_enhancement_overlay,
            'agent_perception': self._draw_agent_perception,
            'holographic_view': self._draw_holographic_view,
            'multiverse_view': self._draw_multiverse_view
        }

        self.main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane.pack(expand=True, fill="both", padx=10, pady=10)

        self.left_frame = ttk.Frame(self.main_pane, width=350)
        self.main_pane.add(self.left_frame, weight=0)

        self.right_frame = ttk.Frame(self.main_pane)
        self.main_pane.add(self.right_frame, weight=1)

        self.notebook = ttk.Notebook(self.right_frame)
        self.notebook.pack(expand=True, fill="both")

        self.universe_frame = ttk.Frame(self.notebook)
        self.metrics_frame = ttk.Frame(self.notebook)
        self.console_frame = ttk.Frame(self.notebook)
        self.entity_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.universe_frame, text="Universe View")
        self.notebook.add(self.metrics_frame, text="Metrics Dashboard")
        self.notebook.add(self.console_frame, text="Command Console")
        self.notebook.add(self.entity_frame, text="Entity Manager")

        self._setup_left_panel()
        self._setup_universe_view()
        self._setup_metrics_dashboard()
        self._setup_command_console()
        self._setup_entity_manager()

        self.ani = None
        self.active_agent_idx = 0
        self.camera_mode = 'follow_agent'
        self.is_running = True

    def _setup_left_panel(self):
        hud_frame = ttk.LabelFrame(self.left_frame, text="Simulation Info")
        hud_frame.pack(fill="x", padx=10, pady=5)
        self.time_label = ttk.Label(hud_frame, text="Time: 0.00 years")
        self.time_label.pack(anchor="w")
        self.agent_label = ttk.Label(hud_frame, text="Agents: 0")
        self.agent_label.pack(anchor="w")
        self.fps_label = ttk.Label(hud_frame, text="FPS: 0.00")
        self.fps_label.pack(anchor="w")

        control_frame = ttk.LabelFrame(self.left_frame, text="View Controls")
        control_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(control_frame, text="Visualization Mode:").pack(anchor="w")
        self.mode_var = tk.StringVar(value='particle_view')
        mode_menu = ttk.OptionMenu(control_frame, self.mode_var, 'particle_view', *self.view_modes.keys(), command=self.set_visualization_mode)
        mode_menu.pack(fill="x")

        self.hud_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Show HUD", variable=self.hud_var).pack(anchor="w")

        self.cam_var = tk.StringVar(value='follow_agent')
        cam_menu = ttk.OptionMenu(control_frame, self.cam_var, 'follow_agent', 'follow_agent', 'global', command=self.set_camera_mode)
        cam_menu.pack(fill="x")

        self.pause_button = ttk.Button(self.left_frame, text="Pause Simulation", command=self.toggle_simulation)
        self.pause_button.pack(fill="x", padx=10, pady=5)

    def toggle_simulation(self):
        if self.is_running:
            self.ani.pause()
            self.pause_button.config(text="Resume Simulation")
            self.is_running = False
        else:
            self.ani.resume()
            self.pause_button.config(text="Pause Simulation")
            self.is_running = True

    def set_visualization_mode(self, mode):
        self.visualization_mode = mode

    def set_camera_mode(self, mode):
        self.camera_mode = mode

    def _setup_universe_view(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_facecolor('#000000')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.universe_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.layers = {
            'particles': self.ax.scatter([], [], s=1, c='blue', alpha=0.6),
            'agents': self.ax.scatter([], [], s=50, c='red'),
            'goals': self.ax.scatter([], [], s=30, c='green', marker='x'),
            'entanglement': self.ax.plot([], [], ':', c='cyan')[0],
            'exotic': self.ax.scatter([], [], s=5, c='magenta', alpha=0.8, marker='s'),
            'dark_matter': self.ax.scatter([], [], s=10, c='purple', alpha=0.4, marker='o'),
            'holographic_projection': self.ax.plot([], [], '-', c='white')[0],
            'supersymmetry_partners': self.ax.plot([], [], '--', c='yellow')[0],
            'cosmic_strings': self.ax.plot([], [], 'w--', alpha=0.5)[0],
            'hud_text': self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, fontsize=10, color='white')
        }

        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

    def on_press(self, event):
        if event.button == 1:
            self.is_panning = True
            self.pan_start_x = event.xdata
            self.pan_start_y = event.ydata

    def on_release(self, event):
        if event.button == 1:
            self.is_panning = False

    def on_motion(self, event):
        if self.is_panning and event.xdata and event.ydata:
            dx = event.xdata - self.pan_start_x
            dy = event.ydata - self.pan_start_y
            self.ax.set_xlim(self.ax.get_xlim()[0] - dx, self.ax.get_xlim()[1] - dx)
            self.ax.set_ylim(self.ax.get_ylim()[0] - dy, self.ax.get_ylim()[1] - dy)
            self.pan_start_x = event.xdata
            self.pan_start_y = event.ydata
            self.canvas.draw_idle()

    def on_scroll(self, event):
        base_scale = 1.2
        xdata, ydata = event.xdata, event.ydata
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return

        if xdata is None or ydata is None:
            return

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rely = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])
        new_xlim = [xdata - new_width * relx, xdata + new_width * (1-relx)]
        new_ylim = [ydata - new_height * rely, ydata + new_height * (1-rely)]
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw_idle()

    def _setup_metrics_dashboard(self):
        self.metrics_fig, self.metrics_ax = plt.subplots(2, 2, figsize=(10, 8))
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, master=self.metrics_frame)
        self.metrics_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.ax_perf = self.metrics_ax[0, 0]
        self.line_fps, = self.ax_perf.plot([], [], label='FPS')
        self.line_mem, = self.ax_perf.plot([], [], label='Memory (%)')
        self.ax_perf.set_title("Performance Metrics")
        self.ax_perf.legend()

        self.ax_agi = self.metrics_ax[0, 1]
        self.line_lr, = self.ax_agi.plot([], [], label='Avg Learning Rate')
        self.line_reward, = self.ax_agi.plot([], [], label='Avg Reward')
        self.ax_agi.set_title("AGI Progression")
        self.ax_agi.legend()

        self.ax_phys = self.metrics_ax[1, 0]
        self.line_energy, = self.ax_phys.plot([], [], label='Total Energy')
        self.line_entropy, = self.ax_phys.plot([], [], label='Entropy')
        self.ax_phys.set_title("Universe State")
        self.ax_phys.legend()

        self.ax_adv_agi = self.metrics_ax[1, 1]
        self.line_intellectual_entropy, = self.ax_adv_agi.plot([], [], label='Intellectual Entropy')
        self.line_cohesion, = self.ax_adv_agi.plot([], [], label='Sociometric Cohesion')
        self.ax_adv_agi.set_title("Emergent AGI Metrics")
        self.ax_adv_agi.legend()

    def update_metrics_charts(self):
        metrics = self.universe.metrics
        times = metrics['time']
        if not times: return

        self.line_fps.set_data(times, metrics['fps'])
        self.line_mem.set_data(times, metrics['memory_usage'])
        self.ax_perf.relim(); self.ax_perf.autoscale_view()

        self.line_lr.set_data(times, metrics['avg_learning_rate'])
        self.line_reward.set_data(times, metrics['avg_reward'])
        self.ax_agi.relim(); self.ax_agi.autoscale_view()

        self.line_energy.set_data(times, metrics['total_energy'])
        self.line_entropy.set_data(times, metrics['entropy'])
        self.ax_phys.relim(); self.ax_phys.autoscale_view()

        self.line_intellectual_entropy.set_data(times, metrics['intellectual_entropy'])
        self.line_cohesion.set_data(times, metrics['sociometric_cohesion'])
        self.ax_adv_agi.relim(); self.ax_adv_agi.autoscale_view()

        self.metrics_fig.tight_layout()
        self.metrics_canvas.draw_idle()

    def _setup_command_console(self):
        self.console_label = ttk.Label(self.console_frame, text="Command Console", font=("TkDefaultFont", 12, "bold"))
        self.console_label.pack(pady=5)
        self.console_output = scrolledtext.ScrolledText(self.console_frame, wrap=tk.WORD, height=20, bg="#2d2d2d", fg="#ffffff")
        self.console_output.pack(padx=10, pady=5, fill=tk.BOTH, expand=1)
        self.console_output.insert(tk.END, "Welcome to the Universe Simulation v0.4!\n")
        self.console_output.config(state=tk.DISABLED)

        self.input_frame = ttk.Frame(self.console_frame)
        self.input_frame.pack(fill=tk.X, padx=10, pady=5)
        self.input_label = ttk.Label(self.input_frame, text="Command:")
        self.input_label.pack(side=tk.LEFT)
        self.input_entry = ttk.Entry(self.input_frame, style="TEntry")
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=1, padx=5)
        self.input_entry.bind("<Return>", self.on_submit)

        self.cmd_history = deque(maxlen=20)
        self.history_idx = -1
        self.input_entry.bind("<Up>", self.history_up)
        self.input_entry.bind("<Down>", self.history_down)

    def history_up(self, event):
        if self.history_idx < len(self.cmd_history) - 1:
            self.history_idx += 1
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, self.cmd_history[self.history_idx])

    def history_down(self, event):
        if self.history_idx > 0:
            self.history_idx -= 1
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, self.cmd_history[self.history_idx])
        elif self.history_idx == 0:
            self.history_idx = -1
            self.input_entry.delete(0, tk.END)

    def on_submit(self, event=None):
        command = self.input_entry.get()
        if command:
            self.cmd_history.appendleft(command)
            self.history_idx = -1
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

    def _setup_entity_manager(self):
        entity_pane = ttk.PanedWindow(self.entity_frame, orient=tk.VERTICAL)
        entity_pane.pack(expand=True, fill="both", padx=10, pady=10)

        agent_select_frame = ttk.LabelFrame(entity_pane, text="Select Agent")
        agent_select_frame.pack(fill="x", padx=10, pady=5)
        entity_pane.add(agent_select_frame, weight=1)

        self.agent_var = tk.StringVar()
        self.agent_menu = ttk.OptionMenu(agent_select_frame, self.agent_var, "Select an Agent", "")
        self.agent_menu.pack(fill="x")
        self.agent_var.trace_add("write", self.update_agent_panel)

        self.agent_details_frame = ttk.LabelFrame(entity_pane, text="Agent Details")
        self.agent_details_frame.pack(fill="both", expand=True, padx=10, pady=5)
        entity_pane.add(self.agent_details_frame, weight=3)

    def update_agent_selection(self):
        menu = self.agent_menu["menu"]
        menu.delete(0, "end")

        if not self.universe.agents:
            self.agent_var.set("No Agents")
            menu.add_command(label="No Agents")
            return

        agent_ids = [f"Agent {a.id}" for a in self.universe.agents]
        for agent_id in agent_ids:
            menu.add_command(label=agent_id, command=partial(self.agent_var.set, agent_id))

        if self.agent_var.get() not in agent_ids:
            self.agent_var.set("Select an Agent")

    def update_agent_panel(self, *args):
        for widget in self.agent_details_frame.winfo_children():
            widget.destroy()

        selected_agent_str = self.agent_var.get()
        if not selected_agent_str.startswith("Agent"):
            return

        agent_id = int(selected_agent_str.split()[1])
        agent = next((a for a in self.universe.agents if a.id == agent_id), None)
        if not agent:
            return

        details_frame = ttk.Frame(self.agent_details_frame)
        details_frame.pack(fill="both", expand=True)

        info_label = ttk.Label(details_frame, text=f"ID: {agent.id}\nTotal Reward: {agent.total_reward:.2f}\nPosition: ({agent.position[0]:.2e}, {agent.position[1]:.2e})\nReality Resource: {agent.reality_editing_resource:.2f}\nSelf-Awareness: {agent.self_awareness_score:.2f}")
        info_label.pack(anchor="w", padx=5, pady=5)

        nt_frame = ttk.LabelFrame(details_frame, text="Neurotransmitter Levels")
        nt_frame.pack(fill="x", padx=5, pady=5)
        for nt, level in agent.neurotransmitters.levels.items():
            ttk.Label(nt_frame, text=f"{nt.title()}: {level:.2f}").pack(anchor="w")

        rel_frame = ttk.LabelFrame(details_frame, text="Social Relationships")
        rel_frame.pack(fill="x", padx=5, pady=5)
        if not agent.relationships:
            ttk.Label(rel_frame, text="No relationships yet.").pack(anchor="w")
        else:
            for other_id, score in agent.relationships.items():
                ttk.Label(rel_frame, text=f"Agent {other_id}: {score:.2f}").pack(anchor="w")

    def update_universe_view(self, frame):
        self.universe.step()

        for patch in self.ax.patches:
            patch.remove()

        for layer in self.layers.values():
            if isinstance(layer, plt.matplotlib.collections.PathCollection) or isinstance(layer, plt.matplotlib.lines.Line2D):
                layer.set_alpha(1.0)

        self.view_modes[self.visualization_mode]()

        agent_pos = np.array([a.position for a in self.universe.agents])
        self.layers['agents'].set_offsets(agent_pos)

        goal_pos = np.array([a.goal for a in self.universe.agents])
        self.layers['goals'].set_offsets(goal_pos)

        entangled_x = []
        entangled_y = []
        for i in range(len(self.universe.particles)):
            if self.universe.particles[i].is_entangled:
                j = int(self.universe.particles[i].entangled_pair)
                if j != -1 and j > i and j < len(self.universe.particles):
                    p1 = self.universe.particles[i].position
                    p2 = self.universe.particles[j].position
                    entangled_x.extend([p1[0], p2[0], np.nan])
                    entangled_y.extend([p1[1], p2[1], np.nan])
        self.layers['entanglement'].set_data(entangled_x, entangled_y)

        susy_x = []
        susy_y = []
        for i in range(len(self.universe.particles)):
            if self.universe.particles[i].is_supersymmetric:
                j = int(self.universe.particles[i].super_partner_id)
                if j != -1 and j > i and j < len(self.universe.particles):
                    p1 = self.universe.particles[i].position
                    p2 = self.universe.particles[j].position
                    susy_x.extend([p1[0], p2[0], np.nan])
                    susy_y.extend([p1[1], p2[1], np.nan])
        self.layers['supersymmetry_partners'].set_data(susy_x, susy_y)

        strings_x = []
        strings_y = []
        for s in self.universe.cosmic_strings:
            strings_x.extend([s[0][0], s[1][0], np.nan])
            strings_y.extend([s[0][1], s[1][1], np.nan])
        self.layers['cosmic_strings'].set_data(strings_x, strings_y)

        years = self.universe.time / (60 * 60 * 24 * 365.25)
        hud_text = f"Time: {years:.2f} years\n"
        hud_text += f"Agents: {len(self.universe.agents)}\n"
        hud_text += f"Avg Reward: {np.mean([a.total_reward for a in self.universe.agents]) if self.universe.agents else 0:.4f}\n"
        hud_text += "\nActive Enhancements:\n"
        for k, v in self.universe.active_enhancements.items():
            if v: hud_text += f" - {k.replace('_', ' ').title()}\n"
        self.layers['hud_text'].set_text(hud_text if self.hud_var.get() else '')

        if self.camera_mode == 'follow_agent' and self.universe.agents:
            self.active_agent_idx = (self.active_agent_idx + 1) % len(self.universe.agents)
            agent_pos = self.universe.agents[self.active_agent_idx].position
            self.ax.set_xlim(agent_pos[0] - self.universe.size/10, agent_pos[0] + self.universe.size/10)
            self.ax.set_ylim(agent_pos[1] - self.universe.size/10, agent_pos[1] + self.universe.size/10)
        else:
            self.ax.set_xlim(0, self.universe.size)
            self.ax.set_ylim(0, self.universe.size)

        self.update_metrics_charts()
        self.update_agent_selection()

        return list(self.layers.values()) + [self.layers['hud_text']]

    def run(self):
        self.ani = animation.FuncAnimation(self.fig, self.update_universe_view, interval=50, blit=False)
        self.root.mainloop()

if __name__ == "__main__":
    # Use a try-except block to handle missing config.json or keys
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Warning: config.json not found or is invalid. Using default values.")
        config = {}

    universe_size = config.get('universe_size', 1e18)
    num_particles = config.get('num_particles', 1000)
    time_step = config.get('time_step', 1e-10)
    num_agents = config.get('num_agents', 10)

    universe_config = {
        'num_particles': num_particles,
        'size': universe_size,
        'time_step': time_step
    }

    universe = Universe(
        config=universe_config
    )

    for i in range(num_agents):
        agent = AGIAgent(universe, id=i)
        universe.agents.append(agent)

    visualizer = UniverseVisualizer(universe)
    visualizer.run()
