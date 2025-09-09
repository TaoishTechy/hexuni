"""
Universe Simulation v0.4

This is the main application file for a universe simulation with AGI agents.
It handles the Tkinter GUI, Matplotlib visualization, and manages the simulation
engine. The application is designed for robustness and performance, with a
focus on safe startup, responsive controls, and clear error handling.

Acceptance Tests:
1. Startup safety: No AttributeErrors; no “no such column” pre-migration errors.
2. Console robustness: Paste a complex batch of commands and see clean outputs;
   prompts keep semicolons verbatim.
3. Pan/pause/speed/theme:
    - Pan works at (0.0, 0.0).
    - Pause/resume doesn’t crash if clicked early.
    - Speed slider doesn’t error before animation exists.
    - Theme switch prints a friendly message if missing.
4. HUD/metrics: No flicker; no “division by zero”; enhancement text updates only
   when flags change.
5. Avatar resizing: If an agent’s avatar is not 16x16, the preview still renders
   without index errors.
"""

# Smoke Test
# The following commands can be pasted directly into the console to test core functionality.
#
# # Test a batch of commands with different formats and comments
# activate temporal flux; activate exotic matter
# deactivate quantum entanglement # inline comment
# activate psionic energy
# set goal for agent 0 to 1.00e21,2.00e21
# set_goal 1 1.50e21,1.50e21
# set_goal all
#
# # Prompting, avatars, and debate
# prompt agent 0 Store: seek gradients near strings; report coherent corridors
# query agent 0 knowledge
# mutate avatar 0 anneal; mutate avatar 0 noise
# start debate on emergence under temporal flux with agents 0,1
# end debate
#
# # Reward shaping check
# set_goal 0 1e20,1e20; query agent 0 reward

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import json
import tkinter as tk
from tkinter import ttk, scrolledtext
from tkinter import filedialog, messagebox
import matplotlib.animation as animation
import os
import sys
import psutil
from collections import deque
import re
from functools import partial
from PIL import Image, ImageTk
import queue
import warnings as _warnings
from typing import List, Dict, Any, Optional, Tuple
_warnings.filterwarnings("ignore", message="Unable to import Axes3D")

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
    DebateArena,
    MAX_DEBATE_TURNS
)
from mods_runtime import (
    ensure_project_dirs,
    load_laws,
    load_mods,
    active_feature_index_map,
    apply_laws_to_config,
    apply_mods_to_state
)

# --- Visualization and Controls (Tkinter-based) ---
class UniverseVisualizer:
    def __init__(self, universe):
        self.universe = universe
        self.root = tk.Tk()
        self.root.title("Universe Simulation v0.4")
        self.root.geometry("1400x900")

        self.cached_enhancements = None
        self.hud_text_enhancements = ""

        self.root.option_add("*tearOff", False)
        self._create_menu()

        # --- UI Setup ---
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

        # Now that self.console_output exists, apply the theme safely.
        self._apply_theme('dark')

        # Initialize animation after all UI elements are created.
        self.ani = None
        self._setup_animation()

        self.active_agent_idx = 0
        self.camera_mode = 'follow_agent'
        self.is_running = True

        # Now that self.universe.nlp.output_queue exists, start checking the queue.
        self.root.after(100, self.check_queue)

    def _setup_animation(self):
        """Initializes the matplotlib animation."""
        self.ani = animation.FuncAnimation(self.fig, self.update_universe_view, interval=50, blit=False)

    def _create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar)
        view_menu = tk.Menu(menubar)
        help_menu = tk.Menu(menubar)

        menubar.add_cascade(menu=file_menu, label="File")
        menubar.add_cascade(menu=view_menu, label="View")
        menubar.add_cascade(menu=help_menu, label="Help")

        file_menu.add_command(label="New Simulation", command=self._new_simulation_prompt)
        file_menu.add_command(label="Load State...", command=self.load_state_dialog)
        file_menu.add_command(label="Save State...", command=self.save_state_dialog)
        file_menu.add_command(label="Export Agent Summary", command=self.export_agent_summary)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)

        view_menu.add_command(label="Toggle Theme", command=self.toggle_theme)

        help_menu.add_command(label="Commands", command=self._show_commands)


    def export_agent_summary(self):
        """Exports a structured JSON summary of all agents."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export Agent Summary"
        )
        if not file_path:
            return

        summary = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "agents": []
        }
        for agent in self.universe.agents:
            agent_data = {
                "id": agent.id,
                "position": agent.position.tolist(),
                "goal": agent.goal.tolist(),
                "total_reward": agent.total_reward,
                "role": agent.role,
                "self_awareness_score": agent.self_awareness_score,
                "relationships": agent.relationships,
                "coalition_name": agent.coalition_name,
                "neurotransmitter_levels": agent.neurotransmitters.levels,
                "kb_size": len(agent.kb.vector_store) if hasattr(agent, 'kb') else 0
            }
            summary["agents"].append(agent_data)

        try:
            with open(file_path, 'w') as f:
                json.dump(summary, f, indent=4)
            self.console_print(f"Summary of all agents exported to {file_path}", tag="success")
        except Exception as e:
            self.console_print(f"Failed to export agent summary: {e}", tag="error")

    def _apply_theme(self, theme):
        try:
            self.root.tk.call('source', 'azure.tcl')
            self.root.tk.call('set_theme', theme)
            self.current_theme = theme
        except tk.TclError:
            self.console_print("Warning: azure.tcl not found. Using default Tkinter theme.", tag="warning")
            self.current_theme = 'default'


    def toggle_theme(self):
        new_theme = 'light' if self.current_theme == 'dark' else 'dark'
        self._apply_theme(new_theme)


    def _new_simulation_prompt(self):
        if self.is_running:
            if messagebox.askyesno("New Simulation", "Are you sure you want to start a new simulation? All unsaved progress will be lost."):
                self.reinitialize_simulation()
        else:
            self.reinitialize_simulation()


    def reinitialize_simulation(self):
        if self.ani and self.ani.event_source:
            self.ani.event_source.stop()
        self.universe.__init__(num_particles=750, size=2.5e22, time_step=5e9)
        self._reinitialize_agents()
        self.root.destroy()
        self.__init__(self.universe)
        self.run()


    def _reinitialize_agents(self):
        self.universe.agents = []
        for i in range(10):
            agent = AGIAgent(self.universe, id=i)
            agent.kb.semantic_search("")
            self.universe.agents.append(agent)


    def load_state_dialog(self):
        file_path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.universe.load_state(file_path)
                self.console_print(f"Loaded simulation state from {file_path}", tag="success")
            except Exception as e:
                self.console_print(f"Failed to load state: {e}", tag="error")


    def save_state_dialog(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.universe.save_state(file_path)
                self.console_print(f"Saved simulation state to {file_path}", tag="success")
            except Exception as e:
                self.console_print(f"Failed to save state: {e}", tag="error")


    def _show_commands(self):
        help_text = self.universe.nlp.get_help_text()
        self.console_print("--- Commands Help ---")
        self.console_print(help_text, tag="info")
        self.console_print("--- End Help ---")


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

        # New Feature 1: Simulation Speed Slider
        ttk.Label(control_frame, text="Speed:").pack(anchor="w")
        self.speed_slider = ttk.Scale(control_frame, from_=10, to=200, orient=tk.HORIZONTAL, command=self.set_sim_speed)
        self.speed_slider.set(50)
        self.speed_slider.pack(fill="x")

        # New Feature 2: Theme Switcher
        ttk.Label(control_frame, text="Theme:").pack(anchor="w")
        self.theme_var = tk.StringVar(value='dark')
        theme_menu = ttk.OptionMenu(control_frame, self.theme_var, 'dark', 'dark', 'light', command=self.set_theme)
        theme_menu.pack(fill="x")

        self.pause_button = ttk.Button(self.left_frame, text="Pause Simulation", command=self.toggle_simulation)
        self.pause_button.pack(fill="x", padx=10, pady=5)

        # New Feature 3: Snapshot Button
        ttk.Button(self.left_frame, text="Save Snapshot", command=self.save_snapshot).pack(fill="x", padx=10, pady=5)

        # New Feature 4: Auto-save Toggle
        self.autosave_var = tk.BooleanVar(value=self.universe.autosave_enabled)
        self.autosave_var.trace_add("write", self._set_autosave_state)
        ttk.Checkbutton(self.left_frame, text="Auto-save every 100 steps", variable=self.autosave_var).pack(fill="x", padx=10, pady=5)

        # New Feature 9: Debate Progress Bar
        debate_frame = ttk.LabelFrame(self.left_frame, text="Active Debate")
        debate_frame.pack(fill="x", padx=10, pady=5)
        self.debate_progress_bar = ttk.Progressbar(debate_frame, orient=tk.HORIZONTAL, length=200, mode='determinate', maximum=MAX_DEBATE_TURNS)
        self.debate_progress_bar.pack(fill="x", padx=5, pady=5)
        self.debate_status_label = ttk.Label(debate_frame, text="No active debate.")
        self.debate_status_label.pack(fill="x", padx=5)

    def _set_autosave_state(self, *args):
        self.universe.autosave_enabled = self.autosave_var.get()

    def set_sim_speed(self, value):
        if self.ani and self.ani.event_source:
            self.ani.event_source.interval = int(float(value))

    def set_theme(self, theme):
        try:
            self.root.tk.call("set_theme", theme)
        except tk.TclError:
            self.console_print(f"Theme switch failed; using default.", tag="warning")

    def save_snapshot(self):
        self.universe.save_state('manual_snapshot.json')
        self.universe.nlp.output_queue.put({"message": "Manual snapshot saved.", "tag": "success"})

    def toggle_simulation(self):
        if self.ani and self.ani.event_source:
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
            'cosmic_strings': self.ax.plot([], [], 'w--', alpha=0.5)[0],
            'supersymmetry_partners': self.ax.plot([], [], '--', c='yellow')[0],
            'dark_matter': self.ax.scatter([], [], s=10, c='purple', alpha=0.4, marker='o'),
            'holographic_projection': self.ax.plot([], [], '-', c='white')[0],
            'hud_text': self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, fontsize=10, color='white')
        }

        self.last_hovered_entity = None
        self.annotation = self.ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                                          bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
                                          arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))
        self.annotation.set_visible(False)
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
        if self.is_panning and event.xdata is not None and event.ydata is not None:
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

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        if xdata is not None and ydata is not None:
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

        self.anomaly_alert_frame = ttk.LabelFrame(self.metrics_frame, text="Anomaly Alerts")
        self.anomaly_alert_frame.pack(fill="x", padx=10, pady=5)
        self.anomaly_text = scrolledtext.ScrolledText(self.anomaly_alert_frame, height=5, bg="#2d2d2d", fg="red")
        self.anomaly_text.pack(fill="both")

    def _setup_command_console(self):
        self.console_label = ttk.Label(self.console_frame, text="Command Console", font=("TkDefaultFont", 12, "bold"))
        self.console_label.pack(pady=5)

        self.console_input = scrolledtext.ScrolledText(self.console_frame, wrap=tk.WORD, height=10, bg="#2d2d2d", fg="#ffffff")
        self.console_input.pack(padx=10, pady=5, fill=tk.X, expand=False)
        self.console_input.bind("<Control-Return>", self.on_submit_multiline)
        self.console_input.insert(tk.END, "Enter commands here (Ctrl+Enter to run):\n")

        run_button = ttk.Button(self.console_frame, text="Run Commands", command=self.on_submit_multiline)
        run_button.pack(padx=10, pady=5, fill=tk.X)

        self.console_output = scrolledtext.ScrolledText(self.console_frame, wrap=tk.WORD, height=20, bg="#2d2d2d", fg="#ffffff")
        self.console_output.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        self.console_output.insert(tk.END, "Welcome to the Universe Simulation v0.4!\n")

        # Configure tags for console output
        self.console_output.tag_config('error', foreground='red')
        self.console_output.tag_config('warning', foreground='yellow')
        self.console_output.tag_config('success', foreground='green')
        self.console_output.tag_config('info', foreground='cyan')
        self.console_output.tag_config('agent', foreground='#87CEFA')
        self.console_output.tag_config('debug', foreground='grey')

        self.console_output.config(state=tk.DISABLED)

    def console_print(self, message, tag=None):
        self.console_output.config(state=tk.NORMAL)
        if tag:
            self.console_output.insert(tk.END, f"{message}\n", tag)
        else:
            self.console_output.insert(tk.END, f"{message}\n")
        self.console_output.config(state=tk.DISABLED)
        self.console_output.see(tk.END)

    def on_submit_multiline(self, event=None):
        commands = self.console_input.get("1.0", tk.END).strip()
        if not commands or commands == "Enter commands here (Ctrl+Enter to run):":
            return

        try:
            self.universe.nlp.parse_command(commands)
            self.console_input.delete("1.0", tk.END)
        except Exception as e:
            self.console_print(f"Error: {e}", tag="error")

        self.check_queue()
        self.root.after_idle(self.check_queue)

    def check_queue(self):
        try:
            while True:
                # The message now includes a tag
                output = self.universe.nlp.output_queue.get_nowait()
                if isinstance(output, dict) and 'message' in output:
                    self.console_print(output['message'], tag=output.get('tag'))
                    # Check for structured updates
                    if output.get('action') == 'update_agent_panel':
                        self.update_agent_panel()
                    if output.get('action') == 'update_debate_transcript':
                        self.update_debate_transcript()
                    if output.get('action') == 'update_metrics':
                        self.update_metrics_charts()
                    if output.get('action') == 'highlight_agent':
                        self._highlight_agent(output['agent_id'])
                else:
                    self.console_print(output)
        except queue.Empty:
            pass
        self.root.after(100, self.check_queue)

    def _highlight_agent(self, agent_id):
        # Placeholder for highlighting logic
        self.console_print(f"Highlighting agent {agent_id} in visualization.", tag="info")

    def _setup_entity_manager(self):
        entity_pane = ttk.PanedWindow(self.entity_frame, orient=tk.HORIZONTAL)
        entity_pane.pack(expand=True, fill="both", padx=10, pady=10)

        left_side = ttk.Frame(entity_pane, width=200)
        entity_pane.add(left_side, weight=1)

        right_side = ttk.Frame(entity_pane)
        entity_pane.add(right_side, weight=2)

        # Top right: Agent details
        self.agent_details_frame = ttk.LabelFrame(right_side, text="Agent Details")
        self.agent_details_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Bottom right: Debate transcript
        debate_transcript_frame = ttk.LabelFrame(right_side, text="Debate Transcript")
        debate_transcript_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.debate_transcript_text = scrolledtext.ScrolledText(debate_transcript_frame, wrap=tk.WORD, height=10, bg="#2d2d2d", fg="#ffffff")
        self.debate_transcript_text.pack(fill="both", expand=True)
        self.debate_transcript_text.config(state=tk.DISABLED)

        agent_select_frame = ttk.LabelFrame(left_side, text="Select Agent")
        agent_select_frame.pack(fill="x", padx=10, pady=5)

        self.agent_var = tk.StringVar()
        self.agent_menu = ttk.OptionMenu(agent_select_frame, self.agent_var, "Select an Agent", "")
        self.agent_menu.pack(fill="x")
        self.agent_var.trace_add("write", self.update_agent_panel)

        self.avatar_canvas = tk.Canvas(left_side, width=160, height=160, bg="#2d2d2d", highlightthickness=1, highlightbackground="white")
        self.avatar_canvas.pack(padx=10, pady=5, fill=tk.X)

        avatar_legend = ttk.Label(left_side, text="Avatar Legend: 1=Filled, 0=Empty")
        avatar_legend.pack(padx=10)

        mutate_frame = ttk.LabelFrame(left_side, text="Mutate Avatar")
        mutate_frame.pack(fill="x", padx=10, pady=5)

        self.auto_mutate_var = tk.BooleanVar(value=False)
        self.auto_mutate_cb = ttk.Checkbutton(mutate_frame, text="Auto-mutate", variable=self.auto_mutate_var, command=self._toggle_auto_mutate)
        self.auto_mutate_cb.pack(fill="x")

        ttk.Button(mutate_frame, text="Flip", command=lambda: self._trigger_mutation('flip')).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(mutate_frame, text="Rotate", command=lambda: self._trigger_mutation('rotate')).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(mutate_frame, text="Noise", command=lambda: self._trigger_mutation('noise')).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(mutate_frame, text="Anneal", command=lambda: self._trigger_mutation('anneal')).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # New Feature 7: Agent Goal Reset Button
        ttk.Button(left_side, text="Reset Agent Goal", command=self.reset_agent_goal).pack(fill="x", padx=10, pady=5)

        self.update_agent_selection()


    def reset_agent_goal(self):
        selected_agent = self.get_selected_agent()
        if selected_agent:
            selected_agent.set_goal(np.random.rand(2) * self.universe.size)
            self.universe.nlp.output_queue.put({"message": f"Agent {selected_agent.id}'s goal randomized.", "tag": "info"})

    def _toggle_auto_mutate(self):
        selected_agent = self.get_selected_agent()
        if selected_agent:
            selected_agent.auto_mutate = self.auto_mutate_var.get()

    def _trigger_mutation(self, mode):
        selected_agent = self.get_selected_agent()
        if selected_agent:
            selected_agent.mutate_avatar(mode)
            self._draw_avatar(selected_agent.avatar)
            self.universe.nlp.output_queue.put({"message": f"Agent {selected_agent.id}'s avatar mutated with mode '{mode}'.", "tag": "info"})


    def get_selected_agent(self):
        selected_agent_str = self.agent_var.get()
        if selected_agent_str.startswith("Agent"):
            agent_id = int(selected_agent_str.split()[1])
            return next((a for a in self.universe.agents if a.id == agent_id), None)
        return None

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
            self._clear_agent_panel()

    def update_agent_panel(self, *args):
        selected_agent = self.get_selected_agent()
        if not selected_agent:
            self._clear_agent_panel()
            return

        for widget in self.agent_details_frame.winfo_children():
            widget.destroy()

        if hasattr(selected_agent, 'auto_mutate'):
            self.auto_mutate_var.set(selected_agent.auto_mutate)

        details_frame = ttk.Frame(self.agent_details_frame)
        details_frame.pack(fill="both", expand=True)

        info_text = (
            f"ID: {selected_agent.id}\n"
            f"Total Reward: {selected_agent.total_reward:.2f}\n"
            f"Position: ({selected_agent.position[0]:.2e}, {selected_agent.position[1]:.2e})\n"
            f"Reality Resource: {selected_agent.reality_editing_resource:.2f}\n"
            f"Self-Awareness: {selected_agent.self_awareness_score:.2f}\n"
            f"Last Mutation: {selected_agent.last_avatar_mutation.capitalize()}\n"
            f"Entropy: {selected_agent.get_intellectual_entropy():.4f}\n"
            f"Goal Alignment: {self.universe.goal_alignment():.4f}\n"
            f"Sociometric Cohesion: {self.universe.get_sociometric_cohesion():.4f}"
        )
        info_label = ttk.Label(details_frame, text=info_text)
        info_label.pack(anchor="w", padx=5, pady=5)

        nt_frame = ttk.LabelFrame(details_frame, text="Neurotransmitter Levels")
        nt_frame.pack(fill="x", padx=5, pady=5)
        for nt, level in selected_agent.neurotransmitters.levels.items():
            ttk.Label(nt_frame, text=f"{nt.title()}: {level:.2f}").pack(anchor="w")

        rel_frame = ttk.LabelFrame(details_frame, text="Social Relationships")
        rel_frame.pack(fill="x", padx=5, pady=5)
        if not selected_agent.relationships:
            ttk.Label(rel_frame, text="No relationships yet.").pack(anchor="w")
        else:
            for other_id, score in selected_agent.relationships.items():
                ttk.Label(rel_frame, text=f"Agent {other_id}: {score:.2f}").pack(anchor="w")

        # Dynamically size avatar canvas
        avatar_height, avatar_width = selected_agent.avatar.shape
        pixel_size = min(160 // avatar_height, 160 // avatar_width)
        self.avatar_canvas.config(width=avatar_width * pixel_size, height=avatar_height * pixel_size)
        self._draw_avatar(selected_agent.avatar)

    def update_debate_transcript(self):
        """Updates the debate transcript UI with the latest arguments."""
        if self.universe.debate_arena.active_debate:
            self.debate_transcript_text.config(state=tk.NORMAL)
            self.debate_transcript_text.delete("1.0", tk.END)
            transcript = self.universe.debate_arena.active_debate.get('arguments', [])
            for arg in transcript:
                message = f"Agent {arg['agent']} ({arg['score']:.2f}): {arg['text']}\n"
                self.debate_transcript_text.insert(tk.END, message, 'agent')
            self.debate_transcript_text.config(state=tk.DISABLED)
            self.debate_transcript_text.see(tk.END)
        else:
            self.debate_transcript_text.config(state=tk.NORMAL)
            self.debate_transcript_text.delete("1.0", tk.END)
            self.debate_transcript_text.insert(tk.END, "No active debate.", 'info')
            self.debate_transcript_text.config(state=tk.DISABLED)


    def _clear_agent_panel(self):
        for widget in self.agent_details_frame.winfo_children():
            widget.destroy()
        self.avatar_canvas.delete("all")
        self.auto_mutate_var.set(False)

    def _draw_avatar(self, avatar):
        self.avatar_canvas.delete("all")
        avatar_height, avatar_width = avatar.shape
        pixel_size = min(160 // avatar_height, 160 // avatar_width)
        for i in range(avatar_height):
            for j in range(avatar_width):
                color = "white" if avatar[i, j] == 1 else "#2d2d2d"
                self.avatar_canvas.create_rectangle(j * pixel_size, i * pixel_size, (j + 1) * pixel_size, (i + 1) * pixel_size, fill=color, outline=color)

    def update_metrics_charts(self):
        metrics = self.universe.metrics_collector.metrics
        times = metrics.get('time', [])
        if not times: return

        self.line_fps.set_data(times, metrics.get('fps', []))
        self.line_mem.set_data(times, metrics.get('memory_usage', []))
        self.ax_perf.relim(); self.ax_perf.autoscale_view()

        self.line_lr.set_data(times, metrics.get('avg_learning_rate', []))
        self.line_reward.set_data(times, metrics.get('avg_reward', []))
        self.ax_agi.relim(); self.ax_agi.autoscale_view()

        self.line_energy.set_data(times, metrics.get('total_energy', []))
        self.line_entropy.set_data(times, metrics.get('entropy', []))
        self.ax_phys.relim(); self.ax_phys.autoscale_view()

        self.line_intellectual_entropy.set_data(times, metrics.get('intellectual_entropy', []))
        self.line_cohesion.set_data(times, metrics.get('sociometric_cohesion', []))
        self.ax_adv_agi.relim(); self.ax_adv_agi.autoscale_view()

        self.metrics_fig.tight_layout()
        self.metrics_canvas.draw_idle()

        for anomaly in list(self.universe.metrics_collector.anomalies_detected):
            self.console_print(f"ALERT: Anomaly detected! {anomaly['type']} for Agent {anomaly['agent_id']} at time {anomaly['time']:.2e}", tag="warning")
            self.universe.metrics_collector.anomalies_detected.remove(anomaly)

    def _draw_particle_view(self):
        particle_pos = np.array([p.position for p in self.universe.particles])
        if len(particle_pos) > 0:
            self.layers['particles'].set_offsets(particle_pos)
        exotic_pos = np.array([p.position for p in self.universe.particles if p.is_exotic])
        if len(exotic_pos) > 0:
            self.layers['exotic'].set_offsets(exotic_pos)
        dark_matter_pos = np.array([p.position for p in self.universe.particles if p.is_dark])
        if len(dark_matter_pos) > 0:
            self.layers['dark_matter'].set_offsets(dark_matter_pos)
        self.layers['entanglement'].set_alpha(1.0)
        self.layers['cosmic_strings'].set_alpha(1.0)
        self.layers['supersymmetry_partners'].set_alpha(1.0)
        self.layers['holographic_projection'].set_alpha(0.0)
        self.ax.set_facecolor('#000000')

    def _draw_energy_field(self):
        energies = np.array([0.5 * p.mass * np.sum(p.velocity**2) for p in self.universe.particles])
        if len(energies) > 0 and energies.max() != energies.min():
            norm_energies = (energies - energies.min()) / (energies.max() - energies.min() + 1e-10)
            self.layers['particles'].set_color(plt.cm.jet(norm_energies))
            exotic_mask = np.array([p.is_exotic for p in self.universe.particles], dtype=bool)
            if np.any(exotic_mask):
                self.layers['exotic'].set_color(plt.cm.jet(norm_energies[exotic_mask]))
            dark_mask = np.array([p.is_dark for p in self.universe.particles], dtype=bool)
            if np.any(dark_mask):
                self.layers['dark_matter'].set_color(plt.cm.jet(norm_energies[dark_mask]))
        self.layers['entanglement'].set_alpha(0.0)
        self.layers['cosmic_strings'].set_alpha(0.0)
        self.layers['supersymmetry_partners'].set_alpha(0.0)
        self.layers['holographic_projection'].set_alpha(0.0)

    def _draw_info_density(self):
        if len(self.universe.particles) > 0:
            info_density = np.random.uniform(1, 10, len(self.universe.particles))
            self.layers['particles'].set_sizes(info_density * 10)
            exotic_mask = np.array([p.is_exotic for p in self.universe.particles], dtype=bool)
            if np.any(exotic_mask):
                self.layers['exotic'].set_sizes(info_density[exotic_mask] * 10)
            dark_mask = np.array([p.is_dark for p in self.universe.particles], dtype=bool)
            if np.any(dark_mask):
                self.layers['dark_matter'].set_sizes(info_density[dark_matter_mask] * 10)
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
        for patch in list(self.ax.patches):
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

        if self.universe.step_counter % 10 == 0:
            holographic_data = self.universe.holographic_data.get('boundary_projection')
            if holographic_data is not None and holographic_data.shape[1] >= 2:
                x, y = holographic_data[:, 0], holographic_data[:, 1]
                self.layers['holographic_projection'].set_data(x, y)
                self.layers['holographic_projection'].set_alpha(1.0)
            else:
                self.layers['holographic_projection'].set_data([], [])
                self.layers['holographic_projection'].set_alpha(0.0)

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

    def update_universe_view(self, frame):
        self.universe.step()

        for patch in list(self.ax.patches):
            patch.remove()

        for layer in self.layers.values():
            if isinstance(layer, plt.matplotlib.collections.PathCollection) or isinstance(layer, plt.matplotlib.lines.Line2D):
                layer.set_alpha(1.0)

        self.view_modes[self.visualization_mode]()

        agent_pos = np.array([a.position for a in self.universe.agents])
        if len(agent_pos) > 0:
            self.layers['agents'].set_offsets(agent_pos)

        goal_pos = np.array([a.goal for a in self.universe.agents])
        if len(goal_pos) > 0:
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
        if hasattr(self.universe, 'cosmic_strings'):
            for s in self.universe.cosmic_strings:
                strings_x.extend([s[0][0], s[1][0], np.nan])
                strings_y.extend([s[0][1], s[1][1], np.nan])
        self.layers['cosmic_strings'].set_data(strings_x, strings_y)

        years = self.universe.time / (60 * 60 * 24 * 365.25)

        current_enhancements = tuple(sorted([k for k, v in self.universe.active_enhancements.items() if v]))
        if current_enhancements != self.cached_enhancements:
            self.cached_enhancements = current_enhancements
            hud_enhancements_list = "\n".join([f" - {k.replace('_', ' ').title()}" for k in current_enhancements])

            compact_mode = self.universe.ui_opts.get("hud_enhancements_compact", False)
            if compact_mode:
                self.hud_text_enhancements = "Enh: " + ", ".join([f"{k.split('_')[0]}(ON)" for k in current_enhancements])
            else:
                self.hud_text_enhancements = f"\nActive Enhancements:\n{hud_enhancements_list}"

        avg_reward_val = np.mean([a.total_reward for a in self.universe.agents]) if self.universe.agents else 0
        hud_text = f"Time: {years:.2f} years\n"
        hud_text += f"Agents: {len(self.universe.agents)}\n"
        hud_text += f"Avg Reward: {avg_reward_val:.4f}\n"
        hud_text += self.hud_text_enhancements

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

        if self.notebook.tab(self.notebook.select(), "text") == "Entity Manager":
            selected_agent = self.get_selected_agent()
            if selected_agent:
                self._draw_avatar(selected_agent.avatar)

        self.update_debate_transcript()


        if self.universe.debate_arena.active_debate:
            if 'turn' in self.universe.debate_arena.active_debate:
                self.debate_progress_bar.config(value=self.universe.debate_arena.active_debate['turn'])
                self.debate_status_label.config(text=f"Turn {self.universe.debate_arena.active_debate['turn']} / {MAX_DEBATE_TURNS}")
            else:
                self.debate_progress_bar.config(value=0)
                self.debate_status_label.config(text="Debate starting...")
        else:
            self.debate_progress_bar.config(value=0)
            self.debate_status_label.config(text="No active debate.")

        return list(self.layers.values()) + [self.layers['hud_text']]

    def run(self):
        # We now create the animation in the __init__ method after UI setup
        self.root.mainloop()

if __name__ == "__main__":

    # --- Boot Sequence & Wiring ---
    try:
        dirs = ensure_project_dirs(os.getcwd())
        laws = load_laws(dirs['laws'])
        mods = load_mods(dirs['mods'])
        with open('config.json', 'r') as f:
            base_config = json.load(f)
    except FileNotFoundError:
        base_config = {
            "num_particles": 750,
            "universe_size": 2.5e22,
            "time_step": 5e9,
            "num_agents": 10
        }
        laws = load_laws(os.getcwd()) # Use defaults if no mods/laws dirs
        mods = []

    config = apply_laws_to_config(laws, base_config)

    # Create the output queue before the Universe or Visualizer
    output_queue = queue.Queue()

    universe = Universe(
        num_particles=config['num_particles'],
        size=config['universe_size'],
        time_step=config['time_step']
    )
    if not hasattr(universe, 'autosave_enabled'):
        universe.autosave_enabled = False

    apply_mods_to_state(mods, universe)

    universe._enh_order = list(active_feature_index_map(laws, universe.physics_overrides.get("enhancement_index_overrides", None)).keys())

    # Wire the output queue to the universe's NLP
    universe.nlp.output_queue = output_queue

    for i in range(config['num_agents']):
        agent = AGIAgent(universe, id=i)
        agent.kb.semantic_search("")
        universe.agents.append(agent)

    visualizer = UniverseVisualizer(universe)
    visualizer.run()
