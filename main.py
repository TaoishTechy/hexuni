import tkinter as tk
from tkinter import filedialog, ttk, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import time
import queue
import json
import argparse
import random
import os
from datetime import datetime
import threading

# Assuming these imports exist from the project structure
from engine.sim_core import SimCore
from engine.nlp_console import NlpConsole
from components import UniverseVisualizer, MetricsCollector, Agent
# Make sure to import any new classes for rich console and commands
# from components import RichConsole, CommandPalette

class MainApp(tk.Tk):
    def __init__(self, seed=None, headless=False):
        # Enhancement 1: Deterministic boots with seed
        self.headless = headless
        self.seed = seed if seed is not None else random.randint(1, 100000)
        np.random.seed(self.seed)

        if not self.headless:
            super().__init__()
            self.title("Hexuni Simulation")
            self.geometry("1200x800")
            self.state('zoomed')
            self.configure(bg="#2c3e50")
            
            try:
                self.tk.call("source", "azure.tcl")
                self.tk.call("set_theme", "dark")
            except tk.TclError:
                print("Azure theme not found. Using default Tkinter theme.")
                # Enhancement 11: Safe theme fallback messaging
                self.show_theme_install_message()

            self.create_widgets()
            self.bind_shortcuts()

        # Engine components
        self.app_id = "hexuni-sim-01" # This would be a dynamic ID in a real system
        self.sim_core = SimCore(self.app_id)
        self.log_queue = queue.Queue(maxsize=1000) # Bug 9: Add maxsize
        self.console = NlpConsole(self.sim_core, self.log_queue)
        self.metrics = MetricsCollector()

        # Simulation state
        self.running = True
        self.pause_requested = False
        self.current_step = 0
        self.autosave_cadence = None
        self.last_autosave_time = time.time()
        self.is_updating = False # Bug 3: Add flag to prevent mid-update saves
        self.agent_var = tk.StringVar(self)
        self.agents = []
        self.selected_agent = None

        # Enhancement 2: Eager agent bootstrap on first run
        self._reinitialize_agents(10)
        self.update_agent_panel()
        
        # Camera bookmarks
        self.camera_bookmarks = {}
        self.camera_follow_agent = None
        self.is_following = False

        # Live debate transcript
        self.transcript_queue = queue.Queue(maxsize=100) # Enhancement 6: Live debate transcript
        self.console_input_history = []

        if not self.headless:
            self.start_gui_loop()
        else:
            self.start_headless_loop()

    # Enhancement 11: Safe theme fallback messaging
    def show_theme_install_message(self):
        messagebox_root = tk.Toplevel(self)
        messagebox_root.title("Theme Not Found")
        tk.Label(messagebox_root, text="The 'Azure' theme was not found.").pack(padx=20, pady=10)
        tk.Label(messagebox_root, text="This may cause visual inconsistencies.").pack(padx=20, pady=5)
        
        button_frame = tk.Frame(messagebox_root)
        button_frame.pack(pady=10)
        tk.Button(button_frame, text="OK", command=messagebox_root.destroy).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Install Theme", command=lambda: self.open_theme_url()).pack(side=tk.LEFT, padx=5)

    def open_theme_url(self):
        import webbrowser
        webbrowser.open("https://github.com/rdbende/Azure-ttk-theme")
    
    def create_widgets(self):
        # Master frames for layout presets
        self.pane_frame = tk.Frame(self, bg="#2c3e50")
        self.pane_frame.pack(fill=tk.BOTH, expand=True)

        # Left Column Frame
        left_frame = tk.Frame(self.pane_frame, width=300, bg="#2c3e50")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left_frame.pack_propagate(False)

        # Universe View Frame (Center)
        center_frame = tk.Frame(self.pane_frame, bg="#34495e")
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Right Column Frame
        right_frame = tk.Frame(self.pane_frame, width=350, bg="#2c3e50")
        right_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        right_frame.pack_propagate(False)

        self.create_universe_view(center_frame)
        self.create_metrics_dashboard(right_frame)
        self.create_control_panel(left_frame)
        self.create_entity_panel(right_frame)
        self.create_debate_transcript(left_frame)

        # Enhancement 18: Pane layout presets
        preset_frame = tk.Frame(self.pane_frame)
        preset_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        tk.Button(preset_frame, text="Save Preset 1", command=lambda: self.save_layout(1)).pack(side=tk.LEFT, padx=2)
        tk.Button(preset_frame, text="Load Preset 1", command=lambda: self.load_layout(1)).pack(side=tk.LEFT, padx=2)

    def save_layout(self, preset_id):
        width, height = self.winfo_width(), self.winfo_height()
        # Save pane widths
        layout = {
            'geometry': f"{width}x{height}",
            'left_width': self.pane_frame.winfo_children()[0].winfo_width(),
            'right_width': self.pane_frame.winfo_children()[2].winfo_width(),
        }
        with open(f"ui_preset_{preset_id}.json", "w") as f:
            json.dump(layout, f)
        print(f"Layout saved to ui_preset_{preset_id}.json")

    def load_layout(self, preset_id):
        try:
            with open(f"ui_preset_{preset_id}.json", "r") as f:
                layout = json.load(f)
            self.geometry(layout['geometry'])
            # Setting width might need more complex logic with grid/pack
            # For simplicity, we just restore the window geometry
            print(f"Loaded layout preset {preset_id}.")
        except FileNotFoundError:
            print(f"Preset {preset_id} not found.")

    def create_universe_view(self, parent):
        self.universe_visualizer = UniverseVisualizer(parent)
        self.canvas = self.universe_visualizer.canvas
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Enhancement 4: Click-to-focus on canvas
        self.canvas.get_tk_widget().bind("<Button-1>", self.on_canvas_click)
        # Enhancement 16: Goal setting gizmo
        self.canvas.get_tk_widget().bind("<Alt-Button-1>", self.on_alt_click_goal)

        # Enhancement 5: Mini-map + camera bookmarks
        self.mini_map_frame = tk.Frame(parent)
        self.mini_map_frame.place(relx=0.98, rely=0.02, anchor="ne")
        self.mini_map_canvas = FigureCanvasTkAgg(self.universe_visualizer.fig, master=self.mini_map_frame)
        self.mini_map_canvas.get_tk_widget().config(width=150, height=150, borderwidth=2, relief="solid")
        self.mini_map_canvas.get_tk_widget().pack()

        bookmark_frame = tk.Frame(self.mini_map_frame)
        bookmark_frame.pack()
        for i in range(1, 6):
            tk.Button(bookmark_frame, text=f"Save {i}", command=lambda i=i: self.save_camera_bookmark(i)).pack(side=tk.LEFT)
            tk.Button(bookmark_frame, text=f"Load {i}", command=lambda i=i: self.load_camera_bookmark(i)).pack(side=tk.LEFT)

        # Enhancement 7: Metrics quick-pins
        self.quick_metrics_frame = tk.Frame(parent, bg="#2c3e50", relief="groove", borderwidth=2)
        self.quick_metrics_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.fps_label = tk.Label(self.quick_metrics_frame, text="FPS: 0", fg="white", bg="#2c3e50")
        self.fps_label.pack(side=tk.LEFT, padx=5)
        self.mem_label = tk.Label(self.quick_metrics_frame, text="MEM: 0 MB", fg="white", bg="#2c3e50")
        self.mem_label.pack(side=tk.LEFT, padx=5)
        self.entropy_label = tk.Label(self.quick_metrics_frame, text="ENTROPY: 0.00", fg="white", bg="#2c3e50")
        self.entropy_label.pack(side=tk.LEFT, padx=5)
        self.alignment_label = tk.Label(self.quick_metrics_frame, text="ALIGNMENT: 0.00", fg="white", bg="#2c3e50")
        self.alignment_label.pack(side=tk.LEFT, padx=5)
        self.cohesion_label = tk.Label(self.quick_metrics_frame, text="COHESION: 0.00", fg="white", bg="#2c3e50")
        self.cohesion_label.pack(side=tk.LEFT, padx=5)
        self.guardrail_tooltip = tk.Label(parent, text="", bg="yellow", fg="black", relief="solid", borderwidth=1) # Enhancement 17
        self.last_frame_time = time.time()

    def on_canvas_click(self, event):
        x = self.universe_visualizer.axes.get_xbound()[0] + event.x / self.canvas.get_tk_widget().winfo_width() * (self.universe_visualizer.axes.get_xbound()[1] - self.universe_visualizer.axes.get_xbound()[0])
        y = self.universe_visualizer.axes.get_ybound()[0] + (1 - event.y / self.canvas.get_tk_widget().winfo_height()) * (self.universe_visualizer.axes.get_ybound()[1] - self.universe_visualizer.axes.get_ybound()[0])
        
        min_dist = float('inf')
        closest_agent = None

        for agent in self.agents:
            dist = np.linalg.norm(np.array(agent.position) - np.array([x, y]))
            if dist < min_dist:
                min_dist = dist
                closest_agent = agent
        
        if closest_agent and min_dist < 25: # Arbitrary proximity pick threshold
            self.agent_var.set(closest_agent.display_name)
            self.selected_agent = closest_agent
            self.update_agent_panel()
            self.log_queue.put(f"Selected agent: {closest_agent.display_name}")

    def on_alt_click_goal(self, event):
        x_click = self.universe_visualizer.axes.get_xbound()[0] + event.x / self.canvas.get_tk_widget().winfo_width() * (self.universe_visualizer.axes.get_xbound()[1] - self.universe_visualizer.axes.get_xbound()[0])
        y_click = self.universe_visualizer.axes.get_ybound()[0] + (1 - event.y / self.canvas.get_tk_widget().winfo_height()) * (self.universe_visualizer.axes.get_ybound()[1] - self.universe_visualizer.axes.get_ybound()[0])
        
        if self.selected_agent:
            self.selected_agent.set_goal(np.array([x_click, y_click]))
            self.log_queue.put(f"Set goal for {self.selected_agent.display_name} at ({x_click:.2f}, {y_click:.2f})")
            
    def save_camera_bookmark(self, bookmark_id):
        self.camera_bookmarks[bookmark_id] = {
            'xlim': self.universe_visualizer.axes.get_xlim(),
            'ylim': self.universe_visualizer.axes.get_ylim()
        }
        self.log_queue.put(f"Saved camera bookmark {bookmark_id}.")

    def load_camera_bookmark(self, bookmark_id):
        if bookmark_id in self.camera_bookmarks:
            bookmark = self.camera_bookmarks[bookmark_id]
            self.universe_visualizer.axes.set_xlim(bookmark['xlim'])
            self.universe_visualizer.axes.set_ylim(bookmark['ylim'])
            self.canvas.draw_idle()
            self.log_queue.put(f"Loaded camera bookmark {bookmark_id}.")
        else:
            self.log_queue.put(f"Bookmark {bookmark_id} does not exist.")

    def create_metrics_dashboard(self, parent):
        self.metrics_notebook = ttk.Notebook(parent)
        self.metrics_notebook.pack(fill=tk.BOTH, expand=True)

        metrics_frame = ttk.Frame(self.metrics_notebook)
        self.metrics_notebook.add(metrics_frame, text="Metrics")

        self.fig, self.axes = plt.subplots(2, 2, figsize=(6, 6)) # Bug 7: Use class variable
        plt.tight_layout(pad=2)
        self.fig.set_facecolor("#2c3e50")
        self.canvas_plots = FigureCanvasTkAgg(self.fig, master=metrics_frame)
        self.canvas_plots.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Enhancement 10: Export metrics & plots
        export_frame = tk.Frame(metrics_frame)
        export_frame.pack(fill=tk.X, pady=5)
        tk.Button(export_frame, text="Export Metrics", command=self.export_metrics).pack(side=tk.LEFT, expand=True, padx=2)

    def export_metrics(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if filepath:
            self.metrics.export_metrics(filepath)
            self.log_queue.put(f"Metrics data exported to {filepath}")
            # Also save the plots
            self.fig.savefig(filepath.replace(".json", ".png"))
            self.log_queue.put(f"Metrics plots saved to {filepath.replace('.json', '.png')}")

    def update_metrics_charts(self):
        # Bug 8: Ensure this is called in the main loop
        pass

    def create_control_panel(self, parent):
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=10)

        # Bug 1: Deterministic boots UI
        seed_frame = tk.Frame(control_frame)
        seed_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(seed_frame, text="Seed:").pack(side=tk.LEFT)
        self.seed_entry = ttk.Entry(seed_frame)
        self.seed_entry.insert(0, str(self.seed))
        self.seed_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        ttk.Button(control_frame, text="Start/Resume", command=self.start_resume_simulation).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(control_frame, text="Pause", command=self.pause_simulation).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(control_frame, text="Reinitialize", command=self.reinitialize_simulation).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(control_frame, text="Save Snapshot", command=self.save_snapshot).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(control_frame, text="Load Snapshot", command=self.load_snapshot).pack(fill=tk.X, padx=5, pady=5)
        
        # Enhancement 9: Autosave cadence control
        autosave_frame = ttk.Frame(control_frame)
        autosave_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(autosave_frame, text="Autosave:").pack(side=tk.LEFT)
        self.autosave_cadence_var = tk.StringVar(self)
        self.autosave_cadence_var.set("Off")
        cadences = ["Off", "10 steps", "50 steps", "100 steps", "60s", "300s"]
        ttk.Combobox(autosave_frame, textvariable=self.autosave_cadence_var, values=cadences).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.last_autosave_label = tk.Label(autosave_frame, text="Last saved: N/A")
        self.last_autosave_label.pack(side=tk.BOTTOM, fill=tk.X)

    def save_snapshot(self):
        # Enhancement 8: Snapshot annotations
        note = simpledialog.askstring("Snapshot Annotation", "Enter a short note for this snapshot:")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.json"
        
        snapshot = {
            "meta": {
                "timestamp": timestamp,
                "note": note,
                "seed": self.seed
            },
            "agents": [a.to_dict() for a in self.agents],
            "simulation_state": {
                "step": self.current_step,
                "universe_state": self.sim_core.get_state()
            }
        }
        
        try:
            with open(filename, "w") as f:
                json.dump(snapshot, f, indent=4)
            self.log_queue.put(f"Saved snapshot to {filename}")
        except Exception as e:
            self.log_queue.put(f"Error saving snapshot: {e}")

    def load_snapshot(self):
        filepath = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if filepath:
            try:
                with open(filepath, "r") as f:
                    snapshot = json.load(f)
                
                # Load agents
                self.agents = [Agent.from_dict(d) for d in snapshot["agents"]]
                
                # Load simulation state
                self.sim_core.load_state(snapshot["simulation_state"]["universe_state"])
                self.current_step = snapshot["simulation_state"]["step"]

                self.log_queue.put(f"Loaded snapshot from {filepath}")
                self.update_agent_panel()

            except Exception as e:
                self.log_queue.put(f"Error loading snapshot: {e}")

    def create_entity_panel(self, parent):
        entity_frame = ttk.Frame(parent)
        entity_frame.pack(fill=tk.BOTH, expand=True)
        self.entity_notebook = ttk.Notebook(entity_frame)
        self.entity_notebook.pack(fill=tk.BOTH, expand=True)

        entity_tab = ttk.Frame(self.entity_notebook)
        self.entity_notebook.add(entity_tab, text="Entity")

        # Enhancement 3: Entity dropdown search + type-ahead
        ttk.Label(entity_tab, text="Selected Agent:").pack(pady=5)
        self.agent_dropdown = ttk.Combobox(entity_tab, textvariable=self.agent_var)
        self.agent_dropdown.pack(fill=tk.X, padx=10, pady=5)
        self.agent_dropdown.bind("<<ComboboxSelected>>", self.on_agent_select)
        
        self.agent_info_frame = ttk.Frame(entity_tab, relief="groove", borderwidth=2)
        self.agent_info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Enhancement 14: Agent panel charts
        self.fig_agent, self.axes_agent = plt.subplots(1, 2, figsize=(6, 2))
        plt.tight_layout(pad=2)
        self.fig_agent.set_facecolor("#2c3e50")
        self.canvas_agent_plots = FigureCanvasTkAgg(self.fig_agent, master=self.agent_info_frame)
        self.canvas_agent_plots.get_tk_widget().pack(side=tk.TOP, fill=tk.X, expand=True)

        # Enhancement 15: Avatar editor
        avatar_frame = ttk.Frame(self.agent_info_frame)
        avatar_frame.pack(fill=tk.X)
        self.avatar_canvas = tk.Canvas(avatar_frame, width=160, height=160, bg="white")
        self.avatar_canvas.pack(side=tk.LEFT, padx=5)
        ttk.Button(avatar_frame, text="Edit Avatar", command=self.edit_avatar).pack(side=tk.LEFT)
        ttk.Button(avatar_frame, text="Reset", command=self.reset_avatar).pack(side=tk.LEFT)

    def on_agent_select(self, event):
        selected_name = self.agent_var.get()
        # Bug 9: Use a dictionary for robust lookup
        self.selected_agent = next((a for a in self.agents if a.display_name == selected_name), None)
        self.update_agent_panel()
    
    def update_agent_panel(self):
        # Update dropdown values
        agent_names = [a.display_name for a in self.agents]
        self.agent_dropdown['values'] = agent_names
        
        if self.selected_agent:
            # Clear old info
            for widget in self.agent_info_frame.winfo_children():
                if widget not in [self.canvas_agent_plots.get_tk_widget(), self.avatar_canvas.master]:
                    widget.destroy()
            
            # Display new info
            info_text = f"ID: {self.selected_agent.agent_id}\n" \
                        f"Role: {self.selected_agent.role}\n" \
                        f"Status: {self.selected_agent.status}\n" \
                        f"Energy: {self.selected_agent.energy:.2f}\n" \
                        f"Position: ({self.selected_agent.position[0]:.2f}, {self.selected_agent.position[1]:.2f})\n" \
                        f"Memories: {len(self.selected_agent.memory_stream)}"
            tk.Label(self.agent_info_frame, text=info_text, justify=tk.LEFT).pack(pady=5, padx=10)
            
            # Update sparklines
            self.update_agent_charts()
            self.draw_avatar()
        else:
            for widget in self.agent_info_frame.winfo_children():
                widget.destroy()
            tk.Label(self.agent_info_frame, text="No agent selected.").pack(pady=20)

    def draw_avatar(self):
        self.avatar_canvas.delete("all")
        if not self.selected_agent:
            return
        avatar_data = self.selected_agent.avatar
        h, w = avatar_data.shape
        pixel_size = max(1, min(160 // h, 160 // w)) # Bug 10: clamp pixel size
        for i in range(h):
            for j in range(w):
                if avatar_data[i, j] == 1:
                    self.avatar_canvas.create_rectangle(
                        j * pixel_size, i * pixel_size,
                        (j + 1) * pixel_size, (i + 1) * pixel_size,
                        fill=self.selected_agent.color, outline="")

    def edit_avatar(self):
        if not self.selected_agent:
            self.log_queue.put("Please select an agent to edit its avatar.")
            return

        editor_window = tk.Toplevel(self)
        editor_window.title(f"Edit Avatar for {self.selected_agent.display_name}")
        
        avatar_copy = np.copy(self.selected_agent.avatar)
        
        editor_canvas = tk.Canvas(editor_window, width=320, height=320, bg="white")
        editor_canvas.pack(pady=10)
        
        def draw_preview():
            editor_canvas.delete("all")
            h, w = avatar_copy.shape
            pixel_size = 320 // max(h, w)
            for i in range(h):
                for j in range(w):
                    if avatar_copy[i, j] == 1:
                        editor_canvas.create_rectangle(
                            j * pixel_size, i * pixel_size,
                            (j + 1) * pixel_size, (i + 1) * pixel_size,
                            fill=self.selected_agent.color, outline="")
        
        def on_click(event):
            h, w = avatar_copy.shape
            pixel_size = 320 // max(h, w)
            j = int(event.x / pixel_size)
            i = int(event.y / pixel_size)
            if 0 <= i < h and 0 <= j < w:
                avatar_copy[i, j] = 1 - avatar_copy[i, j]
                draw_preview()

        def save_avatar():
            self.selected_agent.avatar = avatar_copy
            self.draw_avatar()
            self.log_queue.put(f"Avatar for {self.selected_agent.display_name} updated.")
            editor_window.destroy()

        editor_canvas.bind("<B1-Motion>", on_click)
        editor_canvas.bind("<Button-1>", on_click)
        
        tk.Button(editor_window, text="Save", command=save_avatar).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(editor_window, text="Cancel", command=editor_window.destroy).pack(side=tk.LEFT, padx=5, pady=5)
        
        draw_preview()

    def reset_avatar(self):
        if self.selected_agent:
            self.selected_agent.generate_procedural_avatar()
            self.draw_avatar()
            self.log_queue.put(f"Avatar for {self.selected_agent.display_name} reset.")

    def create_debate_transcript(self, parent):
        debate_frame = ttk.Frame(parent)
        debate_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.debate_transcript_label = ttk.Label(debate_frame, text="Live Debate Transcript", font=("Helvetica", 12))
        self.debate_transcript_label.pack(fill=tk.X, pady=(0, 5))
        
        self.transcript_text = tk.Text(debate_frame, wrap="word", bg="#1d2d3e", fg="white", insertbackground="white")
        self.transcript_text.pack(fill=tk.BOTH, expand=True)
        self.transcript_text.config(state=tk.DISABLED)

    def create_console_panel(self, parent):
        console_frame = ttk.Frame(parent)
        console_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Label(console_frame, text="Console").pack(pady=5)
        
        self.console_text = tk.Text(console_frame, wrap="word", bg="#1d2d3e", fg="white", insertbackground="white")
        self.console_text.pack(fill=tk.BOTH, expand=True)
        self.console_text.config(state=tk.DISABLED)

        self.console_input = ttk.Entry(console_frame)
        self.console_input.pack(fill=tk.X, pady=5)
        self.console_input.bind("<Return>", self.on_console_submit)
        
        # Enhancement 13: Rich-console formatting
        self.console_text.tag_config("system", foreground="lightblue")
        self.console_text.tag_config("error", foreground="red")
        self.console_text.tag_config("warning", foreground="yellow")
        self.console_text.tag_config("info", foreground="lightgreen")
        self.console_text.tag_config("bold", font=("Helvetica", 10, "bold"))
        
        # Enhancement 12: Command palette
        self.bind("<Control-k>", self.show_command_palette)
        # Enhancement 19: Keyboard shortcuts cheat-sheet
        ttk.Button(console_frame, text="?", command=self.show_shortcuts).pack(side=tk.RIGHT, padx=5)

    def show_command_palette(self, event=None):
        palette_window = tk.Toplevel(self)
        palette_window.title("Command Palette")
        palette_window.geometry("400x300")
        
        search_entry = ttk.Entry(palette_window)
        search_entry.pack(fill=tk.X, padx=10, pady=5)
        
        cmd_listbox = tk.Listbox(palette_window)
        cmd_listbox.pack(fill=tk.BOTH, expand=True, padx=10)
        
        commands = {
            "Toggle Theme": "toggle theme",
            "Pause/Resume": "pause",
            "Randomize Goals": "randomize goals",
            "Start Debate": "start debate",
            "Save Snapshot": "save snapshot"
        }
        
        for cmd in commands.keys():
            cmd_listbox.insert(tk.END, cmd)
        
        def on_select(event):
            selection = cmd_listbox.get(cmd_listbox.curselection())
            if selection:
                command_text = commands[selection]
                self.console_input.delete(0, tk.END)
                self.console_input.insert(0, command_text)
                palette_window.destroy()

        cmd_listbox.bind("<<ListboxSelect>>", on_select)
        
    def show_shortcuts(self):
        shortcuts_window = tk.Toplevel(self)
        shortcuts_window.title("Keyboard Shortcuts")
        shortcuts_window.geometry("400x300")
        
        shortcuts_text = tk.Text(shortcuts_window, wrap="word")
        shortcuts_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        shortcuts_text.insert(tk.END, "Hotkeys:\n\n")
        shortcuts_text.insert(tk.END, "P: Pause/Resume\n")
        shortcuts_text.insert(tk.END, "S: Save Snapshot\n")
        shortcuts_text.insert(tk.END, "T: Toggle Theme\n")
        shortcuts_text.insert(tk.END, "Ctrl+K: Open Command Palette\n")
        shortcuts_text.insert(tk.END, "1-5: Save Camera Bookmark\n")
        shortcuts_text.insert(tk.END, "Ctrl+1-5: Load Camera Bookmark\n")
        shortcuts_text.config(state=tk.DISABLED)

    def on_console_submit(self, event):
        command = self.console_input.get()
        if command:
            self.console_input_history.append(command)
            self.log_queue.put(f"> {command}", "info")
            
            # Start a thread for the NLP command to keep the UI responsive
            threading.Thread(target=self.console.process_input, args=(command,)).start()
            
            self.console_input.delete(0, tk.END)

    def _reinitialize_agents(self, count):
        self.agents = [Agent.create_procedural_agent(i) for i in range(count)]

    def start_gui_loop(self):
        self.log_queue.put(f"Simulation started with seed: {self.seed}")
        self.after(10, self.update_gui)
        self.mainloop()
        
    def start_headless_loop(self):
        self.log_queue.put(f"Headless simulation started with seed: {self.seed}")
        # Enhancement 20: Headless mode
        while self.running:
            self.run_simulation_step()
            self.check_queue()
            if self.pause_requested:
                while self.pause_requested:
                    time.sleep(0.1)
        self.dump_final_metrics()

    def update_gui(self):
        self.run_simulation_step()
        self.update_universe_view()
        self.check_queue()
        self.check_autosave()
        self.update_quick_metrics()
        self.after(10, self.update_gui)
        
    def run_simulation_step(self):
        if not self.pause_requested:
            self.is_updating = True
            
            # Bug 6: Guard against None coords
            if self.universe_visualizer.pan_start_x is not None and self.universe_visualizer.pan_start_y is not None:
                self.universe_visualizer.pan_canvas()
            
            self.sim_core.run_step(self.agents)
            self.metrics.collect(self.current_step, self.agents)
            self.current_step += 1
            self.is_updating = False

    def update_universe_view(self):
        start_time = time.time()
        self.universe_visualizer.update_view(self.agents)
        self.canvas.draw_idle()
        
        # Update the mini-map viewport rectangle
        self.universe_visualizer.draw_mini_map_viewport(self.mini_map_canvas.get_tk_widget().winfo_width(), self.mini_map_canvas.get_tk_widget().winfo_height())
        self.mini_map_canvas.draw_idle()
        
        # Enhancement 17: Performance guardrails
        end_time = time.time()
        frame_time = (end_time - start_time) * 1000 # in ms
        if frame_time > 50 and len(self.agents) > 50:
            self.guardrail_tooltip.config(text=f"Lag detected ({frame_time:.1f}ms). Consider reducing agents or hiding layers.")
            self.guardrail_tooltip.place(relx=0.5, rely=0.1, anchor=tk.CENTER)
        else:
            self.guardrail_tooltip.place_forget()
            
    def check_queue(self):
        # Bug 5: Single, robust queue polling mechanism
        try:
            while True:
                log_entry = self.log_queue.get_nowait()
                self.insert_to_console(log_entry)
        except queue.Empty:
            pass
            
        try:
            while True:
                transcript_entry = self.transcript_queue.get_nowait()
                self.insert_to_transcript(transcript_entry)
        except queue.Empty:
            pass

    def insert_to_console(self, entry):
        self.console_text.config(state=tk.NORMAL)
        self.console_text.insert(tk.END, entry + "\n", "system") # Example tag
        self.console_text.config(state=tk.DISABLED)
        self.console_text.see(tk.END)
        
    def insert_to_transcript(self, entry):
        self.transcript_text.config(state=tk.NORMAL)
        self.transcript_text.insert(tk.END, entry + "\n")
        self.transcript_text.config(state=tk.DISABLED)
        self.transcript_text.see(tk.END)

    def check_autosave(self):
        cadence = self.autosave_cadence_var.get()
        if cadence != "Off":
            try:
                if "steps" in cadence:
                    steps = int(cadence.split()[0])
                    if self.current_step % steps == 0 and self.current_step > 0:
                        self.save_snapshot_silent()
                elif "s" in cadence:
                    seconds = int(cadence.split()[0])
                    if time.time() - self.last_autosave_time > seconds:
                        self.save_snapshot_silent()
            except (ValueError, IndexError):
                pass
                
    def save_snapshot_silent(self):
        if not self.is_updating:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"autosave_{timestamp}.json"
            snapshot = {
                "meta": {"timestamp": timestamp, "note": "Autosave"},
                "agents": [a.to_dict() for a in self.agents],
                "simulation_state": {"step": self.current_step}
            }
            try:
                with open(filename, "w") as f:
                    json.dump(snapshot, f)
                self.last_autosave_time = time.time()
                self.last_autosave_label.config(text=f"Last saved: {datetime.now().strftime('%H:%M:%S')}")
                self.log_queue.put(f"Autosaved simulation state to {filename}")
            except Exception as e:
                self.log_queue.put(f"Error autosaving: {e}")

    def update_quick_metrics(self):
        metrics = self.metrics.get_last()
        self.fps_label.config(text=f"FPS: {metrics.get('fps', 0):.1f}")
        self.mem_label.config(text=f"MEM: {metrics.get('mem', 0):.1f} MB")
        self.entropy_label.config(text=f"ENTROPY: {metrics.get('entropy', 0):.2f}")
        self.alignment_label.config(text=f"ALIGNMENT: {metrics.get('alignment', 0):.2f}")
        self.cohesion_label.config(text=f"COHESION: {metrics.get('cohesion', 0):.2f}")

    def bind_shortcuts(self):
        self.bind("<Escape>", lambda e: self.quit())
        self.bind("p", lambda e: self.pause_simulation())
        self.bind("P", lambda e: self.pause_simulation())
        
    def start_resume_simulation(self):
        self.pause_requested = False
        self.log_queue.put("Simulation resumed.", "system")

    def pause_simulation(self):
        self.pause_requested = True
        self.log_queue.put("Simulation paused.", "system")
        
    def reinitialize_simulation(self):
        self.pause_requested = True
        self.sim_core.reset()
        self._reinitialize_agents(10)
        self.current_step = 0
        self.update_agent_panel()
        self.log_queue.put("Simulation reinitialized.", "system")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hexuni Simulation")
    parser.add_argument("--seed", type=int, help="Seed for reproducible simulations.")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without a GUI.")
    args = parser.parse_args()
    
    # Check if we are running in headless mode first
    if args.headless:
        # Enhancement 20: Headless mode
        app = MainApp(seed=args.seed, headless=True)
    else:
        app = MainApp(seed=args.seed)
        
