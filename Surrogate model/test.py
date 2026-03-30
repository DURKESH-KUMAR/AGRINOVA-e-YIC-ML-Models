"""
Surrogate Model — Nozzle CFD Optimization
Professional Light-Theme Tkinter GUI with CSV Input/Output
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import numpy as np
import pandas as pd
from datetime import datetime
import os

# ──────────────────────────────────────────────────────────────
# PROFESSIONAL LIGHT COLOUR PALETTE
# ──────────────────────────────────────────────────────────────
BG          = "#F5F6FA"
PANEL       = "#FFFFFF"
CARD        = "#FFFFFF"
CARD_ALT    = "#F0F2F8"
BORDER      = "#DDE1EE"
BORDER_DARK = "#C4C9DC"

ACCENT      = "#1A56DB"
ACCENT_DK   = "#1344B8"
ACCENT_LT   = "#EBF0FD"
GREEN       = "#0E9F6E"
GREEN_LT    = "#EDFAF4"
WARN        = "#D97706"
WARN_LT     = "#FFF8EB"
RED         = "#E02424"
RED_LT      = "#FEF2F2"
PURPLE      = "#7E3AF2"

TEXT        = "#111827"
TEXT_MED    = "#374151"
TEXT_DIM    = "#6B7280"
TEXT_LIGHT  = "#9CA3AF"
WHITE       = "#FFFFFF"

F_TITLE   = ("Segoe UI", 13, "bold")
F_HEAD    = ("Segoe UI", 10, "bold")
F_BODY    = ("Segoe UI", 9)
F_SMALL   = ("Segoe UI", 8)
F_MONO    = ("Consolas", 9)
F_BIG     = ("Segoe UI", 22, "bold")
F_MED     = ("Segoe UI", 14, "bold")

# ──────────────────────────────────────────────────────────────
# DEFAULT DATA (for reference)
# ──────────────────────────────────────────────────────────────
DEFAULT_X = [
    [30, 100, 50, 70, 20],
    [15, 113, 43, 85, 25],
    [38,  95, 60, 80, 21],
    [24, 110, 48, 75, 23],
    [40,  80, 32, 55, 18],
]
DEFAULT_Y = [
    [0.641, 0.898, 0.615, 0.498, 0.0439],
    [0.777, 0.728, 0.640, 0.206, 0.0398],
    [0.818, 0.635, 0.635, 0.304, 0.0453],
    [0.707, 1.480, 0.602, 0.361, 0.0414],
    [1.080, 0.841, 0.659, 1.170, 0.0522],
]
PARAM_NAMES   = ["Angle (°)", "Protrusion (mm)", "Small Dia (mm)", "Big Dia (mm)", "Hole Dia (mm)"]
OUTPUT_NAMES  = ["Max Vel (m/s)", "Avg Vel (m/s)", "Flow Uniformity", "Pressure Drop (Pa)", "Volume Flow"]
OUTPUT_KEYS   = ["max_vel", "avg_vel", "flow_uniformity", "pressure_drop", "volume_flow"]
DESIGN_LABELS = ["D1 — Base", "D2 — Flat long", "D3 — Steep short", "D4 — Gentle long", "D5 — Steep narrow"]
BOUNDS_DEF    = [(15, 50), (60, 130), (32, 65), (55, 85), (18, 25)]
PARAM_KEYS    = ["angle", "protrusion", "small_dia", "big_dia", "hole_dia"]
WEIGHT_META   = [
    ("avg_velocity",    "Avg Velocity",    "higher = better", 0.35),
    ("flow_uniformity", "Flow Uniformity", "higher = better", 0.25),
    ("pressure_drop",   "Pressure Drop",   "lower  = better", 0.20),
    ("max_velocity",    "Max Velocity",    "higher = better", 0.15),
    ("volume_flow",     "Volume Flow",     "higher = better", 0.05),
]

# CSV column names
X_COLUMNS = ["angle", "protrusion", "small_dia", "big_dia", "hole_dia"]
Y_COLUMNS = ["max_vel", "avg_vel", "flow_uniformity", "pressure_drop", "volume_flow"]


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def flat_entry(parent, width=9, val=""):
    e = tk.Entry(
        parent, width=width, bg=WHITE, fg=TEXT,
        insertbackground=ACCENT, relief="flat", font=F_MONO,
        highlightthickness=1, highlightbackground=BORDER,
        highlightcolor=ACCENT, bd=2,
    )
    e.insert(0, str(val))
    return e


def card_frame(parent, padx=12, pady=8):
    outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
    inner = tk.Frame(outer, bg=CARD, padx=padx, pady=pady)
    inner.pack(fill="both", expand=True)
    return outer, inner


def section_label(parent, text):
    row = tk.Frame(parent, bg=BG)
    row.pack(fill="x", padx=12, pady=(14, 4))
    tk.Label(row, text=text, bg=BG, fg=ACCENT, font=F_HEAD).pack(side="left")
    tk.Frame(row, bg=BORDER, height=1).pack(side="left", fill="x", expand=True, padx=(10, 0))


# ══════════════════════════════════════════════════════════════
class App(tk.Tk):
# ══════════════════════════════════════════════════════════════
    def __init__(self):
        super().__init__()
        self.title("Surrogate Model — Nozzle CFD Optimization")
        self.configure(bg=BG)
        self.geometry("1380x880")
        self.minsize(1100, 720)
        self.running = False
        self.current_csv_file = None
        self.X_data = None
        self.Y_data = None
        self._apply_ttk_styles()
        self._build_header()
        self._build_body()
        self._build_statusbar()

    def _apply_ttk_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        
        # Configure progress bar
        s.configure("TProgressbar",
                    troughcolor=CARD_ALT, background=ACCENT,
                    bordercolor=BORDER, lightcolor=ACCENT, darkcolor=ACCENT,
                    thickness=4)
        
        # Configure custom scrollbar style
        s.configure("Thin.Vertical.TScrollbar",
                    troughcolor=BG, background=BORDER_DARK,
                    borderwidth=0, width=8,
                    arrowcolor=TEXT_DIM)
        
        s.configure("Thin.Horizontal.TScrollbar",
                    troughcolor=BG, background=BORDER_DARK,
                    borderwidth=0, height=8,
                    arrowcolor=TEXT_DIM)

    # ── HEADER ──────────────────────────────────────────────
    def _build_header(self):
        h = tk.Frame(self, bg=WHITE, height=80)
        h.pack(fill="x")
        h.pack_propagate(False)

        tk.Frame(h, bg=ACCENT, width=5).pack(side="left", fill="y")

        logo = tk.Frame(h, bg=WHITE)
        logo.pack(side="left", padx=22, fill="y")
        tk.Label(logo, text="CFD OPTIMIZER", bg=WHITE, fg=ACCENT,
                 font=("Segoe UI", 15, "bold")).pack(anchor="w", pady=(14, 0))
        tk.Label(logo, text="Surrogate Model  ·  Gaussian Process Regression  ·  LHS Sampling",
                 bg=WHITE, fg=TEXT_DIM, font=F_SMALL).pack(anchor="w")

        # CSV File controls
        file_frame = tk.Frame(h, bg=WHITE)
        file_frame.pack(side="left", padx=(30, 0), fill="y")
        
        self.file_label = tk.Label(file_frame, text="No file loaded", bg=WHITE, 
                                   fg=TEXT_DIM, font=F_SMALL)
        self.file_label.pack(side="top", pady=(15, 2))
        
        btn_frame = tk.Frame(file_frame, bg=WHITE)
        btn_frame.pack()
        
        self.load_btn = tk.Button(
            btn_frame, text="📁 Load CSV", bg=WHITE, fg=ACCENT,
            font=F_SMALL, relief="solid", bd=1,
            highlightbackground=BORDER, cursor="hand2",
            command=self._load_csv_file, padx=12, pady=2
        )
        self.load_btn.pack(side="left", padx=2)
        
        self.export_btn = tk.Button(
            btn_frame, text="💾 Export Results", bg=WHITE, fg=GREEN,
            font=F_SMALL, relief="solid", bd=1,
            highlightbackground=BORDER, cursor="hand2",
            command=self._export_results, padx=12, pady=2,
            state="disabled"
        )
        self.export_btn.pack(side="left", padx=2)

        tk.Frame(h, bg=BORDER, width=1).pack(side="left", fill="y", padx=16, pady=16)

        badges = tk.Frame(h, bg=WHITE)
        badges.pack(side="left", fill="y")
        badge_data = [
            ("GP Regression", ACCENT, ACCENT_LT),
            ("LHS Sampling",  GREEN,  GREEN_LT),
            ("Multi-Objective", PURPLE, "#F5F0FF"),
        ]
        for txt, col, lt in badge_data:
            f = tk.Frame(badges, bg=lt, padx=10, pady=3)
            f.pack(side="left", padx=4, pady=22)
            tk.Label(f, text=txt, bg=lt, fg=col, font=F_SMALL).pack()

        self.run_btn = tk.Button(
            h, text="  ▶   RUN OPTIMIZATION  ",
            bg=ACCENT, fg=WHITE, font=("Segoe UI", 10, "bold"),
            relief="flat", bd=0,
            activebackground=ACCENT_DK, activeforeground=WHITE,
            cursor="hand2", padx=8, pady=0,
            command=self._run_threaded,
        )
        self.run_btn.pack(side="right", padx=24, pady=16, ipady=8)

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

    # ── BODY ────────────────────────────────────────────────
    def _build_body(self):
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        left_wrap = tk.Frame(body, bg=BG, width=590)
        left_wrap.pack(side="left", fill="y", padx=(10, 5), pady=10)
        left_wrap.pack_propagate(False)
        self._build_left(left_wrap)

        tk.Frame(body, bg=BORDER, width=1).pack(side="left", fill="y", pady=10)

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(5, 10), pady=10)
        self._build_right(right)

    # ──────────────────────────────────────────────────────
    # LEFT — INPUTS
    # ──────────────────────────────────────────────────────
    def _build_left(self, parent):
        # Create a canvas with scrollbar
        canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient="vertical",
                            command=canvas.yview, 
                            style="Thin.Vertical.TScrollbar")
        canvas.configure(yscrollcommand=vsb.set)
        
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Create scrollable frame
        scroll_frame = tk.Frame(canvas, bg=BG)
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw", width=canvas.winfo_reqwidth())
        
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        def configure_canvas_width(event):
            canvas.itemconfig(1, width=event.width)
        
        scroll_frame.bind("<Configure>", configure_scroll_region)
        canvas.bind("<Configure>", configure_canvas_width)
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._build_file_info(scroll_frame)
        self._build_x_table(scroll_frame)
        self._build_y_table(scroll_frame)
        self._build_weights(scroll_frame)
        self._build_bounds(scroll_frame)
        self._build_sampling(scroll_frame)

    def _build_file_info(self, parent):
        """Display loaded file information"""
        info_frame = tk.Frame(parent, bg=BG)
        info_frame.pack(fill="x", padx=12, pady=(0, 8))
        
        self.file_info_label = tk.Label(info_frame, text="📄 No CSV file loaded. Using default data.", 
                                        bg=BG, fg=TEXT_DIM, font=F_SMALL)
        self.file_info_label.pack(anchor="w")

    def _build_x_table(self, parent):
        section_label(parent, "①  CFD DESIGN POINTS  (X Inputs)")
        outer, inner = card_frame(parent, padx=0, pady=0)
        outer.pack(fill="x", padx=12, pady=(0, 4))

        hdr = tk.Frame(inner, bg=CARD_ALT)
        hdr.pack(fill="x")
        tk.Label(hdr, text="  Design", bg=CARD_ALT, fg=TEXT_DIM,
                 font=F_SMALL, anchor="w", width=17).pack(side="left", pady=6, padx=4)
        for h in ["Angle°", "Protr mm", "SmDia mm", "BgDia mm", "Hole mm"]:
            tk.Label(hdr, text=h, bg=CARD_ALT, fg=TEXT_DIM,
                     font=F_SMALL, width=9, anchor="center").pack(side="left")

        self.x_entries = []
        for r in range(5):
            bg = WHITE if r % 2 == 0 else CARD_ALT
            row = tk.Frame(inner, bg=bg)
            row.pack(fill="x")
            tk.Label(row, text=f"  {DESIGN_LABELS[r]}", bg=bg, fg=TEXT_MED,
                     font=F_SMALL, width=17, anchor="w").pack(side="left", pady=5, padx=4)
            entries = []
            for c in range(5):
                e = flat_entry(row, width=8, val=DEFAULT_X[r][c])
                e.pack(side="left", padx=3, pady=4)
                entries.append(e)
            self.x_entries.append(entries)

    def _build_y_table(self, parent):
        section_label(parent, "②  CFD RESULTS  (Y Outputs)")
        outer, inner = card_frame(parent, padx=0, pady=0)
        outer.pack(fill="x", padx=12, pady=(0, 4))

        hdr = tk.Frame(inner, bg=CARD_ALT)
        hdr.pack(fill="x")
        tk.Label(hdr, text="  Design", bg=CARD_ALT, fg=TEXT_DIM,
                 font=F_SMALL, anchor="w", width=17).pack(side="left", pady=6, padx=4)
        for h in ["MaxVel", "AvgVel", "FlowUnif", "PresDrop", "VolFlow"]:
            tk.Label(hdr, text=h, bg=CARD_ALT, fg=TEXT_DIM,
                     font=F_SMALL, width=9, anchor="center").pack(side="left")

        self.y_entries = []
        for r in range(5):
            bg = WHITE if r % 2 == 0 else CARD_ALT
            row = tk.Frame(inner, bg=bg)
            row.pack(fill="x")
            tk.Label(row, text=f"  {DESIGN_LABELS[r]}", bg=bg, fg=TEXT_MED,
                     font=F_SMALL, width=17, anchor="w").pack(side="left", pady=5, padx=4)
            entries = []
            for c in range(5):
                e = flat_entry(row, width=8, val=DEFAULT_Y[r][c])
                e.pack(side="left", padx=3, pady=4)
                entries.append(e)
            self.y_entries.append(entries)

    def _build_weights(self, parent):
        section_label(parent, "③  OBJECTIVE WEIGHTS  (must sum to 1.0)")
        outer, inner = card_frame(parent)
        outer.pack(fill="x", padx=12, pady=(0, 4))
        self.weight_entries = {}
        for key, label, hint, default in WEIGHT_META:
            row = tk.Frame(inner, bg=CARD)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label, bg=CARD, fg=TEXT, font=F_BODY,
                     width=18, anchor="w").pack(side="left")
            e = flat_entry(row, width=7, val=default)
            e.pack(side="left", padx=8)
            tk.Label(row, text=hint, bg=CARD, fg=TEXT_LIGHT, font=F_SMALL).pack(side="left")
            self.weight_entries[key] = e

    def _build_bounds(self, parent):
        section_label(parent, "④  PARAMETER BOUNDS")
        outer, inner = card_frame(parent, padx=0, pady=0)
        outer.pack(fill="x", padx=12, pady=(0, 4))

        hdr = tk.Frame(inner, bg=CARD_ALT)
        hdr.pack(fill="x")
        for txt, w in [("  Parameter", 22), ("Min", 10), ("Max", 10)]:
            tk.Label(hdr, text=txt, bg=CARD_ALT, fg=TEXT_DIM,
                     font=F_SMALL, width=w, anchor="w").pack(side="left", pady=5, padx=4)

        self.bound_entries = []
        bnames = ["Angle (°)", "Protrusion (mm)", "Small Dia (mm)", "Big Dia (mm)", "Hole Dia (mm)"]
        for r, (lo, hi) in enumerate(BOUNDS_DEF):
            bg = WHITE if r % 2 == 0 else CARD_ALT
            row = tk.Frame(inner, bg=bg)
            row.pack(fill="x")
            tk.Label(row, text=f"  {bnames[r]}", bg=bg, fg=TEXT_MED,
                     font=F_SMALL, width=22, anchor="w").pack(side="left", pady=5, padx=4)
            elo = flat_entry(row, width=9, val=lo); elo.pack(side="left", padx=4, pady=4)
            ehi = flat_entry(row, width=9, val=hi); ehi.pack(side="left", padx=4, pady=4)
            self.bound_entries.append((elo, ehi))

    def _build_sampling(self, parent):
        section_label(parent, "⑤  SAMPLING CONFIGURATION")
        outer, inner = card_frame(parent)
        outer.pack(fill="x", padx=12, pady=(0, 12))
        row = tk.Frame(inner, bg=CARD)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="LHS Candidates:", bg=CARD, fg=TEXT, font=F_BODY,
                 width=18, anchor="w").pack(side="left")
        self.samples_entry = flat_entry(row, width=10, val="20000")
        self.samples_entry.pack(side="left", padx=8)
        tk.Label(row, text="higher = slower, more thorough",
                 bg=CARD, fg=TEXT_LIGHT, font=F_SMALL).pack(side="left")

    # ──────────────────────────────────────────────────────
    # CSV FILE HANDLING
    # ──────────────────────────────────────────────────────
    def _load_csv_file(self):
        """Load CSV file with CFD data"""
        filename = filedialog.askopenfilename(
            title="Select CFD Data CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            df = pd.read_csv(filename)
            
            # Check required columns
            missing_x = [col for col in X_COLUMNS if col not in df.columns]
            missing_y = [col for col in Y_COLUMNS if col not in df.columns]
            
            if missing_x or missing_y:
                error_msg = "Missing required columns:\n"
                if missing_x:
                    error_msg += f"X columns: {missing_x}\n"
                if missing_y:
                    error_msg += f"Y columns: {missing_y}\n"
                error_msg += "\nRequired X columns: angle, protrusion, small_dia, big_dia, hole_dia\n"
                error_msg += "Required Y columns: max_vel, avg_vel, flow_uniformity, pressure_drop, volume_flow"
                messagebox.showerror("Invalid CSV Format", error_msg)
                return
            
            # Extract data
            self.X_data = df[X_COLUMNS].values
            self.Y_data = df[Y_COLUMNS].values
            self.current_csv_file = filename
            
            # Update table displays
            self._update_tables_from_csv()
            
            # Update file info
            n_designs = len(self.X_data)
            self.file_info_label.config(
                text=f"📄 Loaded: {os.path.basename(filename)} ({n_designs} design points)",
                fg=GREEN
            )
            self.file_label.config(text=os.path.basename(filename), fg=GREEN)
            self.export_btn.config(state="normal")
            
            self._log(f"✓ Loaded CSV file: {os.path.basename(filename)}", "ok")
            self._log(f"  {n_designs} design points loaded", "dim")
            
        except Exception as e:
            messagebox.showerror("Error Loading File", f"Failed to load CSV file:\n{str(e)}")
            self._log(f"✗ Error loading CSV: {str(e)}", "err")
    
    def _update_tables_from_csv(self):
        """Update the table entries with CSV data"""
        if self.X_data is None or self.Y_data is None:
            return
        
        n_points = min(5, len(self.X_data))  # Show first 5 points
        
        # Update X table
        for r in range(n_points):
            for c in range(5):
                self.x_entries[r][c].delete(0, tk.END)
                self.x_entries[r][c].insert(0, f"{self.X_data[r, c]:.2f}")
        
        # Clear remaining rows if fewer than 5 points
        for r in range(n_points, 5):
            for c in range(5):
                self.x_entries[r][c].delete(0, tk.END)
                self.x_entries[r][c].insert(0, "0")
        
        # Update Y table
        for r in range(n_points):
            for c in range(5):
                self.y_entries[r][c].delete(0, tk.END)
                self.y_entries[r][c].insert(0, f"{self.Y_data[r, c]:.4f}")
        
        # Clear remaining rows
        for r in range(n_points, 5):
            for c in range(5):
                self.y_entries[r][c].delete(0, tk.END)
                self.y_entries[r][c].insert(0, "0")
    
    def _export_results(self):
        """Export optimization results to CSV file"""
        if not hasattr(self, 'last_optimization_results'):
            messagebox.showwarning("No Results", "Run optimization first before exporting results.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Optimization Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            results = self.last_optimization_results
            
            # Create results dataframe
            df_results = pd.DataFrame({
                'Parameter': PARAM_NAMES,
                'Optimal_Value': results['best_params'],
                'Unit': ['°', 'mm', 'mm', 'mm', 'mm']
            })
            
            df_performance = pd.DataFrame({
                'Output': OUTPUT_NAMES,
                'Predicted_Value': results['best_pred'],
                'Uncertainty': results['best_std'],
                'Reliability_Status': results['reliability_status']
            })
            
            # Save to CSV
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df_results.to_excel(writer, sheet_name='Optimal_Parameters', index=False)
                df_performance.to_excel(writer, sheet_name='Performance', index=False)
                
                # Add summary info
                summary = pd.DataFrame({
                    'Metric': ['Composite_Score', 'Reliability', 'Timestamp'],
                    'Value': [results['score'], results['reliability_msg'], datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                })
                summary.to_excel(writer, sheet_name='Summary', index=False)
            
            self._log(f"✓ Results exported to: {os.path.basename(filename)}", "ok")
            messagebox.showinfo("Export Successful", f"Results saved to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to export results:\n{str(e)}")
            self._log(f"✗ Export error: {str(e)}", "err")

    # ──────────────────────────────────────────────────────
    # RIGHT — RESULTS + CONSOLE
    # ──────────────────────────────────────────────────────
    def _build_right(self, parent):
        # optimal parameter cards
        tk.Label(parent, text="OPTIMAL DESIGN PARAMETERS",
                 bg=BG, fg=TEXT_DIM, font=F_SMALL).pack(anchor="w", padx=4, pady=(2, 4))

        cards_row = tk.Frame(parent, bg=BG)
        cards_row.pack(fill="x", pady=(0, 6))
        self.result_val_lbls = []
        units  = ["°", "mm", "mm", "mm", "mm"]
        colors = [ACCENT, GREEN, WARN, PURPLE, RED]
        for i in range(5):
            outer = tk.Frame(cards_row, bg=BORDER, padx=1, pady=1)
            outer.pack(side="left", fill="x", expand=True, padx=3)
            inner = tk.Frame(outer, bg=WHITE, padx=10, pady=8)
            inner.pack(fill="both", expand=True)
            tk.Frame(inner, bg=colors[i], height=3).pack(fill="x", pady=(0, 6))
            short = PARAM_NAMES[i].split("(")[0].strip().upper()
            tk.Label(inner, text=short, bg=WHITE, fg=TEXT_DIM, font=F_SMALL).pack()
            v = tk.Label(inner, text="—", bg=WHITE, fg=colors[i], font=F_MED)
            v.pack()
            tk.Label(inner, text=units[i], bg=WHITE, fg=TEXT_LIGHT, font=F_SMALL).pack(pady=(0, 2))
            self.result_val_lbls.append(v)

        # score + reliability
        sc_outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
        sc_outer.pack(fill="x", padx=3, pady=(0, 6))
        sc_inner = tk.Frame(sc_outer, bg=WHITE, padx=16, pady=10)
        sc_inner.pack(fill="both")

        ls = tk.Frame(sc_inner, bg=WHITE)
        ls.pack(side="left")
        tk.Label(ls, text="COMPOSITE SCORE", bg=WHITE, fg=TEXT_DIM, font=F_SMALL).pack(anchor="w")
        self.score_lbl = tk.Label(ls, text="—", bg=WHITE, fg=PURPLE,
                                  font=("Segoe UI", 26, "bold"))
        self.score_lbl.pack(anchor="w")

        tk.Frame(sc_inner, bg=BORDER, width=1).pack(side="left", fill="y", padx=20)

        rs = tk.Frame(sc_inner, bg=WHITE)
        rs.pack(side="left", fill="both", expand=True)
        tk.Label(rs, text="RELIABILITY", bg=WHITE, fg=TEXT_DIM, font=F_SMALL).pack(anchor="w")
        self.reliability_lbl = tk.Label(rs, text="—", bg=WHITE, fg=TEXT_MED, font=F_HEAD)
        self.reliability_lbl.pack(anchor="w", pady=(4, 0))
        self.reliability_detail = tk.Label(rs, text="", bg=WHITE, fg=TEXT_LIGHT,
                                           font=F_SMALL, wraplength=320, justify="left")
        self.reliability_detail.pack(anchor="w")

        # performance table
        pf_outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
        pf_outer.pack(fill="x", padx=3, pady=(0, 8))
        pf_inner = tk.Frame(pf_outer, bg=WHITE)
        pf_inner.pack(fill="both")

        hdr = tk.Frame(pf_inner, bg=CARD_ALT)
        hdr.pack(fill="x")
        for txt, w in [("Output", 22), ("Predicted Value", 18), ("Uncertainty (±)", 18), ("Status", 12)]:
            tk.Label(hdr, text=txt, bg=CARD_ALT, fg=TEXT_DIM, font=F_SMALL,
                     width=w, anchor="w").pack(side="left", padx=10, pady=5)

        self.perf_rows = []
        for i, name in enumerate(OUTPUT_NAMES):
            bg = WHITE if i % 2 == 0 else CARD_ALT
            row = tk.Frame(pf_inner, bg=bg)
            row.pack(fill="x")
            tk.Label(row, text=name, bg=bg, fg=TEXT_MED, font=F_BODY,
                     width=22, anchor="w").pack(side="left", padx=10, pady=5)
            pl = tk.Label(row, text="—", bg=bg, fg=TEXT, font=F_MONO, width=18, anchor="w")
            pl.pack(side="left", padx=10)
            sl = tk.Label(row, text="—", bg=bg, fg=TEXT_DIM, font=F_MONO, width=18, anchor="w")
            sl.pack(side="left", padx=10)
            stl = tk.Label(row, text="—", bg=bg, fg=TEXT_LIGHT, font=F_SMALL, width=12, anchor="w")
            stl.pack(side="left", padx=10)
            self.perf_rows.append((pl, sl, stl))

        # console header
        ch = tk.Frame(parent, bg=BG)
        ch.pack(fill="x", pady=(2, 3))
        tk.Label(ch, text="CONSOLE OUTPUT", bg=BG, fg=TEXT_DIM, font=F_SMALL).pack(side="left", padx=4)
        tk.Button(ch, text="Clear", bg=WHITE, fg=TEXT_DIM, font=F_SMALL,
                  relief="flat", cursor="hand2",
                  highlightthickness=1, highlightbackground=BORDER,
                  command=self._clear_console, padx=8, pady=2).pack(side="right", padx=4)

        co_outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
        co_outer.pack(fill="both", expand=True, padx=3, pady=(0, 4))
        self.console = scrolledtext.ScrolledText(
            co_outer, bg=WHITE, fg=TEXT, font=F_MONO,
            insertbackground=ACCENT, relief="flat",
            wrap="word", state="disabled",
            selectbackground=ACCENT_LT, selectforeground=TEXT,
        )
        self.console.pack(fill="both", expand=True)
        self.console.tag_configure("ok",   foreground=GREEN,  font=F_MONO)
        self.console.tag_configure("warn", foreground=WARN,   font=F_MONO)
        self.console.tag_configure("err",  foreground=RED,    font=F_MONO)
        self.console.tag_configure("head", foreground=ACCENT, font=("Consolas", 9, "bold"))
        self.console.tag_configure("dim",  foreground=TEXT_DIM, font=F_MONO)

    # ── STATUS BAR ──────────────────────────────────────────
    def _build_statusbar(self):
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        bar = tk.Frame(self, bg=WHITE, height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.status_dot = tk.Label(bar, text="●", bg=WHITE, fg=TEXT_LIGHT, font=("Segoe UI", 8))
        self.status_dot.pack(side="left", padx=(14, 4), pady=6)
        self.status_lbl = tk.Label(bar, text="Ready", bg=WHITE, fg=TEXT_DIM, font=F_SMALL)
        self.status_lbl.pack(side="left")
        self.progress = ttk.Progressbar(bar, length=180, mode="indeterminate")
        self.progress.pack(side="right", padx=14, pady=7)
        tk.Label(bar, text="Surrogate Model v2.0", bg=WHITE,
                 fg=TEXT_LIGHT, font=F_SMALL).pack(side="right", padx=20)

    # ──────────────────────────────────────────────────────
    def _log(self, msg, tag=None):
        self.console.configure(state="normal")
        self.console.insert("end", msg + "\n", tag or "")
        self.console.see("end")
        self.console.configure(state="disabled")

    def _clear_console(self):
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    def _set_status(self, msg, color=TEXT_DIM, dot=TEXT_LIGHT):
        self.status_lbl.configure(text=msg, fg=color)
        self.status_dot.configure(fg=dot)

    def _read_inputs(self):
        # Read from table entries
        X = [[float(self.x_entries[r][c].get()) for c in range(5)] for r in range(5)]
        Y = [[float(self.y_entries[r][c].get()) for c in range(5)] for r in range(5)]
        
        # If CSV loaded with more than 5 points, use all points
        if self.X_data is not None and len(self.X_data) > 5:
            self._log(f"Using {len(self.X_data)} design points from CSV", "dim")
            X = self.X_data.tolist()
            Y = self.Y_data.tolist()
        
        weights = {k: float(self.weight_entries[k].get()) for k, *_ in WEIGHT_META}
        bounds  = [(float(lo.get()), float(hi.get())) for lo, hi in self.bound_entries]
        n       = int(self.samples_entry.get())
        return np.array(X), np.array(Y), weights, bounds, n

    # ──────────────────────────────────────────────────────
    # RUN
    # ──────────────────────────────────────────────────────
    def _run_threaded(self):
        if self.running:
            return
        self.running = True
        self.run_btn.configure(state="disabled", text="  ⏳  Running…  ", bg=BORDER_DARK)
        self.progress.start(10)
        self._set_status("Running optimization…", ACCENT, ACCENT)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            X_cfd, Y_cfd, weights, bounds_list, n_samples = self._read_inputs()
        except ValueError as ex:
            self.after(0, lambda: self._finish_error(f"Input error: {ex}"))
            return

        total_w = sum(weights.values())
        if abs(total_w - 1.0) > 0.01:
            self.after(0, lambda: self._finish_error(
                f"Weights sum to {total_w:.3f} — must equal 1.0"))
            return

        self.after(0, lambda: self._log("─" * 58, "dim"))
        self.after(0, lambda: self._log("  SURROGATE MODEL  ·  Nozzle CFD Optimization", "head"))
        self.after(0, lambda: self._log("─" * 58 + "\n", "dim"))

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.metrics import r2_score
            from scipy.stats import qmc
        except ImportError as ex:
            self.after(0, lambda: self._finish_error(
                f"Missing library: {ex}\nRun: pip install scikit-learn scipy pandas openpyxl"))
            return

        lb   = np.array([b[0] for b in bounds_list])
        ub   = np.array([b[1] for b in bounds_list])
        norm = lambda X: (X - lb) / (ub - lb)
        X_norm = norm(X_cfd)

        self.after(0, lambda: self._log("● Training GP surrogate models…", "dim"))
        models = []
        for i in range(5):
            gp = GaussianProcessRegressor(Matern(nu=2.5), n_restarts_optimizer=10,
                                          normalize_y=True, random_state=42)
            gp.fit(X_norm, Y_cfd[:, i])
            models.append(gp)
            self.after(0, lambda i=i: self._log(
                f"  ✓  Model {i+1}/5 trained  ({OUTPUT_KEYS[i]})", "ok"))

        self.after(0, lambda: self._log("\n● Leave-One-Out Validation", "dim"))
        n = len(X_norm)
        for j in range(5):
            preds = []
            for i in range(n):
                idx = [k for k in range(n) if k != i]
                gp_l = GaussianProcessRegressor(Matern(nu=2.5), n_restarts_optimizer=5,
                                                normalize_y=True, random_state=42)
                gp_l.fit(X_norm[idx], Y_cfd[idx, j])
                preds.append(gp_l.predict(X_norm[[i]])[0])
            r2  = r2_score(Y_cfd[:, j], preds)
            tag = "ok" if r2 > 0.85 else ("warn" if r2 > 0.70 else "err")
            sym = "✓" if r2 > 0.85 else ("⚠" if r2 > 0.70 else "✗")
            lbl = "Good" if r2 > 0.85 else ("Marginal" if r2 > 0.70 else "Needs more CFD runs")
            self.after(0, lambda nm=OUTPUT_KEYS[j], r=r2, s=sym, l=lbl, t=tag:
                       self._log(f"  {s}  {nm:<18}  R² = {r:.3f}   {l}", t))

        self.after(0, lambda: self._log(f"\n● Generating {n_samples:,} LHS candidates…", "dim"))
        sampler = qmc.LatinHypercube(d=5, seed=42)
        X_cand  = qmc.scale(sampler.random(n=n_samples), lb, ub)
        X_cn    = norm(X_cand)

        self.after(0, lambda: self._log("● Predicting outputs for all candidates…", "dim"))
        Y_pred = np.column_stack([models[i].predict(X_cn) for i in range(5)])
        Y_std  = np.column_stack([models[i].predict(X_cn, return_std=True)[1] for i in range(5)])

        scaler = MinMaxScaler()
        def ns(col, inv=False):
            v = scaler.fit_transform(Y_pred[:, col:col+1]).flatten()
            return 1 - v if inv else v

        score = (weights["avg_velocity"]    * ns(1) +
                 weights["flow_uniformity"] * ns(2) +
                 weights["pressure_drop"]   * ns(3, inv=True) +
                 weights["max_velocity"]    * ns(0) +
                 weights["volume_flow"]     * ns(4))

        best      = np.argmax(score)
        best_p    = X_cand[best]
        best_pred = Y_pred[best]
        best_std  = Y_std[best]
        best_sc   = score[best]
        high_unc  = any(best_std[i] > 0.2 * abs(best_pred[i]) for i in range(5))

        # Store results for export
        self.last_optimization_results = {
            'best_params': best_p,
            'best_pred': best_pred,
            'best_std': best_std,
            'score': best_sc,
            'reliability_msg': "High uncertainty — recommend verification" if high_unc else "Low uncertainty — reliable",
            'reliability_status': ["✓ Reliable" if best_std[i] < 0.2 * abs(best_pred[i]) else "⚠ Verify" for i in range(5)]
        }

        self.after(0, lambda: self._log("\n── OPTIMAL DESIGN ──────────────────────────", "head"))
        for i, nm in enumerate(PARAM_KEYS):
            self.after(0, lambda n=nm, v=best_p[i]:
                       self._log(f"  {n:<18}  {v:.2f}"))

        self.after(0, lambda: self._log("\n── PREDICTED PERFORMANCE ───────────────────", "head"))
        for i, nm in enumerate(OUTPUT_KEYS):
            self.after(0, lambda n=nm, v=best_pred[i], s=best_std[i]:
                       self._log(f"  {n:<18}  {v:.4f}  ±  {s:.4f}"))

        rel_msg = ("⚠  High uncertainty — recommend 1 verification CFD run"
                   if high_unc else "✓  Low uncertainty — prediction is reliable")
        rel_tag = "warn" if high_unc else "ok"
        self.after(0, lambda m=rel_msg, t=rel_tag: self._log(f"\n  {m}", t))
        self.after(0, lambda: self._log(f"\n  Composite Score:  {best_sc:.4f}\n", "head"))

        self.after(0, lambda: self._update_results(
            best_p, best_pred, best_std, best_sc, high_unc))

        self.after(0, lambda: self._log("● Generating plots…", "dim"))
        try:
            self._make_plots(X_cfd, Y_cfd, score, best,
                             best_p, best_pred, X_norm, models, norm)
            self.after(0, lambda: self._log("  ✓  Plots displayed", "ok"))
        except Exception as ex:
            self.after(0, lambda: self._log(f"  ⚠  Plot error: {ex}", "warn"))

        self.after(0, self._finish_ok)

    # ──────────────────────────────────────────────────────
    def _update_results(self, best_p, best_pred, best_std, best_sc, high_unc):
        for i, lbl in enumerate(self.result_val_lbls):
            lbl.configure(text=f"{best_p[i]:.1f}")
        self.score_lbl.configure(text=f"{best_sc:.4f}")
        if high_unc:
            self.reliability_lbl.configure(text="⚠  High Uncertainty", fg=WARN)
            self.reliability_detail.configure(
                text="High variance detected. Recommend one additional verification CFD run near the optimum.")
        else:
            self.reliability_lbl.configure(text="✓  Low Uncertainty", fg=GREEN)
            self.reliability_detail.configure(
                text="All outputs have low standard deviation. The surrogate is reliable.")
        for i, (pl, sl, stl) in enumerate(self.perf_rows):
            pl.configure(text=f"{best_pred[i]:.4f}")
            sl.configure(text=f"± {best_std[i]:.4f}")
            rel = best_std[i] / max(abs(best_pred[i]), 1e-9)
            if rel < 0.05:
                stl.configure(text="✓ Reliable", fg=GREEN)
            elif rel < 0.20:
                stl.configure(text="~ Marginal",  fg=WARN)
            else:
                stl.configure(text="⚠ Verify",    fg=RED)

    # ──────────────────────────────────────────────────────
    def _make_plots(self, X_cfd, Y_cfd, scores, best_idx,
                    best_params, best_preds, X_norm, models, norm_fn):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import seaborn as sns
        import pandas as pd

        mpl.rcParams.update({
            "figure.facecolor": WHITE, "axes.facecolor": WHITE,
            "axes.edgecolor": BORDER_DARK, "axes.labelcolor": TEXT_DIM,
            "xtick.color": TEXT_DIM, "ytick.color": TEXT_DIM,
            "text.color": TEXT, "grid.color": BORDER,
            "grid.linestyle": "--", "grid.linewidth": 0.5,
            "font.family": "DejaVu Sans",
        })

        COLORS = ["#1A56DB", "#0E9F6E", "#D97706", "#7E3AF2", "#E02424"]
        designs = [f"D{i+1}" for i in range(len(X_cfd))]

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.patch.set_facecolor("#F5F6FA")
        fig.suptitle("Surrogate Model — Analysis Dashboard",
                     fontsize=14, fontweight="bold", color=TEXT, y=0.99)

        for ax in axes.flat:
            ax.set_facecolor(WHITE)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, axis="y", alpha=0.4)

        x = np.arange(len(X_cfd))

        ax = axes[0, 0]
        bars = ax.bar(x, Y_cfd[:, 1], color=COLORS[:len(x)], width=0.55, zorder=3)
        ax.set_title("Avg Velocity per Design", fontsize=11, color=TEXT, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(designs)
        ax.set_ylabel("Avg velocity (m/s)", fontsize=9)
        for bar, val in zip(bars, Y_cfd[:, 1]):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT_DIM)

        ax = axes[0, 1]
        bars = ax.bar(x, Y_cfd[:, 3], color=COLORS[:len(x)], width=0.55, zorder=3)
        ax.set_title("Pressure Drop (lower = better)", fontsize=11, color=TEXT, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(designs)
        ax.set_ylabel("Pressure drop (Pa)", fontsize=9)
        for bar, val in zip(bars, Y_cfd[:, 3]):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT_DIM)

        ax = axes[0, 2]
        ax.hist(scores, bins=80, color=ACCENT, alpha=0.7, edgecolor="none", zorder=3)
        ax.axvline(scores[best_idx], color=RED, linewidth=1.5,
                   label=f"Best: {scores[best_idx]:.3f}", zorder=4)
        ax.set_title("Candidate Score Distribution", fontsize=11, color=TEXT, pad=8)
        ax.set_xlabel("Score", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=9)

        ax = axes[1, 0]
        axes[1, 0].grid(False)
        df_corr = pd.DataFrame(np.hstack([X_cfd, Y_cfd]),
                               columns=PARAM_KEYS + OUTPUT_KEYS)
        corr = df_corr.corr().loc[PARAM_KEYS, OUTPUT_KEYS]
        sns.heatmap(corr, ax=ax, annot=True, fmt=".2f",
                    cmap=sns.diverging_palette(230, 20, as_cmap=True),
                    center=0, linewidths=0.4, annot_kws={"size": 9},
                    linecolor=BORDER)
        ax.set_title("Sensitivity — Inputs vs Outputs", fontsize=11, color=TEXT, pad=8)
        ax.set_xticklabels(OUTPUT_KEYS, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(PARAM_KEYS,  rotation=0,  fontsize=8)

        ax = axes[1, 1]
        axes[1, 1].grid(False)
        AA, PP = np.meshgrid(np.linspace(best_params[0]-10, best_params[0]+10, 60), 
                            np.linspace(best_params[1]-20, best_params[1]+20, 60))
        grid = np.column_stack([
            AA.ravel(), PP.ravel(),
            np.full(3600, best_params[2]),
            np.full(3600, best_params[3]),
            np.full(3600, best_params[4]),
        ])
        vel_surf = models[1].predict(norm_fn(grid)).reshape(60, 60)
        c = ax.contourf(AA, PP, vel_surf, levels=20, cmap="Blues")
        plt.colorbar(c, ax=ax)
        ax.set_xlabel("Angle (°)", fontsize=9)
        ax.set_ylabel("Protrusion (mm)", fontsize=9)
        ax.set_title("Predicted Avg Velocity Surface", fontsize=11, color=TEXT, pad=8)
        ax.plot(best_params[0], best_params[1], "r*", markersize=14, label="Optimum", zorder=5)
        ax.scatter(X_cfd[:, 0], X_cfd[:, 1], c="white", edgecolors=TEXT_DIM,
                   s=50, zorder=5, label="CFD pts")
        ax.legend(fontsize=9)

        ax = axes[1, 2]
        d1_pred = np.array([models[i].predict(X_norm[[0]])[0] for i in range(5)])
        d1_base = d1_pred.copy()
        d1_base[d1_base == 0] = 1e-9
        rel_opt = best_preds / d1_base
        x2 = np.arange(5)
        w = 0.35
        ax.bar(x2 - w/2, np.ones(5), w, label="Baseline", color=BORDER_DARK, zorder=3)
        ax.bar(x2 + w/2, rel_opt,    w, label="Optimum",     color=GREEN,       zorder=3)
        ax.axhline(1.0, color=TEXT_LIGHT, linewidth=0.8, linestyle="--")
        ax.set_xticks(x2)
        ax.set_xticklabels(OUTPUT_KEYS, rotation=20, ha="right", fontsize=8)
        ax.set_title("Optimum vs Baseline (relative)", fontsize=11, color=TEXT, pad=8)
        ax.set_ylabel("Ratio (1.0 = baseline)", fontsize=9)
        ax.legend(fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    def _finish_ok(self):
        self.running = False
        self.progress.stop()
        self.run_btn.configure(state="normal",
                               text="  ▶   RUN OPTIMIZATION  ", bg=ACCENT)
        self._set_status("Optimization complete", GREEN, GREEN)
        self.export_btn.config(state="normal")

    def _finish_error(self, msg):
        self.running = False
        self.progress.stop()
        self.run_btn.configure(state="normal",
                               text="  ▶   RUN OPTIMIZATION  ", bg=ACCENT)
        self._set_status(f"Error — {msg[:60]}", RED, RED)
        self._log(f"\n[ERROR] {msg}", "err")
        messagebox.showerror("Error", msg)


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()