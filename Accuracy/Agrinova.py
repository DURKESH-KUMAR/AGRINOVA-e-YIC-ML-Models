"""
=============================================================================
AGRINOVA — Inference UI  (Tkinter + Matplotlib)
=============================================================================
Tabs
  1. Dashboard   — Load pkl files, run inference, see best-model badge
  2. Metrics     — Full metric table for every session × model
  3. Confusion   — Interactive confusion-matrix viewer (select session/model)
  4. Scatter     — Actual vs Predicted scatter plots
  5. Ranking     — Ranked bar chart + heatmap across all pkl files
  6. Log         — Console output

Run:
  python agrinova_inference_ui.py

Requirements:
  pip install scikit-learn matplotlib numpy pandas seaborn
=============================================================================
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, pickle, threading, warnings, datetime
warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, confusion_matrix, accuracy_score,
)

# ── tkinter ───────────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PKLS   = ["agrinova_model_1.pkl",
                  "agrinova_model_2.pkl",
                  "agrinova_model_3.pkl"]
FEATURE_COLS   = ["DayIndex","Lag1","Lag2","Lag3",
                  "Roll3_Mean","Roll3_Std","dG","dT","dH"]
WARNING_PPM    = 220
CRITICAL_PPM   = 260
CLASS_LABELS   = ["Safe", "Warning", "Critical"]
TOLERANCE_PCT  = 5.0

# Colour palette
C = {
    "bg":        "#0F1117",
    "panel":     "#1A1D27",
    "card":      "#22263A",
    "accent":    "#4F8EF7",
    "rf":        "#4F8EF7",
    "gbm":       "#FF9800",
    "safe":      "#4CAF50",
    "warn":      "#FF9800",
    "crit":      "#F44336",
    "text":      "#E8EDF5",
    "subtext":   "#8892A4",
    "border":    "#2E3450",
    "gold":      "#FFD700",
    "silver":    "#C0C0C0",
    "bronze":    "#CD7F32",
}

# ─────────────────────────────────────────────────────────────────────────────
# BACKEND — pure functions (shared with CLI script)
# ─────────────────────────────────────────────────────────────────────────────

def load_pkl(path):
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)

def ppm_to_class(arr):
    arr = np.asarray(arr, dtype=float)
    c = np.zeros(len(arr), dtype=int)
    c[arr >= WARNING_PPM]  = 1
    c[arr >= CRITICAL_PPM] = 2
    return c

def regression_accuracy(y_true, y_pred, tol=TOLERANCE_PCT):
    tol_v = np.maximum(np.abs(y_true) * (tol / 100.0), 1.0)
    return float((np.abs(y_true - y_pred) <= tol_v).mean()) * 100.0

def mape_score(y_true, y_pred):
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100)

def compute_metrics(y_true, y_pred, label=""):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2    = r2_score(y_true, y_pred)
    ra    = regression_accuracy(y_true, y_pred)
    mp    = mape_score(y_true, y_pred)
    ct    = ppm_to_class(y_true)
    cp    = ppm_to_class(y_pred)
    ca    = accuracy_score(ct, cp) * 100.0
    cm    = confusion_matrix(ct, cp, labels=[0,1,2])
    return dict(label=label, mae=mae, rmse=rmse, r2=r2,
                reg_acc=ra, mape=mp, cls_acc=ca, cm=cm,
                c_true=ct, c_pred=cp, y_true=y_true, y_pred=y_pred)

def run_inference(session_name, session, pkl_file):
    if not session.get("ml_available"):
        return None
    X = np.array(session["X"])
    y = np.array(session["y"])
    if len(X) == 0:
        return None
    res = dict(session_name=session_name, pkl_file=pkl_file,
               n_samples=len(y), rf=None, gbm=None,
               rf_loo=None, gbm_loo=None)
    for key, model_key, loo_key, label in [
        ("rf",  "rf_model",  "rf_metrics",  "RF"),
        ("gbm", "gbm_model", "gbm_metrics", "GBM"),
    ]:
        model = session.get(model_key)
        if model is None:
            continue
        res[key] = compute_metrics(y, model.predict(X), f"{label} (Train)")
        m_store  = session.get(loo_key) or {}
        if "loo_preds" in m_store:
            res[f"{key}_loo"] = compute_metrics(
                y, np.array(m_store["loo_preds"]), f"{label} (LOO-CV)")
    return res

def score_entry(m):
    if m is None: return -999.0
    r2  = max(m.get("r2",0), 0)
    ca  = m.get("cls_acc",0) / 100.0
    ra  = m.get("reg_acc",0) / 100.0
    pen = 1.0 / (1.0 + m.get("mae", 999))
    return 0.40*r2 + 0.30*ca + 0.20*ra + 0.10*pen

def build_ranking(all_results):
    rows = []
    for res in all_results:
        for mk, lk, ml in [("rf","rf_loo","Random Forest"),
                            ("gbm","gbm_loo","Gradient Boost")]:
            m = res.get(lk) or res.get(mk)
            if m is None: continue
            rows.append(dict(
                pkl_file  = res["pkl_file"],
                session   = res["session_name"],
                model     = ml,
                eval_type = m["label"],
                MAE       = round(m["mae"],3),
                RMSE      = round(m["rmse"],3),
                R2        = round(m["r2"],4),
                RegAcc    = round(m["reg_acc"],2),
                ClsAcc    = round(m["cls_acc"],2),
                MAPE      = round(m["mape"],2),
                Score     = round(score_entry(m),4),
            ))
    return pd.DataFrame(rows).sort_values("Score",ascending=False).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# STYLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def apply_dark_style():
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(".",
        background=C["bg"], foreground=C["text"],
        fieldbackground=C["panel"], troughcolor=C["panel"],
        selectbackground=C["accent"], selectforeground="white",
        font=("Segoe UI", 10))
    style.configure("TNotebook",
        background=C["bg"], borderwidth=0, tabmargins=[0,0,0,0])
    style.configure("TNotebook.Tab",
        background=C["panel"], foreground=C["subtext"],
        padding=[18, 8], font=("Segoe UI", 10, "bold"), borderwidth=0)
    style.map("TNotebook.Tab",
        background=[("selected", C["card"]), ("active", C["card"])],
        foreground=[("selected", C["text"]), ("active", C["text"])])
    style.configure("TFrame",       background=C["bg"])
    style.configure("Card.TFrame",  background=C["card"],  relief="flat")
    style.configure("Panel.TFrame", background=C["panel"], relief="flat")
    style.configure("TLabel",       background=C["bg"],   foreground=C["text"])
    style.configure("Card.TLabel",  background=C["card"], foreground=C["text"])
    style.configure("Sub.TLabel",   background=C["bg"],   foreground=C["subtext"],
        font=("Segoe UI", 9))
    style.configure("Header.TLabel",
        background=C["bg"], foreground=C["text"],
        font=("Segoe UI", 14, "bold"))
    style.configure("Title.TLabel",
        background=C["card"], foreground=C["text"],
        font=("Segoe UI", 11, "bold"))
    style.configure("Accent.TButton",
        background=C["accent"], foreground="white",
        font=("Segoe UI", 10, "bold"), padding=[12, 6], borderwidth=0)
    style.map("Accent.TButton",
        background=[("active", "#3A78E8"), ("pressed", "#2A60D0")])
    style.configure("TButton",
        background=C["card"], foreground=C["text"],
        font=("Segoe UI", 9), padding=[10, 5], borderwidth=0)
    style.map("TButton",
        background=[("active", C["border"]), ("pressed", C["panel"])])
    style.configure("Treeview",
        background=C["panel"], foreground=C["text"],
        fieldbackground=C["panel"], rowheight=28,
        font=("Segoe UI", 9))
    style.configure("Treeview.Heading",
        background=C["card"], foreground=C["accent"],
        font=("Segoe UI", 9, "bold"), relief="flat")
    style.map("Treeview",
        background=[("selected", C["accent"])],
        foreground=[("selected", "white")])
    style.configure("TCombobox",
        background=C["card"], foreground=C["text"],
        selectbackground=C["accent"], fieldbackground=C["card"])
    style.configure("TScrollbar",
        background=C["card"], troughcolor=C["panel"],
        arrowcolor=C["subtext"])

def mpl_dark(fig):
    """Apply dark theme to a matplotlib figure."""
    fig.patch.set_facecolor(C["bg"])
    for ax in fig.get_axes():
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["subtext"])
        ax.xaxis.label.set_color(C["subtext"])
        ax.yaxis.label.set_color(C["subtext"])
        ax.title.set_color(C["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])

def embed_figure(fig, parent):
    """Embed matplotlib figure in a tk widget, return canvas."""
    mpl_dark(fig)
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().configure(bg=C["bg"], highlightthickness=0)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return canvas

def clear_frame(frame):
    for w in frame.winfo_children():
        w.destroy()

def metric_card(parent, title, value, color=None, width=14):
    color = color or C["accent"]
    f = tk.Frame(parent, bg=C["card"], padx=12, pady=10,
                 relief="flat", bd=0, highlightthickness=1,
                 highlightbackground=C["border"])
    f.pack(side="left", padx=6, pady=4, fill="y")
    tk.Label(f, text=title, bg=C["card"], fg=C["subtext"],
             font=("Segoe UI", 8)).pack(anchor="w")
    tk.Label(f, text=value, bg=C["card"], fg=color,
             font=("Segoe UI", 15, "bold")).pack(anchor="w")
    return f

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

class AgrinovaInferenceUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Agrinova — Inference & Model Evaluation")
        self.geometry("1300x820")
        self.minsize(1100, 700)
        self.configure(bg=C["bg"])
        apply_dark_style()

        # State
        self.pkl_vars     = []          # StringVar list for pkl entries
        self.all_results  = []          # list of inference result dicts
        self.df_rank      = pd.DataFrame()
        self._canvases    = {}          # tab_name → FigureCanvasTkAgg

        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # TOP-LEVEL LAYOUT
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header bar
        hdr = tk.Frame(self, bg=C["panel"], pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🌱  AGRINOVA", bg=C["panel"], fg=C["accent"],
                 font=("Segoe UI", 17, "bold")).pack(side="left", padx=20)
        tk.Label(hdr, text="Onion Spoilage · Inference & Model Evaluation",
                 bg=C["panel"], fg=C["subtext"],
                 font=("Segoe UI", 11)).pack(side="left")
        self._status_var = tk.StringVar(value="Ready — load .pkl files to begin")
        tk.Label(hdr, textvariable=self._status_var,
                 bg=C["panel"], fg=C["subtext"],
                 font=("Segoe UI", 9)).pack(side="right", padx=20)

        # Notebook
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=0, pady=0)

        self.tab_dash    = ttk.Frame(self.nb, style="TFrame")
        self.tab_metrics = ttk.Frame(self.nb, style="TFrame")
        self.tab_cm      = ttk.Frame(self.nb, style="TFrame")
        self.tab_scatter = ttk.Frame(self.nb, style="TFrame")
        self.tab_rank    = ttk.Frame(self.nb, style="TFrame")
        self.tab_log     = ttk.Frame(self.nb, style="TFrame")

        for tab, label in [
            (self.tab_dash,    "  📊  Dashboard  "),
            (self.tab_metrics, "  📋  Metrics    "),
            (self.tab_cm,      "  🔲  Confusion  "),
            (self.tab_scatter, "  🔵  Scatter    "),
            (self.tab_rank,    "  🏆  Ranking    "),
            (self.tab_log,     "  📝  Log        "),
        ]:
            self.nb.add(tab, text=label)

        self._build_dashboard()
        self._build_metrics_tab()
        self._build_cm_tab()
        self._build_scatter_tab()
        self._build_rank_tab()
        self._build_log_tab()

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 — DASHBOARD
    # ─────────────────────────────────────────────────────────────────────────

    def _build_dashboard(self):
        f = self.tab_dash
        f.columnconfigure(0, weight=0)
        f.columnconfigure(1, weight=1)
        f.rowconfigure(1, weight=1)

        # ── Left panel: file loader ───────────────────────────────────────────
        left = tk.Frame(f, bg=C["card"], width=300, padx=14, pady=14,
                        highlightthickness=1,
                        highlightbackground=C["border"])
        left.pack(side="left", fill="y", padx=(12,6), pady=12)
        left.pack_propagate(False)

        tk.Label(left, text="Model Files (.pkl)",
                 bg=C["card"], fg=C["text"],
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0,8))

        self.pkl_frame = tk.Frame(left, bg=C["card"])
        self.pkl_frame.pack(fill="x")

        for path in DEFAULT_PKLS:
            self._add_pkl_row(path)

        tk.Button(left, text="＋  Add PKL File",
                  bg=C["border"], fg=C["accent"],
                  font=("Segoe UI", 9), relief="flat",
                  cursor="hand2",
                  command=self._add_pkl_browse).pack(fill="x", pady=(8,4))

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        self.run_btn = tk.Button(left, text="▶  Run Inference",
                                  bg=C["accent"], fg="white",
                                  font=("Segoe UI", 11, "bold"),
                                  relief="flat", cursor="hand2",
                                  activebackground="#3A78E8",
                                  pady=10,
                                  command=self._run_inference_threaded)
        self.run_btn.pack(fill="x", pady=4)

        self.progress = ttk.Progressbar(left, mode="indeterminate",
                                         length=260)
        self.progress.pack(fill="x", pady=6)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        tk.Label(left, text="Thresholds",
                 bg=C["card"], fg=C["subtext"],
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        for label, val, col in [
            (f"⚠  Warning  ≥ {WARNING_PPM} ppm",  str(WARNING_PPM),  C["warn"]),
            (f"🔴 Critical ≥ {CRITICAL_PPM} ppm", str(CRITICAL_PPM), C["crit"]),
        ]:
            row = tk.Frame(left, bg=C["card"])
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, bg=C["card"], fg=col,
                     font=("Segoe UI", 9)).pack(side="left")

        # ── Right panel: best model badge + summary cards ─────────────────────
        right = tk.Frame(f, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True,
                   padx=(6,12), pady=12)

        tk.Label(right, text="Inference Dashboard",
                 bg=C["bg"], fg=C["text"],
                 font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0,6))

        # Badge area
        self.badge_frame = tk.Frame(right, bg=C["bg"])
        self.badge_frame.pack(fill="x", pady=(0, 8))
        self._draw_placeholder_badge()

        # Summary cards
        self.cards_frame = tk.Frame(right, bg=C["bg"])
        self.cards_frame.pack(fill="x")

        # Results tree
        tree_lbl = tk.Label(right, text="Session Results",
                             bg=C["bg"], fg=C["subtext"],
                             font=("Segoe UI", 9, "bold"))
        tree_lbl.pack(anchor="w", pady=(10,2))

        tree_frame = tk.Frame(right, bg=C["bg"])
        tree_frame.pack(fill="both", expand=True)

        cols = ("pkl","session","model","eval","MAE","RMSE","R²",
                "RegAcc%","ClsAcc%","MAPE%","Score")
        self.dash_tree = ttk.Treeview(tree_frame, columns=cols,
                                       show="headings", selectmode="browse")
        widths = [140,140,120,110,70,70,70,80,80,70,70]
        for col, w in zip(cols, widths):
            self.dash_tree.heading(col, text=col)
            self.dash_tree.column(col, width=w, anchor="center")
        self.dash_tree.column("pkl",     anchor="w")
        self.dash_tree.column("session", anchor="w")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                             command=self.dash_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal",
                             command=self.dash_tree.xview)
        self.dash_tree.configure(yscrollcommand=vsb.set,
                                  xscrollcommand=hsb.set)
        self.dash_tree.pack(fill="both", expand=True, side="left")
        vsb.pack(side="right", fill="y")

        # row colour tags
        self.dash_tree.tag_configure("best",   background="#1A3A1A",
                                               foreground=C["safe"])
        self.dash_tree.tag_configure("rf",     background="#111D33")
        self.dash_tree.tag_configure("gbm",    background="#1F1708")
        self.dash_tree.tag_configure("loo",    foreground=C["subtext"])

    def _draw_placeholder_badge(self):
        clear_frame(self.badge_frame)
        card = tk.Frame(self.badge_frame, bg=C["card"], padx=20, pady=14,
                        highlightthickness=1,
                        highlightbackground=C["border"])
        card.pack(side="left", padx=(0,8))
        tk.Label(card, text="🏆  Best Model",
                 bg=C["card"], fg=C["gold"],
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(card, text="Run inference to see results",
                 bg=C["card"], fg=C["subtext"],
                 font=("Segoe UI", 12)).pack(anchor="w", pady=4)

    def _draw_best_badge(self, best_row):
        clear_frame(self.badge_frame)
        card = tk.Frame(self.badge_frame, bg=C["card"], padx=20, pady=14,
                        highlightthickness=2,
                        highlightbackground=C["gold"])
        card.pack(side="left", padx=(0,8))
        tk.Label(card, text="🏆  BEST MODEL",
                 bg=C["card"], fg=C["gold"],
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(card, text=f"{best_row['model']}",
                 bg=C["card"], fg=C["text"],
                 font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(card, text=f"{best_row['pkl_file']}  ·  {best_row['session']}",
                 bg=C["card"], fg=C["subtext"],
                 font=("Segoe UI", 9)).pack(anchor="w")

        # mini metric strip
        strip = tk.Frame(card, bg=C["card"])
        strip.pack(anchor="w", pady=(8,0))
        for title, val, col in [
            ("Score",    f"{best_row['Score']:.4f}",  C["gold"]),
            ("R²",       f"{best_row['R2']:.4f}",     C["accent"]),
            ("ClassAcc", f"{best_row['ClsAcc']:.1f}%",C["safe"]),
            ("MAE",      f"{best_row['MAE']:.2f} ppm",C["warn"]),
        ]:
            b = tk.Frame(strip, bg=C["panel"], padx=10, pady=6,
                         highlightthickness=1,
                         highlightbackground=C["border"])
            b.pack(side="left", padx=4)
            tk.Label(b, text=title, bg=C["panel"], fg=C["subtext"],
                     font=("Segoe UI", 7)).pack()
            tk.Label(b, text=val,   bg=C["panel"], fg=col,
                     font=("Segoe UI", 11, "bold")).pack()

        # Recommendation text
        model_name = best_row["model"]
        reason = (
            "Random Forest excels with small tabular datasets and noisy "
            "sensor data — it builds many uncorrelated trees, reducing "
            "variance and avoiding overfit on limited daily samples."
        ) if "Forest" in model_name else (
            "Gradient Boosting captures smooth exponential trends by "
            "sequentially minimising residuals — ideal when gas accumulation "
            "follows a consistent growth curve across days."
        )
        rec = tk.Frame(self.badge_frame, bg=C["card"], padx=16, pady=14,
                       highlightthickness=1,
                       highlightbackground=C["border"])
        rec.pack(side="left", fill="both", expand=True)
        tk.Label(rec, text="Why this model?",
                 bg=C["card"], fg=C["accent"],
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Label(rec, text=reason, bg=C["card"], fg=C["text"],
                 font=("Segoe UI", 9), wraplength=420,
                 justify="left").pack(anchor="w", pady=4)
        score_info = (
            "Composite Score = 0.40×R² + 0.30×ClassAcc "
            "+ 0.20×RegAcc + 0.10×(1/(1+MAE))\n"
            "Spoilage classes — Safe: <220 ppm  |  "
            "Warning: 220–259 ppm  |  Critical: ≥260 ppm"
        )
        tk.Label(rec, text=score_info,
                 bg=C["card"], fg=C["subtext"],
                 font=("Segoe UI", 8), justify="left").pack(anchor="w")

    def _add_pkl_row(self, path=""):
        row = tk.Frame(self.pkl_frame, bg=C["card"])
        row.pack(fill="x", pady=2)
        var = tk.StringVar(value=path)
        self.pkl_vars.append(var)
        entry = tk.Entry(row, textvariable=var, bg=C["panel"],
                         fg=C["text"], font=("Segoe UI", 9),
                         relief="flat", bd=4,
                         insertbackground=C["text"], width=22)
        entry.pack(side="left", fill="x", expand=True)

        def browse(v=var):
            p = filedialog.askopenfilename(
                filetypes=[("Pickle files","*.pkl"),("All","*.*")])
            if p:
                v.set(p)

        tk.Button(row, text="…", bg=C["border"], fg=C["text"],
                  font=("Segoe UI", 9), relief="flat", cursor="hand2",
                  command=browse).pack(side="left", padx=(3,0))

        def remove(r=row, v=var):
            self.pkl_vars.remove(v)
            r.destroy()

        tk.Button(row, text="✕", bg=C["border"], fg=C["crit"],
                  font=("Segoe UI", 9), relief="flat", cursor="hand2",
                  command=remove).pack(side="left", padx=2)

    def _add_pkl_browse(self):
        p = filedialog.askopenfilename(
            filetypes=[("Pickle files","*.pkl"),("All","*.*")])
        self._add_pkl_row(p if p else "")

    # ─────────────────────────────────────────────────────────────────────────
    # INFERENCE RUNNER
    # ─────────────────────────────────────────────────────────────────────────

    def _run_inference_threaded(self):
        self.run_btn.configure(state="disabled", text="Running…")
        self.progress.start(12)
        t = threading.Thread(target=self._run_inference, daemon=True)
        t.start()

    def _run_inference(self):
        self.all_results.clear()
        paths = [v.get().strip() for v in self.pkl_vars if v.get().strip()]

        self._log(f"\n{'='*60}")
        self._log(f"  AGRINOVA INFERENCE  —  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
        self._log(f"{'='*60}")

        for pkl_path in paths:
            self._log(f"\n► {pkl_path}")
            store = load_pkl(pkl_path)
            if not store:
                self._log(f"  [SKIP] File missing or empty.")
                continue
            self._log(f"  Sessions: {list(store.keys())}")
            for sname, session in store.items():
                self._log(f"\n  ── {sname}")
                res = run_inference(sname, session, os.path.basename(pkl_path))
                if res is None:
                    self._log("    [SKIP] ml_available=False or no data")
                    continue
                for key in ("rf","rf_loo","gbm","gbm_loo"):
                    m = res.get(key)
                    if m:
                        self._log(f"    {m['label']:20s} "
                                  f"MAE={m['mae']:.3f}  R²={m['r2']:.4f}  "
                                  f"ClsAcc={m['cls_acc']:.1f}%")
                self.all_results.append(res)

        self.df_rank = build_ranking(self.all_results)
        self._log(f"\n{'='*60}")
        self._log(f"  Total valid sessions: {len(self.all_results)}")
        if not self.df_rank.empty:
            best = self.df_rank.iloc[0]
            self._log(f"\n  ★ BEST MODEL: {best['pkl_file']} / "
                      f"{best['session']} / {best['model']}")
            self._log(f"    Score={best['Score']:.4f}  R²={best['R2']:.4f}  "
                      f"ClsAcc={best['ClsAcc']:.1f}%  MAE={best['MAE']:.2f}")
        self._log(f"{'='*60}")

        self.after(0, self._post_inference)

    def _post_inference(self):
        self.progress.stop()
        self.run_btn.configure(state="normal", text="▶  Run Inference")

        if self.df_rank.empty:
            self._status("No valid sessions found — check pkl files.")
            messagebox.showwarning("No Results",
                "No valid sessions found.\n"
                "Make sure the pkl files exist and contain ml_available=True sessions.")
            return

        self._status(f"Done — {len(self.all_results)} session(s) evaluated.")
        self._populate_dash_tree()
        self._draw_best_badge(self.df_rank.iloc[0])
        self._refresh_metric_combos()
        self._refresh_cm_tab()
        self._refresh_scatter_tab()
        self._refresh_rank_tab()
        self.nb.select(0)

    def _populate_dash_tree(self):
        for row in self.dash_tree.get_children():
            self.dash_tree.delete(row)

        best_key = None
        if not self.df_rank.empty:
            br = self.df_rank.iloc[0]
            best_key = (br["pkl_file"], br["session"], br["model"])

        for _, row in self.df_rank.iterrows():
            key = (row["pkl_file"], row["session"], row["model"])
            tag = "best" if key == best_key else \
                  ("rf" if "Forest" in row["model"] else "gbm")
            self.dash_tree.insert("", "end", values=(
                row["pkl_file"], row["session"], row["model"],
                row["eval_type"],
                f"{row['MAE']:.3f}", f"{row['RMSE']:.3f}",
                f"{row['R2']:.4f}",  f"{row['RegAcc']:.1f}",
                f"{row['ClsAcc']:.1f}", f"{row['MAPE']:.1f}",
                f"{row['Score']:.4f}",
            ), tags=(tag,))

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 — METRICS
    # ─────────────────────────────────────────────────────────────────────────

    def _build_metrics_tab(self):
        f = self.tab_metrics
        tk.Label(f, text="Full Metrics Table",
                 bg=C["bg"], fg=C["text"],
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=16, pady=(12,4))
        tk.Label(f, text=(
            "All models × sessions × eval types.  "
            f"RegAcc = within ±{TOLERANCE_PCT}% of actual.  "
            "Score = 0.40×R² + 0.30×ClsAcc + 0.20×RegAcc + 0.10×(1/(1+MAE))"),
            bg=C["bg"], fg=C["subtext"],
            font=("Segoe UI", 9)).pack(anchor="w", padx=16)

        cols = ("pkl","session","model","eval","MAE","RMSE","R²",
                "RegAcc%","ClsAcc%","MAPE%","Score")
        frm = tk.Frame(f, bg=C["bg"])
        frm.pack(fill="both", expand=True, padx=16, pady=8)
        self.metrics_tree = ttk.Treeview(frm, columns=cols,
                                          show="headings", selectmode="browse")
        widths = [140,140,120,110,70,70,70,80,80,70,80]
        for col, w in zip(cols, widths):
            self.metrics_tree.heading(col, text=col,
                command=lambda c=col: self._sort_metrics(c))
            self.metrics_tree.column(col, width=w, anchor="center")
        self.metrics_tree.column("pkl",     anchor="w")
        self.metrics_tree.column("session", anchor="w")
        vsb = ttk.Scrollbar(frm, orient="vertical",
                             command=self.metrics_tree.yview)
        hsb = ttk.Scrollbar(frm, orient="horizontal",
                             command=self.metrics_tree.xview)
        self.metrics_tree.configure(yscrollcommand=vsb.set,
                                     xscrollcommand=hsb.set)
        self.metrics_tree.pack(fill="both", expand=True, side="left")
        vsb.pack(side="right", fill="y")
        self.metrics_tree.tag_configure("best",background="#1A3A1A",
                                                foreground=C["safe"])
        self.metrics_tree.tag_configure("odd", background=C["panel"])
        self.metrics_tree.tag_configure("even",background=C["card"])

    def _refresh_metric_combos(self):
        if self.df_rank.empty:
            return
        best_key = (self.df_rank.iloc[0]["pkl_file"],
                    self.df_rank.iloc[0]["session"],
                    self.df_rank.iloc[0]["model"])
        for row in self.metrics_tree.get_children():
            self.metrics_tree.delete(row)
        for i, (_, row) in enumerate(self.df_rank.iterrows()):
            key = (row["pkl_file"], row["session"], row["model"])
            tag = "best" if key == best_key else \
                  ("odd" if i % 2 else "even")
            self.metrics_tree.insert("", "end", values=(
                row["pkl_file"], row["session"], row["model"],
                row["eval_type"],
                f"{row['MAE']:.3f}", f"{row['RMSE']:.3f}",
                f"{row['R2']:.4f}",  f"{row['RegAcc']:.1f}",
                f"{row['ClsAcc']:.1f}", f"{row['MAPE']:.1f}",
                f"{row['Score']:.4f}",
            ), tags=(tag,))

    def _sort_metrics(self, col):
        pass  # future: add sort by column

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 — CONFUSION MATRIX
    # ─────────────────────────────────────────────────────────────────────────

    def _build_cm_tab(self):
        f = self.tab_cm

        # Controls
        ctrl = tk.Frame(f, bg=C["panel"], padx=14, pady=10)
        ctrl.pack(fill="x")
        tk.Label(ctrl, text="Session:",
                 bg=C["panel"], fg=C["subtext"],
                 font=("Segoe UI", 9)).pack(side="left")
        self.cm_session_var = tk.StringVar()
        self.cm_session_cb  = ttk.Combobox(ctrl, textvariable=self.cm_session_var,
                                            state="readonly", width=36)
        self.cm_session_cb.pack(side="left", padx=(4,16))

        tk.Label(ctrl, text="View:",
                 bg=C["panel"], fg=C["subtext"],
                 font=("Segoe UI", 9)).pack(side="left")
        self.cm_view_var = tk.StringVar(value="All 4 matrices")
        self.cm_view_cb  = ttk.Combobox(ctrl, textvariable=self.cm_view_var,
                                         state="readonly", width=22,
                                         values=["All 4 matrices",
                                                 "RF (Train)",
                                                 "RF (LOO-CV)",
                                                 "GBM (Train)",
                                                 "GBM (LOO-CV)"])
        self.cm_view_cb.pack(side="left", padx=(4,12))
        tk.Button(ctrl, text="Update",
                  bg=C["accent"], fg="white",
                  font=("Segoe UI", 9, "bold"),
                  relief="flat", cursor="hand2",
                  command=self._refresh_cm_tab).pack(side="left", padx=4)

        # Class legend
        for label, col in [("🟢 Safe (<220 ppm)", C["safe"]),
                            ("🟡 Warning (220–259)", C["warn"]),
                            ("🔴 Critical (≥260)", C["crit"])]:
            tk.Label(ctrl, text=label,
                     bg=C["panel"], fg=col,
                     font=("Segoe UI", 9)).pack(side="right", padx=8)

        # Canvas area
        self.cm_canvas_frame = tk.Frame(f, bg=C["bg"])
        self.cm_canvas_frame.pack(fill="both", expand=True, padx=8, pady=8)

        # Stats strip
        self.cm_stats_frame = tk.Frame(f, bg=C["panel"], padx=14, pady=8)
        self.cm_stats_frame.pack(fill="x")

    def _refresh_cm_tab(self):
        if not self.all_results:
            return

        # Populate combo
        options = [f"{r['pkl_file']} | {r['session_name']}"
                   for r in self.all_results]
        self.cm_session_cb["values"] = options
        if not self.cm_session_var.get() or \
                self.cm_session_var.get() not in options:
            self.cm_session_var.set(options[0])

        self._draw_cm()
        self.cm_session_cb.bind("<<ComboboxSelected>>",
                                 lambda _: self._draw_cm())
        self.cm_view_cb.bind("<<ComboboxSelected>>",
                              lambda _: self._draw_cm())

    def _draw_cm(self):
        clear_frame(self.cm_canvas_frame)
        clear_frame(self.cm_stats_frame)

        sel   = self.cm_session_var.get()
        view  = self.cm_view_var.get()
        res   = next((r for r in self.all_results
                      if f"{r['pkl_file']} | {r['session_name']}" == sel), None)
        if res is None:
            return

        view_map = {
            "RF (Train)":    [("rf",  "#4F8EF7", "Random Forest\n(Train-set)")],
            "RF (LOO-CV)":   [("rf_loo","#4F8EF7","Random Forest\n(LOO-CV)")],
            "GBM (Train)":   [("gbm", "#FF9800", "Gradient Boost\n(Train-set)")],
            "GBM (LOO-CV)":  [("gbm_loo","#FF9800","Gradient Boost\n(LOO-CV)")],
            "All 4 matrices":[
                ("rf",      "#4F8EF7", "Random Forest\n(Train)"),
                ("rf_loo",  "#4F8EF7", "Random Forest\n(LOO-CV)"),
                ("gbm",     "#FF9800", "Gradient Boost\n(Train)"),
                ("gbm_loo", "#FF9800", "Gradient Boost\n(LOO-CV)"),
            ],
        }
        panels = view_map.get(view, view_map["All 4 matrices"])
        ncols  = len(panels)

        fig, axes = plt.subplots(1, ncols,
                                  figsize=(5 * ncols, 5.5))
        if ncols == 1:
            axes = [axes]
        fig.patch.set_facecolor(C["bg"])
        fig.suptitle(
            f"Confusion Matrices  —  {res['session_name']}  "
            f"({res['pkl_file']})",
            color=C["text"], fontsize=11, fontweight="bold"
        )

        for ax, (key, color, title) in zip(axes, panels):
            ax.set_facecolor(C["panel"])
            m = res.get(key)
            if m is None:
                ax.text(0.5, 0.5, "Not available",
                        ha="center", va="center",
                        color=C["subtext"], fontsize=11,
                        transform=ax.transAxes)
                ax.set_title(title, color=C["text"],
                             fontsize=10, fontweight="bold")
                for sp in ax.spines.values():
                    sp.set_edgecolor(C["border"])
                continue

            cm_arr = m["cm"]
            row_s  = cm_arr.sum(axis=1, keepdims=True)
            row_s[row_s == 0] = 1
            cm_norm = cm_arr / row_s * 100

            # custom heatmap on dark background
            cmap = plt.cm.Blues
            im   = ax.imshow(cm_norm, cmap=cmap, aspect="auto",
                              vmin=0, vmax=100)

            ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
            ax.set_xticklabels(CLASS_LABELS, color=C["subtext"], fontsize=9)
            ax.set_yticklabels(CLASS_LABELS, color=C["subtext"], fontsize=9,
                               rotation=90, va="center")
            ax.set_xlabel("Predicted", color=C["subtext"], fontsize=9)
            ax.set_ylabel("True",      color=C["subtext"], fontsize=9)
            ax.tick_params(colors=C["subtext"])

            for sp in ax.spines.values():
                sp.set_edgecolor(C["border"])

            # Annotate cells
            class_colors = [C["safe"], C["warn"], C["crit"]]
            for i in range(3):
                for j in range(3):
                    bg_pct = cm_norm[i, j]
                    fc = "white" if bg_pct > 50 else C["text"]
                    border_c = class_colors[i] if i == j else C["subtext"]
                    ax.add_patch(plt.Rectangle(
                        (j-0.5, i-0.5), 1, 1,
                        fill=False,
                        edgecolor=border_c if i==j else C["border"],
                        lw=2.5 if i==j else 0.5))
                    ax.text(j, i,
                            f"{cm_arr[i,j]}\n({bg_pct:.0f}%)",
                            ha="center", va="center",
                            fontsize=10 if ncols <= 2 else 9,
                            fontweight="bold", color=fc)

            acc_str = f"ClsAcc={m['cls_acc']:.1f}%"
            ax.set_title(f"{title}\n{acc_str}",
                          color=C["text"], fontsize=10, fontweight="bold",
                          pad=8)

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        if "cm" in self._canvases:
            try:
                self._canvases["cm"].get_tk_widget().destroy()
            except Exception:
                pass
        canvas = embed_figure(fig, self.cm_canvas_frame)
        self._canvases["cm"] = canvas
        plt.close(fig)

        # Stats strip
        for key, label in [("rf","RF Train"),("rf_loo","RF LOO"),
                            ("gbm","GBM Train"),("gbm_loo","GBM LOO")]:
            m = res.get(key)
            if m is None:
                continue
            col = C["rf"] if "rf" in key else C["gbm"]
            b = tk.Frame(self.cm_stats_frame, bg=C["card"], padx=12, pady=6,
                         highlightthickness=1,
                         highlightbackground=C["border"])
            b.pack(side="left", padx=6, pady=4)
            tk.Label(b, text=label, bg=C["card"], fg=col,
                     font=("Segoe UI", 8, "bold")).pack(anchor="w")
            for t, v in [("ClsAcc", f"{m['cls_acc']:.1f}%"),
                         ("MAE",    f"{m['mae']:.2f} ppm"),
                         ("R²",     f"{m['r2']:.4f}")]:
                row = tk.Frame(b, bg=C["card"])
                row.pack(fill="x")
                tk.Label(row, text=t+":", bg=C["card"], fg=C["subtext"],
                         font=("Segoe UI", 8), width=7, anchor="w").pack(side="left")
                tk.Label(row, text=v,   bg=C["card"], fg=C["text"],
                         font=("Segoe UI", 8, "bold")).pack(side="left")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4 — SCATTER
    # ─────────────────────────────────────────────────────────────────────────

    def _build_scatter_tab(self):
        f = self.tab_scatter
        ctrl = tk.Frame(f, bg=C["panel"], padx=14, pady=10)
        ctrl.pack(fill="x")
        tk.Label(ctrl, text="Session:",
                 bg=C["panel"], fg=C["subtext"],
                 font=("Segoe UI", 9)).pack(side="left")
        self.sc_session_var = tk.StringVar()
        self.sc_session_cb  = ttk.Combobox(ctrl, textvariable=self.sc_session_var,
                                            state="readonly", width=38)
        self.sc_session_cb.pack(side="left", padx=(4,12))
        tk.Button(ctrl, text="Update",
                  bg=C["accent"], fg="white",
                  font=("Segoe UI", 9, "bold"),
                  relief="flat", cursor="hand2",
                  command=self._draw_scatter).pack(side="left", padx=4)
        self.sc_canvas_frame = tk.Frame(f, bg=C["bg"])
        self.sc_canvas_frame.pack(fill="both", expand=True, padx=8, pady=8)

    def _refresh_scatter_tab(self):
        if not self.all_results:
            return
        options = [f"{r['pkl_file']} | {r['session_name']}"
                   for r in self.all_results]
        self.sc_session_cb["values"] = options
        if not self.sc_session_var.get() or \
                self.sc_session_var.get() not in options:
            self.sc_session_var.set(options[0])
        self._draw_scatter()
        self.sc_session_cb.bind("<<ComboboxSelected>>",
                                 lambda _: self._draw_scatter())

    def _draw_scatter(self):
        clear_frame(self.sc_canvas_frame)
        sel = self.sc_session_var.get()
        res = next((r for r in self.all_results
                    if f"{r['pkl_file']} | {r['session_name']}" == sel), None)
        if res is None:
            return

        panels = [
            ("rf",      C["rf"],  "Random Forest (Train)"),
            ("rf_loo",  C["rf"],  "Random Forest (LOO-CV)"),
            ("gbm",     C["gbm"], "Gradient Boost (Train)"),
            ("gbm_loo", C["gbm"], "Gradient Boost (LOO-CV)"),
        ]
        available = [(k,c,l) for k,c,l in panels if res.get(k)]
        nc = len(available)
        if nc == 0:
            return

        fig, axes = plt.subplots(1, nc, figsize=(5*nc, 5))
        if nc == 1:
            axes = [axes]
        fig.patch.set_facecolor(C["bg"])
        fig.suptitle(
            f"Actual vs Predicted  —  {res['session_name']}  ({res['pkl_file']})",
            color=C["text"], fontsize=11, fontweight="bold"
        )

        for ax, (key, color, title) in zip(axes, available):
            m = res[key]
            yt, yp = m["y_true"], m["y_pred"]
            lim = [min(yt.min(), yp.min())-5, max(yt.max(), yp.max())+5]
            ax.set_facecolor(C["panel"])
            ax.scatter(yt, yp, c=color, edgecolors="#ffffff30",
                       s=80, zorder=3, alpha=0.9)
            ax.plot(lim, lim, color=C["subtext"], ls="--",
                    lw=1.2, label="Perfect fit")
            ax.axhline(WARNING_PPM,  color=C["warn"], ls=":", lw=1.2,
                       label=f"Warning ({WARNING_PPM})")
            ax.axhline(CRITICAL_PPM, color=C["crit"], ls=":", lw=1.2,
                       label=f"Critical ({CRITICAL_PPM})")
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel("Actual Gas (ppm)", color=C["subtext"])
            ax.set_ylabel("Predicted Gas (ppm)", color=C["subtext"])
            ax.tick_params(colors=C["subtext"])
            for sp in ax.spines.values():
                sp.set_edgecolor(C["border"])
            ax.grid(alpha=0.15, color=C["border"])
            ax.legend(fontsize=7, facecolor=C["card"],
                      labelcolor=C["text"], edgecolor=C["border"])
            ax.set_title(
                f"{title}\nMAE={m['mae']:.2f} ppm   "
                f"R²={m['r2']:.4f}   ClsAcc={m['cls_acc']:.1f}%",
                color=C["text"], fontsize=9, fontweight="bold"
            )

            # Day-by-day timeline as inset
            ax2 = ax.inset_axes([0.0, -0.52, 1.0, 0.42])
            idx = np.arange(len(yt))
            ax2.plot(idx, yt, "o-", color=C["accent"],
                     lw=1.8, ms=4, label="Actual")
            ax2.plot(idx, yp, "s--", color=color,
                     lw=1.5, ms=3, label="Predicted")
            ax2.axhline(WARNING_PPM,  color=C["warn"], ls=":", lw=1)
            ax2.axhline(CRITICAL_PPM, color=C["crit"], ls=":", lw=1)
            ax2.set_facecolor(C["bg"])
            ax2.tick_params(colors=C["subtext"], labelsize=7)
            for sp in ax2.spines.values():
                sp.set_edgecolor(C["border"])
            ax2.set_xlabel("Day Index", color=C["subtext"], fontsize=7)
            ax2.set_ylabel("ppm",       color=C["subtext"], fontsize=7)
            ax2.grid(alpha=0.12, color=C["border"])
            ax2.legend(fontsize=6, facecolor=C["card"],
                       labelcolor=C["text"], edgecolor=C["border"])

        plt.subplots_adjust(top=0.88, bottom=0.28, wspace=0.35)

        if "scatter" in self._canvases:
            try:
                self._canvases["scatter"].get_tk_widget().destroy()
            except Exception:
                pass
        canvas = embed_figure(fig, self.sc_canvas_frame)
        self._canvases["scatter"] = canvas
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 5 — RANKING
    # ─────────────────────────────────────────────────────────────────────────

    def _build_rank_tab(self):
        f = self.tab_rank
        tk.Label(f, text="Model Ranking — Composite Score",
                 bg=C["bg"], fg=C["text"],
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=16, pady=(12,4))
        self.rank_canvas_frame = tk.Frame(f, bg=C["bg"])
        self.rank_canvas_frame.pack(fill="both", expand=True, padx=8, pady=4)

    def _refresh_rank_tab(self):
        clear_frame(self.rank_canvas_frame)
        if self.df_rank.empty:
            return

        df = self.df_rank
        n  = len(df)
        labels = [f"{r['pkl_file']}\n{r['session']}\n{r['model'][:3]}"
                  for _, r in df.iterrows()]
        scores = df["Score"].values
        bar_c  = [C["rf"] if "Forest" in m else C["gbm"]
                  for m in df["model"]]

        rank_colors = [C["gold"], C["silver"], C["bronze"]]

        fig = plt.figure(figsize=(16, max(5.5, n*0.85+2)))
        fig.patch.set_facecolor(C["bg"])
        gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)
        fig.suptitle("All-Model Ranking", color=C["text"],
                     fontsize=12, fontweight="bold")

        # ── Left: horizontal bar ──────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor(C["panel"])
        y_pos = np.arange(n)
        bars  = ax1.barh(y_pos, scores[::-1],
                          color=bar_c[::-1], edgecolor=C["bg"],
                          height=0.65)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels[::-1], fontsize=8, color=C["text"])
        ax1.set_xlabel("Composite Score", color=C["subtext"])
        ax1.tick_params(colors=C["subtext"])
        for sp in ax1.spines.values():
            sp.set_edgecolor(C["border"])
        ax1.grid(alpha=0.15, axis="x", color=C["border"])
        ax1.set_title("Composite Score  (Higher = Better)",
                       color=C["text"], fontsize=10)

        for i, (bar, s) in enumerate(zip(bars, scores[::-1])):
            rank_idx = n - 1 - i
            ax1.text(bar.get_width() + 0.003,
                     bar.get_y() + bar.get_height()/2,
                     f"{s:.4f}", va="center", fontsize=8,
                     color=rank_colors[rank_idx] if rank_idx < 3 else C["subtext"])
            if rank_idx < 3:
                medal = ["🥇","🥈","🥉"][rank_idx]
                ax1.text(-0.005, bar.get_y() + bar.get_height()/2,
                          medal, va="center", ha="right", fontsize=10)

        leg = [mpatches.Patch(color=C["rf"],  label="Random Forest"),
               mpatches.Patch(color=C["gbm"], label="Gradient Boost")]
        ax1.legend(handles=leg, fontsize=8,
                   facecolor=C["card"], edgecolor=C["border"],
                   labelcolor=C["text"], loc="lower right")

        # ── Right: metric heatmap ─────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor(C["panel"])
        heat_cols = ["R2","RegAcc","ClsAcc","MAE","Score"]
        display_names = ["R²","Reg Acc%","Cls Acc%","MAE (ppm)","Score"]
        heat_df   = df[heat_cols].copy().astype(float)
        heat_norm = (heat_df - heat_df.min()) / \
                    (heat_df.max() - heat_df.min() + 1e-9)
        heat_norm["MAE"] = 1 - heat_norm["MAE"]   # lower MAE = better

        short_lb  = [f"{r['pkl_file']} | {r['model'][:3]}"
                     for _, r in df.iterrows()]

        im = ax2.imshow(heat_norm.values, cmap="RdYlGn",
                         aspect="auto", vmin=0, vmax=1)

        ax2.set_xticks(range(len(display_names)))
        ax2.set_xticklabels(display_names, rotation=30, ha="right",
                             fontsize=8, color=C["subtext"])
        ax2.set_yticks(range(n))
        ax2.set_yticklabels(short_lb, fontsize=8, color=C["text"])
        ax2.tick_params(colors=C["subtext"])
        for sp in ax2.spines.values():
            sp.set_edgecolor(C["border"])

        for i in range(n):
            for j, col in enumerate(heat_cols):
                v_raw  = heat_df.iloc[i][col]
                v_norm = heat_norm.iloc[i][col]
                fc = "white" if v_norm < 0.35 or v_norm > 0.72 else "#1A1A2E"
                ax2.text(j, i, f"{v_raw:.2f}",
                          ha="center", va="center",
                          fontsize=8, color=fc, fontweight="bold")

        ax2.set_title("Normalised Metric Heatmap\n(Greener = Better)",
                       color=C["text"], fontsize=10)

        cb = plt.colorbar(im, ax=ax2, fraction=0.035, pad=0.02)
        cb.ax.tick_params(colors=C["subtext"], labelsize=7)
        cb.outline.set_edgecolor(C["border"])

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        if "rank" in self._canvases:
            try:
                self._canvases["rank"].get_tk_widget().destroy()
            except Exception:
                pass
        canvas = embed_figure(fig, self.rank_canvas_frame)
        self._canvases["rank"] = canvas
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 6 — LOG
    # ─────────────────────────────────────────────────────────────────────────

    def _build_log_tab(self):
        f = self.tab_log
        toolbar = tk.Frame(f, bg=C["panel"], pady=6, padx=12)
        toolbar.pack(fill="x")
        tk.Label(toolbar, text="Console Log",
                 bg=C["panel"], fg=C["text"],
                 font=("Segoe UI", 11, "bold")).pack(side="left")
        tk.Button(toolbar, text="Clear",
                  bg=C["border"], fg=C["text"],
                  font=("Segoe UI", 9), relief="flat", cursor="hand2",
                  command=self._clear_log).pack(side="right", padx=4)
        self.log_box = scrolledtext.ScrolledText(
            f, bg=C["panel"], fg=C["text"],
            font=("Consolas", 9), relief="flat",
            insertbackground=C["text"], borderwidth=0)
        self.log_box.pack(fill="both", expand=True, padx=8, pady=6)
        self.log_box.tag_configure("good",  foreground=C["safe"])
        self.log_box.tag_configure("warn",  foreground=C["warn"])
        self.log_box.tag_configure("error", foreground=C["crit"])
        self.log_box.tag_configure("head",  foreground=C["accent"],
                                            font=("Consolas", 9, "bold"))

    def _log(self, msg):
        def _insert():
            tag = ("good"  if "Done" in msg or "★" in msg or "BEST" in msg
                   else "error" if "ERROR" in msg or "SKIP" in msg
                   else "warn"  if "WARN" in msg
                   else "head"  if msg.startswith("=") or msg.startswith("►")
                   else "")
            self.log_box.insert("end", msg + "\n", tag)
            self.log_box.see("end")
        self.after(0, _insert)

    def _clear_log(self):
        self.log_box.delete("1.0", "end")

    # ─────────────────────────────────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────────────────────────────────

    def _status(self, msg):
        self.after(0, lambda: self._status_var.set(msg))


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = AgrinovaInferenceUI()
    app.mainloop()