"""
=============================================================================
AGRINOVA — Onion Spoilage Prediction  |  Tkinter UI + Pickle Model Store
=============================================================================
How it works:
  • Pick a CSV file via the Browse button
  • Give the session a name (e.g. "C1_Exhaust_Day1")
  • Click "Run Pipeline"
  • Models (RF + GBM) and all results are saved to agrinova_models.pkl
  • Every session you run is appended — nothing is ever overwritten
  • The History tab shows all past sessions stored in the pickle file
  • The Results tab shows metrics, spoilage forecast, and the plot

Run:
  python agrinova_app.py

Requirements:
  pip install scikit-learn matplotlib numpy pandas
=============================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import pickle
import datetime
import threading
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")          # must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut

# ── Tkinter ───────────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PICKLE_FILE    = "agrinova_model_1.pkl"

HARD_LIMITS    = {"Temp": (0.0, 50.0), "Hum": (0.0, 100.0), "Gas": (0.0, 999.0)}
IQR_FENCE      = 3.5
NEIGHBOUR_MULT = 3.0

WARNING_PPM    = 220
CRITICAL_PPM   = 250
CONSEC_DAYS    = 3
FORECAST_DAYS  = 180
MIN_ROWS_ML    = 3

RF_SEED        = 42
GBM_SEED       = 42

FEATURE_COLS   = ["DayIndex", "Lag1", "Lag2", "Lag3",
                  "Roll3_Mean", "Roll3_Std", "dG", "dT", "dH"]
TARGET_COL     = "AvgGas"

# ─────────────────────────────────────────────────────────────────────────────
# PICKLE HELPERS  — load / save the session store
# ─────────────────────────────────────────────────────────────────────────────

def load_store(path: str) -> dict:
    """
    Load the pickle store from disk.
    Returns an empty dict if file does not exist yet.
    Structure: { session_name: session_dict, ... }
    """
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def save_store(store: dict, path: str) -> None:
    """Persist the full store dict to disk as a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(store, f)


def save_session(session_name: str, session_data: dict, path: str) -> None:
    """
    Append (or overwrite) one session inside the pickle store.
    All other sessions are left untouched.
    """
    store = load_store(path)
    store[session_name] = session_data
    save_store(store, path)

# ─────────────────────────────────────────────────────────────────────────────
# ML PIPELINE  — pure functions, no UI calls
# ─────────────────────────────────────────────────────────────────────────────

def step_clean(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Spike removal: hard limits → IQR fence → neighbour spikes → gap-fill."""
    df = df_raw.copy()
    counts = {}

    for col, (lo, hi) in HARD_LIMITS.items():
        mask = ~df[col].between(lo, hi)
        df.loc[mask, col] = np.nan
        counts[f"{col}_hard"] = int(mask.sum())

    for col in ["Temp", "Hum", "Gas"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        mask = (df[col] < q1 - IQR_FENCE * iqr) | (df[col] > q3 + IQR_FENCE * iqr)
        df.loc[mask, col] = np.nan
        counts[f"{col}_iqr"] = int(mask.sum())

    for col in ["Temp", "Hum", "Gas"]:
        s   = df[col].copy()
        std = s.rolling(10, center=True, min_periods=3).std()
        thr = NEIGHBOUR_MULT * std
        mask = ((s - s.shift(1)).abs() > thr) & ((s - s.shift(-1)).abs() > thr)
        df.loc[mask, col] = np.nan
        counts[f"{col}_neighbour"] = int(mask.sum())

    for col in ["Temp", "Hum", "Gas"]:
        df[col] = df[col].ffill(limit=3).bfill(limit=3)

    df = df.dropna(subset=["Temp", "Hum", "Gas"]).reset_index(drop=True)
    return df, counts


def step_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned 10-sec readings to one row per calendar day."""
    df["Date"] = df["Timestamp"].dt.date
    daily = (df.groupby("Date")
               .agg(AvgTemp=("Temp", "mean"),
                    AvgHum =("Hum",  "mean"),
                    AvgGas =("Gas",  "mean"))
               .reset_index())
    daily["Date"]     = pd.to_datetime(daily["Date"])
    daily["DayIndex"] = (daily["Date"] - daily["Date"].min()).dt.days
    return daily


def step_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Create lag, rolling, and delta features on the daily table."""
    n   = len(daily)
    lag = min(3, max(n - 1, 0))

    for k in range(1, lag + 1):
        daily[f"Lag{k}"] = daily["AvgGas"].shift(k)
    for k in range(1, 4):
        if f"Lag{k}" not in daily.columns:
            daily[f"Lag{k}"] = np.nan

    daily["Roll3_Mean"] = (daily["AvgGas"].shift(1)
                                          .rolling(3, min_periods=1).mean())
    daily["Roll3_Std"]  = (daily["AvgGas"].shift(1)
                                          .rolling(3, min_periods=1)
                                          .std().fillna(0.0))
    daily["dG"] = daily["AvgGas"].diff().fillna(0.0)
    daily["dT"] = daily["AvgTemp"].diff().fillna(0.0)
    daily["dH"] = daily["AvgHum"].diff().fillna(0.0)

    daily[FEATURE_COLS] = daily[FEATURE_COLS].ffill(axis=1).bfill(axis=1)
    return daily.dropna(subset=[TARGET_COL]).reset_index(drop=True)


def loo_evaluate(model, X, y, model_name: str) -> dict:
    """LOO-CV: train on n-1, predict 1, repeat for all rows."""
    loo   = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(X):
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
    model.fit(X, y)
    train_preds = model.predict(X)
    return {
        "model":       model,
        "model_name":  model_name,
        "loo_mae":     float(mean_absolute_error(y, preds)),
        "loo_r2":      float(r2_score(y, preds)),
        "train_r2":    float(r2_score(y, train_preds)),
        "loo_preds":   preds.tolist(),
        "train_preds": train_preds.tolist(),
    }


def fit_exponential(day_index, gas_values) -> dict:
    """Fit Gas(t) = G0·exp(b·t) via log-linearisation."""
    t     = np.array(day_index, dtype=float)
    g     = np.array(gas_values, dtype=float)
    logG  = np.log(np.maximum(g, 1e-9))
    coeff = np.polyfit(t, logG, 1)
    b     = float(coeff[0])
    G0    = float(g[0])   # anchor to real first value
    return {
        "G0": G0, "b": b,
        "doubling_time": float(np.log(2) / b) if b != 0 else float("inf"),
        "trend": "accumulating" if b > 0 else ("removing" if b < 0 else "stable"),
    }


def forecast_exp(G0, b, days) -> np.ndarray:
    t = np.arange(0, days + 1, dtype=float)
    return G0 * np.exp(b * t)


def detect_spoilage(forecast_arr, t_arr, consec=3) -> dict:
    """
    Return first day of CONSEC consecutive days >= each threshold.
    If the forecast never sustains CONSEC consecutive days above the threshold
    but does cross it at least once, fall back to the first crossing day.
    Returns None only if the threshold is never reached at all.
    """
    out = {}
    for label, thr in [("warning", WARNING_PPM), ("critical", CRITICAL_PPM)]:
        streak      = 0
        found       = None   # first day of CONSEC-consecutive streak
        first_cross = None   # first day gas ever crosses the threshold (fallback)
        for i, v in enumerate(forecast_arr):
            if v >= thr:
                if first_cross is None:
                    first_cross = int(t_arr[i])   # record very first crossing
                streak += 1
                if streak >= consec and found is None:
                    found = int(t_arr[i - consec + 1])
                    break
            else:
                streak = 0
        # fall back to first single crossing when consecutive streak never completes
        out[label] = found if found is not None else first_cross
    return out


def run_pipeline(csv_path: str, session_name: str,
                 log_fn=None) -> dict:
    """
    Full pipeline. log_fn(msg) is called with progress strings so the UI
    can display them in real time.  Returns the complete session dict.
    """

    def log(msg):
        if log_fn:
            log_fn(msg)

    # ── Load ──────────────────────────────────────────────────────────────────
    log("Loading CSV…")
    df_raw = pd.read_csv(csv_path)
    # df_raw["Timestamp"] = pd.to_datetime(df_raw["Timestamp"])
    df_raw["Timestamp"] = pd.to_datetime(
    df_raw["Timestamp"],
    format="mixed",
    dayfirst=True,
    errors="coerce")
    df_raw = df_raw.sort_values("Timestamp").reset_index(drop=True)
    log(f"  {len(df_raw):,} rows loaded  |  "
        f"{df_raw['Timestamp'].min().date()} → {df_raw['Timestamp'].max().date()}")

    # ── Clean ─────────────────────────────────────────────────────────────────
    log("Removing spikes…")
    df_clean, removal = step_clean(df_raw)
    for col in ["Temp", "Hum", "Gas"]:
        log(f"  {col}: hard={removal[col+'_hard']}  "
            f"iqr={removal[col+'_iqr']}  "
            f"neighbour={removal[col+'_neighbour']}")
    log(f"  Rows after cleaning: {len(df_clean):,} / {len(df_raw):,}")

    # ── Daily aggregation ─────────────────────────────────────────────────────
    log("Aggregating to daily means…")
    daily = step_daily(df_clean)
    log(f"  {len(daily)} day(s)  |  "
        f"DayIndex {daily['DayIndex'].min()} → {daily['DayIndex'].max()}")

    # ── Feature engineering ───────────────────────────────────────────────────
    log("Engineering features…")
    daily_ml = step_features(daily.copy())
    log(f"  ML-ready rows: {len(daily_ml)}  (need ≥{MIN_ROWS_ML} for LOO-CV)")

    X = daily_ml[FEATURE_COLS].values
    y = daily_ml[TARGET_COL].values
    ml_available = len(X) >= MIN_ROWS_ML

    # ── ML training ───────────────────────────────────────────────────────────
    rf_result = gbm_result = None
    feat_imp  = {f: 0.0 for f in FEATURE_COLS}

    if ml_available:
        log("Training Random Forest (LOO-CV)…")
        rf_result = loo_evaluate(
            RandomForestRegressor(n_estimators=500, random_state=RF_SEED),
            X, y, "Random Forest"
        )
        log(f"  LOO-MAE = {rf_result['loo_mae']:.3f} ppm  |  "
            f"LOO-R² = {rf_result['loo_r2']:.4f}  |  "
            f"Train-R² = {rf_result['train_r2']:.4f}")

        log("Training Gradient Boosting (LOO-CV)…")
        gbm_result = loo_evaluate(
            GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                      max_depth=3, subsample=0.8,
                                      random_state=GBM_SEED),
            X, y, "Gradient Boosting"
        )
        log(f"  LOO-MAE = {gbm_result['loo_mae']:.3f} ppm  |  "
            f"LOO-R² = {gbm_result['loo_r2']:.4f}  |  "
            f"Train-R² = {gbm_result['train_r2']:.4f}")

        feat_imp = dict(zip(FEATURE_COLS,
                            rf_result["model"].feature_importances_.tolist()))
        log("  Feature importances (RF):")
        for feat, imp in sorted(feat_imp.items(), key=lambda x: -x[1]):
            bar = "█" * int(imp * 30)
            log(f"    {feat:12s} {imp:.4f}  {bar}")
    else:
        log(f"  ⚠ Only {len(X)} row(s) — ML skipped. Need ≥{MIN_ROWS_ML} days.")

    # ── Exponential forecast ──────────────────────────────────────────────────
    log("Fitting exponential model…")
    day_arr  = daily["DayIndex"].values.astype(float)
    gas_arr  = daily["AvgGas"].values
    exp_fit  = fit_exponential(day_arr, gas_arr)
    t_future = np.arange(0, FORECAST_DAYS + 1)
    gas_fc   = forecast_exp(exp_fit["G0"], exp_fit["b"], FORECAST_DAYS)
    spoilage = detect_spoilage(gas_fc, t_future, CONSEC_DAYS)

    log(f"  Gas(t) = {exp_fit['G0']:.2f} × exp({exp_fit['b']:+.5f} × t)")
    start_date = daily["Date"].min()
    for lv, thr in [("WARNING", WARNING_PPM), ("CRITICAL", CRITICAL_PPM)]:
        d = spoilage[lv.lower()]
        if d is not None:
            ed = (start_date + pd.Timedelta(days=d)).date()
            log(f"  {lv}: Day {d} ({ed})")
        else:
            log(f"  {lv}: Not reached within {FORECAST_DAYS} days")

    # ── Build session dict ────────────────────────────────────────────────────
    session = {
        # ── metadata ──────────────────────────────────────────────────────────
        "session_name": session_name,
        "csv_path":     csv_path,
        "timestamp":    datetime.datetime.now().isoformat(timespec="seconds"),
        # ── data snapshots ────────────────────────────────────────────────────
        "raw_shape":    df_raw.shape,
        "clean_shape":  df_clean.shape,
        "daily_records": len(daily),
        "daily_table":  daily.to_dict(orient="records"),
        "removal_counts": removal,
        # ── features ──────────────────────────────────────────────────────────
        "feature_cols": FEATURE_COLS,
        "X":            X.tolist(),
        "y":            y.tolist(),
        "ml_available": ml_available,
        # ── models (stored as sklearn objects inside pickle) ──────────────────
        "rf_model":     rf_result["model"]  if rf_result  else None,
        "gbm_model":    gbm_result["model"] if gbm_result else None,
        # ── metrics ───────────────────────────────────────────────────────────
        "rf_metrics":  {k: v for k, v in rf_result.items()  if k != "model"}
                        if rf_result  else None,
        "gbm_metrics": {k: v for k, v in gbm_result.items() if k != "model"}
                        if gbm_result else None,
        "feature_importances": feat_imp,
        # ── exponential forecast ──────────────────────────────────────────────
        "exp_fit":      exp_fit,
        "forecast_gas": gas_fc.tolist(),
        "spoilage":     spoilage,
        # ── raw arrays for plotting ───────────────────────────────────────────
        "plot_timestamps": df_clean["Timestamp"].astype(str).tolist(),
        "plot_gas_raw":    df_clean["Gas"].tolist(),
        "plot_day_idx":    day_arr.tolist(),
        "plot_gas_daily":  gas_arr.tolist(),
    }

    log("Saving to pickle…")
    save_session(session_name, session, PICKLE_FILE)
    log(f"  Saved → {PICKLE_FILE}  (session: '{session_name}')")
    log("Done ✓")
    return session


# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPER  — build the matplotlib figure from a session dict
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {"rf": "#2196F3", "gbm": "#FF9800", "actual": "#1A1A2E",
          "forecast": "#9C27B0", "warn": "#FF9800", "crit": "#F44336",
          "raw": "#90CAF9"}


def build_figure(session: dict) -> plt.Figure:
    fig = plt.figure(figsize=(16, 13))
    fig.suptitle(f"Agrinova — {session['session_name']}",
                 fontsize=14, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.46, wspace=0.36)

    day_arr  = np.array(session["plot_day_idx"])
    gas_arr  = np.array(session["plot_gas_daily"])
    ts       = pd.to_datetime(session["plot_timestamps"])
    gas_raw  = np.array(session["plot_gas_raw"])
    t_future = np.arange(0, FORECAST_DAYS + 1)
    gas_fc   = np.array(session["forecast_gas"])
    spoilage = session["spoilage"]
    ep       = session["exp_fit"]
    ml       = session["ml_available"]

    # ── Panel 1: raw + daily ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(ts, gas_raw, color=COLORS["raw"], lw=0.5, alpha=0.6,
             label="10-sec readings")
    ax1.plot(pd.to_datetime([r["Date"] for r in session["daily_table"]]),
             gas_arr, "o-", color=COLORS["actual"], lw=2, ms=7,
             label="Daily mean")
    ax1.axhline(WARNING_PPM,  color=COLORS["warn"], ls="--", lw=1.2,
                label=f"WARNING ({WARNING_PPM} ppm)")
    ax1.axhline(CRITICAL_PPM, color=COLORS["crit"], ls="--", lw=1.2,
                label=f"CRITICAL ({CRITICAL_PPM} ppm)")
    ax1.set_title("Cleaned Gas Readings & Daily Averages")
    ax1.set_xlabel("Timestamp"); ax1.set_ylabel("Gas (ppm)")
    ax1.legend(loc="upper right", fontsize=8); ax1.grid(alpha=0.3)

    # ── Panel 2: RF LOO ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    if ml and session["rf_metrics"]:
        y_act  = np.array(session["y"])
        y_pred = np.array(session["rf_metrics"]["loo_preds"])
        lim = [min(y_act.min(), y_pred.min()) - 5,
               max(y_act.max(), y_pred.max()) + 5]
        ax2.scatter(y_act, y_pred, color=COLORS["rf"],
                    edgecolors="white", s=80, zorder=3)
        ax2.plot(lim, lim, "k--", lw=1.2, label="Perfect fit")
        ax2.set_title(f"Random Forest  |  LOO-MAE = "
                      f"{session['rf_metrics']['loo_mae']:.2f} ppm")
        ax2.set_xlim(lim); ax2.set_ylim(lim)
        ax2.set_xlabel("Actual (ppm)"); ax2.set_ylabel("LOO Predicted (ppm)")
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "ML not available\n(need ≥3 days)",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=11, color="gray")
        ax2.set_title("Random Forest — Insufficient Data")
    ax2.grid(alpha=0.3)

    # ── Panel 3: GBM LOO ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    if ml and session["gbm_metrics"]:
        y_act  = np.array(session["y"])
        y_pred = np.array(session["gbm_metrics"]["loo_preds"])
        lim = [min(y_act.min(), y_pred.min()) - 5,
               max(y_act.max(), y_pred.max()) + 5]
        ax3.scatter(y_act, y_pred, color=COLORS["gbm"],
                    edgecolors="white", s=80, zorder=3)
        ax3.plot(lim, lim, "k--", lw=1.2, label="Perfect fit")
        ax3.set_title(f"Gradient Boosting  |  LOO-MAE = "
                      f"{session['gbm_metrics']['loo_mae']:.2f} ppm")
        ax3.set_xlim(lim); ax3.set_ylim(lim)
        ax3.set_xlabel("Actual (ppm)"); ax3.set_ylabel("LOO Predicted (ppm)")
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "ML not available\n(need ≥3 days)",
                 ha="center", va="center", transform=ax3.transAxes,
                 fontsize=11, color="gray")
        ax3.set_title("Gradient Boosting — Insufficient Data")
    ax3.grid(alpha=0.3)

    # ── Panel 4: Feature importances ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    fi = session["feature_importances"]
    fi_s = sorted(fi.items(), key=lambda x: x[1])
    bars = ax4.barh([k for k, v in fi_s], [v for k, v in fi_s],
                    color=COLORS["rf"], edgecolor="white", height=0.6)
    for bar, (_, val) in zip(bars, fi_s):
        if val > 0:
            ax4.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=8)
    ax4.set_title("Feature Importances (Random Forest)")
    ax4.set_xlabel("Importance"); ax4.grid(alpha=0.3, axis="x")

    # ── Panel 5: 180-day forecast ─────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(day_arr, gas_arr, "o", color=COLORS["actual"],
             ms=8, zorder=4, label="Observed")
    ax5.plot(t_future, gas_fc, "-", color=COLORS["forecast"], lw=2,
             label=f"Forecast  b={ep['b']:+.5f}/day")
    ax5.axhline(WARNING_PPM,  color=COLORS["warn"], ls="--",
                lw=1.5, label=f"WARNING ({WARNING_PPM} ppm)")
    ax5.axhline(CRITICAL_PPM, color=COLORS["crit"], ls="--",
                lw=1.5, label=f"CRITICAL ({CRITICAL_PPM} ppm)")
    start = pd.to_datetime(session["daily_table"][0]["Date"])
    for lv, col in [("warning", "warn"), ("critical", "crit")]:
        d = spoilage[lv]
        if d is not None and d <= FORECAST_DAYS:
            ax5.axvline(d, color=COLORS[col], ls=":", lw=1.5, alpha=0.8)
            ax5.text(d + 1, gas_fc[d] * 1.03,
                     f"Day {d}", color=COLORS[col], fontsize=8, fontweight="bold")
    ax5.set_title("180-Day Gas Forecast")
    ax5.set_xlabel("Day Index"); ax5.set_ylabel("Gas (ppm)")
    ax5.set_xlim(-2, FORECAST_DAYS + 5)
    ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TKINTER APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

class AgrinovaApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Agrinova — Onion Spoilage Prediction")
        self.geometry("1100x760")
        self.minsize(900, 650)
        self.configure(bg="#F5F6FA")

        self._current_session = None
        self._canvas_widget   = None

        self._build_styles()
        self._build_ui()
        self._refresh_history()

    # ── Styles ────────────────────────────────────────────────────────────────
    def _build_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("TNotebook",        background="#F5F6FA", borderwidth=0)
        style.configure("TNotebook.Tab",    background="#DDE1EC", foreground="#333",
                         padding=[14, 6], font=("Helvetica", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", "#4A6FA5")],
                  foreground=[("selected", "white")])

        style.configure("Card.TFrame",      background="white",
                         relief="flat", borderwidth=1)
        style.configure("TLabel",           background="white", foreground="#333",
                         font=("Helvetica", 10))
        style.configure("Header.TLabel",    background="white", foreground="#4A6FA5",
                         font=("Helvetica", 12, "bold"))
        style.configure("Run.TButton",      font=("Helvetica", 11, "bold"),
                         foreground="white", background="#4A6FA5",
                         padding=[18, 8])
        style.map("Run.TButton",
                  background=[("active", "#3A5F95"), ("disabled", "#AAB4CC")])
        style.configure("Small.TButton",    font=("Helvetica", 9),
                         padding=[8, 4])
        style.configure("TEntry",           font=("Helvetica", 10),
                         padding=[6, 4])
        style.configure("Treeview",         font=("Helvetica", 9),
                         rowheight=24)
        style.configure("Treeview.Heading", font=("Helvetica", 9, "bold"))

    # ── Main layout ───────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top banner ────────────────────────────────────────────────────────
        banner = tk.Frame(self, bg="#4A6FA5", height=54)
        banner.pack(fill="x")
        banner.pack_propagate(False)
        tk.Label(banner, text="🌱  AGRINOVA  —  Onion Spoilage Prediction Model 1",
                 bg="#4A6FA5", fg="white",
                 font=("Helvetica", 14, "bold")).pack(side="left", padx=18)
        tk.Label(banner, text=f"Store: {PICKLE_FILE}",
                 bg="#4A6FA5", fg="#C8D6F0",
                 font=("Helvetica", 9)).pack(side="right", padx=18)

        # ── Notebook ──────────────────────────────────────────────────────────
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=8)

        self.tab_run     = ttk.Frame(nb)
        self.tab_results = ttk.Frame(nb)
        self.tab_history = ttk.Frame(nb)

        nb.add(self.tab_run,     text="  ▶  Run Pipeline  ")
        nb.add(self.tab_results, text="  📊  Results  ")
        nb.add(self.tab_history, text="  🗂  History  ")

        self._build_tab_run()
        self._build_tab_results()
        self._build_tab_history()

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 — Run Pipeline
    # ─────────────────────────────────────────────────────────────────────────
    def _build_tab_run(self):
        f = self.tab_run
        f.configure(style="TFrame")

        # ── Input card ────────────────────────────────────────────────────────
        card = ttk.Frame(f, style="Card.TFrame", padding=20)
        card.pack(fill="x", padx=20, pady=(16, 8))

        ttk.Label(card, text="New Pipeline Session",
                  style="Header.TLabel").grid(row=0, column=0, columnspan=3,
                                              sticky="w", pady=(0, 12))

        # CSV file picker
        ttk.Label(card, text="CSV File:").grid(row=1, column=0, sticky="w", pady=4)
        self.csv_var = tk.StringVar()
        e_csv = ttk.Entry(card, textvariable=self.csv_var, width=55)
        e_csv.grid(row=1, column=1, sticky="ew", padx=(8, 6), pady=4)
        ttk.Button(card, text="Browse…", style="Small.TButton",
                   command=self._browse_csv).grid(row=1, column=2, pady=4)

        # Session name
        ttk.Label(card, text="Session Name:").grid(row=2, column=0, sticky="w", pady=4)
        self.name_var = tk.StringVar(
            value=f"Session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        ttk.Entry(card, textvariable=self.name_var, width=55).grid(
            row=2, column=1, sticky="ew", padx=(8, 6), pady=4)

        # Help text
        ttk.Label(card,
                  text="ℹ  Each session is saved independently in the pickle store.",
                  font=("Helvetica", 9), foreground="#888"
                  ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))

        card.columnconfigure(1, weight=1)

        # ── Run button ────────────────────────────────────────────────────────
        btn_row = tk.Frame(f, bg="#F5F6FA")
        btn_row.pack(pady=6)
        self.run_btn = ttk.Button(btn_row, text="▶  Run Pipeline",
                                  style="Run.TButton",
                                  command=self._on_run)
        self.run_btn.pack(side="left", padx=6)
        ttk.Button(btn_row, text="Clear Log", style="Small.TButton",
                   command=lambda: self.log_box.delete("1.0", "end")).pack(
                       side="left", padx=6)

        # ── Progress bar ──────────────────────────────────────────────────────
        self.progress = ttk.Progressbar(f, mode="indeterminate", length=400)
        self.progress.pack(pady=(0, 6))

        # ── Log box ───────────────────────────────────────────────────────────
        log_frame = ttk.Frame(f, style="Card.TFrame", padding=2)
        log_frame.pack(fill="both", expand=True, padx=20, pady=(0, 14))
        ttk.Label(log_frame, text="Pipeline Log",
                  style="Header.TLabel", font=("Helvetica", 10, "bold")
                  ).pack(anchor="w", padx=8, pady=(6, 2))
        self.log_box = scrolledtext.ScrolledText(
            log_frame, font=("Courier", 9), height=16,
            bg="#1E1E2E", fg="#C8D8F0",
            insertbackground="white", relief="flat", borderwidth=0)
        self.log_box.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _browse_csv(self):
        path = filedialog.askopenfilename(
            title="Select sensor CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.csv_var.set(path)
            # Auto-fill session name from filename
            base = os.path.splitext(os.path.basename(path))[0]
            ts   = datetime.datetime.now().strftime("%H%M%S")
            self.name_var.set(f"{base}_{ts}")

    def _log(self, msg: str):
        """Thread-safe log append."""
        self.after(0, self._log_main, msg)

    def _log_main(self, msg: str):
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")

    def _on_run(self):
        csv_path     = self.csv_var.get().strip()
        session_name = self.name_var.get().strip()

        if not csv_path:
            messagebox.showwarning("No file", "Please select a CSV file first.")
            return
        if not os.path.exists(csv_path):
            messagebox.showerror("File not found",
                                 f"Cannot find:\n{csv_path}")
            return
        if not session_name:
            messagebox.showwarning("No name", "Please enter a session name.")
            return

        # Check for duplicate session name
        store = load_store(PICKLE_FILE)
        if session_name in store:
            if not messagebox.askyesno(
                    "Overwrite?",
                    f"Session '{session_name}' already exists.\nOverwrite it?"):
                return

        self.run_btn.config(state="disabled")
        self.progress.start(10)
        self._log(f"{'='*55}")
        self._log(f"Session : {session_name}")
        self._log(f"File    : {csv_path}")
        self._log(f"{'='*55}")

        def worker():
            try:
                session = run_pipeline(csv_path, session_name, log_fn=self._log)
                self.after(0, self._on_pipeline_done, session)
            except Exception as exc:
                self.after(0, self._on_pipeline_error, str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_pipeline_done(self, session):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self._current_session = session
        self._refresh_history()
        self._show_results(session)
        # Switch to Results tab
        self.nametowidget(self.winfo_children()[1]).select(1)  # tab index 1

    def _on_pipeline_error(self, err):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self._log(f"\n❌ ERROR: {err}")
        messagebox.showerror("Pipeline error", err)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 — Results
    # ─────────────────────────────────────────────────────────────────────────
    def _build_tab_results(self):
        f = self.tab_results

        # Top metrics strip
        self.metrics_frame = tk.Frame(f, bg="#F5F6FA")
        self.metrics_frame.pack(fill="x", padx=18, pady=(12, 4))
        self.metrics_label = tk.Label(
            self.metrics_frame,
            text="Run a pipeline session to see results.",
            bg="#F5F6FA", fg="#888", font=("Helvetica", 10))
        self.metrics_label.pack(anchor="w")

        # Plot area
        self.plot_frame = tk.Frame(f, bg="#F5F6FA")
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=4)

        placeholder = tk.Label(
            self.plot_frame,
            text="📊  Results will appear here after running the pipeline.",
            bg="#E8ECF5", fg="#888", font=("Helvetica", 11),
            relief="flat")
        placeholder.pack(fill="both", expand=True)
        self._placeholder = placeholder

    def _show_results(self, session):
        # ── Metrics strip ─────────────────────────────────────────────────────
        ep  = session["exp_fit"]
        sp  = session["spoilage"]
        ml  = session["ml_available"]
        sdt = pd.to_datetime(session["daily_table"][0]["Date"])

        lines = [
            f"Session: {session['session_name']}  |  "
            f"Loaded: {session['timestamp']}  |  "
            f"Days: {session['daily_records']}",
        ]
        if ml and session["rf_metrics"]:
            rm = session["rf_metrics"]
            gm = session["gbm_metrics"]
            lines.append(
                f"RF   → LOO-MAE={rm['loo_mae']:.2f} ppm  "
                f"LOO-R²={rm['loo_r2']:.4f}  Train-R²={rm['train_r2']:.4f}")
            lines.append(
                f"GBM  → LOO-MAE={gm['loo_mae']:.2f} ppm  "
                f"LOO-R²={gm['loo_r2']:.4f}  Train-R²={gm['train_r2']:.4f}")
        else:
            lines.append("ML: not available (need ≥3 days of data)")

        lines.append(
            f"Exp: G0={ep['G0']:.2f} ppm  b={ep['b']:+.5f}/day  "
            f"trend={ep['trend']}")

        for lv, thr in [("WARNING", WARNING_PPM), ("CRITICAL", CRITICAL_PPM)]:
            d = sp[lv.lower()]
            if d is not None:
                ed = (sdt + pd.Timedelta(days=d)).date()
                lines.append(f"{lv}: Day {d} ({ed})")
            else:
                lines.append(f"{lv}: Not reached within {FORECAST_DAYS} days")

        self.metrics_label.config(
            text="\n".join(lines), fg="#1A1A2E",
            font=("Courier", 9), justify="left")

        # ── Plot ──────────────────────────────────────────────────────────────
        if self._placeholder:
            self._placeholder.destroy()
            self._placeholder = None
        if self._canvas_widget:
            self._canvas_widget.get_tk_widget().destroy()

        fig = build_figure(session)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas_widget = canvas
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 — History
    # ─────────────────────────────────────────────────────────────────────────
    def _build_tab_history(self):
        f = self.tab_history

        # Toolbar
        toolbar = tk.Frame(f, bg="#F5F6FA")
        toolbar.pack(fill="x", padx=16, pady=(10, 4))
        ttk.Label(toolbar, text="Saved Sessions",
                  style="Header.TLabel",
                  background="#F5F6FA").pack(side="left")
        ttk.Button(toolbar, text="↻  Refresh", style="Small.TButton",
                   command=self._refresh_history).pack(side="right", padx=4)
        ttk.Button(toolbar, text="📂  Load Selected", style="Small.TButton",
                   command=self._load_selected).pack(side="right", padx=4)
        ttk.Button(toolbar, text="🗑  Delete Selected", style="Small.TButton",
                   command=self._delete_selected).pack(side="right", padx=4)

        # Treeview
        cols = ("session", "timestamp", "days", "rf_mae", "gbm_mae",
                "warning_day", "critical_day", "csv")
        self.tree = ttk.Treeview(f, columns=cols, show="headings",
                                 selectmode="browse")
        headers = {
            "session":      ("Session Name",   200),
            "timestamp":    ("Saved At",        150),
            "days":         ("Days",             50),
            "rf_mae":       ("RF MAE",           70),
            "gbm_mae":      ("GBM MAE",          70),
            "warning_day":  ("WARNING Day",      90),
            "critical_day": ("CRITICAL Day",     90),
            "csv":          ("CSV File",         240),
        }
        for col, (heading, width) in headers.items():
            self.tree.heading(col, text=heading)
            self.tree.column(col, width=width, anchor="center")
        self.tree.column("session", anchor="w")
        self.tree.column("csv",     anchor="w")

        vsb = ttk.Scrollbar(f, orient="vertical",   command=self.tree.yview)
        hsb = ttk.Scrollbar(f, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.pack(fill="both", expand=True, padx=16, pady=4, side="top")
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

        self.tree.bind("<Double-1>", lambda e: self._load_selected())

        # Detail box
        det_frame = ttk.Frame(f, style="Card.TFrame", padding=8)
        det_frame.pack(fill="x", padx=16, pady=(4, 12))
        ttk.Label(det_frame, text="Session Detail",
                  font=("Helvetica", 9, "bold")).pack(anchor="w")
        self.detail_box = scrolledtext.ScrolledText(
            det_frame, font=("Courier", 8), height=7,
            bg="#1E1E2E", fg="#C8D8F0", relief="flat", borderwidth=0)
        self.detail_box.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

    def _refresh_history(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        store = load_store(PICKLE_FILE)
        for name, s in sorted(store.items(),
                               key=lambda x: x[1].get("timestamp", ""),
                               reverse=True):
            rf_mae  = (f"{s['rf_metrics']['loo_mae']:.2f}"
                       if s.get("rf_metrics") else "N/A")
            gbm_mae = (f"{s['gbm_metrics']['loo_mae']:.2f}"
                       if s.get("gbm_metrics") else "N/A")
            sp       = s.get("spoilage", {})
            warn_day = sp.get("warning")
            crit_day = sp.get("critical")
            self.tree.insert("", "end", iid=name, values=(
                name,
                s.get("timestamp", ""),
                s.get("daily_records", ""),
                rf_mae,
                gbm_mae,
                f"Day {warn_day}" if warn_day is not None else "Not reached",
                f"Day {crit_day}" if crit_day is not None else "Not reached",
                os.path.basename(s.get("csv_path", "")),
            ))

        count = len(store)
        self.tree.heading("session",
                          text=f"Session Name  ({count} stored)")

    def _on_tree_select(self, _event=None):
        sel = self.tree.selection()
        if not sel:
            return
        name  = sel[0]
        store = load_store(PICKLE_FILE)
        s     = store.get(name)
        if not s:
            return

        ep = s.get("exp_fit", {})
        sp = s.get("spoilage", {})
        lines = [
            f"Session  : {s.get('session_name')}",
            f"Saved    : {s.get('timestamp')}",
            f"CSV      : {s.get('csv_path')}",
            f"Raw rows : {s.get('raw_shape')}  →  clean: {s.get('clean_shape')}",
            f"Days     : {s.get('daily_records')}",
            f"ML avail : {s.get('ml_available')}",
        ]
        if s.get("rf_metrics"):
            rm = s["rf_metrics"]
            lines.append(f"RF  LOO-MAE={rm['loo_mae']:.3f}  "
                         f"LOO-R²={rm['loo_r2']:.4f}  Train-R²={rm['train_r2']:.4f}")
        if s.get("gbm_metrics"):
            gm = s["gbm_metrics"]
            lines.append(f"GBM LOO-MAE={gm['loo_mae']:.3f}  "
                         f"LOO-R²={gm['loo_r2']:.4f}  Train-R²={gm['train_r2']:.4f}")
        lines += [
            f"Exp fit  : G0={ep.get('G0', '?'):.2f} ppm  "
            f"b={ep.get('b', 0):+.5f}/day  trend={ep.get('trend')}",
            f"WARNING  : " + (f"Day {sp['warning']}" if sp.get('warning') is not None else "Not reached"),
            f"CRITICAL : " + (f"Day {sp['critical']}" if sp.get('critical') is not None else "Not reached"),
            "",
            "Feature importances:",
        ]
        fi = s.get("feature_importances", {})
        for feat, imp in sorted(fi.items(), key=lambda x: -x[1]):
            bar = "█" * int(imp * 20)
            lines.append(f"  {feat:12s} {imp:.4f}  {bar}")

        self.detail_box.delete("1.0", "end")
        self.detail_box.insert("end", "\n".join(lines))

    def _load_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Nothing selected",
                                "Click a session in the table first.")
            return
        name  = sel[0]
        store = load_store(PICKLE_FILE)
        s     = store.get(name)
        if not s:
            return
        self._current_session = s
        self._show_results(s)
        # Switch to Results tab
        nb = self.winfo_children()[1]
        nb.select(1)

    def _delete_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Nothing selected",
                                "Click a session in the table first.")
            return
        name = sel[0]
        if not messagebox.askyesno("Delete session",
                                   f"Permanently delete '{name}'?"):
            return
        store = load_store(PICKLE_FILE)
        store.pop(name, None)
        save_store(store, PICKLE_FILE)
        self._refresh_history()
        self.detail_box.delete("1.0", "end")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = AgrinovaApp()
    app.mainloop()