"""
██████╗ ██████╗  ██████╗      ██╗███████╗ ██████╗████████╗    ██████╗ ██████╗ ██╗   ██╗███╗   ███╗
██╔══██╗██╔══██╗██╔═══██╗     ██║██╔════╝██╔════╝╚══██╔══╝    ██╔══██╗██╔══██╗██║   ██║████╗ ████║
██████╔╝██████╔╝██║   ██║     ██║█████╗  ██║        ██║       ██║  ██║██████╔╝██║   ██║██╔████╔██║
██╔═══╝ ██╔══██╗██║   ██║██   ██║██╔══╝  ██║        ██║       ██║  ██║██╔══██╗██║   ██║██║╚██╔╝██║
██║     ██║  ██║╚██████╔╝╚█████╔╝███████╗╚██████╗   ██║       ██████╔╝██║  ██║╚██████╔╝██║ ╚═╝ ██║
╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝       ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝

PROJECT DRUM v4.0 — Enterprise Data Science Dashboard
Features:
  ✅ .ipynb / .csv / .xlsx / .json / .parquet / .py support
  ✅ Full EDA with 2D + 3D visualisations
  ✅ Supervised ML  — regression lines, residuals, prediction bands, CV curves
  ✅ Unsupervised ML — DBSCAN, Agglomerative, K-Means, PCA, t-SNE, UMAP-lite
  ✅ Deep Learning  — Keras MLP (classification & regression) with live loss curves
  ✅ Reinforcement Learning demo (Q-Learning Grid-World) with training plot
  ✅ Smart dataset-aware PDF reports with auto-insights & recommendations
  ✅ SQL Lab, Pipeline tracker, Export suite, Power BI, Python reader
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, re, sys, time, queue, json, pickle, sqlite3, zipfile
import warnings, threading, textwrap, ast, math, traceback
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D          # noqa
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr

import tkinter as tk
from tkinter import scrolledtext, ttk, filedialog, messagebox, simpledialog

warnings.filterwarnings("ignore")

# ── Optional heavy deps ───────────────────────────────────────────────────────
try:
    from sklearn.model_selection import (train_test_split, cross_val_score,
                                          learning_curve, validation_curve)
    from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                                   GradientBoostingClassifier,
                                   GradientBoostingRegressor, IsolationForest,
                                   AdaBoostClassifier, AdaBoostRegressor)
    from sklearn.linear_model import (Ridge, LogisticRegression, Lasso,
                                       LinearRegression, ElasticNet)
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import (accuracy_score, r2_score, confusion_matrix,
                                  classification_report, mean_squared_error,
                                  mean_absolute_error, roc_curve, auc,
                                  precision_recall_curve)
    from sklearn.preprocessing import (LabelEncoder, StandardScaler,
                                        MinMaxScaler, RobustScaler,
                                        PolynomialFeatures)
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.pipeline import Pipeline as SKPipeline
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("⚠  scikit-learn not found — pip install scikit-learn")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_OK = True
    print(f"✅ TensorFlow {tf.__version__} loaded")
except Exception:
    TF_OK = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
    print(f"✅ PyTorch {torch.__version__} loaded")
except Exception:
    TORCH_OK = False

try:
    import pyttsx3
    _ve = pyttsx3.init()
    _ve.setProperty("rate", 170)
    _ve.setProperty("volume", 0.85)
    _vs = _ve.getProperty("voices")
    if len(_vs) > 1:
        _ve.setProperty("voice", _vs[1].id)
    VOICE_OK = True
except Exception:
    VOICE_OK = False

try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

try:
    import nbformat
    NBFORMAT_OK = True
except ImportError:
    NBFORMAT_OK = False
    print("⚠  nbformat not found — pip install nbformat  (needed for .ipynb)")

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
P = {
    "bg":       "#0d1117",
    "surface":  "#161b22",
    "surface2": "#21262d",
    "border":   "#30363d",
    "accent":   "#58a6ff",
    "accent2":  "#3fb950",
    "accent3":  "#d2a8ff",
    "warn":     "#e3b341",
    "danger":   "#f85149",
    "fg":       "#e6edf3",
    "fg_dim":   "#8b949e",
    "header":   "#1c2128",
    "deep":     "#388bfd",
}
PALETTE = P  # alias for compat

CC = ["#58a6ff","#3fb950","#d2a8ff","#e3b341","#f85149",
      "#79c0ff","#56d364","#bc8cff","#ffa657","#ff7b72"]

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def dark_fig(fig):
    fig.patch.set_facecolor(P["surface"])
    for ax in fig.get_axes():
        ax.set_facecolor(P["surface2"])
        ax.tick_params(colors=P["fg_dim"])
        ax.xaxis.label.set_color(P["fg"])
        ax.yaxis.label.set_color(P["fg"])
        ax.title.set_color(P["fg"])
        for sp in ax.spines.values():
            sp.set_edgecolor(P["border"])
    return fig

def embed_fig(fig, parent, toolbar=True):
    """Embed a matplotlib figure into a tk parent widget."""
    for w in parent.winfo_children():
        w.destroy()
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    if toolbar:
        tf = tk.Frame(parent, bg=P["surface"])
        tf.pack(fill=tk.X)
        NavigationToolbar2Tk(canvas, tf)
    return canvas

def dataset_context(df: pd.DataFrame) -> dict:
    """Analyse dataset and return rich context dict for smart reporting."""
    ctx = {}
    ctx["n_rows"]    = len(df)
    ctx["n_cols"]    = len(df.columns)
    ctx["miss_total"]= int(df.isnull().sum().sum())
    ctx["miss_pct"]  = df.isnull().sum().sum() / df.size * 100 if df.size else 0
    ctx["dup_rows"]  = int(df.duplicated().sum())
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category","string"]).columns.tolist()
    dt_cols  = df.select_dtypes(include="datetime").columns.tolist()
    ctx["num_cols"] = num_cols
    ctx["cat_cols"] = cat_cols
    ctx["dt_cols"]  = dt_cols

    # Guess domain
    all_cols = " ".join(df.columns).lower()
    if any(k in all_cols for k in ["salary","wage","income","revenue","profit","price","cost","sales","spend"]):
        ctx["domain"] = "financial"
    elif any(k in all_cols for k in ["age","gender","bmi","blood","patient","diagnosis","disease","health"]):
        ctx["domain"] = "healthcare"
    elif any(k in all_cols for k in ["churn","customer","product","order","purchase","cart","click","session"]):
        ctx["domain"] = "ecommerce_crm"
    elif any(k in all_cols for k in ["temp","humidity","weather","rainfall","wind","pressure"]):
        ctx["domain"] = "environmental"
    elif any(k in all_cols for k in ["fraud","transaction","amount","balance","account","credit","debit"]):
        ctx["domain"] = "fraud_finance"
    elif any(k in all_cols for k in ["score","grade","student","teacher","school","exam","test","marks"]):
        ctx["domain"] = "education"
    elif any(k in all_cols for k in ["employee","department","hire","tenure","attrition","manager"]):
        ctx["domain"] = "hr"
    else:
        ctx["domain"] = "general"

    # Correlations
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs()
        np.fill_diagonal(corr.values, 0)
        max_corr = corr.unstack().sort_values(ascending=False)
        ctx["top_corr"] = max_corr.head(6).to_dict()
    else:
        ctx["top_corr"] = {}

    # Skewness
    ctx["skewed_cols"] = {c: round(df[c].skew(), 3)
                          for c in num_cols if abs(df[c].skew()) > 1.0}
    # High cardinality
    ctx["high_card"] = [c for c in cat_cols if df[c].nunique() > 50]

    return ctx


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═════════════════════════════════════════════════════════════════════════════
class ProjectDrum:

    def __init__(self):
        self.datasets:          dict[str, pd.DataFrame] = {}
        self.active_dataset_name: str | None = None
        self.current_df:          pd.DataFrame | None  = None
        self.current_source:      str | None = None

        self.pipeline_steps:   list = []
        self.model_history:    list = []
        self.chart_windows:    list = []
        self.predictor_entries: dict = {}

        self.current_model             = None
        self.current_model_features:   list = []
        self.current_model_target:     str | None = None
        self.current_model_encoders:   dict = {}
        self._target_encoder           = None
        self._is_clf                   = False
        self._y_test                   = None
        self._preds_test               = None
        self.last_prediction_input     = None

        self.project_title     = "Untitled Data Science Project"
        self.problem_statement = "No problem statement defined yet."

        self.sql_conn: sqlite3.Connection | None = None
        self.gui_queue        = queue.Queue()
        self.workflow_thread  = None
        self._ml_training_done = False
        self._py_filepath: str | None = None

        # Deep learning state
        self.dl_model    = None
        self.dl_history  = None
        self.dl_features: list = []
        self.dl_target:   str | None = None
        self.dl_is_clf    = False
        self.dl_le        = None

        # RL state
        self.rl_rewards: list = []

        self._build_ui()
        self._poll_gui_queue()
        self.say_it("PROJECT DRUM v4 ready.")

    # ── voice / log ──────────────────────────────────────────────────────────
    def say_it(self, text: str, quiet: bool = False):
        print(text)
        if hasattr(self, "log_area") and self.log_area:
            def _log():
                self.log_area.insert(tk.END, f"[{time.strftime('%H:%M:%S')}]  {text}\n")
                self.log_area.see(tk.END)
            if self.root:
                self.root.after(0, _log)
        if VOICE_OK and not quiet:
            clean = re.sub(r"[^\w\s\.]", "", text)
            try:
                _ve.say(clean); _ve.runAndWait()
            except Exception:
                pass

    # ── gui queue ────────────────────────────────────────────────────────────
    def _poll_gui_queue(self):
        try:
            while True:
                task, args, cb_q = self.gui_queue.get_nowait()
                result = task(*args)
                if cb_q:
                    cb_q.put(result)
        except queue.Empty:
            pass
        finally:
            try:
                if self.root and self.root.winfo_exists():
                    self.root.after(100, self._poll_gui_queue)
            except Exception:
                pass

    def _queue_gui_task(self, task, *args):
        cb_q = queue.Queue()
        self.gui_queue.put((task, args, cb_q))
        return cb_q.get()

    def get_gui_input(self, prompt, title="Input"):
        return self._queue_gui_task(simpledialog.askstring, title, prompt)

    def get_gui_choice(self, prompt, options, title="Select"):
        def _task():
            popup = tk.Toplevel(self.root)
            popup.title(title); popup.geometry("460x360")
            popup.configure(bg=P["surface"]); popup.grab_set()
            selected = tk.StringVar()
            tk.Label(popup, text=prompt, font=("Segoe UI",11,"bold"),
                     bg=P["surface"], fg=P["fg"], wraplength=420, pady=12).pack()
            for i, opt in enumerate(options, 1):
                tk.Button(popup, text=f"  {i}.  {opt}",
                          font=("Segoe UI",10), anchor="w",
                          bg=P["surface2"], fg=P["fg"],
                          activebackground=P["accent"], relief=tk.FLAT,
                          bd=0, pady=8,
                          command=lambda v=str(i): [selected.set(v), popup.destroy()]
                          ).pack(fill=tk.X, padx=20, pady=3)
            self.root.wait_window(popup)
            return selected.get()
        return self._queue_gui_task(_task)

    def get_file_path(self, file_types):
        def _task():
            return filedialog.askopenfilename(parent=self.root, filetypes=file_types)
        return self._queue_gui_task(_task)

    # ══════════════════════════════════════════════════════════════════════════
    #  BUILD UI
    # ══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("PROJECT DRUM v4.0  |  Enterprise Data Science Dashboard")
        self.root.geometry("1440x900")
        self.root.configure(bg=P["bg"])

        style = ttk.Style(); style.theme_use("clam")
        style.configure("TNotebook",        background=P["bg"],        borderwidth=0)
        style.configure("TNotebook.Tab",    background=P["surface2"],  foreground=P["fg_dim"],
                        font=("Segoe UI",9,"bold"), padding=[12,5])
        style.map("TNotebook.Tab",
                  background=[("selected",P["accent"])],
                  foreground=[("selected","#000000")])
        style.configure("TFrame",           background=P["bg"])
        style.configure("TCombobox",        fieldbackground=P["surface2"],
                        background=P["surface2"], foreground=P["fg"])
        style.configure("Treeview",         background=P["surface2"], fieldbackground=P["surface2"],
                        foreground=P["fg"], rowheight=24)
        style.configure("Treeview.Heading", background=P["header"],   foreground=P["accent"],
                        font=("Segoe UI",9,"bold"))
        style.configure("Vertical.TScrollbar",   troughcolor=P["surface"], background=P["border"])
        style.configure("Horizontal.TScrollbar", troughcolor=P["surface"], background=P["border"])

        # ── HEADER ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=P["header"], pady=8); hdr.pack(fill=tk.X)
        tk.Label(hdr, text="🥁  PROJECT DRUM  v4.0",
                 font=("Consolas",15,"bold"),
                 bg=P["header"], fg=P["accent"]).pack(side=tk.LEFT, padx=16)

        def _btn(parent, text, cmd, color, side=tk.LEFT):
            b = tk.Button(parent, text=text, command=cmd, bg=color,
                          fg="#000" if color in (P["accent2"],P["warn"]) else "#fff",
                          font=("Segoe UI",9,"bold"), relief=tk.FLAT,
                          padx=12, pady=5, cursor="hand2")
            b.pack(side=side, padx=4); return b

        self.add_btn  = _btn(hdr, "➕ Add Dataset",  self.begin_project, P["accent2"])
        self.save_btn = _btn(hdr, "💾 Save Session", self.save_session,  P["accent"])
        self.load_btn = _btn(hdr, "📂 Load Session", self.load_session,  P["accent"])

        tk.Label(hdr, text="Active Dataset:", bg=P["header"], fg=P["fg_dim"],
                 font=("Segoe UI",9)).pack(side=tk.LEFT, padx=(18,4))
        self.dataset_selector = ttk.Combobox(hdr, state="readonly", width=28)
        self.dataset_selector.pack(side=tk.LEFT, padx=4)
        self.dataset_selector.bind("<<ComboboxSelected>>", self._on_dataset_switch)

        # version badges
        badges = []
        if SKLEARN_OK: badges.append("✅ sklearn")
        if TF_OK:      badges.append("✅ TensorFlow")
        if TORCH_OK:   badges.append("✅ PyTorch")
        if NBFORMAT_OK:badges.append("✅ nbformat")
        if badges:
            tk.Label(hdr, text="  ".join(badges),
                     bg=P["header"], fg=P["accent2"],
                     font=("Segoe UI",8)).pack(side=tk.RIGHT, padx=14)

        # ── BODY ──────────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=P["bg"]); body.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        log_frame = tk.LabelFrame(body, text=" 📋 System Log ",
                                  font=("Segoe UI",9,"bold"),
                                  bg=P["surface"], fg=P["accent"], bd=1, relief=tk.FLAT, width=310)
        log_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,6)); log_frame.pack_propagate(False)
        self.log_area = scrolledtext.ScrolledText(log_frame, width=34, height=52,
                                                   font=("Consolas",8),
                                                   bg=P["surface"], fg=P["accent2"],
                                                   insertbackground=P["fg"], bd=0)
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        right = tk.Frame(body, bg=P["bg"]); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.notebook = ttk.Notebook(right); self.notebook.pack(fill=tk.BOTH, expand=True)

        self._build_data_tab()
        self._build_stats_tab()
        self._build_visuals_tab()
        self._build_3d_tab()
        self._build_eda_tab()
        self._build_sql_tab()
        self._build_custom_chart_tab()
        self._build_ml_tab()
        self._build_deep_tab()
        self._build_rl_tab()
        self._build_clean_tab()
        self._build_export_tab()
        self._build_pipeline_tab()
        self._build_quality_tab()
        self._build_predictor_tab()
        self._build_story_tab()
        self._build_profiler_tab()
        self._build_comparison_tab()
        self._build_python_tab()

    # ── tab / widget helpers ─────────────────────────────────────────────────
    def _tab(self, label):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=label)
        inner = tk.Frame(frame, bg=P["bg"]); inner.pack(fill=tk.BOTH, expand=True)
        return inner

    def _sec(self, parent, text):
        tk.Label(parent, text=text, font=("Segoe UI",13,"bold"),
                 bg=P["bg"], fg=P["accent"]).pack(anchor="w", padx=14, pady=(10,2))

    def _scroll(self, parent, **kw):
        return scrolledtext.ScrolledText(parent, font=("Consolas",9),
                                         bg=P["surface"], fg=P["fg"],
                                         insertbackground=P["fg"], bd=0, **kw)

    def _tree(self, parent):
        c = tk.Frame(parent, bg=P["bg"]); c.pack(fill=tk.BOTH, expand=True)
        tree = ttk.Treeview(c, show="headings")
        vsb  = ttk.Scrollbar(c, orient="vertical",   command=tree.yview)
        hsb  = ttk.Scrollbar(c, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb .grid(row=0, column=1, sticky="ns")
        hsb .grid(row=1, column=0, sticky="ew")
        c.grid_columnconfigure(0, weight=1); c.grid_rowconfigure(0, weight=1)
        return tree

    def _btn(self, parent, text, cmd, color=None, side=tk.LEFT, **kw):
        color = color or P["accent"]
        b = tk.Button(parent, text=text, command=cmd, bg=color,
                      fg="#000" if color == P["accent2"] else "#fff",
                      font=("Segoe UI",9,"bold"), relief=tk.FLAT,
                      padx=12, pady=6, cursor="hand2", **kw)
        b.pack(side=side, padx=5, pady=4); return b

    # ══════════════════════════════════════════════════════════════════════════
    #  TABS
    # ══════════════════════════════════════════════════════════════════════════
    def _build_data_tab(self):
        p = self._tab("📊 Data")
        self._sec(p, "Dataset Preview")
        self.data_tree = self._tree(p)

    def _build_stats_tab(self):
        p = self._tab("📈 Statistics")
        self._sec(p, "Statistical Report")
        bar = tk.Frame(p, bg=P["bg"]); bar.pack(fill=tk.X, padx=10)
        self._btn(bar, "🔬 Hypothesis Testing",      self._run_statistical_deep_dive, P["accent3"])
        self._btn(bar, "🔗 Cross-Dataset Influence", self._run_cross_dataset_analysis, P["accent2"])
        self._btn(bar, "📐 Full Descriptive Stats",  self._run_full_stats, P["accent"])
        self.stats_log = self._scroll(p); self.stats_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

    def _build_visuals_tab(self):
        p = self._tab("🎨 Visuals")
        self._sec(p, "Automated Summary Charts")
        bar = tk.Frame(p, bg=P["bg"]); bar.pack(fill=tk.X, padx=10)
        self._btn(bar, "🪟 Pop-out Charts", lambda: self._plot_summary(pop_out=True), P["accent"])
        self._btn(bar, "🔄 Refresh",        lambda: self._plot_summary(), P["accent2"])
        self.visuals_container = tk.Frame(p, bg=P["surface"]); self.visuals_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        ins = tk.LabelFrame(p, text=" 💡 Auto Insights ", font=("Segoe UI",9,"bold"),
                            bg=P["surface"], fg=P["warn"], bd=1)
        ins.pack(fill=tk.X, padx=10, pady=4)
        self.insight_log = scrolledtext.ScrolledText(ins, height=5, font=("Segoe UI",9),
                                                     bg=P["surface"], fg=P["warn"], bd=0)
        self.insight_log.pack(fill=tk.X, padx=4, pady=4)

    def _build_3d_tab(self):
        p = self._tab("🌐 3D Visuals")
        self._sec(p, "3D Exploration Charts")
        ctrl = tk.Frame(p, bg=P["bg"]); ctrl.pack(fill=tk.X, padx=14, pady=6)
        for lbl, attr in [("X:","d3_x"),("Y:","d3_y"),("Z:","d3_z")]:
            tk.Label(ctrl, text=lbl, bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
            cb = ttk.Combobox(ctrl, state="readonly", width=16); cb.pack(side=tk.LEFT, padx=4)
            setattr(self, attr, cb)
        tk.Label(ctrl, text="Type:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT, padx=(8,0))
        self.d3_type = ttk.Combobox(ctrl, state="readonly", width=16,
                                    values=["3D Scatter","3D Surface","3D Bar","3D Line","3D Wireframe","3D Trisurf"])
        self.d3_type.set("3D Scatter"); self.d3_type.pack(side=tk.LEFT, padx=4)
        tk.Label(ctrl, text="Color by:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT, padx=(8,0))
        self.d3_color_by = ttk.Combobox(ctrl, state="readonly", width=16); self.d3_color_by.pack(side=tk.LEFT, padx=4)
        self._btn(ctrl, "🚀 Generate", self._generate_3d_chart, P["accent3"])
        self._btn(ctrl, "🪟 Pop-out",  self._pop_3d, P["accent"])
        self.d3_container = tk.Frame(p, bg=P["surface"]); self.d3_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

    def _build_eda_tab(self):
        p = self._tab("🔭 EDA Suite")
        self._sec(p, "Advanced Exploratory Data Analysis")
        bar = tk.Frame(p, bg=P["bg"]); bar.pack(fill=tk.X, padx=10)
        self._btn(bar, "📊 Distribution Grid",  self._eda_distribution_grid, P["accent"])
        self._btn(bar, "🌡️ Heatmap Suite",      self._eda_heatmap_suite,     P["accent3"])
        self._btn(bar, "🎻 Violin + Swarm",      self._eda_violin_swarm,      P["accent2"])
        self._btn(bar, "📉 Pairplot",            self._eda_pairplot,          P["warn"])
        self._btn(bar, "📦 Missing Heatmap",     self._eda_missing_heatmap,   P["danger"])
        self._btn(bar, "🔢 Numeric Scatter Matrix", self._eda_scatter_matrix, P["deep"])
        self.eda_container = tk.Frame(p, bg=P["surface"]); self.eda_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

    def _build_sql_tab(self):
        p = self._tab("🗄️ SQL Lab")
        self._sec(p, "SQL Query Lab  (table = 'data')")
        pane = tk.Frame(p, bg=P["bg"]); pane.pack(fill=tk.BOTH, expand=True, padx=10)
        sidebar = tk.Frame(pane, bg=P["surface2"], width=210)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0,8)); sidebar.pack_propagate(False)
        tk.Label(sidebar, text="📐 Schema", font=("Segoe UI",9,"bold"),
                 bg=P["surface2"], fg=P["accent"]).pack(pady=6)
        self.schema_tree = ttk.Treeview(sidebar, show="tree")
        self.schema_tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._btn(sidebar, "🔄 Refresh", self._refresh_sql_schema, P["accent"])
        right = tk.Frame(pane, bg=P["bg"]); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        le = tk.LabelFrame(right, text=" ✍️ SQL Editor ", font=("Segoe UI",9,"bold"),
                           bg=P["surface"], fg=P["accent2"], bd=1)
        le.pack(fill=tk.BOTH, expand=True, pady=(0,4))
        self.sql_editor = scrolledtext.ScrolledText(le, height=9, font=("Consolas",11),
                                                     bg="#0d1117", fg=P["accent"],
                                                     insertbackground=P["fg"], bd=0)
        self.sql_editor.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.sql_editor.insert("1.0","SELECT * FROM data LIMIT 20;")
        bb = tk.Frame(right, bg=P["bg"]); bb.pack(fill=tk.X, pady=2)
        self._btn(bb, "▶ Run",    self._execute_gui_sql, P["accent2"])
        self._btn(bb, "🗑 Clear", lambda: self.sql_editor.delete("1.0",tk.END), P["danger"])
        rf = tk.LabelFrame(right, text=" 📊 Results ", font=("Segoe UI",9,"bold"),
                           bg=P["surface"], fg=P["accent2"], bd=1)
        rf.pack(fill=tk.BOTH, expand=True, pady=4)
        self.sql_tree = self._tree(rf)

    def _build_custom_chart_tab(self):
        p = self._tab("🎭 Custom Charts")
        self._sec(p, "Custom Chart Builder")
        ctrl = tk.Frame(p, bg=P["bg"]); ctrl.pack(fill=tk.X, padx=14, pady=6)
        for lbl, attr in [("X:","x_combo"),("Y:","y_combo")]:
            tk.Label(ctrl, text=lbl, bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
            cb = ttk.Combobox(ctrl, state="readonly", width=16); cb.pack(side=tk.LEFT, padx=4)
            setattr(self, attr, cb)
        tk.Label(ctrl, text="Type:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.type_combo = ttk.Combobox(ctrl, state="readonly", width=14,
                                       values=["Scatter","Bar","Line","Histogram","Pie","KDE","Box","Area","Step"])
        self.type_combo.set("Scatter"); self.type_combo.pack(side=tk.LEFT, padx=4)
        tk.Label(ctrl, text="Color:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.color_combo = ttk.Combobox(ctrl, state="readonly", width=12,
                                        values=["steelblue","mediumseagreen","mediumpurple","tomato","goldenrod"])
        self.color_combo.set("steelblue"); self.color_combo.pack(side=tk.LEFT, padx=4)
        tk.Label(ctrl, text="Title:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.title_entry = tk.Entry(ctrl, width=18, bg=P["surface2"], fg=P["fg"], insertbackground=P["fg"])
        self.title_entry.insert(0,"Custom Chart"); self.title_entry.pack(side=tk.LEFT, padx=4)
        self._btn(ctrl, "🎨 Generate", self._generate_custom_chart, P["accent3"])
        self._btn(ctrl, "🗑 Close All", self._close_all_charts, P["danger"])

    def _build_ml_tab(self):
        p = self._tab("🤖 ML Lab")
        self._sec(p, "Machine Learning Lab — Full Visualisation")
        ctrl = tk.Frame(p, bg=P["bg"]); ctrl.pack(fill=tk.X, padx=14)
        tk.Label(ctrl, text="Target:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.ml_target_combo = ttk.Combobox(ctrl, state="readonly", width=20); self.ml_target_combo.pack(side=tk.LEFT, padx=6)
        tk.Label(ctrl, text="Algorithm:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.ml_algo_combo = ttk.Combobox(ctrl, state="readonly", width=22,
                                           values=["Random Forest","Gradient Boosting","Logistic / Linear Regression",
                                                   "SVM","KNN","Decision Tree","AdaBoost","Naive Bayes","ElasticNet"])
        self.ml_algo_combo.set("Random Forest"); self.ml_algo_combo.pack(side=tk.LEFT, padx=6)
        self._btn(ctrl, "🚀 Train",    self._run_gui_ml,                    P["warn"])
        self._btn(ctrl, "⚗️ Auto-ML",  lambda: self._run_automl_optimization(self.current_df) if self.current_df is not None else None, P["accent3"])
        self._btn(ctrl, "📈 Learning Curve",    self._plot_learning_curve,   P["accent"])
        self._btn(ctrl, "🔁 K-Fold CV",         self._plot_cv_scores,        P["deep"])
        self._btn(ctrl, "💾 Save Model",        self._gui_save_model,        P["accent2"])
        panes = tk.Frame(p, bg=P["bg"]); panes.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.ml_metrics_area = self._scroll(panes, width=38); self.ml_metrics_area.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0,6))
        self.ml_plot_container = tk.Frame(panes, bg=P["surface"]); self.ml_plot_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        hf = tk.LabelFrame(p, text=" 📜 Training History ", font=("Segoe UI",9,"bold"),
                           bg=P["surface"], fg=P["accent2"], bd=1)
        hf.pack(fill=tk.X, padx=10, pady=4)
        self.ml_history_area = self._scroll(hf, height=4); self.ml_history_area.pack(fill=tk.X, padx=4, pady=4)

    def _build_deep_tab(self):
        p = self._tab("🧠 Deep Learning")
        self._sec(p, "Neural Network Training (Keras / PyTorch)")

        ctrl = tk.Frame(p, bg=P["bg"]); ctrl.pack(fill=tk.X, padx=14, pady=6)
        tk.Label(ctrl, text="Target:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.dl_target_combo = ttk.Combobox(ctrl, state="readonly", width=20); self.dl_target_combo.pack(side=tk.LEFT, padx=6)

        tk.Label(ctrl, text="Layers:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.dl_layers_entry = tk.Entry(ctrl, width=14, bg=P["surface2"], fg=P["fg"], insertbackground=P["fg"])
        self.dl_layers_entry.insert(0,"128,64,32"); self.dl_layers_entry.pack(side=tk.LEFT, padx=4)

        tk.Label(ctrl, text="Epochs:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.dl_epochs_entry = tk.Entry(ctrl, width=6, bg=P["surface2"], fg=P["fg"], insertbackground=P["fg"])
        self.dl_epochs_entry.insert(0,"50"); self.dl_epochs_entry.pack(side=tk.LEFT, padx=4)

        tk.Label(ctrl, text="LR:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.dl_lr_entry = tk.Entry(ctrl, width=8, bg=P["surface2"], fg=P["fg"], insertbackground=P["fg"])
        self.dl_lr_entry.insert(0,"0.001"); self.dl_lr_entry.pack(side=tk.LEFT, padx=4)

        tk.Label(ctrl, text="Dropout:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.dl_dropout_entry = tk.Entry(ctrl, width=6, bg=P["surface2"], fg=P["fg"], insertbackground=P["fg"])
        self.dl_dropout_entry.insert(0,"0.2"); self.dl_dropout_entry.pack(side=tk.LEFT, padx=4)

        tk.Label(ctrl, text="Framework:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.dl_framework = ttk.Combobox(ctrl, state="readonly", width=12,
                                          values=["TensorFlow/Keras","PyTorch (manual)"])
        self.dl_framework.set("TensorFlow/Keras"); self.dl_framework.pack(side=tk.LEFT, padx=4)

        self._btn(ctrl, "🚀 Train DL", self._train_deep_learning, P["accent3"])
        self._btn(ctrl, "🔮 DL Predict", self._dl_predict_tab, P["warn"])

        panes = tk.Frame(p, bg=P["bg"]); panes.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.dl_metrics_area = self._scroll(panes, width=40); self.dl_metrics_area.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0,6))
        self.dl_plot_container = tk.Frame(panes, bg=P["surface"]); self.dl_plot_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def _build_rl_tab(self):
        p = self._tab("🎮 Reinf. Learning")
        self._sec(p, "Reinforcement Learning — Q-Learning Grid-World + Custom Env")

        ctrl = tk.Frame(p, bg=P["bg"]); ctrl.pack(fill=tk.X, padx=14, pady=6)
        tk.Label(ctrl, text="Grid Size:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.rl_grid_entry = tk.Entry(ctrl, width=6, bg=P["surface2"], fg=P["fg"], insertbackground=P["fg"])
        self.rl_grid_entry.insert(0,"6"); self.rl_grid_entry.pack(side=tk.LEFT, padx=4)

        tk.Label(ctrl, text="Episodes:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.rl_episodes_entry = tk.Entry(ctrl, width=8, bg=P["surface2"], fg=P["fg"], insertbackground=P["fg"])
        self.rl_episodes_entry.insert(0,"2000"); self.rl_episodes_entry.pack(side=tk.LEFT, padx=4)

        tk.Label(ctrl, text="ε:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.rl_eps_entry = tk.Entry(ctrl, width=6, bg=P["surface2"], fg=P["fg"], insertbackground=P["fg"])
        self.rl_eps_entry.insert(0,"1.0"); self.rl_eps_entry.pack(side=tk.LEFT, padx=4)

        tk.Label(ctrl, text="Mode:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.rl_mode = ttk.Combobox(ctrl, state="readonly", width=20,
                                     values=["Q-Learning Grid-World","Tabular Cliff Walking","Multi-Armed Bandit"])
        self.rl_mode.set("Q-Learning Grid-World"); self.rl_mode.pack(side=tk.LEFT, padx=4)

        self._btn(ctrl, "▶ Run RL", self._run_rl_training, P["accent2"])
        self._btn(ctrl, "🗺️ Show Policy",self._show_rl_policy, P["accent"])

        panes = tk.Frame(p, bg=P["bg"]); panes.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.rl_metrics_area = self._scroll(panes, width=36); self.rl_metrics_area.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0,6))
        self.rl_plot_container = tk.Frame(panes, bg=P["surface"]); self.rl_plot_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def _build_clean_tab(self):
        p = self._tab("🧹 Cleaning")
        self._sec(p, "Data Cleaning & Feature Engineering")
        df = tk.LabelFrame(p, text=" 🗑 Drop Columns ", font=("Segoe UI",9,"bold"),
                           bg=P["surface"], fg=P["danger"], bd=1)
        df.pack(fill=tk.X, padx=14, pady=6)
        tk.Label(df, text="Ctrl+click for multi-select:", bg=P["surface"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT, padx=8)
        self.drop_listbox = tk.Listbox(df, selectmode=tk.MULTIPLE, height=4,
                                       bg=P["surface2"], fg=P["fg"],
                                       selectbackground=P["accent"], bd=0)
        self.drop_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self._btn(df, "Remove Selected", self._gui_drop_cols, P["danger"])

        of = tk.LabelFrame(p, text=" 📉 Outlier Removal ", font=("Segoe UI",9,"bold"),
                           bg=P["surface"], fg=P["warn"], bd=1)
        of.pack(fill=tk.X, padx=14, pady=4)
        tk.Label(of, text="Z-Score threshold:", bg=P["surface"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT, padx=8)
        self.outlier_threshold = tk.Entry(of, width=7, bg=P["surface2"], fg=P["fg"], insertbackground=P["fg"])
        self.outlier_threshold.insert(0,"3.0"); self.outlier_threshold.pack(side=tk.LEFT, padx=6)
        self._btn(of, "Remove Outliers", self._gui_remove_outliers, P["warn"])

        sf = tk.LabelFrame(p, text=" ⚖️ Feature Scaling ", font=("Segoe UI",9,"bold"),
                           bg=P["surface"], fg=P["accent"], bd=1)
        sf.pack(fill=tk.X, padx=14, pady=4)
        tk.Label(sf, text="Method:", bg=P["surface"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT, padx=8)
        self.scaling_method = ttk.Combobox(sf, state="readonly", width=28,
                                            values=["StandardScaler (Z-Score)","MinMaxScaler (0–1)","RobustScaler (IQR)"])
        self.scaling_method.set("StandardScaler (Z-Score)"); self.scaling_method.pack(side=tk.LEFT, padx=6)
        self._btn(sf, "Apply Scaling", self._gui_apply_scaling, P["accent"])

        fe = tk.LabelFrame(p, text=" 🛠 Auto Feature Engineering ", font=("Segoe UI",9,"bold"),
                           bg=P["surface"], fg=P["accent2"], bd=1)
        fe.pack(fill=tk.X, padx=14, pady=4)
        self._btn(fe, "Run Auto Feat-Eng", self._gui_auto_feat_eng, P["accent2"])
        self._btn(fe, "🔢 Polynomial (deg 2)", self._gui_poly_features, P["accent"])
        self._btn(fe, "🏷️ One-Hot Encode Cats", self._gui_one_hot, P["accent3"])

        pf = tk.LabelFrame(p, text=" 👁 Live Preview ", font=("Segoe UI",9,"bold"),
                           bg=P["surface"], fg=P["fg_dim"], bd=1)
        pf.pack(fill=tk.BOTH, expand=True, padx=14, pady=6)
        self.clean_tree = self._tree(pf)

    def _build_export_tab(self):
        p = self._tab("📤 Export")
        self._sec(p, "Export & Reporting Suite")
        fr = tk.Frame(p, bg=P["bg"]); fr.pack(fill=tk.X, padx=14, pady=6)
        for lbl, fmt in [("📄 CSV","csv"),("📊 Excel","excel"),("🗃 JSON","json")]:
            self._btn(fr, lbl, lambda f=fmt: self._export_data(f), P["accent2"])
        self._btn(fr, "🏛 SQL DDL",        self._export_sql_schema,      P["accent"])
        self._btn(fr, "📖 Data Dictionary", self._export_data_dictionary, P["accent3"])
        self._btn(fr, "📦 Full ZIP",        self._export_project_package, P["danger"])
        pf = tk.LabelFrame(p, text=" ⚡ Power BI Integration ", font=("Segoe UI",9,"bold"),
                           bg=P["surface"], fg=P["warn"], bd=1)
        pf.pack(fill=tk.X, padx=14, pady=8)
        fr2 = tk.Frame(pf, bg=P["surface"]); fr2.pack(pady=6)
        self._btn(fr2, "🚀 PBI CSV Export",  self._export_powerbi_ready,    P["warn"])
        self._btn(fr2, "📜 Copy PBI Script", self._generate_powerbi_script, P["accent"])
        self._btn(fr2, "📅 Calendar Table",  self._generate_calendar_table, P["accent2"])
        self._btn(fr2, "🎨 PBI Theme JSON",  self._export_pbi_theme,        P["accent3"])

    def _build_pipeline_tab(self):
        p = self._tab("⛓ Pipeline")
        self._sec(p, "Analytical Pipeline Tracker")
        self.pipeline_area = self._scroll(p); self.pipeline_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        bar = tk.Frame(p, bg=P["bg"]); bar.pack(fill=tk.X, padx=10)
        self._btn(bar, "📜 Export .py Script", self._export_pipeline_script, P["accent"])
        self._btn(bar, "🗑 Clear Pipeline",     self._clear_pipeline,         P["danger"])

    def _build_quality_tab(self):
        p = self._tab("🧪 Data Health")
        self._sec(p, "Data Quality Dashboard")
        self.quality_score_label = tk.Label(p, text="Quality Score: N/A",
                                            font=("Consolas",22,"bold"),
                                            bg=P["bg"], fg=P["accent2"])
        self.quality_score_label.pack(pady=10)
        self.quality_log = self._scroll(p, height=16); self.quality_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        bar = tk.Frame(p, bg=P["bg"]); bar.pack(fill=tk.X, padx=10)
        self._btn(bar, "🔍 Anomaly Detection",   self._run_anomaly_detection, P["warn"])
        self._btn(bar, "📊 Quality Chart",        self._quality_chart,         P["accent"])

    def _build_predictor_tab(self):
        p = self._tab("🔮 Predictor")
        self._sec(p, "Live Prediction Lab")
        self.predictor_status = tk.Label(p, text="No model trained yet.",
                                         font=("Segoe UI",11,"bold"), bg=P["bg"], fg=P["danger"])
        self.predictor_status.pack(pady=8)
        self.predictor_input_container = tk.Frame(p, bg=P["bg"]); self.predictor_input_container.pack(fill=tk.BOTH, expand=True, padx=14)
        bb = tk.Frame(p, bg=P["bg"]); bb.pack()
        self._btn(bb, "🚀 Predict Now",         self._run_live_prediction, P["accent2"])
        self._btn(bb, "💡 Explain (XAI)",       self._explain_prediction,  P["accent"])
        self._btn(bb, "📊 Partial Dependence",  self._plot_partial_dependence, P["accent3"])
        self.prediction_result_label = tk.Label(p, text="",
                                                font=("Consolas",18,"bold"), bg=P["bg"], fg=P["accent2"])
        self.prediction_result_label.pack(pady=16)

    def _build_story_tab(self):
        p = self._tab("📖 Storyteller")
        self._sec(p, "AI-Powered Data Story & Smart Report")
        self._btn(p, "✍️ Generate Data Story", self._generate_data_story, P["accent3"])
        self._btn(p, "📄 Export PDF Report",   self._export_smart_pdf,    P["danger"])
        self.story_area = scrolledtext.ScrolledText(p, font=("Segoe UI",10),
                                                    bg=P["surface"], fg=P["fg"],
                                                    insertbackground=P["fg"], bd=0, wrap=tk.WORD)
        self.story_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

    def _build_profiler_tab(self):
        p = self._tab("🕵 Profiler")
        self._sec(p, "Advanced Column Profiler")
        self._btn(p, "🔍 Run Profiler", self._run_profiler_analysis, P["accent"])
        cols = ["Column","Type","Non-Null","Nulls%","Unique","Mean","Median","Min","Max","Skewness","Kurtosis"]
        self.profiler_tree = self._tree(p)
        self.profiler_tree.master.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        self.profiler_tree["columns"] = cols
        for c in cols:
            self.profiler_tree.heading(c, text=c)
            self.profiler_tree.column(c, width=90, anchor="center")

    def _build_comparison_tab(self):
        p = self._tab("⚖️ Comparison")
        self._sec(p, "Side-by-Side Dataset Comparison")
        ctrl = tk.Frame(p, bg=P["bg"]); ctrl.pack(fill=tk.X, padx=14, pady=6)
        tk.Label(ctrl, text="Dataset 1:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.comp_ds1 = ttk.Combobox(ctrl, state="readonly", width=22); self.comp_ds1.pack(side=tk.LEFT, padx=6)
        tk.Label(ctrl, text="Dataset 2:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.comp_ds2 = ttk.Combobox(ctrl, state="readonly", width=22); self.comp_ds2.pack(side=tk.LEFT, padx=6)
        self._btn(ctrl, "📊 Compare", self._run_comparison_analysis, P["accent"])
        self.comp_log = self._scroll(p, height=8); self.comp_log.pack(fill=tk.X, padx=10, pady=4)
        self.comp_plot_container = tk.Frame(p, bg=P["surface"]); self.comp_plot_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

    def _build_python_tab(self):
        p = self._tab("🐍 Python Reader")
        self._sec(p, "Python + Notebook File Inspector & Runner")
        bar = tk.Frame(p, bg=P["bg"]); bar.pack(fill=tk.X, padx=14, pady=6)
        self._btn(bar, "📂 Open .py / .ipynb", self._load_python_file, P["accent2"])
        self._btn(bar, "▶ Run Python File",    self._run_python_file,  P["warn"])
        self._btn(bar, "🔍 Analyse AST",       self._analyse_python_ast, P["accent3"])
        self._btn(bar, "📓 Extract Notebook Data", self._extract_notebook_to_dataset, P["accent"])
        self._btn(bar, "🗑 Clear",              lambda: (self.py_editor.delete("1.0",tk.END),
                                                         self.py_output.delete("1.0",tk.END)), P["danger"])
        panes = tk.PanedWindow(p, orient=tk.HORIZONTAL, bg=P["bg"], sashwidth=6)
        panes.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        ef = tk.LabelFrame(panes, text=" 📝 Source ", font=("Segoe UI",9,"bold"),
                           bg=P["surface"], fg=P["accent2"], bd=1)
        self.py_editor = scrolledtext.ScrolledText(ef, font=("Consolas",10),
                                                    bg="#0d1117", fg="#79c0ff",
                                                    insertbackground=P["fg"], bd=0, tabs=("1c",))
        self.py_editor.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        panes.add(ef, minsize=400)
        of = tk.LabelFrame(panes, text=" 📤 Output ", font=("Segoe UI",9,"bold"),
                           bg=P["surface"], fg=P["accent2"], bd=1)
        self.py_output = scrolledtext.ScrolledText(of, font=("Consolas",9),
                                                    bg=P["surface"], fg=P["accent2"],
                                                    insertbackground=P["fg"], bd=0)
        self.py_output.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        panes.add(of, minsize=300)

    # ══════════════════════════════════════════════════════════════════════════
    #  FILE LOADING — supports .ipynb, .csv, .xlsx, .json, .parquet, .py
    # ══════════════════════════════════════════════════════════════════════════
    def _load_file(self, path: str) -> pd.DataFrame | None:
        path = path.strip("'\" ")
        ext  = os.path.splitext(path)[1].lower()
        try:
            if ext == ".csv":
                # Try different encodings and separators
                for enc in ("utf-8","latin-1","cp1252"):
                    try:
                        df = pd.read_csv(path, encoding=enc, low_memory=False)
                        if len(df.columns) == 1:
                            # Try semicolon
                            df2 = pd.read_csv(path, sep=";", encoding=enc, low_memory=False)
                            if len(df2.columns) > 1:
                                df = df2
                        return df
                    except UnicodeDecodeError:
                        continue
                return pd.read_csv(path, encoding="latin-1", errors="replace", low_memory=False)

            elif ext in (".xlsx",".xls"):
                return pd.read_excel(path)

            elif ext == ".json":
                try:
                    return pd.read_json(path)
                except Exception:
                    with open(path,"r") as f:
                        raw = json.load(f)
                    if isinstance(raw, list):
                        return pd.json_normalize(raw)
                    elif isinstance(raw, dict):
                        return pd.json_normalize([raw])
                    return pd.DataFrame([raw])

            elif ext == ".parquet":
                return pd.read_parquet(path)

            elif ext == ".ipynb":
                return self._load_ipynb(path)

            elif ext == ".py":
                # Load .py as a structured DataFrame AND open in Python reader
                with open(path,"r",encoding="utf-8",errors="replace") as f:
                    src = f.read()
                lines = src.splitlines()
                records = [{"line_no": i+1, "content": ln, "length": len(ln),
                            "is_comment": ln.strip().startswith("#"),
                            "is_blank":   ln.strip() == "",
                            "indent":     len(ln) - len(ln.lstrip())}
                           for i, ln in enumerate(lines)]
                return pd.DataFrame(records)

            else:
                messagebox.showerror("Unsupported", f"File type '{ext}' is not supported yet.")
                return None
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            return None

    def _load_ipynb(self, path: str) -> pd.DataFrame:
        """
        Load a Jupyter Notebook:
        1. If nbformat is available, properly parse cells.
        2. Otherwise, fall back to raw JSON parsing.
        Returns a DataFrame of cells for exploration.
        """
        if NBFORMAT_OK:
            with open(path,"r",encoding="utf-8",errors="replace") as f:
                nb = nbformat.read(f, as_version=4)
            records = []
            for i, cell in enumerate(nb.cells):
                src    = "".join(cell["source"])
                output = ""
                for out in cell.get("outputs",[]):
                    if "text" in out:
                        output += "".join(out["text"])
                    elif "data" in out:
                        txt = out["data"].get("text/plain","")
                        if isinstance(txt, list):
                            txt = "".join(txt)
                        output += txt
                records.append({
                    "cell_index":  i,
                    "cell_type":   cell["cell_type"],
                    "source":      src,
                    "output":      output.strip()[:500],
                    "line_count":  src.count("\n") + 1,
                    "char_count":  len(src),
                    "has_output":  bool(output.strip()),
                    "has_import":  "import" in src,
                    "has_plot":    any(k in src for k in ["plot","figure","show","plt","sns","px"]),
                    "has_model":   any(k in src for k in ["fit","train","predict","model","sklearn","keras","torch"]),
                })
            self.say_it(f"Notebook loaded: {len(records)} cells, "
                        f"{sum(1 for r in records if r['cell_type']=='code')} code cells")
            # Also populate Python Reader
            full_src = ""
            for i, cell in enumerate(nb.cells):
                if cell["cell_type"] == "code":
                    full_src += f"# === Cell {i+1} ===\n" + "".join(cell["source"]) + "\n\n"
            def _open():
                self.py_editor.delete("1.0",tk.END)
                self.py_editor.insert("1.0", full_src)
                self._py_filepath = path
            self.root.after(0, _open)
            return pd.DataFrame(records)
        else:
            # fallback: raw JSON
            with open(path,"r",encoding="utf-8",errors="replace") as f:
                nb = json.load(f)
            records = []
            cells = nb.get("cells",[])
            for i, cell in enumerate(cells):
                src = "".join(cell.get("source",[]))
                records.append({
                    "cell_index": i,
                    "cell_type":  cell.get("cell_type","unknown"),
                    "source":     src,
                    "line_count": src.count("\n")+1,
                    "char_count": len(src),
                    "has_import": "import" in src,
                    "has_plot":   any(k in src for k in ["plot","plt","sns","figure"]),
                })
            self.say_it(f"Notebook (raw JSON): {len(records)} cells loaded")
            return pd.DataFrame(records)

    def _extract_notebook_to_dataset(self):
        """Try to extract variable DataFrames defined in notebook cells."""
        code = self.py_editor.get("1.0",tk.END)
        if not code.strip():
            messagebox.showwarning("Empty","Load a notebook or .py file first."); return
        self.py_output.delete("1.0",tk.END)
        self.py_output.insert(tk.END,"Attempting to extract DataFrames from notebook code...\n\n")
        sandbox = {"pd": pd, "np": np}
        old_stdout, old_stderr = sys.stdout, sys.stderr
        cap = StringIO(); sys.stdout = sys.stderr = cap
        try:
            exec(compile(code, self._py_filepath or "<nb>", "exec"), sandbox)
        except Exception as e:
            self.py_output.insert(tk.END, f"Partial exec: {e}\n")
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
        found = 0
        for name, val in sandbox.items():
            if isinstance(val, pd.DataFrame) and len(val) > 0:
                self.datasets[f"nb_{name}"] = val
                found += 1
                self.py_output.insert(tk.END, f"✅ Extracted '{name}': {val.shape}\n")
        if found:
            self.active_dataset_name = f"nb_{list(sandbox.keys())[-1]}"
            for k, v in sandbox.items():
                if isinstance(v, pd.DataFrame) and len(v) > 0:
                    self.active_dataset_name = f"nb_{k}"
                    self.current_df = v
                    break
            self._update_all_gui_elements()
            self.say_it(f"Extracted {found} DataFrame(s) from notebook.")
        else:
            self.py_output.insert(tk.END, "No DataFrames found (cells may require data files to run).\n")

    # ══════════════════════════════════════════════════════════════════════════
    #  AUTO CLEAN
    # ══════════════════════════════════════════════════════════════════════════
    def clean_and_format(self, df: pd.DataFrame) -> pd.DataFrame:
        self.say_it("Auto-cleaning data…")
        df = df.copy().drop_duplicates()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for col in df.columns:
                if df[col].dtype == object:
                    converted = pd.to_datetime(df[col], errors="coerce")
                    if converted.notnull().mean() > 0.8:
                        df[col] = converted; continue
                    converted = pd.to_numeric(df[col], errors="coerce")
                    if converted.notnull().mean() > 0.8:
                        df[col] = converted
        num_cols = df.select_dtypes(include="number").columns
        if not num_cols.empty:
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        cat_cols = df.select_dtypes(include=["object","category"]).columns
        for col in cat_cols:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if not mode.empty else "Unknown")
        df = df.convert_dtypes()
        self.say_it(f"Cleaning complete. Shape: {df.shape}")
        self._add_pipeline_step("Auto-Clean",
            "df = df.drop_duplicates()\n"
            "df[num_cols] = df[num_cols].fillna(df[num_cols].median())")
        return df

    # ══════════════════════════════════════════════════════════════════════════
    #  DATASET SWITCH / UPDATE
    # ══════════════════════════════════════════════════════════════════════════
    def _on_dataset_switch(self, _=None):
        name = self.dataset_selector.get()
        if name and name in self.datasets:
            self.active_dataset_name = name
            self.current_df          = self.datasets[name]
            self.current_source      = name
            self.say_it(f"Switched to: {name}")
            self._update_all_gui_elements()

    def _update_all_gui_elements(self):
        if not self.datasets: return
        df   = self.current_df if self.current_df is not None else list(self.datasets.values())[0]
        cols = list(df.columns)
        num_cols = list(df.select_dtypes(include="number").columns)

        self.dataset_selector["values"] = list(self.datasets.keys())
        if self.active_dataset_name:
            self.dataset_selector.set(self.active_dataset_name)
        if hasattr(self,"comp_ds1"):
            ds = list(self.datasets.keys())
            self.comp_ds1["values"] = ds; self.comp_ds2["values"] = ds

        def _fill_tree(tree, dataframe, max_rows=500):
            tree.delete(*tree.get_children())
            tree["columns"] = list(dataframe.columns)
            for c in dataframe.columns:
                tree.heading(c, text=c)
                tree.column(c, width=110, anchor="center")
            for _, row in dataframe.head(max_rows).iterrows():
                tree.insert("", tk.END, values=list(row))

        _fill_tree(self.data_tree, df)
        _fill_tree(self.clean_tree, df, 100)

        self.drop_listbox.delete(0, tk.END)
        for c in cols: self.drop_listbox.insert(tk.END, c)

        for cb in (self.x_combo, self.y_combo, self.ml_target_combo,
                   self.dl_target_combo):
            cb["values"] = cols
        if hasattr(self,"d3_x"):
            for cb in (self.d3_x, self.d3_y, self.d3_z, self.d3_color_by):
                cb["values"] = num_cols

        self.root.after(0, self._refresh_sql_schema)
        self._calculate_data_health(df)
        self.root.after(0, self._plot_summary)
        self.root.after(0, self._refresh_predictor_ui)
        self.root.after(0, self._run_profiler_analysis)
        self.root.after(0, lambda: self._run_full_stats())

    # ══════════════════════════════════════════════════════════════════════════
    #  STATISTICS
    # ══════════════════════════════════════════════════════════════════════════
    def run_full_analysis(self, df):
        self.current_df = df
        self.root.after(0, self._update_all_gui_elements)
        self.say_it("Running full analysis…")
        self._add_pipeline_step("Data Load", f"df = pd.read_csv('{self.current_source}')")

    def _run_full_stats(self):
        if self.current_df is None: return
        df = self.current_df
        self.stats_log.delete("1.0",tk.END)
        buf = StringIO(); df.info(buf=buf)
        self.stats_log.insert(tk.END, "=== DATASET INFO ===\n" + buf.getvalue() + "\n\n")
        self.stats_log.insert(tk.END, "=== NUMERIC SUMMARY ===\n" + df.describe().round(3).to_string() + "\n\n")
        num = df.select_dtypes(include="number")
        if not num.empty:
            self.stats_log.insert(tk.END, "=== SKEWNESS & KURTOSIS ===\n" +
                pd.DataFrame({"Skewness": num.skew(), "Kurtosis": num.kurt()}).round(3).to_string() + "\n\n")
            self.stats_log.insert(tk.END, "=== PEARSON CORRELATIONS ===\n" +
                num.corr().round(3).to_string() + "\n\n")

    def _run_statistical_deep_dive(self):
        if self.current_df is None: return
        df = self.current_df
        num_cols = df.select_dtypes(include="number").columns
        cat_cols = df.select_dtypes(include=["object","category"]).columns
        out = ["=== 🔬 STATISTICAL DEEP DIVE ===\n"]
        out.append("1. NORMALITY (Shapiro-Wilk, first 5000 rows):")
        for col in num_cols[:6]:
            s, p = stats.shapiro(df[col].dropna().head(5000))
            out.append(f"   {col}: W={s:.4f} p={p:.4f}  →  {'Normal ✅' if p>0.05 else 'Non-Normal ⚠'}")
        if len(num_cols) and len(cat_cols):
            out.append("\n2. GROUP DIFFERENCES (ANOVA):")
            for target_num in num_cols[:2]:
                for target_cat in cat_cols[:2]:
                    groups = [g[target_num].dropna().values
                              for _, g in df.groupby(target_cat)
                              if len(g[target_num].dropna()) > 1]
                    if len(groups) > 1:
                        f, p = stats.f_oneway(*groups)
                        out.append(f"   '{target_num}' across '{target_cat}': p={p:.4f}  →  {'Significant ✅' if p<0.05 else 'No difference'}")
        if len(cat_cols) >= 2:
            out.append("\n3. CATEGORICAL LINKS (Chi-Square):")
            ct = pd.crosstab(df[cat_cols[0]], df[cat_cols[1]])
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                chi2, p, dof, _ = stats.chi2_contingency(ct)
                out.append(f"   '{cat_cols[0]}' vs '{cat_cols[1]}': chi2={chi2:.2f} df={dof} p={p:.4f}")
        if len(num_cols) >= 2:
            out.append("\n4. PEARSON & SPEARMAN correlations (top 5):")
            pairs = [(c1,c2) for i,c1 in enumerate(num_cols) for c2 in num_cols[i+1:]][:10]
            for c1, c2 in pairs:
                r, rp = pearsonr(df[c1].dropna(), df[c2].dropna()) if len(df[c1].dropna()) > 3 else (0,1)
                rs, sp = spearmanr(df[c1].dropna(), df[c2].dropna()) if len(df[c1].dropna()) > 3 else (0,1)
                out.append(f"   {c1} ↔ {c2}: Pearson={r:.3f}(p={rp:.3f})  Spearman={rs:.3f}(p={sp:.3f})")
        def _up():
            self.stats_log.insert(tk.END, "\n".join(out) + "\n" + "─"*50 + "\n")
            self.stats_log.see(tk.END)
        self.root.after(0, _up)
        self.say_it("Statistical deep dive complete.")

    def _run_cross_dataset_analysis(self):
        if len(self.datasets) < 2:
            messagebox.showinfo("Need 2+ Datasets","Load at least 2 datasets first."); return
        names = list(self.datasets.keys())
        out   = ["=== 🔗 CROSS-DATASET INFLUENCE ===\n"]
        for i in range(len(names)):
            for j in range(i+1,len(names)):
                d1, d2  = self.datasets[names[i]], self.datasets[names[j]]
                common  = list(set(d1.columns) & set(d2.columns))
                out.append(f"\n'{names[i]}' ↔ '{names[j]}': common cols = {common[:6]}")
                if common:
                    try:
                        merged = pd.merge(d1, d2, on=common[0]).head(5000)
                        num = merged.select_dtypes(include="number")
                        if len(num.columns) >= 2:
                            corr = num.corr().unstack()
                            strong = corr[(abs(corr) > 0.5) & (corr < 1.0)].drop_duplicates()
                            for (c1,c2), v in strong.head(5).items():
                                out.append(f"   → '{c1}' ↔ '{c2}': r={v:.2f}")
                    except Exception as e:
                        out.append(f"   Merge error: {e}")
        def _up():
            self.stats_log.insert(tk.END, "\n".join(out) + "\n" + "─"*50 + "\n")
            self.stats_log.see(tk.END)
        self.root.after(0, _up)

    # ══════════════════════════════════════════════════════════════════════════
    #  EDA SUITE
    # ══════════════════════════════════════════════════════════════════════════
    def _eda_distribution_grid(self):
        if self.current_df is None: return
        df = self.current_df
        num_cols = df.select_dtypes(include="number").columns.tolist()[:12]
        if not num_cols: messagebox.showwarning("No Numeric","No numeric columns."); return
        n = len(num_cols); rows = math.ceil(n/3); cols_per_row = 3
        fig, axs = plt.subplots(rows, cols_per_row, figsize=(15, 4*rows))
        dark_fig(fig); fig.suptitle("Distribution Grid", color=P["fg"], fontsize=14)
        axs = axs.flatten() if n > 1 else [axs]
        for i, col in enumerate(num_cols):
            data = df[col].dropna()
            axs[i].hist(data, bins=30, color=CC[i%len(CC)], alpha=0.7, edgecolor="none", density=True)
            try:
                data.plot(kind="kde", ax=axs[i], color=P["warn"], lw=1.8)
            except Exception: pass
            sk = data.skew()
            axs[i].set_title(f"{col}\nskew={sk:.2f}", color=P["fg"], fontsize=9)
            axs[i].set_facecolor(P["surface2"])
        for j in range(i+1, len(axs)):
            axs[j].set_visible(False)
        plt.tight_layout()
        embed_fig(fig, self.eda_container)

    def _eda_heatmap_suite(self):
        if self.current_df is None: return
        df = self.current_df
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2: messagebox.showwarning("Need 2+","Need at least 2 numeric columns."); return
        fig, axs = plt.subplots(1, 2, figsize=(16,7))
        dark_fig(fig)
        # Pearson
        corr_p = df[num_cols].corr(method="pearson")
        mask = np.triu(np.ones_like(corr_p, dtype=bool))
        sns.heatmap(corr_p, ax=axs[0], cmap="RdYlGn", annot=True, fmt=".2f",
                    mask=mask, annot_kws={"size":8}, cbar=True,
                    linewidths=0.5, linecolor=P["border"])
        axs[0].set_title("Pearson Correlation", color=P["fg"])
        # Spearman
        corr_s = df[num_cols].corr(method="spearman")
        sns.heatmap(corr_s, ax=axs[1], cmap="PuOr", annot=True, fmt=".2f",
                    annot_kws={"size":8}, cbar=True,
                    linewidths=0.5, linecolor=P["border"])
        axs[1].set_title("Spearman Correlation", color=P["fg"])
        plt.tight_layout()
        embed_fig(fig, self.eda_container)

    def _eda_violin_swarm(self):
        if self.current_df is None: return
        df = self.current_df
        num_cols = df.select_dtypes(include="number").columns.tolist()[:6]
        cat_cols = df.select_dtypes(include=["object","category","string"]).columns.tolist()
        if not num_cols: messagebox.showwarning("No Numeric","Need numeric columns."); return
        fig, axs = plt.subplots(1, min(3,len(num_cols)), figsize=(15,6))
        dark_fig(fig); fig.suptitle("Violin + Strip Charts", color=P["fg"], fontsize=13)
        if len(num_cols) == 1: axs = [axs]
        for i, col in enumerate(num_cols[:3]):
            ax = axs[i] if len(num_cols) > 1 else axs[0]
            if cat_cols:
                hue_col = cat_cols[0]
                top_cats = df[hue_col].value_counts().head(6).index
                sub = df[df[hue_col].isin(top_cats)]
                try:
                    sns.violinplot(data=sub, y=col, x=hue_col, ax=ax,
                                   palette=CC[:len(top_cats)], inner="quartile", linewidth=0.8)
                    sns.stripplot(data=sub, y=col, x=hue_col, ax=ax,
                                  color=P["fg_dim"], size=2.5, alpha=0.4, jitter=True)
                    ax.tick_params(axis="x", rotation=30)
                except Exception:
                    df[col].plot(kind="violin", ax=ax)
            else:
                ax.violinplot(df[col].dropna(), showmedians=True)
            ax.set_title(col, color=P["fg"])
        plt.tight_layout()
        embed_fig(fig, self.eda_container)

    def _eda_pairplot(self):
        if self.current_df is None: return
        df = self.current_df
        num_cols = df.select_dtypes(include="number").columns.tolist()[:5]
        if len(num_cols) < 2:
            messagebox.showwarning("Need 2+","Need at least 2 numeric columns."); return
        cat_cols = df.select_dtypes(include=["object","category","string"]).columns.tolist()
        sample = df.head(500)
        try:
            hue = cat_cols[0] if cat_cols and df[cat_cols[0]].nunique() <= 8 else None
            g = sns.pairplot(sample[num_cols + ([hue] if hue else [])],
                             hue=hue, plot_kws={"alpha":0.4,"s":12},
                             diag_kind="kde",
                             palette=CC[:df[hue].nunique()] if hue else None)
            g.fig.patch.set_facecolor(P["surface"])
            for ax in g.axes.flatten():
                if ax: ax.set_facecolor(P["surface2"])
            embed_fig(g.fig, self.eda_container, toolbar=False)
        except Exception as e:
            messagebox.showerror("Pairplot Error", str(e))

    def _eda_missing_heatmap(self):
        if self.current_df is None: return
        df = self.current_df
        miss = df.isnull()
        if miss.sum().sum() == 0:
            messagebox.showinfo("No Missing","Dataset has no missing values! 🎉"); return
        fig, ax = plt.subplots(figsize=(max(10, len(df.columns)*0.8), 6))
        dark_fig(fig)
        sns.heatmap(miss, ax=ax, cbar=False, yticklabels=False,
                    cmap=sns.color_palette([P["surface2"], P["danger"]], as_cmap=True))
        ax.set_title("Missing Values Heatmap (yellow = missing)", color=P["fg"])
        ax.set_xlabel("Columns"); ax.set_ylabel("Rows")
        plt.tight_layout()
        embed_fig(fig, self.eda_container)

    def _eda_scatter_matrix(self):
        if self.current_df is None: return
        df = self.current_df
        num_cols = df.select_dtypes(include="number").columns.tolist()[:6]
        if len(num_cols) < 2: messagebox.showwarning("Need 2+","Need 2+ numeric."); return
        fig, axs = plt.subplots(len(num_cols), len(num_cols),
                                figsize=(max(12,len(num_cols)*2.4), max(10,len(num_cols)*2.4)))
        dark_fig(fig); fig.suptitle("Numeric Scatter Matrix", color=P["fg"], fontsize=13)
        sample = df[num_cols].dropna().sample(min(500,len(df)))
        for i, c1 in enumerate(num_cols):
            for j, c2 in enumerate(num_cols):
                ax = axs[i][j]
                if i == j:
                    ax.hist(sample[c1], bins=20, color=CC[i%len(CC)], alpha=0.8, edgecolor="none")
                    ax.set_title(c1, color=P["fg"], fontsize=8)
                else:
                    ax.scatter(sample[c2], sample[c1], s=5, alpha=0.4, color=CC[i%len(CC)])
                    # Regression line
                    try:
                        m, b, r, p_val, _ = stats.linregress(sample[c2].fillna(0), sample[c1].fillna(0))
                        xr = np.linspace(sample[c2].min(), sample[c2].max(), 50)
                        ax.plot(xr, m*xr + b, color=P["warn"], lw=1.2, alpha=0.8)
                        ax.set_title(f"r={r:.2f}", color=P["fg_dim"], fontsize=7)
                    except Exception:
                        pass
                ax.set_facecolor(P["surface2"])
                ax.tick_params(labelsize=6, colors=P["fg_dim"])
        plt.tight_layout()
        embed_fig(fig, self.eda_container)

    # ══════════════════════════════════════════════════════════════════════════
    #  2D SUMMARY VISUALS
    # ══════════════════════════════════════════════════════════════════════════
    def _plot_summary(self, pop_out=False):
        df = self.current_df
        if df is None: return
        try:
            num_cols = df.select_dtypes(include="number").columns
            cat_cols = df.select_dtypes(include=["object","category","string"]).columns
            fig, axs = plt.subplots(2, 3, figsize=(15,9))
            plt.subplots_adjust(hspace=0.48, wspace=0.34)
            dark_fig(fig)

            if len(cat_cols):
                vc = df[cat_cols[0]].value_counts().head(10)
                vc.plot(kind="bar", ax=axs[0,0], color=CC[:len(vc)], edgecolor="none")
                axs[0,0].set_title(f"Top 10 — {cat_cols[0]}"); axs[0,0].tick_params(axis="x", rotation=40)
                vc.head(5).plot(kind="pie", ax=axs[0,1], autopct="%1.1f%%",
                               colors=CC[:5], wedgeprops={"edgecolor":P["surface2"]})
                axs[0,1].set_title(f"{cat_cols[0]}"); axs[0,1].set_ylabel("")

            if len(num_cols) >= 2:
                s = df.sample(min(800,len(df)))
                sc = axs[0,2].scatter(s[num_cols[0]], s[num_cols[1]], alpha=0.4, s=12, c=CC[0])
                # Regression line
                try:
                    m, b, r, _, _ = stats.linregress(df[num_cols[0]].fillna(0), df[num_cols[1]].fillna(0))
                    xr = np.linspace(df[num_cols[0]].min(), df[num_cols[0]].max(), 100)
                    axs[0,2].plot(xr, m*xr+b, color=P["warn"], lw=2, label=f"r={r:.2f}")
                    axs[0,2].legend(fontsize=8)
                except Exception: pass
                axs[0,2].set_xlabel(num_cols[0]); axs[0,2].set_ylabel(num_cols[1])
                axs[0,2].set_title(f"{num_cols[0]} vs {num_cols[1]}")

            if len(num_cols):
                data = df[num_cols[0]].dropna()
                axs[1,0].hist(data, bins=30, color=CC[2], edgecolor=P["surface"], density=True, alpha=0.8)
                try:
                    data.plot(kind="kde", ax=axs[1,0], color=P["warn"], lw=2)
                except Exception: pass
                axs[1,0].set_title(f"{num_cols[0]} Dist.")

            if len(num_cols) >= 2:
                corr = df[num_cols].corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(corr, annot=True, cmap="RdYlGn", fmt=".2f", ax=axs[1,1],
                            annot_kws={"size":7}, mask=mask, linewidths=0.3)
                axs[1,1].set_title("Correlation Heatmap")
            else:
                axs[1,1].text(0.5,0.5,"Need 2+ numeric cols", ha="center",
                              color=P["fg_dim"], transform=axs[1,1].transAxes)

            if len(num_cols):
                box_cols = num_cols[:min(6,len(num_cols))]
                bp = df[box_cols].plot(kind="box", ax=axs[1,2], vert=True, patch_artist=True)
                for patch, color in zip(axs[1,2].patches, CC):
                    patch.set_facecolor(color); patch.set_alpha(0.7)
                axs[1,2].set_title("Boxplot (Outliers)"); axs[1,2].tick_params(axis="x", rotation=35)

            target = self.visuals_container if not pop_out else None
            if pop_out:
                win = tk.Toplevel(self.root)
                win.title("Summary Visuals"); win.configure(bg=P["surface"])
                self.chart_windows.append(win); target = win
            else:
                for w in self.visuals_container.winfo_children(): w.destroy()

            canvas = FigureCanvasTkAgg(fig, master=target)
            canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            tf = tk.Frame(target, bg=P["surface"]); tf.pack(fill=tk.X)
            NavigationToolbar2Tk(canvas, tf)

            # insights
            if not pop_out:
                self.insight_log.delete("1.0", tk.END)
                if len(cat_cols):
                    top = df[cat_cols[0]].value_counts().idxmax()
                    self.insight_log.insert(tk.END, f"→ '{top}' dominates '{cat_cols[0]}'\n")
                if len(num_cols) >= 2:
                    r = df[num_cols[0]].corr(df[num_cols[1]])
                    strength = "Strong" if abs(r)>0.6 else "Moderate" if abs(r)>0.3 else "Weak"
                    self.insight_log.insert(tk.END,
                        f"→ {strength} {'positive' if r>0 else 'negative'} correlation "
                        f"({r:.2f}) between '{num_cols[0]}' & '{num_cols[1]}'\n")
                if len(num_cols):
                    sk = df[num_cols[0]].skew()
                    if abs(sk) > 1:
                        self.insight_log.insert(tk.END,
                            f"→ '{num_cols[0]}' heavily skewed ({sk:.2f}) — consider log-transform\n")
        except Exception as e:
            self.say_it(f"Visual error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    #  3D CHARTS
    # ══════════════════════════════════════════════════════════════════════════
    def _generate_3d_chart(self, pop_out=False):
        if self.current_df is None:
            messagebox.showwarning("No Data","Load a dataset first!"); return
        x_col = self.d3_x.get(); y_col = self.d3_y.get(); z_col = self.d3_z.get()
        chart_type = self.d3_type.get()
        color_col  = self.d3_color_by.get()
        df = self.current_df.copy()
        num_cols = list(df.select_dtypes(include="number").columns)
        if len(num_cols) < 2:
            messagebox.showwarning("Need Numeric Data","At least 2 numeric columns required."); return
        x_col = x_col or (num_cols[0] if len(num_cols)>0 else None)
        y_col = y_col or (num_cols[1] if len(num_cols)>1 else num_cols[0])
        z_col = z_col or (num_cols[2] if len(num_cols)>2 else num_cols[0])

        def _draw():
            fig = plt.figure(figsize=(10,7), facecolor=P["surface"])
            ax  = fig.add_subplot(111, projection="3d")
            ax.set_facecolor(P["surface2"])
            ax.tick_params(colors=P["fg_dim"])
            for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                pane.fill = False; pane.set_edgecolor(P["border"])

            sample = df[[c for c in [x_col,y_col,z_col,color_col] if c and c in df.columns]].dropna().head(2000)
            xs, ys, zs = sample[x_col].values, sample[y_col].values, sample[z_col].values
            c_vals = sample[color_col].values if color_col and color_col in sample.columns else zs

            if chart_type == "3D Scatter":
                sc = ax.scatter(xs, ys, zs, c=c_vals, cmap="plasma", alpha=0.75, s=18, edgecolors="none")
                fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1, label=color_col or z_col)
                # Regression plane
                try:
                    A = np.column_stack([xs, ys, np.ones(len(xs))])
                    coef, _, _, _ = np.linalg.lstsq(A, zs, rcond=None)
                    xi = np.linspace(xs.min(),xs.max(),20)
                    yi = np.linspace(ys.min(),ys.max(),20)
                    Xi, Yi = np.meshgrid(xi, yi)
                    Zi = coef[0]*Xi + coef[1]*Yi + coef[2]
                    ax.plot_surface(Xi, Yi, Zi, alpha=0.15, color=P["warn"])
                except Exception: pass

            elif chart_type == "3D Surface":
                from scipy.interpolate import griddata
                xi = np.linspace(xs.min(),xs.max(),50)
                yi = np.linspace(ys.min(),ys.max(),50)
                Xi, Yi = np.meshgrid(xi, yi)
                Zi = griddata((xs,ys), zs, (Xi,Yi), method="linear")
                surf = ax.plot_surface(Xi, Yi, Zi, cmap="viridis", alpha=0.88, edgecolor="none")
                fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)

            elif chart_type == "3D Bar":
                cats = sample[x_col].round(1).unique()[:20]
                for i, c in enumerate(cats):
                    mask = sample[x_col].round(1) == c
                    ax.bar3d(i, 0, 0, 0.6, 0.6, float(sample.loc[mask,z_col].mean()),
                             color=CC[i%len(CC)], alpha=0.8)

            elif chart_type == "3D Line":
                idx = np.argsort(xs)
                ax.plot(xs[idx], ys[idx], zs[idx], color=P["accent"], lw=1.5, alpha=0.9)
                ax.scatter(xs[idx], ys[idx], zs[idx], c=zs[idx], cmap="plasma", s=8)

            elif chart_type == "3D Wireframe":
                from scipy.interpolate import griddata
                xi = np.linspace(xs.min(),xs.max(),30); yi = np.linspace(ys.min(),ys.max(),30)
                Xi, Yi = np.meshgrid(xi, yi)
                Zi = griddata((xs,ys), zs, (Xi,Yi), method="linear")
                ax.plot_wireframe(Xi, Yi, Zi, color=P["accent3"], alpha=0.6, linewidth=0.5)

            elif chart_type == "3D Trisurf":
                ax.plot_trisurf(xs, ys, zs, cmap="inferno", alpha=0.8, edgecolor="none")

            ax.set_xlabel(x_col, color=P["fg"]); ax.set_ylabel(y_col, color=P["fg"]); ax.set_zlabel(z_col, color=P["fg"])
            ax.set_title(f"{chart_type}: {x_col}/{y_col}/{z_col}", color=P["fg"], fontsize=11, pad=12)

            if pop_out:
                win = tk.Toplevel(self.root); win.title(f"3D — {chart_type}")
                win.configure(bg=P["surface"]); self.chart_windows.append(win); master = win
            else:
                for w in self.d3_container.winfo_children(): w.destroy(); master = self.d3_container

            canvas = FigureCanvasTkAgg(fig, master=master)
            canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            tf = tk.Frame(master, bg=P["surface"]); tf.pack(fill=tk.X)
            NavigationToolbar2Tk(canvas, tf)

        self.root.after(0, _draw)

    def _pop_3d(self):
        self._generate_3d_chart(pop_out=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  CUSTOM CHART
    # ══════════════════════════════════════════════════════════════════════════
    def _generate_custom_chart(self):
        if self.current_df is None: messagebox.showwarning("No Data","Load a dataset first!"); return
        x = self.x_combo.get(); y = self.y_combo.get()
        ctype = self.type_combo.get(); title = self.title_entry.get() or f"{ctype}: {x} vs {y}"
        color = self.color_combo.get(); df = self.current_df.copy()

        def _draw():
            try:
                win = tk.Toplevel(self.root); win.title(title)
                win.configure(bg=P["surface"]); self.chart_windows.append(win)
                fig, ax = plt.subplots(figsize=(9,6)); dark_fig(fig)
                is_xn = pd.api.types.is_numeric_dtype(df[x]) if x in df.columns else False
                is_yn = pd.api.types.is_numeric_dtype(df[y]) if y in df.columns else False

                if ctype == "Scatter":
                    if is_xn and is_yn:
                        ax.scatter(df[x], df[y], alpha=0.45, color=color, s=15)
                        m, b, r, _, _ = stats.linregress(df[x].fillna(0), df[y].fillna(0))
                        xr = np.linspace(df[x].min(), df[x].max(), 100)
                        ax.plot(xr, m*xr+b, color=P["warn"], lw=2, label=f"Regression (r={r:.2f})")
                        ax.legend()
                    else: messagebox.showwarning("Type","Scatter requires numeric X & Y."); win.destroy(); return
                elif ctype == "Bar":
                    if x in df.columns:
                        (df.groupby(x)[y].mean().head(20) if is_yn else df[x].value_counts().head(20)
                         ).plot(kind="bar", ax=ax, color=color)
                elif ctype == "Line":
                    if x in df.columns and y in df.columns and is_yn:
                        df.sort_values(x).plot(kind="line", x=x, y=y, ax=ax, color=color)
                    else: messagebox.showwarning("Type","Line needs numeric Y."); win.destroy(); return
                elif ctype == "Histogram":
                    col = x if x in df.columns else y
                    data = df[col].dropna()
                    if pd.api.types.is_numeric_dtype(df[col]):
                        ax.hist(data, bins=30, color=color, edgecolor=P["surface"], density=True, alpha=0.8)
                        data.plot(kind="kde", ax=ax, color=P["warn"], lw=2)
                    else:
                        df[col].value_counts().head(20).plot(kind="bar", ax=ax, color=color)
                elif ctype == "Pie":
                    col = x if x in df.columns else y
                    df[col].value_counts().head(8).plot(kind="pie", ax=ax, autopct="%1.1f%%",
                                                        startangle=90, colors=CC); ax.set_ylabel("")
                elif ctype == "KDE":
                    col = x if x in df.columns else y
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col].dropna().plot(kind="kde", ax=ax, color=color, lw=2)
                    else: messagebox.showwarning("Type","KDE needs numeric."); win.destroy(); return
                elif ctype == "Box":
                    col = y if y in df.columns else x
                    if pd.api.types.is_numeric_dtype(df[col]):
                        ax.boxplot(df[col].dropna(), patch_artist=True,
                                   boxprops={"facecolor":color,"alpha":0.7},
                                   medianprops={"color":P["warn"],"lw":2},
                                   whiskerprops={"color":P["fg_dim"]},
                                   capprops={"color":P["fg_dim"]},
                                   flierprops={"marker":"o","markersize":4,"markerfacecolor":P["danger"]})
                    else: messagebox.showwarning("Type","Box needs numeric."); win.destroy(); return
                elif ctype == "Area":
                    col = y if is_yn else x
                    df[col].fillna(0).plot(kind="area", ax=ax, color=color, alpha=0.5)
                elif ctype == "Step":
                    col = y if is_yn else x
                    ax.step(range(len(df)), df[col].fillna(0), color=color, lw=1.5, where="mid")

                ax.set_title(title, color=P["fg"]); plt.tight_layout()
                canvas = FigureCanvasTkAgg(fig, master=win)
                canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                tf = tk.Frame(win, bg=P["surface"]); tf.pack(fill=tk.X)
                NavigationToolbar2Tk(canvas, tf)
                tk.Button(tf, text="💾 Save", bg=P["accent"], fg="#000", relief=tk.FLAT,
                          command=lambda: self._save_chart_image(fig)).pack(side=tk.RIGHT, padx=6)
            except Exception as e:
                messagebox.showerror("Chart Error", str(e))

        self.root.after(0, _draw)

    def _close_all_charts(self):
        for w in self.chart_windows:
            try: w.destroy()
            except: pass
        self.chart_windows = []

    def _save_chart_image(self, fig):
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG","*.png"),("JPG","*.jpg"),("SVG","*.svg")])
        if path: fig.savefig(path, dpi=150, bbox_inches="tight"); messagebox.showinfo("Saved",f"Saved to {path}")

    # ══════════════════════════════════════════════════════════════════════════
    #  MACHINE LEARNING — full visualisation
    # ══════════════════════════════════════════════════════════════════════════
    def _prepare_features(self, df, target):
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not in DataFrame")
        y = df[target].copy(); X = df.drop(columns=[target])
        encoders = {}; dropped = []
        for col in list(X.columns):
            n = X[col].nunique()
            if n <= 1 or n == len(X):
                X = X.drop(columns=[col]); dropped.append(col); continue
            if X[col].dtype == object or str(X[col].dtype) == "string":
                if n < 100:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str).fillna("__na__"))
                    encoders[col] = le
                else:
                    X = X.drop(columns=[col]); dropped.append(col)
        X = X.select_dtypes(include="number").fillna(X.select_dtypes(include="number").median())
        if X.empty:
            raise ValueError("No usable feature columns after preprocessing.")
        return X, y, encoders, dropped

    def _get_algorithm(self, algo_name, is_clf):
        mapping = {
            "Random Forest":                  (RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
                                               RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)),
            "Gradient Boosting":              (GradientBoostingClassifier(n_estimators=100, random_state=42),
                                               GradientBoostingRegressor(n_estimators=100, random_state=42)),
            "Logistic / Linear Regression":   (LogisticRegression(max_iter=500, random_state=42),
                                               LinearRegression()),
            "SVM":                            (SVC(kernel="rbf", probability=True, random_state=42),
                                               SVR(kernel="rbf")),
            "KNN":                            (KNeighborsClassifier(n_neighbors=5),
                                               KNeighborsRegressor(n_neighbors=5)),
            "Decision Tree":                  (DecisionTreeClassifier(max_depth=8, random_state=42),
                                               DecisionTreeRegressor(max_depth=8, random_state=42)),
            "AdaBoost":                       (AdaBoostClassifier(n_estimators=100, random_state=42),
                                               AdaBoostRegressor(n_estimators=100, random_state=42)),
            "Naive Bayes":                    (GaussianNB(), GaussianNB()),
            "ElasticNet":                     (LogisticRegression(penalty="elasticnet", solver="saga",
                                                                  l1_ratio=0.5, max_iter=500),
                                               ElasticNet(alpha=0.1, random_state=42)),
        }
        clf_model, reg_model = mapping.get(algo_name, mapping["Random Forest"])
        return clf_model if is_clf else reg_model

    def _run_gui_ml(self):
        if not SKLEARN_OK: messagebox.showerror("sklearn required","pip install scikit-learn"); return
        target = self.ml_target_combo.get()
        if not target: messagebox.showwarning("No Target","Select a target column!"); return
        if self.current_df is None: return
        self._start_learning_animation(target)
        threading.Thread(target=self._train_supervised,
                         args=(self.current_df.copy(), target), daemon=True).start()

    def _start_learning_animation(self, target):
        win = tk.Toplevel(self.root); win.title("🧠 Training…")
        win.geometry("420x260"); win.configure(bg=P["surface"]); win.resizable(False,False)
        tk.Label(win, text="Machine is Learning…", font=("Segoe UI",13,"bold"),
                 bg=P["surface"], fg=P["accent"]).pack(pady=14)
        tk.Label(win, text=f"Target: {target}", font=("Segoe UI",10),
                 bg=P["surface"], fg=P["fg_dim"]).pack()
        pb = ttk.Progressbar(win, mode="indeterminate", length=320); pb.pack(pady=12); pb.start(12)
        sl = tk.Label(win, text="Initialising…", font=("Segoe UI",9),
                      bg=P["surface"], fg=P["accent2"]); sl.pack()
        phases = ["Splitting data…","Fitting estimator…","Cross-validating…",
                  "Computing importances…","Building confusion matrix…","Finalising…"]
        self._ml_training_done = False
        def _tick():
            if not win.winfo_exists(): return
            sl.config(text=phases[np.random.randint(len(phases))])
            if self._ml_training_done: win.destroy(); self._ml_training_done = False
            else: win.after(600, _tick)
        _tick()

    def _train_supervised(self, df, target):
        try:
            X, y, encoders, dropped = self._prepare_features(df, target)
        except ValueError as e:
            self.root.after(0, lambda: (messagebox.showerror("Prep Error", str(e)),
                                        setattr(self, "_ml_training_done", True)))
            return

        is_clf = (y.dtype == object or y.nunique() <= 20)
        target_encoder = None
        if is_clf and y.dtype == object:
            target_encoder = LabelEncoder()
            y_enc = target_encoder.fit_transform(y.astype(str))
        else:
            y_enc = y.values.astype(float) if not is_clf else y.values

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_enc, test_size=0.2, random_state=42, stratify=(y_enc if is_clf else None))
        except Exception:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=0.2, random_state=42)

        algo_name = self.ml_algo_combo.get()
        model = self._get_algorithm(algo_name, is_clf)

        try:
            model.fit(X_tr, y_tr)
        except Exception as e:
            self.root.after(0, lambda: (messagebox.showerror("Train Error", str(e)),
                                        setattr(self,"_ml_training_done",True)))
            return

        preds = model.predict(X_te)
        score = accuracy_score(y_te, preds) if is_clf else r2_score(y_te, preds)
        metric = "Accuracy" if is_clf else "R²"

        # CV score
        try:
            cv_scores = cross_val_score(model, X, y_enc, cv=5,
                                         scoring="accuracy" if is_clf else "r2")
        except Exception:
            cv_scores = np.array([score])

        self.current_model           = model
        self.current_model_features  = list(X.columns)
        self.current_model_target    = target
        self.current_model_encoders  = encoders
        self._target_encoder         = target_encoder
        self._is_clf                 = is_clf
        self._y_test                 = y_te
        self._preds_test             = preds
        self._X_train                = X_tr
        self._X_test                 = X_te
        self._y_train                = y_tr

        self.model_history.append({
            "time": time.strftime("%H:%M:%S"), "algo": algo_name,
            "model": type(model).__name__, "target": target,
            "metric": f"{metric}={score:.4f}", "cv_mean": f"{cv_scores.mean():.4f}"
        })

        lines = [
            f"=== 🤖 ML REPORT: {algo_name} ===",
            f"Target      : {target}  ({'Classification' if is_clf else 'Regression'})",
            f"Features    : {len(X.columns)}",
            f"Train/Test  : {len(X_tr)} / {len(X_te)}",
            f"{metric}       : {score:.4f}",
            f"CV (5-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
        ]
        if is_clf:
            lines += ["", "Classification Report:",
                      classification_report(y_te, preds, zero_division=0)]
        else:
            rmse = math.sqrt(mean_squared_error(y_te, preds))
            mae  = mean_absolute_error(y_te, preds)
            lines += [f"RMSE        : {rmse:.4f}", f"MAE         : {mae:.4f}"]
        if dropped:
            lines += [f"\nDropped    : {', '.join(dropped[:8])}"]

        self._ml_training_done = True
        self.root.after(0, lambda: self._show_ml_results(model, X, y_te, preds,
                                                          score, metric, is_clf,
                                                          "\n".join(lines), cv_scores))
        self.say_it(f"Training done. {metric}: {score:.2%}")

    def _show_ml_results(self, model, X, y_test, preds, score, metric, is_clf, report_text, cv_scores):
        self.ml_metrics_area.delete("1.0",tk.END)
        self.ml_metrics_area.insert(tk.END, report_text)
        self.ml_history_area.delete("1.0",tk.END)
        self.ml_history_area.insert(tk.END,"📜 TRAINING HISTORY:\n")
        for r in self.model_history[-8:]:
            self.ml_history_area.insert(tk.END, f"[{r['time']}] {r['algo']} → {r['metric']} (CV={r['cv_mean']})\n")

        for w in self.ml_plot_container.winfo_children(): w.destroy()

        if is_clf:
            fig, axs = plt.subplots(2, 2, figsize=(13,10))
        else:
            fig, axs = plt.subplots(2, 2, figsize=(13,10))
        dark_fig(fig)

        # Feature importances
        if hasattr(model,"feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
            colors = [CC[i%len(CC)] for i in range(len(fi))]
            axs[0,0].barh(fi.index[::-1], fi.values[::-1], color=colors[::-1], edgecolor="none")
            axs[0,0].set_title("Feature Importances", color=P["fg"])
            axs[0,0].set_xlabel("Importance")
        elif hasattr(model,"coef_"):
            coefs = model.coef_.flatten()[:len(X.columns)]
            fi = pd.Series(np.abs(coefs), index=X.columns[:len(coefs)]).sort_values(ascending=False).head(15)
            axs[0,0].barh(fi.index[::-1], fi.values[::-1], color=CC[0], edgecolor="none")
            axs[0,0].set_title("Coefficient Magnitudes", color=P["fg"])
        else:
            axs[0,0].text(0.5,0.5,"Feature importances\nnot available", ha="center",
                          color=P["fg_dim"], transform=axs[0,0].transAxes)

        # CV scores
        axs[0,1].bar(range(len(cv_scores)), cv_scores, color=CC, edgecolor="none", alpha=0.85)
        axs[0,1].axhline(cv_scores.mean(), color=P["warn"], lw=2, linestyle="--",
                         label=f"Mean={cv_scores.mean():.3f}")
        axs[0,1].fill_between([-0.5,len(cv_scores)-0.5],
                               cv_scores.mean()-cv_scores.std(),
                               cv_scores.mean()+cv_scores.std(),
                               alpha=0.2, color=P["warn"])
        axs[0,1].set_title("5-Fold CV Scores", color=P["fg"])
        axs[0,1].set_xlabel("Fold"); axs[0,1].set_ylabel(metric)
        axs[0,1].legend(fontsize=9)
        axs[0,1].set_xticks(range(len(cv_scores)))
        axs[0,1].set_xticklabels([f"Fold {i+1}" for i in range(len(cv_scores))], rotation=20)

        if is_clf:
            # Confusion matrix
            cm = confusion_matrix(y_test, preds)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[1,0],
                        linewidths=0.5, linecolor=P["border"])
            axs[1,0].set_title("Confusion Matrix", color=P["fg"])
            axs[1,0].set_xlabel("Predicted"); axs[1,0].set_ylabel("Actual")

            # ROC curve (binary only)
            try:
                if len(np.unique(y_test)) == 2 and hasattr(model,"predict_proba"):
                    proba = model.predict_proba(self._X_test)[:,1]
                    fpr, tpr, _ = roc_curve(y_test, proba)
                    roc_auc = auc(fpr, tpr)
                    axs[1,1].plot(fpr, tpr, color=P["accent"], lw=2, label=f"AUC = {roc_auc:.3f}")
                    axs[1,1].plot([0,1],[0,1],"--", color=P["fg_dim"], lw=1)
                    axs[1,1].fill_between(fpr, tpr, alpha=0.1, color=P["accent"])
                    axs[1,1].set_xlabel("False Positive Rate"); axs[1,1].set_ylabel("True Positive Rate")
                    axs[1,1].set_title("ROC Curve", color=P["fg"])
                    axs[1,1].legend(fontsize=9)
                else:
                    # Multi-class: per-class bar
                    from sklearn.metrics import precision_recall_fscore_support
                    pr, rec, f1, sup = precision_recall_fscore_support(y_test, preds, zero_division=0)
                    x_labels = [str(c) for c in np.unique(y_test)][:len(pr)]
                    xp = np.arange(len(x_labels))
                    w = 0.25
                    axs[1,1].bar(xp-w, pr[:len(xp)],  w, label="Precision", color=CC[0], alpha=0.85)
                    axs[1,1].bar(xp,   rec[:len(xp)], w, label="Recall",    color=CC[1], alpha=0.85)
                    axs[1,1].bar(xp+w, f1[:len(xp)],  w, label="F1",        color=CC[2], alpha=0.85)
                    axs[1,1].set_xticks(xp); axs[1,1].set_xticklabels(x_labels, rotation=30)
                    axs[1,1].set_title("Per-Class Metrics", color=P["fg"])
                    axs[1,1].legend(fontsize=9)
            except Exception: pass
        else:
            # Actual vs Predicted
            axs[1,0].scatter(y_test, preds, alpha=0.35, color=CC[0], s=14)
            lim = [min(float(y_test.min()), float(preds.min())),
                   max(float(y_test.max()), float(preds.max()))]
            axs[1,0].plot(lim, lim, "--", color=P["warn"], lw=2, label="Perfect fit")
            # Regression line through scatter
            m, b, r, _, _ = stats.linregress(y_test, preds)
            xr = np.linspace(lim[0], lim[1], 100)
            axs[1,0].plot(xr, m*xr+b, color=P["danger"], lw=1.5, alpha=0.7, label=f"Fit line r={r:.2f}")
            axs[1,0].set_xlabel("Actual"); axs[1,0].set_ylabel("Predicted")
            axs[1,0].set_title("Actual vs Predicted", color=P["fg"])
            axs[1,0].legend(fontsize=9)

            # Residuals
            residuals = preds - y_test
            axs[1,1].scatter(preds, residuals, alpha=0.35, s=12, color=CC[3])
            axs[1,1].axhline(0, color=P["warn"], lw=1.5, linestyle="--")
            axs[1,1].axhline(residuals.std(),  color=P["danger"], lw=1, linestyle=":", alpha=0.7)
            axs[1,1].axhline(-residuals.std(), color=P["danger"], lw=1, linestyle=":", alpha=0.7)
            axs[1,1].set_xlabel("Predicted"); axs[1,1].set_ylabel("Residual")
            axs[1,1].set_title("Residual Plot", color=P["fg"])

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.ml_plot_container)
        canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        tf = tk.Frame(self.ml_plot_container, bg=P["surface"]); tf.pack(fill=tk.X)
        NavigationToolbar2Tk(canvas, tf)
        self._refresh_predictor_ui()

    def _plot_learning_curve(self):
        if not SKLEARN_OK: return
        target = self.ml_target_combo.get()
        if not target or self.current_df is None: messagebox.showwarning("No Target","Select target + load data."); return
        def _task():
            try:
                X, y, _, _ = self._prepare_features(self.current_df.copy(), target)
                is_clf = (y.dtype == object or y.nunique() <= 20)
                if is_clf and y.dtype == object:
                    le = LabelEncoder(); y = pd.Series(le.fit_transform(y.astype(str)))
                else:
                    y = y.fillna(0)
                model = self._get_algorithm(self.ml_algo_combo.get(), is_clf)
                train_sizes, train_scores, val_scores = learning_curve(
                    model, X, y, cv=5, train_sizes=np.linspace(0.1,1.0,10),
                    scoring="accuracy" if is_clf else "r2", n_jobs=-1)
                def _draw():
                    win = tk.Toplevel(self.root); win.title("Learning Curve"); win.configure(bg=P["surface"])
                    fig, ax = plt.subplots(figsize=(9,6)); dark_fig(fig)
                    tr_mean = train_scores.mean(axis=1); tr_std = train_scores.std(axis=1)
                    va_mean = val_scores.mean(axis=1);   va_std = val_scores.std(axis=1)
                    ax.plot(train_sizes, tr_mean, "o-", color=CC[0], lw=2, label="Training score")
                    ax.fill_between(train_sizes, tr_mean-tr_std, tr_mean+tr_std, alpha=0.15, color=CC[0])
                    ax.plot(train_sizes, va_mean, "s-", color=CC[1], lw=2, label="CV score")
                    ax.fill_between(train_sizes, va_mean-va_std, va_mean+va_std, alpha=0.15, color=CC[1])
                    ax.set_xlabel("Training examples"); ax.set_ylabel("Score")
                    ax.set_title(f"Learning Curve — {self.ml_algo_combo.get()}", color=P["fg"])
                    ax.legend(fontsize=10); plt.tight_layout()
                    embed_fig(fig, tk.Frame(win, bg=P["surface"]))
                    canvas = FigureCanvasTkAgg(fig, master=win)
                    canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                self.root.after(0, _draw)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        threading.Thread(target=_task, daemon=True).start()

    def _plot_cv_scores(self):
        if not SKLEARN_OK or self.current_df is None: return
        target = self.ml_target_combo.get()
        if not target: messagebox.showwarning("No Target","Select target first."); return
        def _task():
            try:
                X, y, _, _ = self._prepare_features(self.current_df.copy(), target)
                is_clf = (y.dtype == object or y.nunique() <= 20)
                if is_clf and y.dtype == object:
                    le = LabelEncoder(); y = pd.Series(le.fit_transform(y.astype(str)))
                else:
                    y = y.fillna(0)
                model = self._get_algorithm(self.ml_algo_combo.get(), is_clf)
                all_scores = cross_val_score(model, X, y, cv=10,
                                              scoring="accuracy" if is_clf else "r2")
                def _draw():
                    win = tk.Toplevel(self.root); win.title("10-Fold CV Scores"); win.configure(bg=P["surface"])
                    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,5)); dark_fig(fig)
                    ax1.bar(range(1,11), all_scores, color=[CC[i%len(CC)] for i in range(10)], edgecolor="none")
                    ax1.axhline(all_scores.mean(), color=P["warn"], lw=2, linestyle="--", label=f"μ={all_scores.mean():.3f}")
                    ax1.fill_between([-0.5,10.5], all_scores.mean()-all_scores.std(), all_scores.mean()+all_scores.std(),
                                     alpha=0.15, color=P["warn"])
                    ax1.set_title("10-Fold CV Scores", color=P["fg"]); ax1.legend()
                    ax1.set_xlabel("Fold"); ax1.set_ylabel("Score")
                    ax2.hist(all_scores, bins=6, color=P["accent3"], edgecolor=P["surface"])
                    ax2.axvline(all_scores.mean(), color=P["warn"], lw=2, linestyle="--")
                    ax2.set_title("Score Distribution", color=P["fg"])
                    ax2.set_xlabel("Score"); ax2.set_ylabel("Count")
                    plt.tight_layout()
                    canvas = FigureCanvasTkAgg(fig, master=win)
                    canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                self.root.after(0, _draw)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("CV Error", str(e)))
        threading.Thread(target=_task, daemon=True).start()

    def _run_automl_optimization(self, df):
        if not SKLEARN_OK or df is None: return
        target = self.ml_target_combo.get()
        if not target: messagebox.showwarning("No Target","Select target."); return
        def _task():
            try:
                X, y, _, _ = self._prepare_features(df.copy(), target)
                is_clf = (y.dtype == object or y.nunique() <= 20)
                if is_clf and y.dtype == object:
                    le = LabelEncoder(); y = pd.Series(le.fit_transform(y.astype(str)))
                else:
                    y = y.fillna(0)
                scoring = "accuracy" if is_clf else "r2"
                models = {
                    "RandomForest":     (RandomForestClassifier if is_clf else RandomForestRegressor)(n_estimators=100, random_state=42),
                    "GradientBoosting": (GradientBoostingClassifier if is_clf else GradientBoostingRegressor)(n_estimators=100, random_state=42),
                    "Ridge/LogReg":     (LogisticRegression(max_iter=400) if is_clf else Ridge()),
                    "KNN":              (KNeighborsClassifier() if is_clf else KNeighborsRegressor()),
                    "DecisionTree":     (DecisionTreeClassifier(random_state=42) if is_clf else DecisionTreeRegressor(random_state=42)),
                }
                results = {}
                for name, mdl in models.items():
                    try:
                        cv = cross_val_score(mdl, X, y, cv=5, scoring=scoring, n_jobs=-1).mean()
                        results[name] = round(cv, 4)
                    except Exception: results[name] = 0.0
                best = max(results, key=results.get)
                text = "=== ⚗️ AUTO-ML BENCHMARK ===\n\n"
                for k, v in sorted(results.items(), key=lambda x: -x[1]):
                    bar = "█" * int(v * 30) + "░" * (30 - int(v * 30))
                    flag = " 🏆 WINNER" if k == best else ""
                    text += f"  {k:22}: {v:.4f}  {bar}{flag}\n"
                text += f"\n✅ Best model: {best} ({results[best]:.4f})\n"
                text += f"Recommendation: Use {best} for '{target}' prediction.\n"
                def _up():
                    self.ml_metrics_area.delete("1.0",tk.END)
                    self.ml_metrics_area.insert(tk.END, text)
                    self._ml_training_done = True
                self.root.after(0, _up)
                self.say_it(f"Auto-ML done. Best: {best} ({results[best]:.4f})")
            except Exception as e:
                self.root.after(0, lambda: (messagebox.showerror("Auto-ML Error", str(e)),
                                            setattr(self,"_ml_training_done",True)))
        self._start_learning_animation(target)
        threading.Thread(target=_task, daemon=True).start()

    def _gui_save_model(self):
        if self.current_model is None: messagebox.showwarning("No Model","Train first!"); return
        path = filedialog.asksaveasfilename(defaultextension=".joblib",
                                            filetypes=[("Joblib","*.joblib"),("Pickle","*.pkl")])
        if path:
            if JOBLIB_OK: joblib.dump(self.current_model, path)
            else:
                with open(path,"wb") as f: pickle.dump(self.current_model, f)
            messagebox.showinfo("Saved", f"Model saved to: {path}")

    # ══════════════════════════════════════════════════════════════════════════
    #  DEEP LEARNING
    # ══════════════════════════════════════════════════════════════════════════
    def _train_deep_learning(self):
        target = self.dl_target_combo.get()
        if not target:
            messagebox.showwarning("No Target","Select a target column!"); return
        if self.current_df is None: return
        if not TF_OK and not TORCH_OK:
            messagebox.showerror("Deep Learning","Neither TensorFlow nor PyTorch is installed.\n"
                                 "pip install tensorflow   OR   pip install torch"); return
        framework = self.dl_framework.get()
        self.dl_metrics_area.delete("1.0",tk.END)
        self.dl_metrics_area.insert(tk.END, f"🧠 Training {framework} neural network...\n")
        self._start_learning_animation(target)
        threading.Thread(target=self._dl_training_thread,
                         args=(self.current_df.copy(), target, framework), daemon=True).start()

    def _dl_training_thread(self, df, target, framework):
        try:
            X, y, encoders, _ = self._prepare_features(df, target)
            is_clf = (y.dtype == object or y.nunique() <= 20)
            self.dl_is_clf  = is_clf
            self.dl_features = list(X.columns)
            self.dl_target   = target

            target_le = None
            if is_clf and y.dtype == object:
                target_le = LabelEncoder()
                y = pd.Series(target_le.fit_transform(y.astype(str)))
            self.dl_le = target_le

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.fillna(0))
            X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y.values,
                                                        test_size=0.2, random_state=42)

            try:
                layer_sizes = [int(x.strip()) for x in self.dl_layers_entry.get().split(",") if x.strip()]
            except Exception:
                layer_sizes = [128, 64, 32]
            try:
                epochs = int(self.dl_epochs_entry.get())
            except Exception:
                epochs = 50
            try:
                lr = float(self.dl_lr_entry.get())
            except Exception:
                lr = 0.001
            try:
                dropout = float(self.dl_dropout_entry.get())
            except Exception:
                dropout = 0.2

            n_classes = len(np.unique(y.values)) if is_clf else 1
            history_data = {"loss":[], "val_loss":[], "acc":[], "val_acc":[]}

            if framework == "TensorFlow/Keras" and TF_OK:
                # Build Keras model
                model_layers = [layers.Dense(layer_sizes[0], activation="relu",
                                             input_shape=(X_tr.shape[1],))]
                if dropout > 0: model_layers.append(layers.Dropout(dropout))
                for sz in layer_sizes[1:]:
                    model_layers.append(layers.Dense(sz, activation="relu"))
                    model_layers.append(layers.BatchNormalization())
                    if dropout > 0: model_layers.append(layers.Dropout(dropout))
                if is_clf:
                    if n_classes == 2:
                        model_layers.append(layers.Dense(1, activation="sigmoid"))
                        loss = "binary_crossentropy"; metrics_list = ["accuracy"]
                    else:
                        model_layers.append(layers.Dense(n_classes, activation="softmax"))
                        loss = "sparse_categorical_crossentropy"; metrics_list = ["accuracy"]
                else:
                    model_layers.append(layers.Dense(1))
                    loss = "mse"; metrics_list = ["mae"]

                nn = keras.Sequential(model_layers)
                nn.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                           loss=loss, metrics=metrics_list)

                class HistoryCallback(keras.callbacks.Callback):
                    def __init__(self_cb):
                        super().__init__()
                    def on_epoch_end(self_cb, epoch, logs=None):
                        if logs:
                            history_data["loss"].append(logs.get("loss",0))
                            history_data["val_loss"].append(logs.get("val_loss",0))
                            if is_clf:
                                history_data["acc"].append(logs.get("accuracy",0))
                                history_data["val_acc"].append(logs.get("val_accuracy",0))

                early_stop = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                nn.fit(X_tr, y_tr, epochs=epochs, batch_size=32,
                       validation_split=0.15, callbacks=[HistoryCallback(), early_stop], verbose=0)

                y_pred_raw = nn.predict(X_te, verbose=0)
                if is_clf:
                    if n_classes == 2:
                        y_pred = (y_pred_raw.flatten() > 0.5).astype(int)
                    else:
                        y_pred = np.argmax(y_pred_raw, axis=1)
                else:
                    y_pred = y_pred_raw.flatten()

                self.dl_model = nn
                self._dl_scaler = scaler

            elif TORCH_OK:
                # PyTorch manual
                X_tr_t = torch.tensor(X_tr.astype(np.float32))
                X_te_t = torch.tensor(X_te.astype(np.float32))
                if is_clf:
                    y_tr_t = torch.tensor(y_tr.astype(np.int64))
                    y_te_t = torch.tensor(y_te.astype(np.int64))
                else:
                    y_tr_t = torch.tensor(y_tr.astype(np.float32)).unsqueeze(1)
                    y_te_t = torch.tensor(y_te.astype(np.float32)).unsqueeze(1)

                class MLP(nn.Module):
                    def __init__(self_m, in_dim, layer_sizes, n_out, clf):
                        super().__init__()
                        layers_list = []
                        prev = in_dim
                        for sz in layer_sizes:
                            layers_list += [nn.Linear(prev, sz), nn.ReLU(), nn.BatchNorm1d(sz)]
                            if dropout > 0: layers_list.append(nn.Dropout(dropout))
                            prev = sz
                        layers_list.append(nn.Linear(prev, n_out))
                        if clf and n_out == 1: layers_list.append(nn.Sigmoid())
                        elif clf: layers_list.append(nn.Softmax(dim=1))
                        self_m.net = nn.Sequential(*layers_list)
                    def forward(self_m, x): return self_m.net(x)

                n_out = n_classes if is_clf else 1
                net = MLP(X_tr.shape[1], layer_sizes, n_out if n_classes > 2 else 1, is_clf)
                optimizer = optim.Adam(net.parameters(), lr=lr)
                criterion = (nn.BCELoss() if (is_clf and n_classes == 2) else
                             nn.CrossEntropyLoss() if is_clf else nn.MSELoss())

                dataset = TensorDataset(X_tr_t, y_tr_t)
                loader  = DataLoader(dataset, batch_size=32, shuffle=True)
                best_val = float("inf"); patience_cnt = 0

                for epoch in range(epochs):
                    net.train(); ep_loss = 0
                    for xb, yb in loader:
                        optimizer.zero_grad()
                        out = net(xb)
                        if is_clf and n_classes > 2:
                            loss = criterion(out, yb)
                        else:
                            loss = criterion(out, yb.float() if not is_clf else yb.unsqueeze(1).float())
                        loss.backward(); optimizer.step()
                        ep_loss += loss.item()
                    ep_loss /= len(loader)
                    history_data["loss"].append(ep_loss)
                    # Simple val
                    net.eval()
                    with torch.no_grad():
                        val_out = net(X_te_t)
                        if is_clf and n_classes > 2:
                            val_loss = criterion(val_out, y_te_t).item()
                        else:
                            val_loss = criterion(val_out, y_te_t.float() if not is_clf else y_te_t.unsqueeze(1).float() if not isinstance(y_te_t, torch.FloatTensor) else y_te_t).item()
                    history_data["val_loss"].append(val_loss)
                    if val_loss < best_val - 1e-4:
                        best_val = val_loss; patience_cnt = 0
                    else:
                        patience_cnt += 1
                    if patience_cnt >= 10: break

                net.eval()
                with torch.no_grad():
                    raw = net(X_te_t)
                if is_clf:
                    if n_classes == 2: y_pred = (raw.numpy().flatten() > 0.5).astype(int)
                    else: y_pred = torch.argmax(raw, dim=1).numpy()
                else:
                    y_pred = raw.numpy().flatten()

                self.dl_model   = net
                self._dl_scaler = scaler
            else:
                raise RuntimeError("No DL framework available.")

            # Metrics
            if is_clf:
                score = accuracy_score(y_te, y_pred)
                report = classification_report(y_te, y_pred, zero_division=0)
            else:
                score = r2_score(y_te, y_pred)
                report = f"R²={score:.4f}  RMSE={math.sqrt(mean_squared_error(y_te,y_pred)):.4f}"

            arch_str = " → ".join([str(X_tr.shape[1])] + [str(s) for s in layer_sizes] + [str(n_classes if is_clf else 1)])
            result_text = (f"=== 🧠 DEEP LEARNING REPORT ===\n"
                           f"Framework : {framework}\n"
                           f"Target    : {target} ({'Classification' if is_clf else 'Regression'})\n"
                           f"Architecture: {arch_str}\n"
                           f"Epochs run: {len(history_data['loss'])}\n"
                           f"{'Accuracy' if is_clf else 'R²'}: {score:.4f}\n\n"
                           f"{report if is_clf else ''}")

            self._ml_training_done = True
            self.root.after(0, lambda: self._show_dl_results(result_text, history_data, y_te, y_pred, is_clf))
            self.say_it(f"Deep learning done. Score: {score:.4f}")

        except Exception as e:
            tb = traceback.format_exc()
            self._ml_training_done = True
            self.root.after(0, lambda: (messagebox.showerror("DL Error", f"{e}\n\n{tb[-500:]}"),))

    def _show_dl_results(self, text, history, y_test, y_pred, is_clf):
        self.dl_metrics_area.delete("1.0",tk.END)
        self.dl_metrics_area.insert(tk.END, text)
        for w in self.dl_plot_container.winfo_children(): w.destroy()

        fig, axs = plt.subplots(2, 2, figsize=(13,9)); dark_fig(fig)

        # Loss curves
        axs[0,0].plot(history["loss"],     color=CC[0], lw=2, label="Train Loss")
        axs[0,0].plot(history["val_loss"], color=CC[1], lw=2, linestyle="--", label="Val Loss")
        axs[0,0].set_title("Loss Curve", color=P["fg"]); axs[0,0].legend()
        axs[0,0].set_xlabel("Epoch"); axs[0,0].set_ylabel("Loss")

        if is_clf and history["acc"]:
            axs[0,1].plot(history["acc"],     color=CC[2], lw=2, label="Train Acc")
            axs[0,1].plot(history["val_acc"], color=CC[3], lw=2, linestyle="--", label="Val Acc")
            axs[0,1].set_title("Accuracy Curve", color=P["fg"]); axs[0,1].legend()
            axs[0,1].set_xlabel("Epoch"); axs[0,1].set_ylabel("Accuracy")
        else:
            # Loss improvement
            axs[0,1].plot(np.array(history["loss"]) - np.array(history["val_loss"]),
                          color=CC[4], lw=1.5)
            axs[0,1].axhline(0, color=P["warn"], lw=1, linestyle="--")
            axs[0,1].set_title("Train−Val Loss Gap", color=P["fg"])
            axs[0,1].set_xlabel("Epoch"); axs[0,1].set_ylabel("Gap")

        if is_clf:
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[1,0],
                        linewidths=0.5, linecolor=P["border"])
            axs[1,0].set_title("Confusion Matrix", color=P["fg"])
            axs[1,0].set_xlabel("Predicted"); axs[1,0].set_ylabel("Actual")
            axs[1,1].bar(range(len(np.unique(y_test))),
                         [(y_pred==c).sum() for c in np.unique(y_test)],
                         color=CC); axs[1,1].set_title("Predictions per Class")
        else:
            axs[1,0].scatter(y_test, y_pred, alpha=0.3, s=12, color=CC[0])
            lim = [min(float(y_test.min()),float(y_pred.min())),
                   max(float(y_test.max()),float(y_pred.max()))]
            axs[1,0].plot(lim, lim, "--", color=P["warn"], lw=2)
            axs[1,0].set_title("Actual vs Predicted", color=P["fg"])
            axs[1,0].set_xlabel("Actual"); axs[1,0].set_ylabel("Predicted")
            residuals = y_pred - y_test
            axs[1,1].scatter(y_pred, residuals, alpha=0.3, s=12, color=CC[3])
            axs[1,1].axhline(0, color=P["warn"], lw=1.5, linestyle="--")
            axs[1,1].set_title("Residuals", color=P["fg"])
            axs[1,1].set_xlabel("Predicted"); axs[1,1].set_ylabel("Residual")

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.dl_plot_container)
        canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        tf = tk.Frame(self.dl_plot_container, bg=P["surface"]); tf.pack(fill=tk.X)
        NavigationToolbar2Tk(canvas, tf)

    def _dl_predict_tab(self):
        if self.dl_model is None: messagebox.showwarning("No DL Model","Train a DL model first!"); return
        val = simpledialog.askstring("DL Predict",
                                     f"Enter comma-separated values for features:\n{', '.join(self.dl_features[:5])}...")
        if not val: return
        try:
            vals = [float(v.strip()) for v in val.split(",")]
            if len(vals) != len(self.dl_features):
                messagebox.showerror("Input Error", f"Expected {len(self.dl_features)} values, got {len(vals)}")
                return
            arr = self._dl_scaler.transform(np.array([vals]))
            if TF_OK and hasattr(self.dl_model, "predict"):
                raw = self.dl_model.predict(arr, verbose=0)
                if self.dl_is_clf:
                    pred = int(np.argmax(raw) if raw.shape[-1] > 1 else int(raw.flatten()[0] > 0.5))
                else:
                    pred = float(raw.flatten()[0])
            else:
                import torch as _torch
                t = _torch.tensor(arr.astype(np.float32))
                with _torch.no_grad(): raw = self.dl_model(t).numpy()
                pred = int(np.argmax(raw)) if self.dl_is_clf else float(raw.flatten()[0])
            if self.dl_le is not None:
                try: pred = self.dl_le.inverse_transform([pred])[0]
                except: pass
            messagebox.showinfo("DL Prediction", f"Predicted {self.dl_target}: {pred}")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    # ══════════════════════════════════════════════════════════════════════════
    #  REINFORCEMENT LEARNING
    # ══════════════════════════════════════════════════════════════════════════
    def _run_rl_training(self):
        mode = self.rl_mode.get()
        threading.Thread(target=self._rl_task, args=(mode,), daemon=True).start()

    def _rl_task(self, mode):
        try:
            n       = int(self.rl_grid_entry.get())
            eps_tot = int(self.rl_episodes_entry.get())
            eps_start = float(self.rl_eps_entry.get())
        except Exception:
            n = 6; eps_tot = 2000; eps_start = 1.0

        if mode == "Q-Learning Grid-World":
            self._run_qlearning_gridworld(n, eps_tot, eps_start)
        elif mode == "Tabular Cliff Walking":
            self._run_cliff_walking(eps_tot, eps_start)
        elif mode == "Multi-Armed Bandit":
            self._run_multi_armed_bandit(eps_tot)

    def _run_qlearning_gridworld(self, n, episodes, eps_start):
        """Grid-World: agent navigates n×n grid from (0,0) to (n-1,n-1). Walls random."""
        rng = np.random.default_rng(42)
        # Place walls
        walls = set()
        for _ in range(int(n*n*0.1)):
            r, c = rng.integers(0,n), rng.integers(0,n)
            if (r,c) not in [(0,0),(n-1,n-1)]:
                walls.add((r,c))

        n_states = n*n; n_actions = 4  # up,down,left,right
        Q = np.zeros((n_states, n_actions))
        alpha=0.1; gamma=0.95; epsilon=eps_start
        rewards_ep = []
        deltas      = []

        def state_idx(r,c): return r*n + c
        def step(r,c,a):
            nr,nc = r,c
            if   a==0: nr=r-1
            elif a==1: nr=r+1
            elif a==2: nc=c-1
            elif a==3: nc=c+1
            if nr<0 or nr>=n or nc<0 or nc>=n or (nr,nc) in walls:
                return r,c,-1,False
            if nr==n-1 and nc==n-1:
                return nr,nc,10,True
            return nr,nc,-0.05,False

        for ep in range(episodes):
            r,c = 0,0; total_rew = 0; done = False; steps = 0
            while not done and steps < n*n*4:
                s = state_idx(r,c)
                if np.random.rand() < epsilon: a = np.random.randint(n_actions)
                else: a = np.argmax(Q[s])
                nr,nc,rew,done = step(r,c,a)
                ns = state_idx(nr,nc)
                old_q = Q[s,a]
                Q[s,a] += alpha*(rew + gamma*np.max(Q[ns]) - Q[s,a])
                deltas.append(abs(Q[s,a] - old_q))
                r,c = nr,nc; total_rew += rew; steps += 1
            rewards_ep.append(total_rew)
            epsilon = max(0.01, eps_start * (1 - ep/episodes))

        self.rl_Q    = Q
        self.rl_grid_n = n
        self.rl_walls  = walls
        self.rl_rewards = rewards_ep

        # Smooth rewards
        window = max(1, episodes//50)
        smooth = np.convolve(rewards_ep, np.ones(window)/window, mode="valid")

        text = (f"=== 🎮 Q-LEARNING GRID-WORLD ===\n"
                f"Grid size  : {n}×{n}  ({n*n} states)\n"
                f"Episodes   : {episodes}\n"
                f"Walls      : {len(walls)}\n"
                f"Final ε    : {epsilon:.3f}\n"
                f"Last 100 avg reward: {np.mean(rewards_ep[-100:]):.2f}\n"
                f"Best episode reward: {max(rewards_ep):.2f}\n"
                f"Q-table converged  : {'Yes ✅' if np.mean(deltas[-1000:]) < 0.001 else 'Partial ⚠'}\n\n"
                f"Greedy policy arrows shown in 'Show Policy' button.")

        def _draw():
            self.rl_metrics_area.delete("1.0",tk.END)
            self.rl_metrics_area.insert(tk.END, text)
            for w in self.rl_plot_container.winfo_children(): w.destroy()
            fig, axs = plt.subplots(1, 2, figsize=(13,5)); dark_fig(fig)

            axs[0].plot(rewards_ep, color=P["fg_dim"], lw=0.5, alpha=0.4, label="Episode reward")
            axs[0].plot(np.arange(len(smooth))+(window//2), smooth, color=CC[1], lw=2, label=f"Smooth ({window}ep)")
            axs[0].set_title("Episode Rewards", color=P["fg"])
            axs[0].set_xlabel("Episode"); axs[0].set_ylabel("Total Reward")
            axs[0].legend()

            # Q-value max heatmap
            qmax = Q.max(axis=1).reshape(n,n)
            sns.heatmap(qmax, ax=axs[1], cmap="plasma", annot=n<=8,
                        fmt=".1f", linewidths=0.3, linecolor=P["border"])
            # Mark walls, start, goal
            for (wr,wc) in walls:
                axs[1].add_patch(mpatches.Rectangle((wc,wr),1,1,color=P["danger"],alpha=0.7))
            axs[1].add_patch(mpatches.Rectangle((0,0),1,1,color=P["accent2"],alpha=0.5))
            axs[1].add_patch(mpatches.Rectangle((n-1,n-1),1,1,color=P["warn"],alpha=0.8))
            axs[1].set_title("Max Q-Values (yellow=goal)", color=P["fg"])

            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self.rl_plot_container)
            canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.root.after(0, _draw)
        self.say_it(f"Q-Learning done. Avg last-100 reward: {np.mean(rewards_ep[-100:]):.2f}")

    def _run_cliff_walking(self, episodes, eps_start):
        """4×12 Cliff Walking (Sutton & Barto)."""
        rows,cols = 4,12; n_actions=4
        Q = np.zeros((rows*cols, n_actions))
        alpha=0.1; gamma=0.99; eps=eps_start
        rewards_ep = []

        def state(r,c): return r*cols+c
        def step(r,c,a):
            nr,nc = r,c
            if a==0: nr=r-1
            elif a==1: nr=r+1
            elif a==2: nc=c-1
            elif a==3: nc=c+1
            nr = np.clip(nr,0,rows-1); nc = np.clip(nc,0,cols-1)
            # Cliff
            if nr==rows-1 and 0<nc<cols-1:
                return rows-1,0,-100,False
            if nr==rows-1 and nc==cols-1:
                return nr,nc,0,True
            return nr,nc,-1,False

        for ep in range(episodes):
            r,c=rows-1,0; tot=0; done=False; steps=0
            while not done and steps<200:
                s=state(r,c)
                a=np.random.randint(n_actions) if np.random.rand()<eps else np.argmax(Q[s])
                nr,nc,rew,done=step(r,c,a)
                ns=state(nr,nc)
                Q[s,a]+=alpha*(rew+gamma*np.max(Q[ns])-Q[s,a])
                r,c=nr,nc; tot+=rew; steps+=1
            rewards_ep.append(tot)
            eps=max(0.01,eps_start*(1-ep/episodes))

        self.rl_rewards = rewards_ep
        smooth=np.convolve(rewards_ep,np.ones(50)/50,mode="valid")
        def _draw():
            self.rl_metrics_area.delete("1.0",tk.END)
            self.rl_metrics_area.insert(tk.END,
                f"=== CLIFF WALKING (4×12) ===\nEpisodes: {episodes}\n"
                f"Best: {max(rewards_ep):.0f}  Last-100 avg: {np.mean(rewards_ep[-100:]):.1f}\n")
            for w in self.rl_plot_container.winfo_children(): w.destroy()
            fig,ax=plt.subplots(figsize=(12,5)); dark_fig(fig)
            ax.plot(rewards_ep, color=P["fg_dim"],lw=0.5,alpha=0.3)
            ax.plot(np.arange(len(smooth))+25,smooth,color=CC[0],lw=2,label="Smoothed reward")
            ax.set_title("Cliff Walking — Episode Rewards",color=P["fg"])
            ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward"); ax.legend()
            plt.tight_layout()
            canvas=FigureCanvasTkAgg(fig,master=self.rl_plot_container)
            canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)
        self.root.after(0, _draw)

    def _run_multi_armed_bandit(self, episodes):
        """10-arm bandit, epsilon-greedy vs UCB comparison."""
        n_arms = 10; true_means = np.random.randn(n_arms)
        results = {}
        for method, eps in [("ε-greedy ε=0.1",0.1),("ε-greedy ε=0.01",0.01)]:
            Q_est=np.zeros(n_arms); N=np.zeros(n_arms); rews=[]
            for t in range(episodes):
                a = np.random.randint(n_arms) if np.random.rand()<eps else np.argmax(Q_est)
                r = true_means[a]+np.random.randn()
                N[a]+=1; Q_est[a]+=(r-Q_est[a])/N[a]; rews.append(r)
            results[method] = np.cumsum(rews)/(np.arange(episodes)+1)
        # UCB
        Q_est=np.zeros(n_arms); N=np.ones(n_arms)*0.001; rews=[]
        for t in range(episodes):
            ucb = Q_est + np.sqrt(2*np.log(t+1)/(N))
            a = np.argmax(ucb); r = true_means[a]+np.random.randn()
            N[a]+=1; Q_est[a]+=(r-Q_est[a])/N[a]; rews.append(r)
        results["UCB c=2"] = np.cumsum(rews)/(np.arange(episodes)+1)

        def _draw():
            self.rl_metrics_area.delete("1.0",tk.END)
            self.rl_metrics_area.insert(tk.END,
                f"=== MULTI-ARMED BANDIT ===\nArms: {n_arms}  Pulls: {episodes}\n"
                f"Optimal arm: #{np.argmax(true_means)} (mean={max(true_means):.2f})\n\n"
                f"Average final reward:\n"
                + "\n".join([f"  {k}: {v[-1]:.4f}" for k,v in results.items()]))
            for w in self.rl_plot_container.winfo_children(): w.destroy()
            fig,ax=plt.subplots(figsize=(12,5)); dark_fig(fig)
            for i,(method,cumavg) in enumerate(results.items()):
                ax.plot(cumavg,color=CC[i],lw=2,label=method)
            ax.set_title("Multi-Armed Bandit — Average Reward",color=P["fg"])
            ax.set_xlabel("Episode"); ax.set_ylabel("Avg Reward"); ax.legend()
            plt.tight_layout()
            canvas=FigureCanvasTkAgg(fig,master=self.rl_plot_container)
            canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)
        self.root.after(0, _draw)

    def _show_rl_policy(self):
        if not hasattr(self,"rl_Q") or self.rl_Q is None:
            messagebox.showwarning("No RL Model","Run Q-Learning first!"); return
        n = self.rl_grid_n; Q = self.rl_Q
        arrows = ["↑","↓","←","→"]
        policy = Q.argmax(axis=1).reshape(n,n)
        fig,ax = plt.subplots(figsize=(max(6,n),max(6,n))); dark_fig(fig)
        ax.set_xlim(0,n); ax.set_ylim(0,n); ax.set_aspect("equal")
        for r in range(n):
            for c in range(n):
                if (r,c) in self.rl_walls:
                    ax.add_patch(mpatches.Rectangle((c,n-1-r),1,1,color=P["danger"],alpha=0.8))
                elif r==n-1 and c==n-1:
                    ax.add_patch(mpatches.Rectangle((c,n-1-r),1,1,color=P["warn"],alpha=0.8))
                    ax.text(c+0.5,n-1-r+0.5,"🏆",ha="center",va="center",fontsize=12)
                elif r==0 and c==0:
                    ax.add_patch(mpatches.Rectangle((c,n-1-r),1,1,color=P["accent2"],alpha=0.5))
                    ax.text(c+0.5,n-1-r+0.5,"S",ha="center",va="center",color="#000",fontsize=10,fontweight="bold")
                else:
                    ax.text(c+0.5,n-1-r+0.5,arrows[policy[r,c]],
                            ha="center",va="center",color=P["fg"],fontsize=14)
        for i in range(n+1):
            ax.axhline(i,color=P["border"],lw=0.5)
            ax.axvline(i,color=P["border"],lw=0.5)
        ax.set_title("Greedy Policy  (🏆=Goal  S=Start  🟥=Wall)",color=P["fg"],pad=12)
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        win=tk.Toplevel(self.root); win.title("RL Policy Map"); win.configure(bg=P["surface"])
        canvas=FigureCanvasTkAgg(fig,master=win); canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  CLEANING ACTIONS
    # ══════════════════════════════════════════════════════════════════════════
    def _gui_drop_cols(self):
        if self.current_df is None: return
        sel = [self.drop_listbox.get(i) for i in self.drop_listbox.curselection()]
        if not sel: messagebox.showwarning("No Selection","Select columns first!"); return
        if messagebox.askyesno("Confirm", f"Drop: {sel}?"):
            self.current_df = self.current_df.drop(columns=sel, errors="ignore")
            self.datasets[self.active_dataset_name] = self.current_df
            self._add_pipeline_step("Drop Columns", f"df = df.drop(columns={sel})")
            self._update_all_gui_elements()

    def _gui_remove_outliers(self):
        if self.current_df is None: return
        try: thresh = float(self.outlier_threshold.get())
        except: messagebox.showerror("Invalid","Threshold must be a number."); return
        num = self.current_df.select_dtypes(include="number")
        if num.empty: return
        z = np.abs(stats.zscore(num.fillna(num.median())))
        mask = (z < thresh).all(axis=1)
        old = len(self.current_df)
        self.current_df = self.current_df[mask].reset_index(drop=True)
        self.datasets[self.active_dataset_name] = self.current_df
        self._add_pipeline_step("Remove Outliers",
            f"from scipy import stats\nz = np.abs(stats.zscore(df.select_dtypes(include='number').fillna(0)))\n"
            f"df = df[(z < {thresh}).all(axis=1)].reset_index(drop=True)")
        self._update_all_gui_elements()
        self.say_it(f"Removed {old - len(self.current_df)} outlier rows.")

    def _gui_auto_feat_eng(self):
        if self.current_df is None: return
        df = self.current_df.copy(); added = 0; code = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                for part in ["year","month","day","dayofweek","quarter"]:
                    df[f"{col}_{part}"] = getattr(df[col].dt, part)
                    code.append(f"df['{col}_{part}'] = df['{col}'].dt.{part}")
                    added += 1
        for col in df.select_dtypes(include="object").columns:
            if df[col].nunique() > 30:
                df[f"{col}_len"] = df[col].astype(str).str.len()
                code.append(f"df['{col}_len'] = df['{col}'].astype(str).str.len()"); added += 1
        if added:
            self.current_df = df; self.datasets[self.active_dataset_name] = df
            self._add_pipeline_step("Feature Engineering", "\n".join(code))
            self._update_all_gui_elements(); self.say_it(f"Added {added} new features.")
        else:
            messagebox.showinfo("No Changes","No suitable columns found.")

    def _gui_poly_features(self):
        if not SKLEARN_OK: messagebox.showerror("sklearn required","pip install scikit-learn"); return
        if self.current_df is None: return
        num_cols = self.current_df.select_dtypes(include="number").columns.tolist()[:5]
        if not num_cols: messagebox.showwarning("No Numeric","No numeric columns."); return
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        arr = poly.fit_transform(self.current_df[num_cols].fillna(0))
        names = poly.get_feature_names_out(num_cols)
        poly_df = pd.DataFrame(arr, columns=names, index=self.current_df.index)
        new_cols = [c for c in poly_df.columns if c not in self.current_df.columns]
        self.current_df = pd.concat([self.current_df.reset_index(drop=True),
                                     poly_df[new_cols].reset_index(drop=True)], axis=1)
        self.datasets[self.active_dataset_name] = self.current_df
        self._add_pipeline_step("Polynomial Features",
            f"from sklearn.preprocessing import PolynomialFeatures\n"
            f"poly = PolynomialFeatures(degree=2, include_bias=False)\n"
            f"df[poly.get_feature_names_out(num_cols)] = poly.fit_transform(df[num_cols])")
        self._update_all_gui_elements()
        self.say_it(f"Added {len(new_cols)} polynomial features.")

    def _gui_one_hot(self):
        if self.current_df is None: return
        cat_cols = self.current_df.select_dtypes(include=["object","category"]).columns.tolist()
        low_card = [c for c in cat_cols if self.current_df[c].nunique() <= 15]
        if not low_card: messagebox.showwarning("No Cats","No low-cardinality categorical columns."); return
        self.current_df = pd.get_dummies(self.current_df, columns=low_card)
        self.datasets[self.active_dataset_name] = self.current_df
        self._add_pipeline_step("One-Hot Encoding", f"df = pd.get_dummies(df, columns={low_card})")
        self._update_all_gui_elements()
        self.say_it(f"One-hot encoded: {low_card}")

    def _gui_apply_scaling(self):
        if not SKLEARN_OK: messagebox.showerror("sklearn required","pip install scikit-learn"); return
        if self.current_df is None: return
        method = self.scaling_method.get()
        num_cols = self.current_df.select_dtypes(include="number").columns
        if num_cols.empty: return
        scaler = StandardScaler() if "Standard" in method else MinMaxScaler() if "MinMax" in method else RobustScaler()
        self.current_df[num_cols] = scaler.fit_transform(self.current_df[num_cols].fillna(self.current_df[num_cols].median()))
        self.datasets[self.active_dataset_name] = self.current_df
        self._add_pipeline_step("Feature Scaling", f"scaler = {type(scaler).__name__}()\ndf[num_cols] = scaler.fit_transform(df[num_cols])")
        self._update_all_gui_elements()
        self.say_it(f"{method} applied.")

    # ══════════════════════════════════════════════════════════════════════════
    #  DATA HEALTH
    # ══════════════════════════════════════════════════════════════════════════
    def _calculate_data_health(self, df):
        if df is None or df.empty: return
        miss_pct = df.isnull().sum().sum() / df.size * 100 if df.size else 0
        dup_pct  = df.duplicated().sum() / len(df) * 100 if len(df) else 0
        score    = max(0.0, min(100.0, 100 - miss_pct*2 - dup_pct*1.5))
        color    = P["accent2"] if score>80 else P["warn"] if score>50 else P["danger"]
        self.quality_score_label.config(text=f"Quality Score: {score:.1f} / 100", fg=color)
        self.quality_log.delete("1.0",tk.END)
        self.quality_log.insert(tk.END,
            f"=== DATA QUALITY REPORT ===\n"
            f"Score       : {score:.1f}%\n"
            f"Missing     : {df.isnull().sum().sum():,} ({miss_pct:.2f}%)\n"
            f"Duplicates  : {df.duplicated().sum():,} ({dup_pct:.2f}%)\n"
            f"Rows        : {len(df):,}   Columns: {len(df.columns)}\n\n"
            f"=== COLUMN-LEVEL ISSUES ===\n")
        for col in df.columns:
            issues = []
            miss_c = df[col].isnull().mean()*100
            if miss_c > 5: issues.append(f"{miss_c:.1f}% missing")
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() <= 1: issues.append("zero variance")
                sk = df[col].skew()
                if abs(sk) > 2: issues.append(f"skew={sk:.1f}")
            if issues:
                self.quality_log.insert(tk.END, f"  ⚠ {col}: {', '.join(issues)}\n")

    def _run_anomaly_detection(self):
        if not SKLEARN_OK: messagebox.showerror("sklearn required","pip install scikit-learn"); return
        if self.current_df is None: return
        df = self.current_df.select_dtypes(include="number").fillna(0)
        if df.empty: return
        iso = IsolationForest(contamination=0.05, random_state=42)
        labels = iso.fit_predict(df)
        count  = (labels == -1).sum()
        self.quality_log.insert(tk.END,
            f"\n=== ANOMALY DETECTION (Isolation Forest) ===\n"
            f"Anomalies: {count} ({count/len(df):.1%}) rows flagged as unusual.\n")
        self.quality_log.see(tk.END)
        self.say_it(f"Anomaly detection: {count} outliers found.")

    def _quality_chart(self):
        if self.current_df is None: return
        df = self.current_df
        miss = df.isnull().sum()
        if miss.sum() == 0:
            messagebox.showinfo("Perfect!","Dataset has no missing values! 🎉"); return
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(13,5)); dark_fig(fig)
        miss_nonzero = miss[miss>0]
        ax1.barh(miss_nonzero.index, miss_nonzero.values/len(df)*100,
                 color=[P["danger"] if v/len(df)>0.2 else P["warn"] if v/len(df)>0.05 else CC[0]
                        for v in miss_nonzero.values])
        ax1.set_title("Missing Value % per Column",color=P["fg"]); ax1.set_xlabel("%")
        ax1.axvline(20,color=P["danger"],lw=1,linestyle="--"); ax1.axvline(5,color=P["warn"],lw=1,linestyle="--")
        types = df.dtypes.astype(str).value_counts()
        ax2.pie(types.values, labels=types.index, autopct="%1.1f%%", colors=CC[:len(types)],
                wedgeprops={"edgecolor":P["surface2"]})
        ax2.set_title("Column Data Types",color=P["fg"])
        plt.tight_layout()
        win = tk.Toplevel(self.root); win.title("Quality Charts"); win.configure(bg=P["surface"])
        canvas=FigureCanvasTkAgg(fig,master=win); canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  PROFILER
    # ══════════════════════════════════════════════════════════════════════════
    def _run_profiler_analysis(self):
        if self.current_df is None: return
        df = self.current_df
        self.profiler_tree.delete(*self.profiler_tree.get_children())
        for col in df.columns:
            dtype    = str(df[col].dtype)
            non_null = int(df[col].count())
            null_pct = f"{df[col].isnull().mean()*100:.1f}%"
            unique   = int(df[col].nunique())
            is_num   = pd.api.types.is_numeric_dtype(df[col])
            mean_v   = f"{df[col].mean():.3f}"   if is_num else "N/A"
            median_v = f"{df[col].median():.3f}" if is_num else "N/A"
            mn       = f"{df[col].min():.3f}"    if is_num else str(df[col].min())[:20]
            mx       = f"{df[col].max():.3f}"    if is_num else str(df[col].max())[:20]
            skew     = f"{df[col].skew():.3f}"   if is_num else "N/A"
            kurt     = f"{df[col].kurt():.3f}"   if is_num else "N/A"
            self.profiler_tree.insert("",tk.END,
                values=(col,dtype,non_null,null_pct,unique,mean_v,median_v,mn,mx,skew,kurt))

    # ══════════════════════════════════════════════════════════════════════════
    #  PREDICTOR
    # ══════════════════════════════════════════════════════════════════════════
    def _refresh_predictor_ui(self):
        for w in self.predictor_input_container.winfo_children(): w.destroy()
        if not self.current_model:
            self.predictor_status.config(text="No model trained yet.", fg=P["danger"]); return
        self.predictor_status.config(
            text=f"Model: {type(self.current_model).__name__}  |  Target: {self.current_model_target}",
            fg=P["accent2"])
        self.predictor_entries = {}
        gf = tk.Frame(self.predictor_input_container, bg=P["bg"]); gf.pack(pady=8)
        for i, feat in enumerate(self.current_model_features):
            r, c = divmod(i, 3)
            tk.Label(gf, text=f"{feat}:", bg=P["bg"], fg=P["fg"],
                     font=("Segoe UI",9)).grid(row=r, column=c*2, padx=8, pady=4, sticky="e")
            if feat in self.current_model_encoders:
                opts = list(self.current_model_encoders[feat].classes_)
                ent  = ttk.Combobox(gf, values=opts, width=14, state="readonly")
                if opts: ent.set(opts[0])
            else:
                ent = tk.Entry(gf, width=16, bg=P["surface2"], fg=P["fg"], insertbackground=P["fg"])
            ent.grid(row=r, column=c*2+1, padx=8, pady=4, sticky="w")
            self.predictor_entries[feat] = ent

    def _run_live_prediction(self):
        if not self.current_model: messagebox.showwarning("No Model","Train a model first!"); return
        try:
            vals = []
            for feat in self.current_model_features:
                raw = self.predictor_entries[feat].get()
                if raw == "": messagebox.showerror("Input",f"Enter value for '{feat}'"); return
                if feat in self.current_model_encoders:
                    try: val = int(self.current_model_encoders[feat].transform([raw])[0])
                    except: val = 0
                else:
                    val = float(raw)
                vals.append(val)
            inp  = pd.DataFrame([vals], columns=self.current_model_features)
            pred = self.current_model.predict(inp)[0]
            self.last_prediction_input = inp
            if self._is_clf and self._target_encoder is not None:
                try: pred = self._target_encoder.inverse_transform([int(pred)])[0]
                except: pass
            if isinstance(pred, float):
                res = f"PREDICTED {self.current_model_target.upper()}: {pred:.4f}"
            else:
                res = f"PREDICTED {self.current_model_target.upper()}: {pred}"
            self.prediction_result_label.config(text=res, fg=P["accent2"])
            self.say_it(f"Prediction: {pred}")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def _explain_prediction(self):
        if self.last_prediction_input is None or self.current_model is None:
            messagebox.showwarning("No Prediction","Make a prediction first!"); return
        if not hasattr(self.current_model,"feature_importances_"):
            messagebox.showinfo("XAI","Feature importances not available for this model type."); return
        fi  = self.current_model.feature_importances_
        ser = pd.Series(fi, index=self.current_model_features).sort_values(ascending=False)
        text = "=== 💡 XAI — Why this prediction? ===\n\n"
        for i,(feat,imp) in enumerate(ser.head(8).items(),1):
            raw = self.last_prediction_input[feat].values[0]
            if feat in self.current_model_encoders:
                try: raw = self.current_model_encoders[feat].inverse_transform([int(raw)])[0]
                except: pass
            text += f"{i}. {feat.upper()}  =  {raw}\n   Importance: {imp:.2%}  {'█'*int(imp*40)}\n\n"
        win = tk.Toplevel(self.root); win.title("Prediction Explanation"); win.configure(bg=P["surface"])
        t = scrolledtext.ScrolledText(win, font=("Segoe UI",10), bg=P["surface"], fg=P["fg"],
                                       width=52, height=20, bd=0)
        t.pack(padx=12, pady=12); t.insert(tk.END, text); t.config(state=tk.DISABLED)

    def _plot_partial_dependence(self):
        if self.current_model is None or self.current_df is None: return
        if not hasattr(self.current_model,"feature_importances_"):
            messagebox.showinfo("PDP","Partial dependence requires a tree-based model."); return
        try:
            from sklearn.inspection import PartialDependenceDisplay
            X, y, _, _ = self._prepare_features(self.current_df.copy(), self.current_model_target)
            top_feat = pd.Series(self.current_model.feature_importances_,
                                  index=X.columns).sort_values(ascending=False).head(3).index.tolist()
            win = tk.Toplevel(self.root); win.title("Partial Dependence Plots")
            win.configure(bg=P["surface"])
            fig,ax = plt.subplots(1,len(top_feat),figsize=(14,5)); dark_fig(fig)
            if len(top_feat)==1: ax=[ax]
            for i,feat in enumerate(top_feat):
                feat_idx = list(X.columns).index(feat)
                vals = np.linspace(X[feat].min(), X[feat].max(), 50)
                X_mod = X.copy(); preds = []
                for v in vals:
                    X_mod[feat] = v
                    p = self.current_model.predict(X_mod)
                    preds.append(p.mean())
                ax[i].plot(vals, preds, color=CC[i], lw=2)
                ax[i].fill_between(vals, preds, alpha=0.15, color=CC[i])
                ax[i].set_title(f"PDP: {feat}",color=P["fg"])
                ax[i].set_xlabel(feat); ax[i].set_ylabel("Avg prediction")
            plt.tight_layout()
            canvas=FigureCanvasTkAgg(fig,master=win); canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("PDP Error", str(e))

    # ══════════════════════════════════════════════════════════════════════════
    #  SMART STORYTELLER — dataset-aware
    # ══════════════════════════════════════════════════════════════════════════
    def _generate_data_story(self):
        if self.current_df is None: return
        df   = self.current_df
        ctx  = dataset_context(df)
        score= self.quality_score_label.cget("text").split(": ")[-1]

        domain_labels = {
            "financial":      "💰 Financial / Revenue",
            "healthcare":     "🏥 Healthcare / Medical",
            "ecommerce_crm":  "🛒 E-commerce / CRM",
            "environmental":  "🌤️ Environmental / Climate",
            "fraud_finance":  "🔍 Fraud Detection / Fintech",
            "education":      "📚 Education / Academic",
            "hr":             "👥 Human Resources",
            "general":        "📊 General Dataset",
        }
        domain_recs = {
            "financial":     ["Monitor revenue KPIs monthly","Segment customers by LTV","Build revenue forecasting model"],
            "healthcare":    ["Validate data with medical experts","Check for HIPAA compliance","Consider class imbalance in target"],
            "ecommerce_crm": ["Build churn prediction model","Segment customers (RFM)","Analyse funnel drop-off rates"],
            "environmental": ["Check for seasonality (STL decomposition)","Interpolate missing measurements","Build time-series forecast"],
            "fraud_finance": ["Use SMOTE for class imbalance","Focus on precision-recall tradeoff","Investigate anomaly clusters"],
            "education":     ["Compare cohorts statistically","Check grade distribution normality","Model predictors of performance"],
            "hr":            ["Analyse attrition drivers","Check for bias across protected groups","Build tenure prediction model"],
            "general":       ["Start with EDA before modelling","Handle class imbalance if present","Validate model on holdout set"],
        }

        s  = f"=== 📊 DATA STORY — {self.project_title.upper()} ===\n"
        s += f"Domain: {domain_labels.get(ctx['domain'],'General')}\n"
        s += f"Problem: {self.problem_statement}\n{'─'*70}\n\n"

        s += f"1️⃣  DATA LANDSCAPE\n"
        s += f"   Records: {ctx['n_rows']:,}   Columns: {ctx['n_cols']}   Quality Score: {score}\n"
        s += f"   Numeric: {len(ctx['num_cols'])}   Categorical: {len(ctx['cat_cols'])}   DateTime: {len(ctx['dt_cols'])}\n"
        s += (f"   ✅ Zero missing values — clean dataset.\n" if ctx['miss_total']==0
              else f"   ⚠️ {ctx['miss_total']:,} missing values ({ctx['miss_pct']:.1f}%) — auto-filled.\n")
        if ctx['dup_rows']:
            s += f"   ⚠️ {ctx['dup_rows']} duplicate rows removed.\n"
        s += "\n"

        s += f"2️⃣  KEY FINDINGS\n"
        if ctx["top_corr"]:
            top_pair = list(ctx["top_corr"].items())[0]
            pair_str = f"{top_pair[0]}" if not isinstance(top_pair[0],tuple) else f"{top_pair[0][0]} ↔ {top_pair[0][1]}"
            s += f"   📈 Strongest relationship: {pair_str} (r={list(ctx['top_corr'].values())[0]:.2f})\n"
        if ctx["skewed_cols"]:
            worst = max(ctx["skewed_cols"], key=lambda c: abs(ctx["skewed_cols"][c]))
            s += f"   📉 '{worst}' is heavily skewed ({ctx['skewed_cols'][worst]:.2f}) — log-transform recommended.\n"
        if ctx["high_card"]:
            s += f"   🏷️ High-cardinality columns: {ctx['high_card'][:3]} — consider encoding carefully.\n"
        if len(ctx["num_cols"]) > 1:
            df_num = df[ctx["num_cols"]]
            for col in ctx["num_cols"][:3]:
                iqr = df_num[col].quantile(0.75)-df_num[col].quantile(0.25)
                out_count = ((df_num[col] > df_num[col].quantile(0.75)+1.5*iqr) |
                             (df_num[col] < df_num[col].quantile(0.25)-1.5*iqr)).sum()
                if out_count > 0:
                    s += f"   📦 '{col}': {out_count} outliers beyond 1.5×IQR\n"
        s += "\n"

        if self.current_model:
            s += f"3️⃣  ML MODEL INSIGHTS\n"
            s += f"   Model: {type(self.current_model).__name__} → Target: '{self.current_model_target}'\n"
            if hasattr(self.current_model,"feature_importances_"):
                fi = pd.Series(self.current_model.feature_importances_,
                               index=self.current_model_features).sort_values(ascending=False)
                s += f"   Top 3 drivers:\n"
                for feat, imp in fi.head(3).items():
                    s += f"      • {feat}: {imp:.2%} importance\n"
            s += "\n"
        else:
            s += f"3️⃣  ML STATUS\n   No model trained yet. Go to ML Lab.\n\n"

        if self.dl_model is not None:
            s += f"4️⃣  DEEP LEARNING\n"
            s += f"   Neural network trained for '{self.dl_target}' ({'Classification' if self.dl_is_clf else 'Regression'}).\n\n"

        s += f"5️⃣  RECOMMENDATIONS — {domain_labels.get(ctx['domain'])}\n"
        for i, rec in enumerate(domain_recs.get(ctx["domain"], domain_recs["general"]), 1):
            s += f"   {i}. {rec}\n"
        if ctx["miss_total"] > 0:
            worst_miss = df.isnull().sum().idxmax()
            s += f"   🛠  Investigate missing-data root cause in '{worst_miss}'\n"
        if ctx["skewed_cols"]:
            s += f"   📊 Apply log1p() transform to: {list(ctx['skewed_cols'].keys())[:3]}\n"
        s += "\n"

        s += f"6️⃣  CONCLUSION\n"
        s += (f"   The dataset is in excellent condition (Score {score}). "
              if float(score.split("/")[0].strip()) > 80
              else f"   Data quality needs attention before modelling (Score {score}). ")
        s += (f"Focus on '{list(ctx['skewed_cols'].keys())[0]}' distribution and correlation insights "
              if ctx['skewed_cols'] else "")
        s += "for highest analytical impact.\n"

        self.story_area.delete("1.0",tk.END)
        self.story_area.insert(tk.END, s)
        self.say_it("Data story generated.")

    def _export_smart_pdf(self):
        if self.current_df is None: return
        path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                            filetypes=[("PDF","*.pdf")])
        if not path: return
        df  = self.current_df
        ctx = dataset_context(df)
        miss_pct = ctx["miss_pct"]
        dup_pct  = df.duplicated().sum()/len(df)*100 if len(df) else 0
        score    = max(0, min(100, 100 - miss_pct*2 - dup_pct*1.5))

        try:
            with PdfPages(path) as pdf:
                # Cover page
                fig, ax = plt.subplots(figsize=(11,8.5)); ax.axis("off")
                y = 0.96
                def wl(t, size=10, bold=False, mono=False, color="#e6edf3"):
                    nonlocal y
                    for ln in textwrap.wrap(str(t), width=100):
                        ax.text(0.04, y, ln, fontsize=size, va="top",
                                family="monospace" if mono else "sans-serif",
                                fontweight="bold" if bold else "normal", color=color)
                        y -= 0.028
                    y -= 0.006
                fig.patch.set_facecolor("#0d1117")
                wl("🥁  PROJECT DRUM v4.0 — Data Science Report", 18, bold=True, color="#58a6ff")
                wl(f"Source: {self.current_source}", 10, color="#8b949e")
                wl(f"Domain: {ctx['domain'].replace('_',' ').title()}", 10, color="#3fb950")
                wl(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", 10, color="#8b949e")
                wl("─"*90, 9, mono=True, color="#30363d")
                wl(f"Rows: {len(df):,}   Columns: {len(df.columns)}   Quality Score: {score:.1f}%", 13, bold=True, color="#e3b341")
                wl(f"Project: {self.project_title}", 11, bold=True, color="#e6edf3")
                wl(self.problem_statement, 10, color="#e6edf3")
                wl("─"*90, 9, mono=True, color="#30363d")
                wl("KEY INSIGHTS:", 12, bold=True, color="#d2a8ff")
                if ctx["top_corr"]:
                    pair = list(ctx["top_corr"].items())[0]
                    pair_key = pair[0] if not isinstance(pair[0],tuple) else f"{pair[0][0]} ↔ {pair[0][1]}"
                    wl(f"  • Strongest correlation: {pair_key} (r={list(ctx['top_corr'].values())[0]:.2f})", color="#e6edf3")
                if ctx["skewed_cols"]:
                    wl(f"  • Skewed columns: {list(ctx['skewed_cols'].keys())[:3]}", color="#e3b341")
                if self.current_model:
                    wl(f"  • Best model: {type(self.current_model).__name__} → Target: {self.current_model_target}", color="#3fb950")
                pdf.savefig(fig); plt.close(fig)

                # Stats page
                fig, ax = plt.subplots(figsize=(11,8.5)); ax.axis("off"); y = 0.96
                fig.patch.set_facecolor("#0d1117")
                ax.text(0.04, y, "Statistical Summary", fontsize=13, fontweight="bold",
                        color="#58a6ff"); y -= 0.05
                desc = df.describe(include="all").round(2).T
                for ln in desc.to_string().split("\n"):
                    ax.text(0.02, y, ln, fontsize=7, family="monospace", color="#e6edf3"); y -= 0.022
                    if y < 0.05: break
                pdf.savefig(fig); plt.close(fig)

                # Visuals page
                num_cols = df.select_dtypes(include="number").columns.tolist()
                cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
                fig, axs = plt.subplots(2,3,figsize=(15,9)); dark_fig(fig)
                if cat_cols:
                    vc = df[cat_cols[0]].value_counts().head(10)
                    vc.plot(kind="bar", ax=axs[0,0], color=CC[:len(vc)], edgecolor="none")
                    axs[0,0].set_title(f"Top {cat_cols[0]}", color=P["fg"])
                if len(num_cols) >= 2:
                    corr = df[num_cols].corr()
                    sns.heatmap(corr, ax=axs[0,1], cmap="RdYlGn", annot=True, fmt=".1f", annot_kws={"size":6})
                    axs[0,1].set_title("Correlation", color=P["fg"])
                if num_cols:
                    df[num_cols[0]].hist(bins=30, ax=axs[0,2], color=CC[0], edgecolor="none", density=True, alpha=0.8)
                    try: df[num_cols[0]].plot(kind="kde", ax=axs[0,2], color=P["warn"])
                    except: pass
                    axs[0,2].set_title(f"{num_cols[0]} Dist", color=P["fg"])
                if len(num_cols) >= 2:
                    sample = df.sample(min(800,len(df)))
                    axs[1,0].scatter(sample[num_cols[0]], sample[num_cols[1]], alpha=0.3, s=8, color=CC[2])
                    try:
                        m,b,r,_,_ = stats.linregress(df[num_cols[0]].fillna(0), df[num_cols[1]].fillna(0))
                        xr = np.linspace(df[num_cols[0]].min(),df[num_cols[0]].max(),100)
                        axs[1,0].plot(xr, m*xr+b, color=P["warn"], lw=2, label=f"r={r:.2f}"); axs[1,0].legend()
                    except: pass
                    axs[1,0].set_title(f"{num_cols[0]} vs {num_cols[1]}", color=P["fg"])
                if num_cols:
                    box_cols = num_cols[:min(5,len(num_cols))]
                    bp = df[box_cols].plot(kind="box", ax=axs[1,1], patch_artist=True)
                    for patch, col in zip(axs[1,1].patches, CC): patch.set_facecolor(col); patch.set_alpha(0.7)
                    axs[1,1].set_title("Boxplots", color=P["fg"])
                # Missing
                miss = df.isnull().sum()
                miss_nz = miss[miss>0]
                if not miss_nz.empty:
                    axs[1,2].barh(miss_nz.index[:10], miss_nz.values[:10]/len(df)*100, color=P["danger"])
                    axs[1,2].set_title("Missing % per Column", color=P["fg"])
                else:
                    axs[1,2].text(0.5,0.5,"✅ No Missing Values",ha="center",color=P["accent2"],
                                  transform=axs[1,2].transAxes, fontsize=13)
                plt.tight_layout()
                pdf.savefig(fig); plt.close(fig)

            messagebox.showinfo("PDF Saved", f"Smart report saved to:\n{path}")
            self.say_it("Smart PDF report exported.")
        except Exception as e:
            messagebox.showerror("PDF Error", str(e))

    # ══════════════════════════════════════════════════════════════════════════
    #  COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    def _run_comparison_analysis(self):
        n1, n2 = self.comp_ds1.get(), self.comp_ds2.get()
        if not n1 or not n2 or n1==n2:
            messagebox.showwarning("Select Two","Choose two different datasets."); return
        d1, d2 = self.datasets[n1], self.datasets[n2]
        self.comp_log.delete("1.0",tk.END)
        self.comp_log.insert(tk.END, f"{'Metric':<24} | {n1[:22]:<22} | {n2[:22]}\n{'─'*72}\n")
        for label, v1, v2 in [("Rows",len(d1),len(d2)),("Columns",len(d1.columns),len(d2.columns)),
                               ("Missing",d1.isnull().sum().sum(),d2.isnull().sum().sum()),
                               ("Duplicates",d1.duplicated().sum(),d2.duplicated().sum())]:
            self.comp_log.insert(tk.END, f"{label:<24} | {str(v1):<22} | {v2}\n")
        common = list(set(d1.select_dtypes(include="number").columns) &
                      set(d2.select_dtypes(include="number").columns))
        for w in self.comp_plot_container.winfo_children(): w.destroy()
        if common:
            tc = common[0]
            fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,5)); dark_fig(fig)
            sns.kdeplot(d1[tc].dropna(), ax=ax1, label=n1[:12], fill=True, color=CC[0], alpha=0.35)
            sns.kdeplot(d2[tc].dropna(), ax=ax1, label=n2[:12], fill=True, color=CC[1], alpha=0.35)
            ax1.set_title(f"Distribution — {tc}",color=P["fg"]); ax1.legend()
            ax2.boxplot([d1[tc].dropna(), d2[tc].dropna()], labels=[n1[:10],n2[:10]],
                        patch_artist=True,
                        boxprops={"facecolor":CC[0],"alpha":0.7},
                        medianprops={"color":P["warn"],"lw":2})
            ax2.set_title(f"Box Comparison — {tc}",color=P["fg"]); plt.tight_layout()
            canvas=FigureCanvasTkAgg(fig,master=self.comp_plot_container)
            canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  SQL
    # ══════════════════════════════════════════════════════════════════════════
    def _refresh_sql_schema(self):
        if not self.datasets: return
        try:
            if self.sql_conn is None:
                self.sql_conn = sqlite3.connect(":memory:", check_same_thread=False)
            for name, df in self.datasets.items():
                safe = re.sub(r"[^a-zA-Z0-9]","_",name).lower()
                df.to_sql(safe, self.sql_conn, index=False, if_exists="replace")
            if self.current_df is not None:
                self.current_df.to_sql("data", self.sql_conn, index=False, if_exists="replace")
            self.schema_tree.delete(*self.schema_tree.get_children())
            cur = self.sql_conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            for (tbl,) in cur.fetchall():
                tid = self.schema_tree.insert("",tk.END,text=f"📋 {tbl}", open=True)
                cur.execute(f"PRAGMA table_info(`{tbl}`);")
                for col in cur.fetchall():
                    self.schema_tree.insert(tid,tk.END,text=f"  🔹 {col[1]} ({col[2]})")
        except Exception as e:
            print(f"Schema refresh error: {e}")

    def _execute_gui_sql(self):
        script = self.sql_editor.get("1.0",tk.END).strip()
        if not script or self.current_df is None: return
        try:
            if self.sql_conn is None:
                self.sql_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self.current_df.to_sql("data", self.sql_conn, index=False, if_exists="replace")
            stmts = [s.strip() for s in script.split(";") if s.strip()]
            last = None
            for stmt in stmts:
                clean = re.sub(r"--.*?\n|/\*.*?\*/","",stmt,flags=re.DOTALL).strip().lower()
                if clean.startswith(("select","with","pragma","explain")):
                    last = pd.read_sql_query(stmt, self.sql_conn)
                else:
                    self.sql_conn.execute(stmt); self.sql_conn.commit()
            self.sql_tree.delete(*self.sql_tree.get_children())
            if last is not None:
                self.sql_tree["columns"] = list(last.columns)
                for c in last.columns: self.sql_tree.heading(c,text=c); self.sql_tree.column(c,width=110,anchor="center")
                for _,row in last.head(1000).iterrows(): self.sql_tree.insert("",tk.END,values=list(row))
                self.say_it(f"Query returned {len(last)} rows.")
            self._refresh_sql_schema()
        except Exception as e:
            messagebox.showerror("SQL Error", str(e))

    # ══════════════════════════════════════════════════════════════════════════
    #  EXPORT
    # ══════════════════════════════════════════════════════════════════════════
    def _export_data(self, fmt):
        if self.current_df is None: messagebox.showwarning("No Data","Load data first!"); return
        ext = ".xlsx" if fmt=="excel" else f".{fmt}"
        path = filedialog.asksaveasfilename(defaultextension=ext, filetypes=[(fmt.upper(),f"*{ext}"),("All","*.*")])
        if not path: return
        try:
            if   fmt=="csv":   self.current_df.to_csv(path,index=False)
            elif fmt=="excel": self.current_df.to_excel(path,index=False)
            elif fmt=="json":  self.current_df.to_json(path,orient="records",indent=2)
            messagebox.showinfo("Exported",f"Saved to: {path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _export_sql_schema(self):
        if self.current_df is None: return
        ddl = "-- SQL DDL generated by PROJECT DRUM v4.0\nCREATE TABLE data_table (\n"
        cols = []
        for col in self.current_df.columns:
            dt = self.current_df[col].dtype
            sql_t = ("INT" if pd.api.types.is_integer_dtype(dt) else
                     "FLOAT" if pd.api.types.is_float_dtype(dt) else
                     "DATETIME" if pd.api.types.is_datetime64_any_dtype(dt) else "VARCHAR(255)")
            cols.append(f"    `{col}` {sql_t}")
        ddl += ",\n".join(cols) + "\n);\n"
        path = filedialog.asksaveasfilename(defaultextension=".sql", filetypes=[("SQL","*.sql")])
        if path:
            with open(path,"w") as f: f.write(ddl)
            messagebox.showinfo("Saved",f"SQL schema saved to: {path}")

    def _export_data_dictionary(self):
        if self.current_df is None: return
        rows = [{"Column":col,"Type":str(self.current_df[col].dtype),
                 "Missing":int(self.current_df[col].isnull().sum()),
                 "Unique":int(self.current_df[col].nunique()),
                 "Sample":str(self.current_df[col].iloc[0]) if len(self.current_df) else ""}
                for col in self.current_df.columns]
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if path: pd.DataFrame(rows).to_csv(path,index=False); messagebox.showinfo("Saved",f"Data dictionary saved.")

    def _export_powerbi_ready(self):
        if self.current_df is None: return
        df = self.current_df.copy()
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if path: df.to_csv(path,index=False); messagebox.showinfo("Saved","Power BI CSV ready.")

    def _generate_powerbi_script(self):
        if self.current_df is None: return
        script = f"# Power BI Python Script — PROJECT DRUM v4\nimport pandas as pd\ndata = pd.DataFrame({self.current_df.head(100).to_dict(orient='list')})\n"
        self.root.clipboard_clear(); self.root.clipboard_append(script)
        messagebox.showinfo("Copied","Power BI script copied to clipboard!")

    def _generate_calendar_table(self):
        if self.current_df is None: return
        date_cols = self.current_df.select_dtypes(include="datetime64").columns
        if not len(date_cols): messagebox.showwarning("No Dates","No datetime columns found."); return
        start, end = self.current_df[date_cols[0]].min(), self.current_df[date_cols[0]].max()
        if pd.isnull(start) or pd.isnull(end): return
        dr = pd.date_range(start,end,freq="D")
        cal = pd.DataFrame({"Date":dr})
        cal["Year"]      = cal.Date.dt.year; cal["Month"] = cal.Date.dt.month
        cal["MonthName"] = cal.Date.dt.strftime("%B"); cal["Quarter"] = "Q"+cal.Date.dt.quarter.astype(str)
        cal["DayOfWeek"] = cal.Date.dt.strftime("%A"); cal["IsWeekend"] = cal.Date.dt.dayofweek >= 5
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if path: cal.to_csv(path,index=False); messagebox.showinfo("Saved",f"{len(cal)} row calendar saved.")

    def _export_pbi_theme(self):
        theme = json.dumps({"name":"PROJECT DRUM Dark",
                             "dataColors":CC,"background":P["bg"],"foreground":P["fg"],"tableAccent":P["accent"]},
                            indent=2)
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if path:
            with open(path,"w") as f: f.write(theme)
            messagebox.showinfo("Saved","PBI theme saved.")

    def _export_project_package(self):
        if self.current_df is None: return
        path = filedialog.asksaveasfilename(defaultextension=".zip", filetypes=[("ZIP","*.zip")])
        if not path: return
        try:
            with zipfile.ZipFile(path,"w") as z:
                z.writestr("cleaned_data.csv",  self.current_df.to_csv(index=False))
                z.writestr("pipeline.py",        self.pipeline_area.get("1.0",tk.END))
                z.writestr("project_info.txt",
                            f"Project: {self.project_title}\nProblem: {self.problem_statement}\n"
                            f"Exported: {time.ctime()}\nDomain: {dataset_context(self.current_df)['domain']}\n")
            messagebox.showinfo("Package Ready",f"Full project package saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    # ══════════════════════════════════════════════════════════════════════════
    #  PIPELINE
    # ══════════════════════════════════════════════════════════════════════════
    def _add_pipeline_step(self, name, code):
        self.pipeline_steps.append({"step":name, "code":code, "time":time.strftime("%H:%M:%S")})
        self._refresh_pipeline_ui()

    def _refresh_pipeline_ui(self):
        self.pipeline_area.delete("1.0",tk.END)
        self.pipeline_area.insert(tk.END,"# PROJECT DRUM v4.0 Auto-Generated Pipeline\nimport pandas as pd\nimport numpy as np\n\n")
        for i, s in enumerate(self.pipeline_steps, 1):
            self.pipeline_area.insert(tk.END, f"# Step {i}: {s['step']} ({s['time']})\n{s['code']}\n\n")

    def _clear_pipeline(self):
        if messagebox.askyesno("Clear","Delete all pipeline steps?"):
            self.pipeline_steps = []; self._refresh_pipeline_ui()

    def _export_pipeline_script(self):
        if not self.pipeline_steps: messagebox.showwarning("Empty","No steps recorded."); return
        path = filedialog.asksaveasfilename(defaultextension=".py", filetypes=[("Python","*.py")])
        if path:
            with open(path,"w") as f: f.write(self.pipeline_area.get("1.0",tk.END))
            messagebox.showinfo("Saved",f"Pipeline saved to: {path}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PYTHON FILE READER
    # ══════════════════════════════════════════════════════════════════════════
    def _load_python_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Python / Notebook","*.py *.ipynb"),("Python","*.py"),("Notebook","*.ipynb"),("All","*.*")])
        if not path: return
        ext = os.path.splitext(path)[1].lower()
        if ext == ".ipynb":
            # Load as dataset AND open in editor
            df = self._load_ipynb(path)
            if df is not None:
                name = os.path.basename(path)
                self.datasets[name] = df; self.active_dataset_name = name
                self.current_df = df; self.current_source = name
                self._update_all_gui_elements()
                self.say_it(f"Notebook loaded: {name}")
        else:
            try:
                with open(path,"r",encoding="utf-8",errors="replace") as f:
                    src = f.read()
                self.py_editor.delete("1.0",tk.END)
                self.py_editor.insert("1.0",src)
                self._py_filepath = path
                self.py_output.delete("1.0",tk.END)
                self.py_output.insert(tk.END, f"✅ Loaded: {path}\n   Lines: {src.count(chr(10))+1}   Chars: {len(src):,}\n\n")
                self.say_it(f"Python file loaded: {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("File Error", str(e))

    def _run_python_file(self):
        code = self.py_editor.get("1.0",tk.END)
        if not code.strip(): messagebox.showwarning("Empty","No Python code to run!"); return
        if not messagebox.askyesno("Run Code","⚠️ This will execute the Python code.\nOnly run code you trust. Continue?"): return
        self.py_output.delete("1.0",tk.END)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        captured = StringIO(); sys.stdout = sys.stderr = captured
        try:
            exec(compile(code, self._py_filepath or "<editor>","exec"), {"__name__":"__main__","pd":pd,"np":np})
            output = captured.getvalue() or "(no output)"
            self.py_output.insert(tk.END, f"✅ Execution complete:\n\n{output}")
        except Exception as exc:
            self.py_output.insert(tk.END, f"❌ Error:\n{exc}\n\n{captured.getvalue()}")
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    def _analyse_python_ast(self):
        code = self.py_editor.get("1.0",tk.END)
        if not code.strip(): return
        self.py_output.delete("1.0",tk.END)
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            self.py_output.insert(tk.END, f"❌ Syntax Error: {e}"); return
        functions = [n.name for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)]
        classes_  = [n.name for n in ast.walk(tree) if isinstance(n,ast.ClassDef)]
        imports_  = set()
        for n in ast.walk(tree):
            if isinstance(n,ast.Import): imports_ |= {a.name for a in n.names}
            elif isinstance(n,ast.ImportFrom): imports_.add(n.module or "?")
        imports_ = sorted(imports_)
        lines = code.count("\n")+1
        depth = sum(1 for _ in ast.walk(tree) if isinstance(_, (ast.For,ast.While,ast.If,ast.Try,ast.With)))
        report = (f"=== 🔍 AST Analysis ===\n\n"
                  f"  Lines      : {lines}\n"
                  f"  Functions  : {len(functions)} — {', '.join(functions[:10]) or 'none'}\n"
                  f"  Classes    : {len(classes_)} — {', '.join(classes_) or 'none'}\n"
                  f"  Imports    : {len(imports_)} — {', '.join(list(imports_)[:12]) or 'none'}\n\n"
                  f"  Complexity : {depth} branching nodes\n"
                  f"  {'⚠️ High complexity — consider refactoring.' if depth>50 else '✅ Complexity OK'}\n")
        self.py_output.insert(tk.END, report)

    # ══════════════════════════════════════════════════════════════════════════
    #  SESSION SAVE / LOAD
    # ══════════════════════════════════════════════════════════════════════════
    def save_session(self):
        if not self.datasets: messagebox.showwarning("Empty","Nothing to save!"); return
        path = filedialog.asksaveasfilename(defaultextension=".drum", filetypes=[("PROJECT DRUM","*.drum")])
        if not path: return
        try:
            with open(path,"wb") as f:
                pickle.dump({"datasets":self.datasets,"active":self.active_dataset_name,
                             "pipeline":self.pipeline_steps,"title":self.project_title,
                             "problem":self.problem_statement}, f)
            messagebox.showinfo("Saved", f"Session saved to: {path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def load_session(self):
        path = filedialog.askopenfilename(filetypes=[("PROJECT DRUM","*.drum"),("All","*.*")])
        if not path: return
        try:
            with open(path,"rb") as f: state = pickle.load(f)
            self.datasets            = state.get("datasets",{})
            self.active_dataset_name = state.get("active")
            self.pipeline_steps      = state.get("pipeline",[])
            self.project_title       = state.get("title","Loaded Project")
            self.problem_statement   = state.get("problem","")
            if self.active_dataset_name and self.active_dataset_name in self.datasets:
                self.current_df = self.datasets[self.active_dataset_name]
                self.current_source = self.active_dataset_name
            self._update_all_gui_elements(); self._refresh_pipeline_ui()
            messagebox.showinfo("Loaded","Session loaded successfully.")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN WORKFLOW
    # ══════════════════════════════════════════════════════════════════════════
    def begin_project(self):
        if self.workflow_thread and self.workflow_thread.is_alive():
            messagebox.showwarning("Busy","An analysis is already in progress!"); return
        self.workflow_thread = threading.Thread(target=self._project_workflow, daemon=True)
        self.workflow_thread.start()
        self.add_btn.config(state=tk.DISABLED)

    def _project_workflow(self):
        try:
            title = self.get_gui_input("Project Title:", "Project Setup")
            if title: self.project_title = title

            problem = self.get_gui_input(
                "What real-world problem are you solving?\n"
                "(e.g. Predict employee churn, detect fraud, forecast sales)",
                "Problem Statement")
            if problem: self.problem_statement = problem

            self.say_it(f"Project: {self.project_title}")

            path = self.get_file_path([
                ("All supported",   "*.csv;*.xlsx;*.xls;*.json;*.parquet;*.py;*.ipynb"),
                ("CSV",             "*.csv"),
                ("Excel",           "*.xlsx;*.xls"),
                ("JSON",            "*.json"),
                ("Parquet",         "*.parquet"),
                ("Python",          "*.py"),
                ("Jupyter Notebook","*.ipynb"),
                ("All",             "*.*"),
            ])

            if not path: self.say_it("No file selected."); return

            ext  = os.path.splitext(path)[1].lower()
            name = os.path.basename(path)

            if ext == ".py":
                def _open_py():
                    try:
                        with open(path,"r",encoding="utf-8",errors="replace") as f:
                            src = f.read()
                        self.py_editor.delete("1.0",tk.END)
                        self.py_editor.insert("1.0",src)
                        self._py_filepath = path
                        self.notebook.select(18)
                    except Exception: pass
                self.root.after(0, _open_py)

            df = self._load_file(path)
            if df is None or df.empty:
                self.say_it("No data loaded."); return

            cleaned = self.clean_and_format(df)
            self.datasets[name]      = cleaned
            self.active_dataset_name = name
            self.current_df          = cleaned
            self.current_source      = name

            self.run_full_analysis(cleaned)

            if messagebox.askyesno("ML Lab","Train a machine learning model now?"):
                self.train_ml_model(cleaned)

            messagebox.showinfo("Done",
                "✅ Analysis complete!\n\nExplore:\n"
                "• 📊 Data   • 📈 Statistics   • 🎨 Visuals\n"
                "• 🌐 3D     • 🔭 EDA Suite    • 🗄️ SQL Lab\n"
                "• 🤖 ML Lab • 🧠 Deep Learning• 🎮 RL\n"
                "• 🔮 Predictor  • 📖 Storyteller • 📓 Notebook Reader")
            self.say_it("Analysis complete! Explore the dashboard.")
        except Exception as e:
            self.say_it(f"Critical error: {e}")
            messagebox.showerror("Error", str(e))
        finally:
            self.root.after(0, lambda: self.add_btn.config(state=tk.NORMAL))

    def train_ml_model(self, df):
        if not SKLEARN_OK: return
        choices = ["Supervised (predict a column)","Unsupervised (K-Means + t-SNE)","Auto-ML (benchmark all)"]
        sel = self.get_gui_choice("Which ML mode?", choices, "ML Mode")
        if not sel: return
        idx = int(sel) - 1
        if idx == 0:
            target = self.get_gui_input("Target column:", "Supervised ML")
            if target and target in df.columns:
                self.ml_target_combo.set(target)
                self.root.after(0, self._run_gui_ml)
        elif idx == 1:
            self.root.after(0, self._run_unsupervised)
        else:
            target = self.get_gui_input("Target column for Auto-ML:", "Auto-ML")
            if target and target in df.columns:
                self.ml_target_combo.set(target)
                self.root.after(0, lambda: self._run_automl_optimization(df))

    def _run_unsupervised(self):
        if not SKLEARN_OK or self.current_df is None: return
        df = self.current_df.select_dtypes(include="number").fillna(0)
        if df.empty: return
        def _task():
            try:
                km  = KMeans(n_clusters=3, random_state=42, n_init=10)
                lbl = km.fit_predict(df)
                pca = PCA(n_components=2)
                red = pca.fit_transform(df)
                try:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30,len(df)//3))
                    red_tsne = tsne.fit_transform(df)
                except Exception:
                    red_tsne = red
                def _show():
                    for w in self.ml_plot_container.winfo_children(): w.destroy()
                    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5)); dark_fig(fig)
                    sc1 = ax1.scatter(red[:,0],red[:,1],c=lbl,cmap="plasma",alpha=0.65,s=16)
                    fig.colorbar(sc1,ax=ax1,label="Cluster")
                    ax1.set_title("K-Means Clusters (PCA)",color=P["fg"])
                    sc2 = ax2.scatter(red_tsne[:,0],red_tsne[:,1],c=lbl,cmap="plasma",alpha=0.65,s=16)
                    fig.colorbar(sc2,ax=ax2,label="Cluster")
                    ax2.set_title("K-Means Clusters (t-SNE)",color=P["fg"])
                    plt.tight_layout()
                    canvas=FigureCanvasTkAgg(fig,master=self.ml_plot_container)
                    canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)
                    self.ml_metrics_area.delete("1.0",tk.END)
                    self.ml_metrics_area.insert(tk.END,"=== UNSUPERVISED: K-MEANS (k=3) ===\n")
                    for i in range(3):
                        self.ml_metrics_area.insert(tk.END,f"Cluster {i}: {(lbl==i).sum()} rows\n")
                    self.ml_metrics_area.insert(tk.END,f"\nPCA var explained: {pca.explained_variance_ratio_.sum():.2%}\n")
                    self._ml_training_done = True
                self.root.after(0, _show)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Cluster Error",str(e)))
                self._ml_training_done = True
        self._start_learning_animation("K-Means Clustering")
        threading.Thread(target=_task, daemon=True).start()


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  PROJECT DRUM v4.0 — Enterprise Data Science Dashboard")
    print("  Dependencies: pandas numpy matplotlib seaborn scipy sklearn")
    print("  Optional:     tensorflow torch nbformat pyttsx3 joblib")
    print("=" * 70)
    app = ProjectDrum()
    app.root.mainloop()