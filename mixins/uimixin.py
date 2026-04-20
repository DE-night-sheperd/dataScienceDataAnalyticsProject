from core import *
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
from scipy import stats
import threading, time, math, os, json, sqlite3, queue, pickle, re, warnings

class UIMixin:
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
        self._build_unsupervised_tab()
        self._build_ts_tab()  # NEW: Time Series
        self._build_nlp_tab() # NEW: NLP
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


    def _build_unsupervised_tab(self):
        p = self._tab("🧩 Unsupervised")
        self._sec(p, "Clustering & Dimensionality Reduction")
        ctrl = tk.Frame(p, bg=P["bg"]); ctrl.pack(fill=tk.X, padx=14, pady=6)
        
        tk.Label(ctrl, text="Algorithm:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.unsup_algo = ttk.Combobox(ctrl, state="readonly", width=18,
                                       values=["K-Means", "DBSCAN", "Agglomerative", "PCA (2D)", "t-SNE (2D)"])
        self.unsup_algo.set("K-Means"); self.unsup_algo.pack(side=tk.LEFT, padx=4)
        
        tk.Label(ctrl, text="Clusters/Params (e.g. k=3 or eps=0.5):", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.unsup_k = tk.Entry(ctrl, width=6, bg=P["surface2"], fg=P["fg"], insertbackground=P["fg"])
        self.unsup_k.insert(0, "3"); self.unsup_k.pack(side=tk.LEFT, padx=4)
        
        self._btn(ctrl, "🚀 Run Analysis", self._run_unsupervised, P["accent3"])
        
        panes = tk.Frame(p, bg=P["bg"]); panes.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.unsup_metrics_area = self._scroll(panes, width=38); self.unsup_metrics_area.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0,6))
        self.unsup_plot_container = tk.Frame(panes, bg=P["surface"]); self.unsup_plot_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

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
        self._btn(ctrl, "⚙️ Hyper-Tune", self._run_hyperparameter_tuning, P["accent"])
        self._btn(ctrl, "📈 Learning Curve",    self._plot_learning_curve,   P["deep"])
        self._btn(ctrl, "🔁 K-Fold CV",         self._plot_cv_scores,        P["deep"])
        self._btn(ctrl, "💾 Save",        self._gui_save_model,        P["accent2"])
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

    def _build_ts_tab(self):
        p = self._tab("📈 Time Series")
        self._sec(p, "Time Series & Forecasting Lab")
        ctrl = tk.Frame(p, bg=P["bg"]); ctrl.pack(fill=tk.X, padx=14, pady=6)
        
        tk.Label(ctrl, text="Time Column:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.ts_time_combo = ttk.Combobox(ctrl, state="readonly", width=18); self.ts_time_combo.pack(side=tk.LEFT, padx=4)
        
        tk.Label(ctrl, text="Value Column:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.ts_val_combo = ttk.Combobox(ctrl, state="readonly", width=18); self.ts_val_combo.pack(side=tk.LEFT, padx=4)
        
        self._btn(ctrl, "📅 Parse Dates", self._parse_dates, P["accent"])
        self._btn(ctrl, "📊 Plot Trend", self._plot_ts_trend, P["warn"])
        self._btn(ctrl, "🔮 Forecast (ARIMA)", self._forecast_arima, P["accent3"])
        
        panes = tk.Frame(p, bg=P["bg"]); panes.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.ts_metrics_area = self._scroll(panes, width=38); self.ts_metrics_area.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0,6))
        self.ts_plot_container = tk.Frame(panes, bg=P["surface"]); self.ts_plot_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def _build_nlp_tab(self):
        p = self._tab("📝 NLP Lab")
        self._sec(p, "Text Analytics & NLP Suite")
        ctrl = tk.Frame(p, bg=P["bg"]); ctrl.pack(fill=tk.X, padx=14, pady=6)
        
        tk.Label(ctrl, text="Text Column:", bg=P["bg"], fg=P["fg"], font=("Segoe UI",9)).pack(side=tk.LEFT)
        self.nlp_text_combo = ttk.Combobox(ctrl, state="readonly", width=22); self.nlp_text_combo.pack(side=tk.LEFT, padx=4)
        
        self._btn(ctrl, "☁️ WordCloud", self._generate_wordcloud, P["accent"])
        self._btn(ctrl, "😊 Sentiment (VADER)", self._analyze_sentiment, P["warn"])
        self._btn(ctrl, "📊 TF-IDF Top Words", self._extract_tfidf, P["accent3"])
        
        panes = tk.Frame(p, bg=P["bg"]); panes.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.nlp_metrics_area = self._scroll(panes, width=42); self.nlp_metrics_area.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0,6))
        self.nlp_plot_container = tk.Frame(panes, bg=P["surface"]); self.nlp_plot_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

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

