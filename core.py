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


