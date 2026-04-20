from core import *
from mixins.uimixin import UIMixin
from mixins.datamixin import DataMixin
from mixins.statsmixin import StatsMixin
from mixins.chartsmixin import ChartsMixin
from mixins.mlmixin import MLMixin
from mixins.toolsmixin import ToolsMixin
from mixins.extralabmixin import ExtraLabMixin
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import threading, time, math, os, json, sqlite3, queue, pickle, re

# ═════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═════════════════════════════════════════════════════════════════════════════
class ProjectDrum(UIMixin, DataMixin, StatsMixin, ChartsMixin, MLMixin, ToolsMixin, ExtraLabMixin):

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
from splash import SplashAnimator

if __name__ == "__main__":
    print("=" * 70)
    print("  PROJECT DRUM v4.0 — Enterprise Data Science Dashboard")
    print("  Dependencies: pandas numpy matplotlib seaborn scipy sklearn")
    print("  Optional:     tensorflow torch nbformat pyttsx3 joblib")
    print("=" * 70)
    
    splash_root = tk.Tk()
    splash_root.withdraw()
    
    def launch_app():
        splash_root.destroy()
        app = ProjectDrum()
        app.root.mainloop()
        
    SplashAnimator(splash_root, launch_app)
    splash_root.mainloop()