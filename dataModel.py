

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import textwrap
import re
import urllib.parse
import sqlite3
import requests
import pyttsx3
import tkinter as tk
from tkinter import scrolledtext, ttk, filedialog, messagebox, simpledialog
from io import StringIO, BytesIO
import time
import docx
import pdfplumber
import os
import zipfile
import warnings
import threading
import queue
from pymongo import MongoClient

# For MySQL and ML (run this once if you haven't):
# pip install scikit-learn sqlalchemy pymysql

try:
    from sqlalchemy import create_engine
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, r2_score
except ImportError:
    print(" Missing some libraries – run: pip install scikit-learn sqlalchemy pymysql")


class SuperMLMode:
    def __init__(self):
        # Set up the voice assistant making my things easier
        try:
            self.voice_engine = pyttsx3.init()
            self.voice_engine.setProperty('rate', 180)
            self.voice_engine.setProperty('volume', 0.9)
            available_voices = self.voice_engine.getProperty('voices')
            if len(available_voices) > 1:
                self.voice_engine.setProperty('voice', available_voices[1].id) 
            self.has_voice = True
        except Exception as err:
            print(f"Heads up: Voice engine error ({err}). Printing instead.")
            self.has_voice = False
        
        # Threading support
        self.gui_queue = queue.Queue()
        self.workflow_thread = None
        self.chart_windows = [] 
        
        # GUI Setup using Tkinter for the dashboard  which 
        self.root = None
        self.log_area = None
        self.notebook = None
        self._setup_dashboard()
        
        # Start checking the GUI queue for requests from the worker thread
        self._poll_gui_queue()
        
        self.say_it("SuperMLMode is ready. Click 'Start New Analysis' on the dashboard to begin!")

    def _setup_dashboard(self):
        """Initializes the main Tkinter dashboard window."""
        self.root = tk.Tk()
        self.root.title("TyranProject - Data Scientist Dashboard")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        # Top Banner with Start Button
        header_frame = tk.Frame(self.root, bg="#2c3e50", pady=10)
        header_frame.pack(fill=tk.X)
        
        header = tk.Label(header_frame, text="TyranProject Dashboard 📊", font=("Arial", 18, "bold"), bg="#2c3e50", fg="white")
        header.pack(side=tk.LEFT, padx=20)

        self.start_btn = tk.Button(header_frame, text="🚀 Start New Analysis", command=self.begin_project, 
                                 bg="#2ecc71", fg="white", font=("Arial", 11, "bold"), padx=20)
        self.start_btn.pack(side=tk.RIGHT, padx=20)

        # Main Layout: Left (Logs) | Right (Tabs for Charts/Data)
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left: Log Area
        log_frame = tk.LabelFrame(main_frame, text="System Status & Voice Transcript", font=("Arial", 10, "bold"), bg="#f0f0f0", width=400)
        log_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5)
        
        self.log_area = scrolledtext.ScrolledText(log_frame, width=45, height=35, font=("Consolas", 9), bg="#1e1e1e", fg="#00ff00")
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right: Notebook for Analysis
        analysis_frame = tk.Frame(main_frame, bg="#f0f0f0")
        analysis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        self.notebook = ttk.Notebook(analysis_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Data Views on the dashboard
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="Data Explorer")
        self.data_log = scrolledtext.ScrolledText(self.data_tab, font=("Consolas", 9))
        self.data_log.pack(fill=tk.BOTH, expand=True)

        # Tab 2: Statistical Report (NEW!)
        self.stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_tab, text="Statistical Report")
        self.stats_log = scrolledtext.ScrolledText(self.stats_tab, font=("Consolas", 9), bg="#f8f9fa")
        self.stats_log.pack(fill=tk.BOTH, expand=True)

        # Tab 3: Visuals
        self.visuals_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.visuals_tab, text="Visual Analysis")
        
        # Add a pop-out button for the summary visuals making it easier for all to understand the data views 
        self.visuals_header = tk.Frame(self.visuals_tab, bg="#f0f0f0")
        self.visuals_header.pack(fill=tk.X)
        tk.Button(self.visuals_header, text="🪟 Pop out Summary Visuals", 
                  command=self._pop_out_summary, bg="#3498db", fg="white").pack(side=tk.RIGHT, padx=10, pady=5)

        self.visuals_container = tk.Frame(self.visuals_tab)
        self.visuals_container.pack(fill=tk.BOTH, expand=True)

        # Tab 4: SQL Lab to allow the query writings and a full Buildin SQL 
        self.sql_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.sql_tab, text="SQL Lab")
        self._setup_sql_tab()

        # Tab 5: Custom Charts to customize any data that one wanna plot
        self.custom_chart_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.custom_chart_tab, text="Custom Charts")
        self._setup_custom_chart_tab()

        # Tab 6: ML Lab so see the machine learning performance nd outputs of it
        self.ml_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.ml_tab, text="ML Lab")
        self._setup_ml_tab()

    def _setup_ml_tab(self):
        """Builds the machine learning control center."""
        frame = tk.Frame(self.ml_tab, bg="#f0f0f0")
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        controls = tk.Frame(frame, bg="#f0f0f0")
        controls.pack(fill=tk.X)

        tk.Label(controls, text="Target Column:", bg="#f0f0f0").grid(row=0, column=0, padx=5)
        self.ml_target_combo = ttk.Combobox(controls, state="readonly", width=25)
        self.ml_target_combo.grid(row=0, column=1, padx=5)

        tk.Button(controls, text=" Train & Evaluate Model", command=self._run_gui_ml,
                  bg="#e67e22", fg="white", font=("Arial", 10, "bold"), padx=15).grid(row=0, column=2, padx=20)

        # Layout for results: Left (Metrics) | Right (Plots)
        res_frame = tk.Frame(frame, bg="#f0f0f0")
        res_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.ml_metrics_area = scrolledtext.ScrolledText(res_frame, width=40, font=("Consolas", 10), bg="white")
        self.ml_metrics_area.pack(side=tk.LEFT, fill=tk.BOTH, padx=5)

        self.ml_plot_container = tk.Frame(res_frame, bg="white")
        self.ml_plot_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

    def _run_gui_ml(self):
        """Triggered by the GUI button to run ML."""
        target = self.ml_target_combo.get()
        if not target:
            messagebox.showwarning("No Target", "Please select a target column first!")
            return
        
        threading.Thread(target=lambda: self.train_ml_model(self.current_df, target_override=target), daemon=True).start()

    def _setup_sql_tab(self):
        """Builds the interactive SQL query interface."""
        frame = tk.Frame(self.sql_tab, bg="#f0f0f0")
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(frame, text="Run SQL Queries on your Cleaned Data (table name is 'data')", 
                 font=("Arial", 10, "bold"), bg="#f0f0f0").pack(anchor="w")
        
        self.sql_entry = tk.Entry(frame, font=("Consolas", 11), width=80)
        self.sql_entry.pack(fill=tk.X, pady=5)
        self.sql_entry.insert(0, "SELECT * FROM data LIMIT 10;")
        
        btn_frame = tk.Frame(frame, bg="#f0f0f0")
        btn_frame.pack(fill=tk.X)
        
        tk.Button(btn_frame, text="▶ Run Query", command=self._execute_gui_sql, 
                  bg="#3498db", fg="white", padx=20).pack(side=tk.LEFT, pady=5)
        
        self.sql_result_area = scrolledtext.ScrolledText(frame, font=("Consolas", 9), bg="white")
        self.sql_result_area.pack(fill=tk.BOTH, expand=True, pady=5)

    def _setup_custom_chart_tab(self):
        """Builds the custom visualization builder with extra customization."""
        frame = tk.Frame(self.custom_chart_tab, bg="#f0f0f0")
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        controls = tk.Frame(frame, bg="#f0f0f0")
        controls.pack(fill=tk.X)
        tk.Label(controls, text="X-Axis:", bg="#f0f0f0").grid(row=0, column=0, padx=5)
        self.x_combo = ttk.Combobox(controls, state="readonly", width=25)
        self.x_combo.grid(row=0, column=1, padx=5)
        tk.Label(controls, text="Y-Axis:", bg="#f0f0f0").grid(row=0, column=2, padx=5)
        self.y_combo = ttk.Combobox(controls, state="readonly", width=25)
        self.y_combo.grid(row=0, column=3, padx=5)
        tk.Label(controls, text="Type:", bg="#f0f0f0").grid(row=0, column=4, padx=5)
        self.type_combo = ttk.Combobox(controls, state="readonly", values=["Scatter", "Bar", "Line", "Histogram", "Pie"], width=15)
        self.type_combo.set("Scatter")
        self.type_combo.grid(row=0, column=5, padx=5)
        tk.Label(controls, text="Chart Title:", bg="#f0f0f0").grid(row=1, column=0, padx=5, pady=10)
        self.title_entry = tk.Entry(controls, width=27)
        self.title_entry.insert(0, "My Custom Analysis")
        self.title_entry.grid(row=1, column=1, padx=5)
        tk.Label(controls, text="Color:", bg="#f0f0f0").grid(row=1, column=2, padx=5)
        self.color_combo = ttk.Combobox(controls, state="readonly", values=["Purple", "Teal", "Skyblue", "Orange", "Red", "Green", "Pink"], width=25)
        self.color_combo.set("Purple")
        self.color_combo.grid(row=1, column=3, padx=5)
        tk.Button(controls, text="🎨 Generate Chart Window", command=self._generate_custom_chart,
                  bg="#9b59b6", fg="white", font=("Arial", 10, "bold"), padx=10).grid(row=1, column=5, padx=10)

        tk.Button(controls, text="🗑️ Close All Charts", command=self._close_all_charts,
                  bg="#e74c3c", fg="white", font=("Arial", 10, "bold"), padx=10).grid(row=1, column=6, padx=10)
        tk.Label(frame, text="Click 'Generate Chart Window' to open as many independent, movable graphs as you want!", 
                 font=("Arial", 10, "italic"), bg="#f0f0f0", fg="#7f8c8d").pack(pady=20)

    def _close_all_charts(self):
        """Closes all open Toplevel chart windows."""
        for win in self.chart_windows:
            try: win.destroy()
            except: pass
        self.chart_windows = []

    def _execute_gui_sql(self):
        """Runs SQL from the GUI tab."""
        query = self.sql_entry.get()
        if not hasattr(self, 'current_df') or self.current_df is None:
            messagebox.showwarning("No Data", "Please load a dataset first!")
            return

        try:
            conn = sqlite3.connect(':memory:')
            self.current_df.to_sql('data', conn, index=False, if_exists='replace')
            res = pd.read_sql_query(query, conn)
            self.sql_result_area.delete('1.0', tk.END)
            self.sql_result_area.insert(tk.END, f"Showing {len(res)} results:\n")
            self.sql_result_area.insert(tk.END, "-" * 50 + "\n")
            self.sql_result_area.insert(tk.END, res.to_string())
            conn.close()
        except Exception as e:
            self.sql_result_area.delete('1.0', tk.END)
            self.sql_result_area.insert(tk.END, f" Error: {e}")

    def _generate_custom_chart(self):
        """Generates a chart based on GUI selections. Thread-safe."""
        if not hasattr(self, 'current_df') or self.current_df is None:
            messagebox.showwarning("No Data", "Please load a dataset first!")
            return

        x = self.x_combo.get()
        y = self.y_combo.get()
        ctype = self.type_combo.get()
        title = self.title_entry.get() or f"{ctype}: {x} vs {y}"
        color = self.color_combo.get().lower()
        df_plot = self.current_df.copy() # Use a copy to avoid modifying the main data

        # We must create the plot on the main thread
        def _plot_task():
            try:
                if x not in df_plot.columns or y not in df_plot.columns:
                    return

                win = tk.Toplevel(self.root)
                win.title(f"Chart: {title}")
                win.geometry("800x600")
                self.chart_windows.append(win)
                
                # Cleanup when window is closed manually
                def _on_win_close():
                    if win in self.chart_windows:
                        self.chart_windows.remove(win)
                    win.destroy()
                win.protocol("WM_DELETE_WINDOW", _on_win_close)

                fig, ax = plt.subplots(figsize=(8, 5))
                
                is_y_numeric = pd.api.types.is_numeric_dtype(df_plot[y])
                is_x_numeric = pd.api.types.is_numeric_dtype(df_plot[x])

                if ctype == "Scatter":
                    if is_x_numeric and is_y_numeric:
                        
                        df_plot.plot(kind='scatter', x=x, y=y, ax=ax, alpha=0.5, color=color)
                    elif (not is_x_numeric and is_y_numeric) or (is_x_numeric and not is_y_numeric):
                        cat_col = x if not is_x_numeric else y
                        num_col = y if not is_x_numeric else x
                        cats = df_plot[cat_col].unique()
                        cat_map = {cat: i for i, cat in enumerate(cats)}
                        jitter = np.random.uniform(-0.15, 0.15, size=len(df_plot))
                        
                        plot_x = df_plot[cat_col].map(cat_map) + jitter if not is_x_numeric else df_plot[num_col]
                        plot_y = df_plot[num_col] if not is_x_numeric else df_plot[cat_col].map(cat_map) + jitter
                        
                        ax.scatter(plot_x, plot_y, alpha=0.4, color=color, s=15)
                        
                        if not is_x_numeric:
                            ax.set_xticks(range(len(cats)))
                            ax.set_xticklabels(cats, rotation=45)
                            ax.set_xlabel(cat_col)
                            ax.set_ylabel(num_col)
                        else:
                            ax.set_yticks(range(len(cats)))
                            ax.set_yticklabels(cats)
                            ax.set_xlabel(num_col)
                            ax.set_ylabel(cat_col)
                    else:
                        messagebox.showwarning("Type Error", "Scatter plots need at least one numeric column. For two categorical columns, try a 'Bar' chart!")
                        win.destroy()
                        return
                    
                elif ctype == "Bar":
                    if not is_y_numeric and not is_x_numeric:
                        pd.crosstab(df_plot[x], df_plot[y]).head(15).plot(kind='bar', stacked=True, ax=ax)
                        ax.set_ylabel("Count")
                    elif is_x_numeric and not is_y_numeric:
                        if df_plot[x].nunique() > 20:
                            df_plot['temp_bin'] = pd.cut(df_plot[x], bins=10)
                            pd.crosstab(df_plot['temp_bin'], df_plot[y]).plot(kind='bar', stacked=True, ax=ax)
                            ax.set_xlabel(f"{x} (Binned)")
                        else:
                            pd.crosstab(df_plot[x], df_plot[y]).plot(kind='bar', stacked=True, ax=ax)
                        ax.set_ylabel(f"Proportion of {y}")
                    elif is_y_numeric:
                       
                        df_plot.groupby(x)[y].mean().head(15).plot(kind='bar', ax=ax, color=color)
                        ax.set_ylabel(f"Average {y}")
                    else:
                    
                        df_plot.groupby(x)[y].count().head(15).plot(kind='bar', ax=ax, color=color)
                        ax.set_ylabel(f"Count of {y}")
                    
                elif ctype == "Line":
                    if not is_y_numeric:
                        messagebox.showwarning("Type Error", f"Line charts need a numeric Y axis.")
                        return
                    df_plot.sort_values(x).plot(kind='line', x=x, y=y, ax=ax, color=color)
                    
                elif ctype == "Histogram":
                    if not is_x_numeric:
                        df_plot[x].value_counts().head(15).plot(kind='bar', ax=ax, color=color)
                        ax.set_ylabel("Frequency (Count)")
                    else:
                        df_plot[x].plot(kind='hist', ax=ax, bins=20, color=color)
                        ax.set_xlabel(x)

                elif ctype == "Pie":
                    if is_x_numeric and df_plot[x].nunique() > 10:
                        messagebox.showwarning("Data Error", "Pie charts don't work well with many unique numbers. Try a 'Histogram' instead!")
                        return
                    
                    df_plot[x].value_counts().head(8).plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90)
                    ax.set_ylabel("") 
                    ax.set_title(f"Distribution of {x}")

                ax.set_title(title, fontweight='bold', fontsize=12)
                plt.tight_layout()
                canvas = FigureCanvasTkAgg(fig, master=win)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                toolbar_frame = tk.Frame(win)
                toolbar_frame.pack(fill=tk.X)
                toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                toolbar.update()
            except Exception as e:
                messagebox.showerror("Plot Error", f"Could not create chart: {e}")

        self.root.after(0, _plot_task)

    def _poll_gui_queue(self):
        """Processes GUI requests sent from the worker thread."""
        try:
            while True:
                task, args, callback_queue = self.gui_queue.get_nowait()
                result = task(*args)
                if callback_queue:
                    callback_queue.put(result)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._poll_gui_queue)

    def _queue_gui_task(self, task, *args):
        """Sends a task to the GUI thread and waits for the result."""
        callback_queue = queue.Queue()
        self.gui_queue.put((task, args, callback_queue))
        return callback_queue.get()

    def say_it(self, text: str, quiet=False):
        """Logs message and speaks it. Thread-safe."""
        print(text)
        if self.log_area:
            def _log():
                self.log_area.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {text}\n")
                self.log_area.see(tk.END)
            self.root.after(0, _log)

        if self.has_voice and not quiet:
            speech_text = re.sub(r'[^\w\s\.]', '', text)
            self.voice_engine.say(speech_text)
            self.voice_engine.runAndWait()

    def show_data_view(self, df, title):
        """Displays data subsets in the GUI tab. Thread-safe."""
        def _task():
            self.data_log.insert(tk.END, f"\n--- {title} ---\n")
            self.data_log.insert(tk.END, df.to_string() + "\n")
            self.data_log.see(tk.END)
        self.root.after(0, _task)

    def get_gui_input(self, prompt, title="Input Required"):
        return self._queue_gui_task(simpledialog.askstring, title, prompt)

    def get_gui_choice(self, prompt, options, title="Select Option"):
        def _task():
            popup = tk.Toplevel(self.root)
            popup.title(title)
            popup.geometry("400x300")
            popup.grab_set()
            selected_value = tk.StringVar()
            tk.Label(popup, text=prompt, font=("Arial", 10, "bold"), wraplength=350, pady=10).pack()
            for i, opt in enumerate(options, 1):
                tk.Button(popup, text=f"{i}) {opt}", 
                          command=lambda v=str(i): [selected_value.set(v), popup.destroy()],
                          width=40, pady=5).pack(pady=2)
            self.root.wait_window(popup)
            return selected_value.get()
        return self._queue_gui_task(_task)

    def get_file_path(self, file_types):
        """Opens a file dialog to pick a file. Thread-safe."""
        def _task():
            return filedialog.askopenfilename(parent=self.root, filetypes=file_types)
        return self._queue_gui_task(_task)

    def get_db_credentials(self, db_type):
        def _task():
            popup = tk.Toplevel(self.root)
            popup.title(f"{db_type.upper()} Connection Details")
            popup.geometry("400x450")
            popup.grab_set()
            results = {}
            fields = [("Host", "localhost"), ("Database Name", ""), ("Username", ""), ("Password", ""), ("Table/Query", "")]
            entries = {}
            for label, default in fields:
                tk.Label(popup, text=label).pack(pady=(10, 0))
                ent = tk.Entry(popup, width=40)
                ent.insert(0, default)
                ent.pack()
                entries[label] = ent
                if label == "Password": ent.config(show="*")
            def on_submit():
                for label, ent in entries.items(): results[label] = ent.get()
                popup.destroy()
            tk.Button(popup, text="Connect", command=on_submit, bg="#4CAF50", fg="white", pady=10, width=20).pack(pady=20)
            self.root.wait_window(popup)
            return results
        return self._queue_gui_task(_task)

    def _pop_out_summary(self):
        """Moves the summary plots to a separate window."""
        if not hasattr(self, 'current_df') or self.current_df is None:
            messagebox.showwarning("No Data", "Please load a dataset first!")
            return
        
        self._plot_summary(pop_out=True)

    def run_full_analysis(self, df):
        """Performs top/last/random views and creates visuals."""
        self.current_df = df 
        
        cols = list(df.columns)
        def _update_combos():
            self.x_combo['values'] = cols
            self.y_combo['values'] = cols
            self.ml_target_combo['values'] = cols
            if cols:
                self.x_combo.set(cols[0])
                self.y_combo.set(cols[1] if len(cols) > 1 else cols[0])
                self.ml_target_combo.set(cols[-1]) # Default to last column
        self.root.after(0, _update_combos)

        self.say_it("Running Deep Analysis on your dataset...")
        
        # 1. Data Subsets
        self.show_data_view(df.head(10), "Top 10 Rows")
        self.show_data_view(df.tail(20), "Last 20 Rows")
        self.show_data_view(df.head(5), "Top 5 Rows")
        self.show_data_view(df.tail(5), "Last 5 Rows")
        self.show_data_view(df.sample(min(20, len(df))), "Random 20 Rows")

        # 2. Statistical Report
        self.say_it("Generating Statistical Reports...")
        def _update_stats():
            self.stats_log.delete('1.0', tk.END)
            buffer = StringIO()
            df.info(buf=buffer)
            self.stats_log.insert(tk.END, "=== DATASET INFO ===\n" + buffer.getvalue() + "\n\n")
            self.stats_log.insert(tk.END, "=== NUMERIC SUMMARY ===\n" + df.describe().round(2).to_string() + "\n\n")
            num_df = df.select_dtypes(include=['number'])
            if not num_df.empty:
                self.stats_log.insert(tk.END, "=== CORRELATION ===\n" + num_df.corr().round(3).to_string() + "\n\n")
        self.root.after(0, _update_stats)

        # 3. Visuals
        self.say_it("Generating Data Visualizations...")
        self.root.after(0, self._plot_summary)
        self.say_it("Analysis complete! Check the dashboard tabs.")

    def _plot_summary(self, pop_out=False):
        """Creates the summary plots grid. Can be in the dashboard or a window."""
        df = self.current_df
        try:
            num_cols = df.select_dtypes(include=['number']).columns
            cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns
            if not num_cols.empty or not cat_cols.empty:
                fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                plt.subplots_adjust(hspace=0.4, wspace=0.3)
                if not cat_cols.empty:
                    df[cat_cols[0]].value_counts().head(10).plot(kind='bar', ax=axs[0,0], color='skyblue')
                    axs[0,0].set_title(f"Top 10 {cat_cols[0]}")
                    df[cat_cols[0]].value_counts().head(5).plot(kind='pie', ax=axs[0,1], autopct='%1.1f%%')
                    axs[0,1].set_title(f"{cat_cols[0]} Distribution")
                if len(num_cols) >= 2:
                    df.sample(min(500, len(df))).plot(kind='scatter', x=num_cols[0], y=num_cols[1], ax=axs[1,0], alpha=0.5, color='green')
                    axs[1,0].set_title(f"{num_cols[0]} vs {num_cols[1]}")
                if not num_cols.empty:
                    df[num_cols[0]].plot(kind='hist', ax=axs[1,1], bins=20, color='orange')
                    axs[1,1].set_title(f"{num_cols[0]} Dist")
                
                if pop_out:
                    win = tk.Toplevel(self.root)
                    win.title("Summary Visual Analysis")
                    win.geometry("1000x800")
                    self.chart_windows.append(win)
                    
                    def _on_win_close():
                        if win in self.chart_windows:
                            self.chart_windows.remove(win)
                        win.destroy()
                    win.protocol("WM_DELETE_WINDOW", _on_win_close)
                    
                    master = win
                else:
                    for widget in self.visuals_container.winfo_children():
                        widget.destroy()
                    master = self.visuals_container
                
                canvas = FigureCanvasTkAgg(fig, master=master)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # Add Navigation Toolbar (New!)
                toolbar_frame = tk.Frame(master)
                toolbar_frame.pack(fill=tk.X)
                toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                toolbar.update()
        except Exception as e:
            self.say_it(f"Visual Error: {e}")

    def load_from_url(self, url: str):
        # Strip quotes and extra spaces
        url = url.strip('"\' ')
        self.say_it(f"Okay, I'm grabbing the data from this URL: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        try:
            if url.endswith(('.csv', '.txt')):
                return pd.read_csv(url, storage_options=headers)
            elif url.endswith(('.xls', '.xlsx')):
                return pd.read_excel(url, storage_options=headers)
        except:
            pass 

        try:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            
            content_type = resp.headers.get('Content-Type', '').lower()
            content = resp.content

            if url.endswith('.zip') or 'zip' in content_type or content[:4] == b'PK\x03\x04':
                self.say_it("Wait! I found a ZIP file. I'll look for a CSV inside it...")
                with zipfile.ZipFile(BytesIO(content)) as z:
                    csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                    if csv_files:
                        self.say_it(f"Found a CSV called '{csv_files[0]}'. Loading it now.")
                        return pd.read_csv(z.open(csv_files[0]))
                    else:
                        self.say_it("I couldn't find any CSV files inside that ZIP.")
                        return None
            
            if url.endswith(('.csv', '.txt')) or 'text/csv' in content_type:
                return pd.read_csv(StringIO(resp.text))
            elif url.endswith('.parquet') or 'parquet' in content_type:
                return pd.read_parquet(BytesIO(content))
            elif url.endswith('.json') or 'application/json' in content_type:
                return pd.read_json(StringIO(resp.text))
            elif url.endswith(('.xls', '.xlsx')) or 'excel' in content_type or 'spreadsheet' in content_type:
                return pd.read_excel(BytesIO(content))
            else:
                try:
                   
                    if b'\x00' in content[:1024]:
                        self.say_it("This looks like a binary file I don't recognize. Is it a CSV?")
                        return None
                    return pd.read_csv(StringIO(resp.text))
                except Exception as guess_err:
                    self.say_it(f"I couldn't guess the format: {guess_err}")
                    return None
        except Exception as e:
            self.say_it(f"Hmm, the URL load still failed: {e}")
            self.say_it("Make sure the link is a direct download link (like from GitHub raw or Dropbox direct).")
            return None

    def load_from_csv(self, file_path: str):
        
        file_path = file_path.strip('"\'')
        self.say_it(f"Loading your CSV file from: {file_path}")
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            self.say_it(f"Could not load the CSV: {e}")
            return None

    def load_from_excel(self, file_path: str):
        file_path = file_path.strip('"\'')
        self.say_it(f"Loading your Excel file from: {file_path}")
        try:
            return pd.read_excel(file_path)
        except Exception as e:
            self.say_it(f"Could not load the Excel file: {e}")
            return None

    def load_from_word(self, file_path: str):
        file_path = file_path.strip('"\'')
        self.say_it(f"Attempting to extract data from a Word document: {file_path}")
        try:
            doc = docx.Document(file_path)
            data = []
            for table in doc.tables:
                for row in table.rows:
                    data.append([cell.text.strip() for cell in row.cells])
            
            if not data:
                self.say_it("I couldn't find any tables in that Word document!")
                return None
                
            df = pd.DataFrame(data[1:], columns=data[0])
            self.say_it(f"Found a table with {len(df)} rows.")
            return df
        except Exception as e:
            self.say_it(f"Word file error: {e}")
            return None

    def load_from_pdf(self, file_path: str):
        file_path = file_path.strip('"\'')
        self.say_it(f"Scanning the PDF for tables: {file_path}")
        try:
            with pdfplumber.open(file_path) as pdf:
                all_tables = []
                for page in pdf.pages:
                    table = page.extract_table()
                    if table:
                        all_tables.extend(table)
            
            if not all_tables:
                self.say_it("I couldn't find any clear tables in that PDF.")
                return None
                
            df = pd.DataFrame(all_tables[1:], columns=all_tables[0])
            self.say_it(f"Extracted {len(df)} rows from the PDF.")
            return df
        except Exception as e:
            self.say_it(f"PDF extraction failed: {e}")
            return None

    def _create_db_engine(self, db_type, host, db, user, pwd, port=None):
        """Helper to create SQLAlchemy engines for different SQL flavors."""
        if db_type == "mysql":
            port = port or 3306
            return create_engine(f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}")
        elif db_type == "postgresql":
            port = port or 5432
            return create_engine(f"postgresql://{user}:{pwd}@{host}:{port}/{db}")
        elif db_type == "sqlite":
            return create_engine(f"sqlite:///{db}.db")
        return None

    def load_from_sql(self, db_type, host, db, user, pwd, port=None, query=None, table=None):
        self.say_it(f"Connecting to your {db_type} database...")
        try:
            engine = self._create_db_engine(db_type, host, db, user, pwd, port)
            if query:
                df = pd.read_sql(query, engine)
            elif table:
                df = pd.read_sql_table(table, engine)
            else:
                self.say_it("I need a query or table name!")
                return None
            self.say_it(f"Loaded {len(df)} rows.")
            return df
        except Exception as e:
            self.say_it(f"SQL connection failed: {e}")
            return None

    def save_to_sql(self, df, db_type, host, db, user, pwd, table_name, port=None):
        self.say_it(f"Saving to {db_type} database...")
        try:
            engine = self._create_db_engine(db_type, host, db, user, pwd, port)
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            self.say_it(f"Data saved to table '{table_name}' successfully!")
        except Exception as e:
            if "10061" in str(e):
                self.say_it(f"Connection refused! Is your {db_type} server running on {host}?")
            else:
                self.say_it(f"Save failed: {e}")

    def save_to_nosql(self, df, host, db, user, pwd, collection_name, port=27017):
        self.say_it("Connecting to MongoDB (NoSQL)...")
        try:
            # Note: MongoDB connection string might vary
            uri = f"mongodb://{user}:{pwd}@{host}:{port}/" if user and pwd else f"mongodb://{host}:{port}/"
            client = MongoClient(uri)
            db_conn = client[db]
            collection = db_conn[collection_name]
            
            # Convert DF to dict for Mongo
            records = df.to_dict(orient='records')
            collection.insert_many(records)
            self.say_it(f"Saved {len(records)} records to MongoDB collection '{collection_name}'.")
        except Exception as e:
            self.say_it(f"MongoDB save failed: {e}")

    def clean_and_format(self, df):
        old_shape = df.shape
        
        self.say_it("Starting the data cleaning process. Check the dashboard for details.")

        # 1. Duplicates
        dupes = df.duplicated().sum()
        df = df.drop_duplicates()
        self.say_it(f"Clean Step 1: Removed {dupes} duplicate rows.")

        # 2. Fix Types
        self.say_it("Clean Step 2: Fixing data types (dates and numbers)...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        converted = pd.to_datetime(df[col], errors='coerce')
                        if converted.notnull().mean() > 0.8:
                            df[col] = converted
                    except: pass
                
                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        converted = pd.to_numeric(df[col], errors='coerce')
                        if converted.notnull().mean() > 0.8:
                            df[col] = converted
                    except: pass

        # 3. Missing Values
        self.say_it("Clean Step 3: Imputing missing values...")
        num_cols = df.select_dtypes(include=['number']).columns
        if not num_cols.empty:
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
            self.say_it(f"-> Filled missing numbers in {len(num_cols)} columns using the median.")

        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else "Missing")
        if not cat_cols.empty:
            self.say_it(f"-> Filled missing categories in {len(cat_cols)} columns using the mode.")

        # 4. Final Polish
        df = df.convert_dtypes()
        self.say_it(f"Cleaning complete! Final data shape is {df.shape}.")
        
        return df

    def safe_filename_from_source(self, source):
        """Generates a clean filename for the PDF report."""
        if isinstance(source, dict):
            name = source.get('database', 'mysql_data')
        else:
            parsed = urllib.parse.urlparse(source)
            path = parsed.path.strip('/').replace('/', '_')
            if not path:
                path = parsed.netloc.replace('.', '_')
            name = path or "dataset"
        
        # Keep only letters, numbers, and underscores
        name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)[:80]
        return f"report_{name}.pdf"

    def generate_report(self, df: pd.DataFrame, source):
        pdf_path = self.safe_filename_from_source(source)
        
        self.say_it("\n" + "=" * 65, quiet=True)
        self.say_it("          FULL DATASET REPORT")
        self.say_it("=" * 65, quiet=True)

        self.say_it(f"Source       : {source}")
        self.say_it(f"PDF saved as : {pdf_path}")
        self.say_it(f"Rows         : {df.shape[0]:,}")
        self.say_it(f"Columns      : {df.shape[1]}")
        
        # Terminal: cleaner column list
        cols_str = ", ".join(list(df.columns))
        if len(cols_str) > 100:
            cols_str = cols_str[:97] + "..."
        self.say_it(f"Column names : {cols_str}", quiet=True)

        self.say_it("\nData types:", quiet=True)
        print(df.dtypes.to_string())

        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            self.say_it("\nMissing values:", quiet=True)
            for col, cnt in missing.items():
                pct = cnt / len(df) * 100
                print(f"  {col:20} {cnt:6,} ({pct:5.2f}%)")
        else:
            self.say_it("\nMissing values: None – nice and clean!")

        self.say_it(f"\nDuplicate rows: {df.duplicated().sum():,}")

        self.say_it("\nQuick numeric summary (Transposed for better reading):", quiet=True)
        desc = df.describe().round(2).T
        print(desc.to_string())

        print("\n" + "-" * 65)

        # Create a professional-looking PDF
        with PdfPages(pdf_path) as pdf:
            # --- PAGE 1: OVERVIEW ---
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            y_pos = 0.95

            def write_pdf_line(txt, size=10, bold=False, wrap=95, font='monospace'):
                nonlocal y_pos
                lines = textwrap.wrap(txt, width=wrap)
                for line in lines:
                    ax.text(0.04, y_pos, line, fontsize=size, va='top', family=font,
                            fontweight='bold' if bold else 'normal')
                    y_pos -= 0.025
                y_pos -= 0.01

            write_pdf_line("SuperMLMode – Dataset Report", 16, bold=True, font='sans-serif')
            write_pdf_line(f"Source: {str(source)[:100]}", 10)
            write_pdf_line(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}", 10)
            write_pdf_line("-" * 80, 10)
            
            write_pdf_line("Basic Info:", 12, bold=True, font='sans-serif')
            write_pdf_line(f"Total Rows:    {df.shape[0]:,}", 10)
            write_pdf_line(f"Total Columns: {df.shape[1]}", 10)
            write_pdf_line(f"Duplicates:    {df.duplicated().sum():,}", 10)
            
            write_pdf_line("Missing Values:", 12, bold=True, font='sans-serif')
            miss_items = [f"{col}: {cnt:,} ({cnt/len(df)*100:.1f}%)" 
                         for col, cnt in df.isnull().sum().items() if cnt > 0]
            if miss_items:
                for item in miss_items[:15]: # Limit to first 15 to avoid overflow
                    write_pdf_line(item, 9)
                if len(miss_items) > 15:
                    write_pdf_line(f"...and {len(miss_items)-15} more", 9)
            else:
                write_pdf_line("None (Perfectly clean!)", 10)

            pdf.savefig(fig)
            plt.close(fig)

            # --- PAGE 2+: NUMERIC SUMMARY (Transposed) ---
            # We split the summary if there are too many variables
            chunk_size = 25
            vars_to_summarize = desc.index.tolist()
            
            for i in range(0, len(vars_to_summarize), chunk_size):
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis('off')
                y_pos = 0.95
                
                chunk = desc.iloc[i : i + chunk_size]
                
                header = "Variable Summary (Stats per Column):"
                ax.text(0.04, y_pos, header, fontsize=12, fontweight='bold', family='sans-serif')
                y_pos -= 0.04
                
                # Convert chunk to a clean table string
                table_str = chunk.to_string()
                
                # Use fixed-width font for the table to keep columns aligned
                for line in table_str.split('\n'):
                    ax.text(0.04, y_pos, line, fontsize=8, family='monospace', va='top')
                    y_pos -= 0.025
                    if y_pos < 0.05: # Safety break
                        break
                
                pdf.savefig(fig)
                plt.close(fig)

        self.say_it(f"Done! Your PDF report is ready and saved as {pdf_path}")
        print("=" * 65 + "\n")

    def sql_mode(self, df: pd.DataFrame, table_name="data"):
        # We'll use SQLite in memory for this part
        conn = sqlite3.connect(':memory:')
        df.to_sql(table_name, conn, index=False, if_exists='replace')

        self.say_it(f"\n🗄️ SQL MODE ACTIVE (table = '{table_name}')")
        self.say_it("You can run SQL queries here. For example: SELECT * FROM data LIMIT 5;")
        self.say_it("Just type 'exit' when you want to leave.\n")

        while True:
            try:
                query = input("SQL> ").strip()
                if query.lower() in ['exit', 'quit', 'q']:
                    self.say_it("Closing SQL mode...")
                    break
                if not query:
                    continue

                res = pd.read_sql_query(query, conn)
                if res.empty:
                    print("(empty result)")
                else:
                    print(res.to_string(max_rows=40))

                print("-" * 60)

            except Exception as e:
                self.say_it(f"Query error: {e}")

        conn.close()

    def train_ml_model(self, df: pd.DataFrame, target_override=None):
        self.say_it("\n🤖 Time for the Machine Learning part!")
        
        if target_override:
            target = target_override
        else:
            self.say_it("Which column should I try to predict? (The 'Target')")
            self.say_it("Check the dashboard popup to choose.")
            raw_input = self.get_gui_input("Enter target column name (or 'auto' for me to pick):", "ML Model Training")
            if raw_input is None: return
            target = raw_input.strip().strip('"\'[] ')

        # --- AUTO MODE ---
        if not target or target.lower() == 'auto':
            target = df.columns[-1]
            self.say_it(f"Auto-Mode ON! I've picked '{target}' as the target.")

        if target not in df.columns:
            self.say_it(f"Error: Target '{target}' not found.")
            return

        # Clear and start Training Log in GUI
        def _clear_ml():
            self.ml_metrics_area.delete('1.0', tk.END)
            self.ml_metrics_area.insert(tk.END, f"--- Starting ML Training ---\nTarget: {target}\n\n")
        self.root.after(0, _clear_ml)

        # Prepare features (X) and target (y)
        X = df.drop(target, axis=1)
        y = df[target]

        # Drop columns that don't help with simple models (like dates)
        dt_cols = X.select_dtypes(include=['datetime64']).columns
        if not dt_cols.empty:
            self.say_it(f"I'm dropping these date columns because they won't work with this model: {list(dt_cols)}")
            X = X.drop(dt_cols, axis=1)

        # Turn categories into numbers (One-Hot Encoding)
        cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns
        if not cat_cols.empty:
            self.say_it(f"I'm encoding these categorical columns into numbers: {list(cat_cols)}")
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

        # Final check for missing targets
        if y.isnull().any():
            self.say_it("Dropping rows where the target is missing...")
            valid_idx = y.dropna().index
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]

        if X.empty:
            self.say_it("I don't have enough data left to train a model. Skipping.")
            return

        # Split data into training and testing sets
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Decide if it's classification or regression
        is_classification = (y.dtype == 'object' or y.dtype.name == 'string' or y.nunique() < 10)

        if is_classification:
            self.say_it("This looks like a classification task. I'll use a Random Forest Classifier.")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            score = accuracy_score(y_test, pred)
            self.say_it(f"The model accuracy is {score:.2%}. Not bad!")
        else:
            self.say_it("This looks like a regression task. I'll use a Random Forest Regressor.")
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            score = r2_score(y_test, pred)
            self.say_it(f"The model R-squared score is {score:.4f}.")

        # Display Results in GUI
        def _show_results():
            self.ml_metrics_area.insert(tk.END, "✅ Training Complete!\n\n")
            if is_classification:
                report = classification_report(y_test, pred)
                self.ml_metrics_area.insert(tk.END, "Classification Report:\n" + report)
            else:
                mse = mean_squared_error(y_test, pred)
                self.ml_metrics_area.insert(tk.END, f"Mean Squared Error: {mse:.4f}\n")
                self.ml_metrics_area.insert(tk.END, f"R2 Score: {score:.4f}\n")
            
            # Plot Feature Importance
            for widget in self.ml_plot_container.winfo_children(): widget.destroy()
            fig, ax = plt.subplots(figsize=(6, 4))
            importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
            importances.plot(kind='barh', ax=ax, color='orange')
            ax.set_title("Top 10 Feature Importance")
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self.ml_plot_container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add Navigation Toolbar (New!)
            toolbar_frame = tk.Frame(self.ml_plot_container)
            toolbar_frame.pack(fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            
        self.root.after(0, _show_results)
        self.say_it("Machine Learning training and evaluation finished!")


    def begin_project(self):
        """Starts the project workflow in a separate thread to keep the GUI responsive."""
        if self.workflow_thread and self.workflow_thread.is_alive():
            messagebox.showwarning("Busy", "An analysis is already running!")
            return
            
        self.workflow_thread = threading.Thread(target=self._project_workflow, daemon=True)
        self.workflow_thread.start()
        self.start_btn.config(state=tk.DISABLED)

    def _project_workflow(self):
        """The actual project logic, running in the background."""
        try:
            self.say_it("Let's get your data loaded. Please choose a source in the popup.")
            load_options = [
                "URL (CSV, Excel, JSON, etc.)",
                "CSV File (Local)",
                "Excel File (Local)",
                "Word Document (Tables)",
                "PDF File (Tables)",
                "SQL Database (MySQL, PostgreSQL, SQLite)"
            ]
            load_choice = self.get_gui_choice("Where is your data coming from?", load_options, "Select Data Source")
            
            data_frame = None
            data_source_name = "unknown"

            if load_choice == "1":
                url = self.get_gui_input("Paste the full URL here:", "URL Data Load")
                if url:
                    data_frame = self.load_from_url(url)
                    data_source_name = url
            elif load_choice == "2":
                path = self.get_file_path([("CSV files", "*.csv")])
                if path:
                    data_frame = self.load_from_csv(path)
                    data_source_name = os.path.basename(path)
            elif load_choice == "3":
                path = self.get_file_path([("Excel files", "*.xlsx;*.xls")])
                if path:
                    data_frame = self.load_from_excel(path)
                    data_source_name = os.path.basename(path)
            elif load_choice == "4":
                path = self.get_file_path([("Word files", "*.docx")])
                if path:
                    data_frame = self.load_from_word(path)
                    data_source_name = os.path.basename(path)
            elif load_choice == "5":
                path = self.get_file_path([("PDF files", "*.pdf")])
                if path:
                    data_frame = self.load_from_pdf(path)
                    data_source_name = os.path.basename(path)
            elif load_choice == "6":
                sql_flavors = ["MySQL", "PostgreSQL", "SQLite"]
                flavor_idx = self.get_gui_choice("Which SQL flavor?", sql_flavors, "SQL Connection")
                if not flavor_idx: return
                db_type = sql_flavors[int(flavor_idx)-1].lower()
                
                creds = self.get_db_credentials(db_type)
                if creds:
                    host, db, user, pwd = creds["Host"], creds["Database Name"], creds["Username"], creds["Password"]
                    query_or_table = creds["Table/Query"]
                    
                    if query_or_table.lower().startswith("select"):
                        data_frame = self.load_from_sql(db_type, host, db, user, pwd, query=query_or_table)
                    else:
                        data_frame = self.load_from_sql(db_type, host, db, user, pwd, table=query_or_table)
                    data_source_name = f"{db_type}_{db}"
            
            if data_frame is None or data_frame.empty:
                self.say_it("I couldn't find any data to work with. Process stopped.")
            else:
                # 1. Clean the data well given 
                cleaned_df = self.clean_and_format(data_frame)
                
                # 2. Deep Analysis
                self.run_full_analysis(cleaned_df)
                
                # 3. Generate the reports
                self.generate_report(cleaned_df, data_source_name)
                
                # 4. Interactive SQL mode
                if messagebox.askyesno("SQL Explorer", "Would you like to explore the cleaned data with SQL?"):
                    self.sql_mode(cleaned_df)

                # 5. Storage Selection
                save_options = ["MySQL", "PostgreSQL", "SQLite", "NoSQL (MongoDB)", "Skip Saving"]
                save_choice = self.get_gui_choice("Where should I save this cleaned data?", save_options, "Save Results")
                
                if save_choice in ["1", "2", "3"]:
                    db_type = ["mysql", "postgresql", "sqlite"][int(save_choice)-1]
                    creds = self.get_db_credentials(db_type)
                    if creds:
                        self.save_to_sql(cleaned_df, db_type, creds["Host"], creds["Database Name"], 
                                         creds["Username"], creds["Password"], creds["Table/Query"] or "cleaned_data")
                    
                elif save_choice == "4":
                    creds = self.get_db_credentials("mongodb")
                    if creds:
                        self.save_to_nosql(cleaned_df, creds["Host"], creds["Database Name"], 
                                           creds["Username"], creds["Password"], creds["Table/Query"] or "cleaned_data")
                self.train_ml_model(cleaned_df)
            self.say_it("\n All set! Your data is processed and results are ready. Great job!")
            messagebox.showinfo("SuperMLMode", "Analysis complete! Check the dashboard tabs for results.")
        except Exception as e:
            self.say_it(f" Critical Error😓😓😓: {e}")
        finally:
            self.start_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    tool = SuperMLMode()
    tool.root.mainloop()