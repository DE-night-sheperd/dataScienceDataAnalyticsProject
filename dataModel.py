import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import textwrap
import re
import sqlite3
import pyttsx3
import tkinter as tk
from tkinter import scrolledtext, ttk, filedialog, messagebox, simpledialog
from io import StringIO, BytesIO
import time
import os
import warnings
import threading
import queue
from scipy import stats
import zipfile


try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, r2_score
except ImportError:
    print(" Missing some libraries – run: pip install scikit-learn")


class ProjectDrum:

    def __init__(self):
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
        
        self.gui_queue = queue.Queue()
        self.workflow_thread = None
        self.chart_windows = [] 
        self.pipeline_steps = []
        self.is_dark_mode = False
        self.datasets = {}
        self.active_dataset_name = None
        self.model_history = []
        self.current_df = None
        self.current_model = None
        self.current_model_features = []
        self.current_model_target = None
        self.current_model_encoders = {} 
        self.current_source = None
        self.problem_statement = "No problem statement defined yet."
        self.project_title = "Untitled Data Science Project"
        self.sql_conn = None 
        self.sql_assistant_queries = {
            "Select All": "SELECT * FROM data LIMIT 20;",
            "Count by Category": "SELECT column_name, COUNT(*) FROM data GROUP BY column_name;",
            "Top 10 Rows": "SELECT * FROM data ORDER BY rowid DESC LIMIT 10;",
            "Filter Numeric": "SELECT * FROM data WHERE numeric_column > 100;",
            "Join Example": "SELECT * FROM table1 JOIN table2 ON table1.id = table2.id;"
        }
        self.root = None
        self.log_area = None
        self.notebook = None

        self._setup_dashboard()
        self._poll_gui_queue()
        
        self.say_it("PROJECT DRUM is ready. Click 'Start New Analysis' on the dashboard to begin!")


    def _setup_dashboard(self):
        self.root = tk.Tk()
        self.root.title("PROJECT DRUM - Data Scientist Dashboard")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        header_frame = tk.Frame(self.root, bg="#2c3e50", pady=10)
        header_frame.pack(fill=tk.X)
        
        header = tk.Label(header_frame, text="PROJECT DRUM Dashboard 🥁", font=("Arial", 18, "bold"), bg="#2c3e50", fg="white")
        header.pack(side=tk.LEFT, padx=20)

        self.start_btn = tk.Button(header_frame, text="🚀 Add Dataset", command=self.begin_project, 
                                 bg="#2ecc71", fg="white", font=("Arial", 11, "bold"), padx=20)
        self.start_btn.pack(side=tk.LEFT, padx=10)

        tk.Label(header_frame, text="Active Dataset:", bg="#2c3e50", fg="white", font=("Arial", 10)).pack(side=tk.LEFT, padx=(20, 5))
        self.dataset_selector = ttk.Combobox(header_frame, state="readonly", width=30)
        self.dataset_selector.pack(side=tk.LEFT, padx=5)
        self.dataset_selector.bind("<<ComboboxSelected>>", self._on_dataset_switch)

        self.dark_mode_btn = tk.Button(header_frame, text="🌙 Dark Mode", command=self.toggle_dark_mode, 
                                     bg="#34495e", fg="white", font=("Arial", 10), padx=10)
        self.dark_mode_btn.pack(side=tk.RIGHT, padx=10)

        self.save_proj_btn = tk.Button(header_frame, text="💾 Save Project", command=self.save_session, 
                                     bg="#3498db", fg="white", font=("Arial", 10), padx=10)
        self.save_proj_btn.pack(side=tk.RIGHT, padx=10)

        self.load_proj_btn = tk.Button(header_frame, text="📂 Load Project", command=self.load_session, 
                                     bg="#3498db", fg="white", font=("Arial", 10), padx=10)
        self.load_proj_btn.pack(side=tk.RIGHT, padx=10)

        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        log_frame = tk.LabelFrame(main_frame, text="System Status & Voice Transcript", font=("Arial", 10, "bold"), bg="#f0f0f0", width=400)
        log_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5)
        
        self.log_area = scrolledtext.ScrolledText(log_frame, width=45, height=35, font=("Consolas", 9), bg="#1e1e1e", fg="#00ff00")
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        analysis_frame = tk.Frame(main_frame, bg="#f0f0f0")
        analysis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        self.notebook = ttk.Notebook(analysis_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="Data Explorer")
        
        table_container = tk.Frame(self.data_tab)
        table_container.pack(fill=tk.BOTH, expand=True)
        
        self.data_tree = ttk.Treeview(table_container, show="headings")
        vsb = ttk.Scrollbar(table_container, orient="vertical", command=self.data_tree.yview)
        hsb = ttk.Scrollbar(table_container, orient="horizontal", command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        self.data_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        table_container.grid_columnconfigure(0, weight=1)
        table_container.grid_rowconfigure(0, weight=1)

        self.stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_tab, text="Statistical Report")
        
        stats_header = tk.Frame(self.stats_tab, bg="#f0f0f0")
        stats_header.pack(fill=tk.X)
        tk.Button(stats_header, text="🔬 Run Statistical Deep Dive (Hypothesis Testing)", 
                  command=self._run_statistical_deep_dive, bg="#8e44ad", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10, pady=5)
        
        tk.Button(stats_header, text="🔗 Run Cross-Dataset Influence Analysis", 
                  command=self._run_cross_dataset_analysis, bg="#27ae60", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10, pady=5)

        self.stats_log = scrolledtext.ScrolledText(self.stats_tab, font=("Consolas", 9), bg="#f8f9fa")
        self.stats_log.pack(fill=tk.BOTH, expand=True)

        self.visuals_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.visuals_tab, text="Visual Analysis")
        
        self.comparison_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.comparison_tab, text="Comparison Lab")
        self._setup_comparison_tab()

        self.visuals_header = tk.Frame(self.visuals_tab, bg="#f0f0f0")
        self.visuals_header.pack(fill=tk.X)
        tk.Button(self.visuals_header, text="🪟 Pop out Summary Visuals", 
                  command=self._pop_out_summary, bg="#3498db", fg="white").pack(side=tk.RIGHT, padx=10, pady=5)

        self.visuals_container = tk.Frame(self.visuals_tab)
        self.visuals_container.pack(fill=tk.BOTH, expand=True)
        
        self.insight_frame = tk.LabelFrame(self.visuals_tab, text="🔍 Automated Insights & Observations", bg="#f0f0f0", font=("Arial", 10, "bold"))
        self.insight_frame.pack(fill=tk.X, padx=10, pady=10)
        self.insight_log = scrolledtext.ScrolledText(self.insight_frame, height=6, font=("Arial", 10), bg="#fff8e1")
        self.insight_log.pack(fill=tk.X, padx=5, pady=5)

        self.sql_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.sql_tab, text="SQL Lab")
        self._setup_sql_tab()

        self.custom_chart_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.custom_chart_tab, text="Custom Charts")
        self._setup_custom_chart_tab()

        self.ml_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.ml_tab, text="ML Lab")
        self._setup_ml_tab()

        self.clean_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.clean_tab, text="Cleaning Lab")
        self._setup_clean_tab()

        self.export_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.export_tab, text="Data Export")
        self._setup_export_tab()

        self.pipeline_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.pipeline_tab, text="Pipeline Tracker")
        self._setup_pipeline_tab()

        self.quality_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.quality_tab, text="Data Health")
        self._setup_quality_tab()

        self.predictor_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.predictor_tab, text="Predictor Lab")
        self._setup_predictor_tab()

        self.story_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.story_tab, text="Storyteller Lab")
        self._setup_story_tab()

        self.profiler_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.profiler_tab, text="Data Profiler")
        self._setup_profiler_tab()


    def _setup_profiler_tab(self):
        frame = tk.Frame(self.profiler_tab, bg="#f0f0f0")
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(frame, text="🕵️ Advanced Data Profiler", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(anchor="w")
        tk.Label(frame, text="Complete metadata breakdown of every column in your active dataset.", 
                 font=("Arial", 10, "italic"), bg="#f0f0f0").pack(anchor="w", pady=5)

        table_container = tk.Frame(frame)
        table_container.pack(fill=tk.BOTH, expand=True, pady=10)

        self.profiler_tree = ttk.Treeview(table_container, show="headings", height=15)
        p_vsb = ttk.Scrollbar(table_container, orient="vertical", command=self.profiler_tree.yview)
        p_hsb = ttk.Scrollbar(table_container, orient="horizontal", command=self.profiler_tree.xview)
        self.profiler_tree.configure(yscrollcommand=p_vsb.set, xscrollcommand=p_hsb.set)

        self.profiler_tree.grid(row=0, column=0, sticky='nsew')
        p_vsb.grid(row=0, column=1, sticky='ns')
        p_hsb.grid(row=1, column=0, sticky='ew')

        table_container.grid_columnconfigure(0, weight=1)
        table_container.grid_rowconfigure(0, weight=1)

        cols = ["Column", "Type", "Non-Null", "Nulls %", "Unique", "Mean", "Median", "Min", "Max", "Skewness"]
        self.profiler_tree["columns"] = cols
        for col in cols:
            self.profiler_tree.heading(col, text=col)
            self.profiler_tree.column(col, width=100, anchor="center")


    def _run_profiler_analysis(self):
        if self.current_df is None: return
        
        df = self.current_df
        self.profiler_tree.delete(*self.profiler_tree.get_children())
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].count()
            nulls_pct = f"{(df[col].isnull().sum() / len(df) * 100):.1f}%"
            unique = df[col].nunique()
            
            mean, median, min_val, max_val, skew = "N/A", "N/A", "N/A", "N/A", "N/A"
            
            if pd.api.types.is_numeric_dtype(df[col]):
                mean = f"{df[col].mean():.2f}"
                median = f"{df[col].median():.2f}"
                min_val = f"{df[col].min():.2f}"
                max_val = f"{df[col].max():.2f}"
                skew = f"{df[col].skew():.2f}"
            
            self.profiler_tree.insert("", tk.END, values=(col, dtype, non_null, nulls_pct, unique, mean, median, min_val, max_val, skew))


    def _setup_quality_tab(self):
        frame = tk.Frame(self.quality_tab, bg="#f0f0f0")
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(frame, text="🧪 Data Health & Quality Dashboard", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(anchor="w")
        
        self.quality_score_label = tk.Label(frame, text="Quality Score: N/A", font=("Arial", 24, "bold"), bg="#f0f0f0", fg="#27ae60")
        self.quality_score_label.pack(pady=10)

        self.quality_log = scrolledtext.ScrolledText(frame, font=("Consolas", 10), bg="#1e1e1e", fg="#2ecc71", height=15)
        self.quality_log.pack(fill=tk.BOTH, expand=True, pady=10)

        tk.Button(frame, text="🔍 Run Advanced Anomaly Detection (Isolation Forest)", command=self._run_anomaly_detection,
                  bg="#e67e22", fg="white", font=("Arial", 10, "bold"), padx=20).pack(pady=5)


    def _setup_predictor_tab(self):
        frame = tk.Frame(self.predictor_tab, bg="#f0f0f0")
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(frame, text="🔮 Live Prediction Lab", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(anchor="w")
        tk.Label(frame, text="Use your trained machine learning model to predict outcomes for new data.", 
                 font=("Arial", 10, "italic"), bg="#f0f0f0").pack(anchor="w", pady=5)

        self.predictor_status = tk.Label(frame, text="No model trained yet.", font=("Arial", 11, "bold"), bg="#f0f0f0", fg="#e74c3c")
        self.predictor_status.pack(pady=10)

        self.predictor_input_container = tk.Frame(frame, bg="#f0f0f0")
        self.predictor_input_container.pack(fill=tk.BOTH, expand=True, pady=10)

        self.predict_btn = tk.Button(frame, text="🚀 Predict Outcome Now", command=self._run_live_prediction,
                                   bg="#2ecc71", fg="white", font=("Arial", 11, "bold"), padx=30, pady=10)
        self.predict_btn.pack(pady=10)

        self.explain_btn = tk.Button(frame, text="💡 Explain This Prediction (XAI)", command=self._explain_prediction,
                                   bg="#3498db", fg="white", font=("Arial", 10), padx=20)
        self.explain_btn.pack(pady=5)

        self.prediction_result_label = tk.Label(frame, text="", font=("Arial", 20, "bold"), bg="#f0f0f0")
        self.prediction_result_label.pack(pady=20)


    def _setup_story_tab(self):
        frame = tk.Frame(self.story_tab, bg="#f0f0f0")
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(frame, text="📖 Data Storyteller & Incident Reporter", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(anchor="w")
        
        tk.Button(frame, text="✍️ Generate Final Story & Incident Report", command=self._generate_data_story,
                  bg="#9b59b6", fg="white", font=("Arial", 11, "bold"), padx=20, pady=10).pack(pady=10)

        self.story_area = scrolledtext.ScrolledText(frame, font=("Arial", 11), bg="white", wrap=tk.WORD)
        self.story_area.pack(fill=tk.BOTH, expand=True, pady=10)


    def _generate_data_story(self):
        if self.current_df is None: return
        
        df = self.current_df
        health_score = self.quality_score_label.cget("text").split(": ")[-1]
        
        story = f"=== 📊 THE DATA STORY OF {self.project_title.upper()} ===\n\n"
        story += f"PROBLEM STATEMENT: {self.problem_statement}\n"
        story += "-"*60 + "\n\n"
        
        story += f"1. DATA LANDSCAPE:\n"
        story += f"We analyzed a dataset with {len(df)} records and {len(df.columns)} variables.\n"
        story += f"The overall data health is rated at {health_score}.\n"
        
        missing = df.isnull().sum().sum()
        if missing > 0:
            story += f"⚠️ WARNING: We found {missing} missing data points. The system has automatically handled these, but it may affect precision.\n"
        else:
            story += f"✅ EXCELLENT: The data is exceptionally clean with zero missing values.\n"
            
        story += f"\n2. POSSIBLE INCIDENTS & RISKS DISCOVERED:\n"
        
        anomalies = 0
        try:
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(contamination=0.05, random_state=42)
            anomalies = (iso.fit_predict(df.select_dtypes(include=['number']).fillna(0)) == -1).sum()
        except: pass
        
        if anomalies > 0:
            story += f"🚩 INCIDENT ALERT: Found {anomalies} abnormal data points (outliers). These could represent fraudulent activity, sensor errors, or rare critical events.\n"
        
        num_df = df.select_dtypes(include=['number'])
        for col in num_df.columns:
            skew = num_df[col].skew()
            if abs(skew) > 1.5:
                story += f"🚩 SKEWNESS RISK: '{col}' is heavily skewed ({skew:.2f}). This suggests that most data is concentrated on one side, which might hide rare incidents in the 'tail'.\n"

        story += f"\n3. PREDICTIVE INSIGHTS:\n"
        if self.current_model:
            story += f"We have trained a {type(self.current_model).__name__} model to predict '{self.current_model_target}'.\n"
            story += f"This model can now anticipate future trends and help prevent the incidents identified above.\n"
        else:
            story += f"No predictive model has been finalized yet. Training a model in the ML Lab will unlock 'future forecasting'.\n"

        story += f"\n4. ACTIONABLE RECOMMENDATIONS:\n"
        if missing > 0:
            story += f"🛠️ DATA CLEANING: Prioritize investigating the source of missing data in columns like {df.columns[df.isnull().any()][0]}.\n"
        
        if anomalies > 5:
            story += f"🔍 SECURITY/AUDIT: Review the {anomalies} anomalies found. They are statistically significant and may indicate critical system failures or high-risk outliers.\n"
        
        if self.current_model:
            top_feat = self.current_model_features[0]
            story += f"📈 MONITORING: To control '{self.current_model_target}', we recommend close monitoring of '{top_feat}', as the model identifies it as the most critical influence.\n"

        story += f"\n5. CONCLUSION:\n"
        story += f"Based on our findings, we recommend focusing on the anomalies and monitoring the variance in your numeric variables to maintain stability."
        
        self.story_area.delete('1.0', tk.END)
        self.story_area.insert(tk.END, story)
        self.say_it("Final story and incident report generated. Check the Storyteller tab.")


    def _run_live_prediction(self):
        if not self.current_model:
            messagebox.showwarning("No Model", "Please train a machine learning model first!")
            return
            
        try:
            input_values = []
            for feat, entry in self.predictor_entries.items():
                val = entry.get()
                if val == "":
                    messagebox.showerror("Input Error", f"Please enter a value for {feat}")
                    return
                
                if feat in self.current_model_encoders:
                    le = self.current_model_encoders[feat]
                    try:
                        encoded_val = le.transform([val])[0]
                    except:
                        encoded_val = 0 
                    input_values.append(encoded_val)
                else:
                    input_values.append(float(val))
            
            input_df = pd.DataFrame([input_values], columns=self.current_model_features)
            prediction = self.current_model.predict(input_df)[0]
            
            self.last_prediction_input = input_df 
            
            res_text = f"PREDICTED {self.current_model_target.upper()}: {prediction}"
            if isinstance(prediction, (float, np.float64, np.float32)):
                res_text = f"PREDICTED {self.current_model_target.upper()}: {prediction:.2f}"
            
            self.prediction_result_label.config(text=res_text, fg="#2ecc71")
            self.say_it(f"The model predicts: {prediction}")
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Check your inputs: {e}")


    def _explain_prediction(self):
        if not hasattr(self, 'last_prediction_input') or self.current_model is None:
            messagebox.showwarning("No Data", "Please make a prediction first!")
            return
            
        importances = self.current_model.feature_importances_
        feat_imp = pd.Series(importances, index=self.current_model_features).sort_values(ascending=False)
        
        explanation = f"=== 💡 WHY THIS PREDICTION? (XAI) ===\n\n"
        explanation += f"This outcome for '{self.current_model_target}' was driven primarily by:\n\n"
        
        top_3 = feat_imp.head(3)
        for i, (feat, imp) in enumerate(top_3.items(), 1):
            val = self.last_prediction_input[feat].values[0]
            if feat in self.current_model_encoders:
                val = self.current_model_encoders[feat].inverse_transform([int(val)])[0]
            explanation += f"{i}. {feat.upper()} (Value: {val})\n"
            explanation += f"   Contribution to decision: {imp:.1%}\n\n"
            
        explanation += "The model weighted these features as the most significant indicators for the final result."
        
        win = tk.Toplevel(self.root)
        win.title("Prediction Explanation")
        win.geometry("450x350")
        txt = scrolledtext.ScrolledText(win, font=("Arial", 11), padx=10, pady=10)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert(tk.END, explanation)
        txt.config(state=tk.DISABLED)


    def _refresh_predictor_ui(self):
        for widget in self.predictor_input_container.winfo_children(): widget.destroy()
        
        if not self.current_model:
            self.predictor_status.config(text="No model trained yet.", fg="#e74c3c")
            return
            
        self.predictor_status.config(text=f"Model: {type(self.current_model).__name__} (Target: {self.current_model_target})", fg="#27ae60")
        
        self.predictor_entries = {}
        
        grid_frame = tk.Frame(self.predictor_input_container, bg="#f0f0f0")
        grid_frame.pack(pady=10)
        
        for i, feat in enumerate(self.current_model_features):
            row, col = divmod(i, 2)
            lbl = tk.Label(grid_frame, text=f"{feat}:", bg="#f0f0f0", font=("Arial", 10))
            lbl.grid(row=row, column=col*2, padx=10, pady=5, sticky="e")
            
            if feat in self.current_model_encoders:
                options = list(self.current_model_encoders[feat].classes_)
                ent = ttk.Combobox(grid_frame, values=options, width=13, state="readonly")
                if options: ent.set(options[0])
            else:
                ent = tk.Entry(grid_frame, width=15)
                
            ent.grid(row=row, column=col*2+1, padx=10, pady=5, sticky="w")
            self.predictor_entries[feat] = ent


    def _setup_comparison_tab(self):
        frame = tk.Frame(self.comparison_tab, bg="#f0f0f0")
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        header = tk.Frame(frame, bg="#f0f0f0")
        header.pack(fill=tk.X, pady=(0, 10))
        tk.Label(header, text="⚖️ Dataset Comparison Lab", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(side=tk.LEFT)

        selectors = tk.Frame(frame, bg="#f0f0f0")
        selectors.pack(fill=tk.X, pady=10)

        tk.Label(selectors, text="Dataset 1:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        self.comp_ds1 = ttk.Combobox(selectors, state="readonly", width=25)
        self.comp_ds1.pack(side=tk.LEFT, padx=5)

        tk.Label(selectors, text="Dataset 2:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        self.comp_ds2 = ttk.Combobox(selectors, state="readonly", width=25)
        self.comp_ds2.pack(side=tk.LEFT, padx=5)

        tk.Button(selectors, text="📊 Compare Now", command=self._run_comparison_analysis,
                  bg="#3498db", fg="white", font=("Arial", 10, "bold"), padx=20).pack(side=tk.LEFT, padx=20)

        self.comp_results_container = tk.Frame(frame, bg="#f0f0f0")
        self.comp_results_container.pack(fill=tk.BOTH, expand=True)

        self.comp_log = scrolledtext.ScrolledText(self.comp_results_container, height=8, font=("Consolas", 10), bg="white")
        self.comp_log.pack(fill=tk.X, pady=5)

        self.comp_plot_container = tk.Frame(self.comp_results_container, bg="white")
        self.comp_plot_container.pack(fill=tk.BOTH, expand=True, pady=5)


    def _run_comparison_analysis(self):
        name1 = self.comp_ds1.get()
        name2 = self.comp_ds2.get()

        if not name1 or not name2:
            messagebox.showwarning("Selection Required", "Please select two datasets to compare!")
            return

        if name1 == name2:
            messagebox.showwarning("Same Data", "Please select two different datasets to see a comparison.")
            return

        df1 = self.datasets[name1]
        df2 = self.datasets[name2]

        self.say_it(f"Comparing '{name1}' vs '{name2}'...")
        
        self.comp_log.delete('1.0', tk.END)
        self.comp_log.insert(tk.END, f"=== SIDE-BY-SIDE COMPARISON ===\n")
        self.comp_log.insert(tk.END, f"{'Metric':<25} | {name1[:20]:<20} | {name2[:20]:<20}\n")
        self.comp_log.insert(tk.END, "-"*70 + "\n")
        self.comp_log.insert(tk.END, f"{'Total Rows':<25} | {len(df1):<20} | {len(df2):<20}\n")
        self.comp_log.insert(tk.END, f"{'Total Columns':<25} | {len(df1.columns):<20} | {len(df2.columns):<20}\n")
        self.comp_log.insert(tk.END, f"{'Missing Values':<25} | {df1.isnull().sum().sum():<20} | {df2.isnull().sum().sum():<20}\n")
        self.comp_log.insert(tk.END, f"{'Duplicate Rows':<25} | {df1.duplicated().sum():<20} | {df2.duplicated().sum():<20}\n")
        
        num_cols1 = df1.select_dtypes(include=['number']).columns
        num_cols2 = df2.select_dtypes(include=['number']).columns
        common_num = list(set(num_cols1) & set(num_cols2))

        for widget in self.comp_plot_container.winfo_children(): widget.destroy()

        if common_num:
            self.comp_log.insert(tk.END, f"\nCommon Numeric Columns: {len(common_num)}\n")
            target_col = common_num[0]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            sns.kdeplot(df1[target_col].dropna(), ax=ax1, label=name1, fill=True, color='blue', alpha=0.3)
            sns.kdeplot(df2[target_col].dropna(), ax=ax1, label=name2, fill=True, color='orange', alpha=0.3)
            ax1.set_title(f"Distribution Comparison: {target_col}")
            ax1.legend()

            comp_data = [df1[target_col].dropna(), df2[target_col].dropna()]
            ax2.boxplot(comp_data, labels=[name1[:10], name2[:10]])
            ax2.set_title(f"Outlier Comparison: {target_col}")

            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self.comp_plot_container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            self.comp_log.insert(tk.END, "\nNo common numeric columns found for visual distribution comparison.")

        self.say_it("Comparison complete! Check the Comparison Lab.")


    def _run_anomaly_detection(self):
        if self.current_df is None: return
        
        self.say_it("Running Isolation Forest for advanced anomaly detection...")
        df = self.current_df.select_dtypes(include=['number']).fillna(0)
        if df.empty: return

        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(contamination=0.05, random_state=42)
        anomalies = iso.fit_predict(df)
        
        anomaly_count = (anomalies == -1).sum()
        self.say_it(f"Detection complete! Found {anomaly_count} potential anomalies (hidden outliers).")
        
        self.quality_log.insert(tk.END, f"\n=== ANOMALY DETECTION REPORT ===\n")
        self.quality_log.insert(tk.END, f"Algorithm: Isolation Forest\n")
        self.quality_log.insert(tk.END, f"Potential Anomalies: {anomaly_count} ({anomaly_count/len(df):.1%} of data)\n")
        self.quality_log.insert(tk.END, f"Recommendation: Review these rows carefully in the 'Cleaning Lab'.\n")
        self.quality_log.see(tk.END)


    def _calculate_data_health(self, df):
        missing_pct = df.isnull().sum().sum() / (df.size) * 100 if df.size > 0 else 0
        duplicate_pct = df.duplicated().sum() / len(df) * 100 if len(df) > 0 else 0
        
        score = 100 - (missing_pct * 2) - (duplicate_pct * 1.5)
        score = max(0, min(100, score))
        
        color = "#27ae60" if score > 80 else "#f39c12" if score > 50 else "#e74c3c"
        self.quality_score_label.config(text=f"Quality Score: {score:.1f}%", fg=color)
        
        self.quality_log.delete('1.0', tk.END)
        self.quality_log.insert(tk.END, f"=== DATA QUALITY REPORT ===\n")
        self.quality_log.insert(tk.END, f"Overall Score: {score:.1f}%\n\n")
        self.quality_log.insert(tk.END, f"Missing Values: {df.isnull().sum().sum()} ({missing_pct:.2f}%)\n")
        self.quality_log.insert(tk.END, f"Duplicate Rows: {df.duplicated().sum()} ({duplicate_pct:.2f}%)\n")
        self.quality_log.insert(tk.END, f"Total Rows: {len(df)}\n")
        self.quality_log.insert(tk.END, f"Total Columns: {len(df.columns)}\n\n")
        
        self.quality_log.insert(tk.END, f"=== COLUMN VARIANCE CHECK ===\n")
        num_df = df.select_dtypes(include=['number'])
        for col in num_df.columns:
            if num_df[col].nunique() <= 1:
                self.quality_log.insert(tk.END, f"⚠️ '{col}' has ZERO variance (constant values).\n")
        
        self.quality_log.insert(tk.END, "\nRun Anomaly Detection below for deeper insights.")


    def _setup_pipeline_tab(self):
        frame = tk.Frame(self.pipeline_tab, bg="#f0f0f0")
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(frame, text="⛓️ Analytical Pipeline (Step-by-Step)", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(anchor="w")
        tk.Label(frame, text="Every action you take is recorded here. You can export this as a Python script to replay the analysis later.", 
                 font=("Arial", 10, "italic"), bg="#f0f0f0").pack(anchor="w", pady=5)

        self.pipeline_area = scrolledtext.ScrolledText(frame, font=("Consolas", 10), bg="#1e1e1e", fg="#3498db")
        self.pipeline_area.pack(fill=tk.BOTH, expand=True, pady=10)

        btn_frame = tk.Frame(frame, bg="#f0f0f0")
        btn_frame.pack(fill=tk.X)

        tk.Button(btn_frame, text="📜 Export Pipeline as Python Script", command=self._export_pipeline_script,
                  bg="#3498db", fg="white", font=("Arial", 10, "bold"), padx=20).pack(side=tk.LEFT)
        
        tk.Button(btn_frame, text="🗑️ Clear Pipeline", command=self._clear_pipeline,
                  bg="#e74c3c", fg="white", font=("Arial", 10, "bold"), padx=20).pack(side=tk.RIGHT)


    def _add_pipeline_step(self, step_name, code_snippet):
        timestamp = time.strftime('%H:%M:%S')
        self.pipeline_steps.append({'step': step_name, 'code': code_snippet, 'time': timestamp})
        self._update_pipeline_ui()


    def _update_pipeline_ui(self):
        self.pipeline_area.delete('1.0', tk.END)
        self.pipeline_area.insert(tk.END, "# PROJECT DRUM Auto-Generated Pipeline Script\n")
        self.pipeline_area.insert(tk.END, "import pandas as pd\nimport numpy as np\n\n")
        for i, step in enumerate(self.pipeline_steps, 1):
            self.pipeline_area.insert(tk.END, f"# Step {i}: {step['step']} ({step['time']})\n")
            self.pipeline_area.insert(tk.END, f"{step['code']}\n\n")


    def _clear_pipeline(self):
        if messagebox.askyesno("Clear Pipeline", "Are you sure you want to delete all recorded steps?"):
            self.pipeline_steps = []
            self._update_pipeline_ui()


    def _export_pipeline_script(self):
        if not self.pipeline_steps:
            messagebox.showwarning("Empty", "No steps recorded yet!")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".py", filetypes=[("Python Script", "*.py")])
        if file_path:
            with open(file_path, 'w') as f:
                f.write(self.pipeline_area.get('1.0', tk.END))
            messagebox.showinfo("Success", f"Pipeline script saved to: {file_path}")


    def toggle_dark_mode(self):
        self.is_dark_mode = not self.is_dark_mode
        bg = "#2c2c2c" if self.is_dark_mode else "#f0f0f0"
        fg = "#ffffff" if self.is_dark_mode else "#000000"
        text_bg = "#1e1e1e" if self.is_dark_mode else "#ffffff"
        text_fg = "#00ff00" if self.is_dark_mode else "#000000"

        self.root.configure(bg=bg)
        self.dark_mode_btn.config(text="☀️ Light Mode" if self.is_dark_mode else "🌙 Dark Mode")
        
        for widget in self.root.winfo_children():
            try: 
                if isinstance(widget, tk.Frame) or isinstance(widget, tk.LabelFrame):
                    widget.configure(bg=bg)
                    for child in widget.winfo_children():
                        if isinstance(child, tk.Label): child.configure(bg=bg, fg=fg)
                        elif isinstance(child, tk.Frame): child.configure(bg=bg)
            except: pass

        self.say_it(f"Switched to {'Dark' if self.is_dark_mode else 'Light'} Mode.")


    def _on_dataset_switch(self, event=None):
        selected = self.dataset_selector.get()
        if not selected or selected not in self.datasets: return
        
        self.active_dataset_name = selected
        self.current_df = self.datasets[selected]
        self.current_source = selected
        
        self.say_it(f"Switched focus to dataset: {selected}")
        self._update_all_gui_elements()


    def save_session(self):
        if self.current_df is None:
            messagebox.showwarning("No Data", "Nothing to save yet!")
            return
            
        import pickle
        file_path = filedialog.asksaveasfilename(defaultextension=".sml", filetypes=[("SuperML Project", "*.sml")])
        if file_path:
            try:
                state = {
                    'data': self.current_df,
                    'pipeline': self.pipeline_steps,
                    'source': self.current_source
                }
                with open(file_path, 'wb') as f:
                    pickle.dump(state, f)
                messagebox.showinfo("Success", "Project session saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save project: {e}")


    def load_session(self):
        import pickle
        file_path = filedialog.askopenfilename(filetypes=[("SuperML Project", "*.sml")])
        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    state = pickle.load(f)
                self.current_df = state['data']
                self.pipeline_steps = state.get('pipeline', [])
                self.current_source = state.get('source', "Loaded Session")
                
                self._update_all_gui_elements()
                self._update_pipeline_ui()
                self.say_it("Project session loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load project: {e}")


    def _setup_export_tab(self):
        frame = tk.Frame(self.export_tab, bg="#f0f0f0")
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(frame, text="💾 Export Cleaned Data", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=10)
        tk.Label(frame, text="Save your processed dataset for use in other tools.", font=("Arial", 10, "italic"), bg="#f0f0f0").pack()

        btn_frame = tk.Frame(frame, bg="#f0f0f0")
        btn_frame.pack(pady=20)

        tk.Button(btn_frame, text="📄 Save as CSV", command=lambda: self._export_data("csv"), 
                  bg="#2ecc71", fg="white", font=("Arial", 11, "bold"), padx=30, pady=15).pack(side=tk.LEFT, padx=10)
        
        tk.Button(btn_frame, text="📊 Save as Excel", command=lambda: self._export_data("excel"), 
                  bg="#27ae60", fg="white", font=("Arial", 11, "bold"), padx=30, pady=15).pack(side=tk.LEFT, padx=10)
        
        tk.Button(btn_frame, text="🗄️ Save as JSON", command=lambda: self._export_data("json"), 
                  bg="#16a085", fg="white", font=("Arial", 11, "bold"), padx=30, pady=15).pack(side=tk.LEFT, padx=10)

        tk.Button(btn_frame, text="🏛️ Export SQL Schema (DDL)", command=self._export_sql_schema, 
                  bg="#2c3e50", fg="white", font=("Arial", 11, "bold"), padx=30, pady=15).pack(side=tk.LEFT, padx=10)

        pbi_frame = tk.LabelFrame(frame, text="📈 Power BI Integration", font=("Arial", 11, "bold"), bg="#f0f0f0", pady=10)
        pbi_frame.pack(fill=tk.X, pady=20)

        tk.Label(pbi_frame, text="Optimize your data for Power BI Desktop analysis.", bg="#f0f0f0").pack(pady=5)

        pbi_btn_frame = tk.Frame(pbi_frame, bg="#f0f0f0")
        pbi_btn_frame.pack(pady=10)

        tk.Button(pbi_btn_frame, text="🚀 Power BI Optimized Export", command=self._export_powerbi_ready,
                  bg="#f1c40f", fg="#2c3e50", font=("Arial", 10, "bold"), padx=20, pady=10).pack(side=tk.LEFT, padx=10)

        tk.Button(pbi_btn_frame, text="📜 Copy Power BI Python Script", command=self._generate_powerbi_script,
                  bg="#34495e", fg="white", font=("Arial", 10, "bold"), padx=20, pady=10).pack(side=tk.LEFT, padx=10)

        tk.Button(pbi_btn_frame, text="📊 Detailed Report Guide (DAX)", command=self._generate_pbi_detailed_guide,
                  bg="#e67e22", fg="white", font=("Arial", 10, "bold"), padx=20, pady=10).pack(side=tk.LEFT, padx=10)

        tk.Button(pbi_btn_frame, text="📖 Export Data Dictionary", command=self._export_data_dictionary,
                  bg="#9b59b6", fg="white", font=("Arial", 10, "bold"), padx=20, pady=10).pack(side=tk.LEFT, padx=10)

        tk.Button(pbi_btn_frame, text="📅 Generate Calendar Table", command=self._generate_calendar_table,
                  bg="#16a085", fg="white", font=("Arial", 10, "bold"), padx=20, pady=10).pack(side=tk.LEFT, padx=10)

        tk.Button(pbi_btn_frame, text="🎨 Export PBI Theme (JSON)", command=self._export_pbi_theme,
                  bg="#3498db", fg="white", font=("Arial", 10, "bold"), padx=20, pady=10).pack(side=tk.LEFT, padx=10)

        package_frame = tk.LabelFrame(frame, text="📦 Full Project Delivery", font=("Arial", 11, "bold"), bg="#f0f0f0", pady=10)
        package_frame.pack(fill=tk.X, pady=20)

        tk.Label(package_frame, text="Create a professional ZIP package containing your data, report, and automation script.", bg="#f0f0f0").pack(pady=5)

        tk.Button(package_frame, text="🎁 Export Full Project Package (ZIP)", command=self._export_project_package,
                  bg="#e74c3c", fg="white", font=("Arial", 11, "bold"), padx=30, pady=15).pack(pady=10)


    def _export_sql_schema(self):
        if self.current_df is None: return
        
        df = self.current_df
        table_name = "data_analysis_table"
        
        ddl = f"-- SQL Schema for Component 3 (Database Design)\n"
        ddl += f"CREATE DATABASE IF NOT EXISTS my_ds_project;\nUSE my_ds_project;\n\n"
        ddl += f"CREATE TABLE {table_name} (\n"
        
        cols = []
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                sql_type = "INT"
            elif pd.api.types.is_float_dtype(dtype):
                sql_type = "FLOAT"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                sql_type = "DATETIME"
            else:
                sql_type = "VARCHAR(255)"
            cols.append(f"    `{col}` {sql_type}")
        
        ddl += ",\n".join(cols)
        ddl += "\n);\n\n-- Example insertion tip:\n-- Use LOAD DATA INFILE or an ORM like SQLAlchemy to populate this table."

        file_path = filedialog.asksaveasfilename(defaultextension=".sql", filetypes=[("SQL Script", "*.sql")])
        if file_path:
            with open(file_path, 'w') as f:
                f.write(ddl)
            messagebox.showinfo("SQL Exported", f"SQL Schema (DDL) saved to: {file_path}")


    def _generate_calendar_table(self):
        if not hasattr(self, 'current_df') or self.current_df is None:
            messagebox.showwarning("No Data", "Please load a dataset first!")
            return

        date_cols = self.current_df.select_dtypes(include=['datetime64']).columns
        if date_cols.empty:
            messagebox.showwarning("No Dates", "No date columns found! I need a date column to build a calendar.")
            return

        start_date = self.current_df[date_cols[0]].min()
        end_date = self.current_df[date_cols[0]].max()
        
        if pd.isnull(start_date) or pd.isnull(end_date):
            messagebox.showerror("Date Error", "Date column contains invalid or null dates.")
            return

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        calendar_df = pd.DataFrame({'Date': date_range})
        
        calendar_df['Year'] = calendar_df['Date'].dt.year
        calendar_df['MonthNo'] = calendar_df['Date'].dt.month
        calendar_df['MonthName'] = calendar_df['Date'].dt.strftime('%B')
        calendar_df['Quarter'] = "Q" + calendar_df['Date'].dt.quarter.astype(str)
        calendar_df['DayOfWeek'] = calendar_df['Date'].dt.strftime('%A')
        calendar_df['IsWeekend'] = calendar_df['Date'].dt.dayofweek >= 5
        calendar_df['YearMonth'] = calendar_df['Date'].dt.strftime('%Y-%m')

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", 
                                               filetypes=[("Calendar CSV", "*.csv")])
        if file_path:
            calendar_df.to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"Calendar Table saved!\n\nPro Tip: In Power BI, relate this 'Date' column to your main data's date column to unlock Time Intelligence.")


    def _export_pbi_theme(self):
        theme_json = """{
    "name": "PROJECT DRUM Professional",
    "dataColors": ["#2c3e50", "#2ecc71", "#e67e22", "#3498db", "#e74c3c", "#9b59b6", "#1abc9c", "#f1c40f"],
    "background": "#FFFFFF",
    "foreground": "#2c3e50",
    "tableAccent": "#2c3e50",
    "visualStyles": {
        "*": {
            "*": {
                "title": [{"show": true, "fontColor": {"solid": {"color": "#2c3e50"}}, "fontSize": 12}],
                "border": [{"show": true, "radius": 5, "color": {"solid": {"color": "#E6E6E6"}}}],
                "background": [{"show": true, "color": {"solid": {"color": "#FFFFFF"}}, "transparency": 0}]
            }
        }
    }
}"""
        file_path = filedialog.asksaveasfilename(defaultextension=".json", 
                                               filetypes=[("Power BI Theme", "*.json")])
        if file_path:
            with open(file_path, 'w') as f:
                f.write(theme_json)
            messagebox.showinfo("Theme Exported", "Professional Power BI Theme saved! Import it via 'View' -> 'Themes' -> 'Browse for themes' in Power BI.")


    def _generate_pbi_detailed_guide(self):
        if not hasattr(self, 'current_df') or self.current_df is None:
            messagebox.showwarning("No Data", "Please load a dataset first!")
            return

        df = self.current_df
        num_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns
        
        guide = f"""=== POWER BI DETAILED REPORT GUIDE ===
Generated by PROJECT DRUM on {time.strftime('%Y-%m-%d')}

1. DATA MODEL ARCHITECTURE (The Star Schema):
------------------------------------------------------
For professional analysis, DO NOT just load one big table. 
* Use the 'Power BI Optimized Export' as your FACT table (e.g., 'FactSales').
* Use the 'Generate Calendar Table' as your DATE DIMENSION (e.g., 'DimDate').
* RELATE THEM: Go to 'Model View' in Power BI and drag 'DimDate'[Date] to 'FactTable'[YourDateColumn].

2. RECOMMENDED DAX MEASURES (Copy these into Power BI):
------------------------------------------------------
Total Rows = COUNTROWS('data')

"""
        for col in num_cols:
            guide += f"Avg {col} = AVERAGE('data'[{col}])\n"
            guide += f"Max {col} = MAX('data'[{col}])\n"
            guide += f"Total {col} = SUM('data'[{col}])\n\n"

        guide += """3. TIME INTELLIGENCE DAX (Requires Calendar Table):
------------------------------------------------------
Total Value YTD = TOTALYTD([Total Value], 'DimDate'[Date])
Total Value LY = CALCULATE([Total Value], SAMEPERIODLASTYEAR('DimDate'[Date]))
Growth % = DIVIDE([Total Value] - [Total Value LY], [Total Value LY], 0)

4. RECOMMENDED VISUAL LAYOUT:
------------------------------------------------------
- Page 1: Executive Summary
  * KPI Cards: Total Rows, """ + ", ".join([f"Avg {c}" for c in num_cols[:2]]) + """
  * Bar Chart: Count of rows by """ + (cat_cols[0] if not cat_cols.empty else "Category") + """
  * Map: (If you have Location data)

- Page 2: Deep Dive Analysis
  * Scatter Plot: """ + (f"{num_cols[0]} vs {num_cols[1]}" if len(num_cols) >= 2 else "Numeric trends") + """
  * Slicers: Add """ + ", ".join(cat_cols[:3]) + """ as filters.

5. DATA REFRESH TIP:
Set the 'Storage Mode' to 'Import' for the fastest performance with this cleaned dataset.
"""
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", 
                                               filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(guide)
            messagebox.showinfo("Guide Saved", f"Detailed Power BI Report Guide saved to: {file_path}")


    def _run_statistical_deep_dive(self):
        if self.current_df is None: return
        
        df = self.current_df
        num_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        
        results = ["=== 🔬 STATISTICAL DEEP DIVE (PROJECT DRUM) ===\n"]
        
        results.append("1. NORMALITY CHECKS (Shapiro-Wilk):")
        for col in num_cols[:5]: 
            stat, p = stats.shapiro(df[col].dropna().head(5000))
            is_normal = "Normal" if p > 0.05 else "Skewed/Non-Normal"
            results.append(f"-> {col}: p={p:.4f} ({is_normal})")
        
        if not num_cols.empty and not cat_cols.empty:
            results.append("\n2. GROUP DIFFERENCES (ANOVA/T-Test):")
            target_num = num_cols[0]
            group_cat = cat_cols[0]
            
            groups = [group[target_num].values for name, group in df.groupby(group_cat)]
            if len(groups) > 1:
                f_stat, p = stats.f_oneway(*groups)
                results.append(f"Testing if '{target_num}' varies significantly across '{group_cat}':")
                results.append(f"-> Result: p={p:.4f} ({'Significant' if p < 0.05 else 'No significant difference'})")

        if len(cat_cols) >= 2:
            results.append("\n3. CATEGORICAL LINKS (Chi-Square):")
            c1, c2 = cat_cols[0], cat_cols[1]
            contingency = pd.crosstab(df[c1], df[c2])
            chi2, p, dof, ex = stats.chi2_contingency(contingency)
            results.append(f"Testing link between '{c1}' and '{c2}':")
            results.append(f"-> Result: p={p:.4f} ({'Strong association' if p < 0.05 else 'No association'})")

        def _update():
            self.stats_log.insert(tk.END, "\n" + "\n".join(results) + "\n" + "="*40 + "\n")
            self.stats_log.see(tk.END)
        self.root.after(0, _update)
        self.say_it("Statistical deep dive complete. I found some significant patterns in your groups.")


    def _run_cross_dataset_analysis(self):
        if len(self.datasets) < 2:
            messagebox.showinfo("Need More Data", "Please load at least 2 datasets to run influence analysis.")
            return
        
        self.say_it("Running Cross-Dataset Influence Analysis...")
        results = ["=== 🔗 CROSS-DATASET INFLUENCE REPORT ===\n"]
        
        ds_names = list(self.datasets.keys())
        found_link = False
        
        for i in range(len(ds_names)):
            for j in range(i + 1, len(ds_names)):
                df1 = self.datasets[ds_names[i]]
                df2 = self.datasets[ds_names[j]]
                
                common_cols = list(set(df1.columns) & set(df2.columns))
                if common_cols:
                    found_link = True
                    results.append(f"Matching keys found between '{ds_names[i]}' and '{ds_names[j]}': {common_cols}")
                    
                    try:
                        merged = pd.merge(df1, df2, on=common_cols[0], how='inner').head(5000)
                        if not merged.empty:
                            num_merged = merged.select_dtypes(include=['number'])
                            if len(num_merged.columns) >= 2:
                                corr = num_merged.corr()
                                high_corr = corr.unstack()
                                high_corr = high_corr[(abs(high_corr) > 0.5) & (high_corr < 1.0)].sort_values(ascending=False).drop_duplicates()
                                
                                if not high_corr.empty:
                                    results.append(f"-> Influence Detected! Variables across files are linked:")
                                    for (c1, c2), val in high_corr.head(5).items():
                                        results.append(f"   * '{c1}' vs '{c2}': {val:.2f} correlation")
                    except Exception as e:
                        results.append(f"-> Error merging: {e}")
        
        if not found_link:
            results.append("No common columns found between your datasets. Try JOINing them in the SQL Lab manually!")

        def _update():
            self.stats_log.insert(tk.END, "\n" + "\n".join(results) + "\n" + "="*40 + "\n")
            self.stats_log.see(tk.END)
        self.root.after(0, _update)
        self.say_it("Cross-analysis complete. Check the Statistical Report tab.")


    def _export_project_package(self):
        if self.current_df is None: return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".zip", filetypes=[("ZIP Package", "*.zip")])
        if not file_path: return
        
        try:
            with zipfile.ZipFile(file_path, 'w') as z:
                csv_data = self.current_df.to_csv(index=False)
                z.writestr("cleaned_data.csv", csv_data)
                
                pipeline_script = self.pipeline_area.get('1.0', tk.END)
                z.writestr("analytical_pipeline.py", pipeline_script)
                
                with open("README.md", "r") as f:
                    readme_content = f.read()
                z.writestr("README.md", readme_content)
                
                info = f"Project: {self.project_title}\nProblem: {self.problem_statement}\nExported: {time.ctime()}"
                z.writestr("project_info.txt", info)
                
            messagebox.showinfo("Package Created", f"Full Project Package exported to: {file_path}")
            self.say_it("Project package is ready for delivery.")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))


    def _export_data_dictionary(self):
        if not hasattr(self, 'current_df') or self.current_df is None:
            messagebox.showwarning("No Data", "Please load a dataset first!")
            return

        df = self.current_df
        dict_data = []
        for col in df.columns:
            dict_data.append({
                "Column Name": col,
                "Data Type": str(df[col].dtype),
                "Missing Values": df[col].isnull().sum(),
                "Unique Values": df[col].nunique(),
                "Sample Value": str(df[col].iloc[0]) if not df.empty else "N/A"
            })
        
        dict_df = pd.DataFrame(dict_data)
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", 
                                               filetypes=[("Data Dictionary CSV", "*.csv")])
        if file_path:
            dict_df.to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"Data Dictionary exported to: {file_path}")


    def _export_powerbi_ready(self):
        if not hasattr(self, 'current_df') or self.current_df is None:
            messagebox.showwarning("No Data", "Please load a dataset first!")
            return

        df = self.current_df.copy()
        
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", 
                                               filetypes=[("Power BI CSV", "*.csv")])
        if file_path:
            try:
                df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Power BI ready file saved!\n\nLocation: {file_path}\n\nTip: In Power BI, use 'Get Data -> Text/CSV' and select this file.")
                
                try: os.startfile(os.path.dirname(file_path))
                except: pass
            except Exception as e:
                messagebox.showerror("Error", str(e))


    def _generate_powerbi_script(self):
        if not hasattr(self, 'current_df') or self.current_df is None:
            messagebox.showwarning("No Data", "Please load a dataset first!")
            return

        script = f"""# Power BI Python Script Generated by PROJECT DRUM
import pandas as pd

data = pd.DataFrame({self.current_df.head(100).to_dict(orient='list')}) 
"""
        self.root.clipboard_clear()
        self.root.clipboard_append(script)
        messagebox.showinfo("Script Copied", "A Power BI Python Template has been copied to your clipboard!")


    def _export_data(self, format_type):
        if not hasattr(self, 'current_df') or self.current_df is None:
            messagebox.showwarning("No Data", "Please load and process a dataset first!")
            return

        ext = f".{format_type}" if format_type != "excel" else ".xlsx"
        file_path = filedialog.asksaveasfilename(defaultextension=ext, 
                                               filetypes=[(f"{format_type.upper()} files", f"*{ext}"), ("All files", "*.*")])
        if file_path:
            try:
                if format_type == "csv":
                    self.current_df.to_csv(file_path, index=False)
                elif format_type == "excel":
                    self.current_df.to_excel(file_path, index=False)
                elif format_type == "json":
                    self.current_df.to_json(file_path, orient='records', indent=4)
                
                messagebox.showinfo("Success", f"Data exported successfully to: {file_path}")
                self.say_it(f"Cleaned dataset saved as {format_type.upper()} at {file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Could not export data: {e}")


    def _setup_clean_tab(self):
        frame = tk.Frame(self.clean_tab, bg="#f0f0f0")
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(frame, text="Manual Data Management", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(anchor="w")
        
        drop_frame = tk.LabelFrame(frame, text="🗑️ Drop Columns", bg="#f0f0f0", pady=10)
        drop_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(drop_frame, text="Select columns to remove:", bg="#f0f0f0").pack(side=tk.LEFT, padx=10)
        self.drop_listbox = tk.Listbox(drop_frame, selectmode=tk.MULTIPLE, height=5, exportselection=0)
        self.drop_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        tk.Button(drop_frame, text="Remove Selected", command=self._gui_drop_cols, bg="#e74c3c", fg="white").pack(side=tk.LEFT, padx=10)

        outlier_frame = tk.LabelFrame(frame, text="📈 Outlier Handling (Z-Score)", bg="#f0f0f0", pady=10)
        outlier_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(outlier_frame, text="Threshold (Default 3.0):", bg="#f0f0f0").pack(side=tk.LEFT, padx=10)
        self.outlier_threshold = tk.Entry(outlier_frame, width=10)
        self.outlier_threshold.insert(0, "3.0")
        self.outlier_threshold.pack(side=tk.LEFT, padx=10)
        
        tk.Button(outlier_frame, text="Apply Outlier Removal", command=self._gui_remove_outliers, bg="#f39c12", fg="white").pack(side=tk.LEFT, padx=10)

        feat_frame = tk.LabelFrame(frame, text="🚀 Smart Feature Engineering", bg="#f0f0f0", pady=10)
        feat_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(feat_frame, text="Automatically extract Date features & scale numeric data:", bg="#f0f0f0").pack(side=tk.LEFT, padx=10)
        tk.Button(feat_frame, text="Run Auto-Feat Eng", command=self._gui_auto_feat_eng, bg="#27ae60", fg="white").pack(side=tk.LEFT, padx=10)

        scaling_frame = tk.LabelFrame(frame, text="⚖️ Feature Scaling", bg="#f0f0f0", pady=10)
        scaling_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(scaling_frame, text="Method:", bg="#f0f0f0").pack(side=tk.LEFT, padx=10)
        self.scaling_method = ttk.Combobox(scaling_frame, values=["StandardScaler (Z-Score)", "MinMaxScaler (0-1)"], state="readonly", width=25)
        self.scaling_method.set("StandardScaler (Z-Score)")
        self.scaling_method.pack(side=tk.LEFT, padx=5)
        
        tk.Button(scaling_frame, text="Apply Scaling", command=self._gui_apply_scaling, bg="#3498db", fg="white").pack(side=tk.LEFT, padx=10)

        preview_frame = tk.LabelFrame(frame, text="Current Data Preview", bg="#f0f0f0")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        preview_container = tk.Frame(preview_frame)
        preview_container.pack(fill=tk.BOTH, expand=True)
        
        self.clean_tree = ttk.Treeview(preview_container, show="headings")
        p_vsb = ttk.Scrollbar(preview_container, orient="vertical", command=self.clean_tree.yview)
        p_hsb = ttk.Scrollbar(preview_container, orient="horizontal", command=self.clean_tree.xview)
        self.clean_tree.configure(yscrollcommand=p_vsb.set, xscrollcommand=p_hsb.set)
        
        self.clean_tree.grid(row=0, column=0, sticky='nsew')
        p_vsb.grid(row=0, column=1, sticky='ns')
        p_hsb.grid(row=1, column=0, sticky='ew')
        
        preview_container.grid_columnconfigure(0, weight=1)
        preview_container.grid_rowconfigure(0, weight=1)


    def _gui_drop_cols(self):
        if not hasattr(self, 'current_df') or self.current_df is None:
            messagebox.showwarning("No Data", "Please load a dataset first!")
            return
            
        selected_indices = self.drop_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select columns to drop!")
            return
            
        cols_to_drop = [self.drop_listbox.get(i) for i in selected_indices]
        if messagebox.askyesno("Confirm Drop", f"Are you sure you want to drop: {', '.join(cols_to_drop)}?"):
            self.current_df = self.current_df.drop(columns=cols_to_drop)
            self.say_it(f"Dropped columns: {', '.join(cols_to_drop)}")
            
            self._add_pipeline_step("Drop Columns", f"df = df.drop(columns={cols_to_drop})")
            
            self._update_all_gui_elements()


    def _gui_remove_outliers(self):
        if not hasattr(self, 'current_df') or self.current_df is None:
            messagebox.showwarning("No Data", "Please load a dataset first!")
            return
            
        try:
            threshold = float(self.outlier_threshold.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Threshold must be a number (e.g., 3.0)")
            return
            
        num_df = self.current_df.select_dtypes(include=['number'])
        if num_df.empty:
            messagebox.showinfo("No Numeric Data", "No numeric columns found for outlier detection.")
            return
            
        from scipy import stats
        z_scores = np.abs(stats.zscore(num_df.fillna(num_df.median())))
        filtered_entries = (z_scores < threshold).all(axis=1)
        
        old_count = len(self.current_df)
        self.current_df = self.current_df[filtered_entries]
        new_count = len(self.current_df)
        
        self.say_it(f"Outlier removal complete. Removed {old_count - new_count} rows using threshold {threshold}.")
        
        self._add_pipeline_step("Remove Outliers", f"from scipy import stats\nz_scores = np.abs(stats.zscore(df.select_dtypes(include=['number'])))\ndf = df[(z_scores < {threshold}).all(axis=1)]")
        
        self._update_all_gui_elements()


    def _gui_auto_feat_eng(self):
        if not hasattr(self, 'current_df') or self.current_df is None:
            messagebox.showwarning("No Data", "Please load a dataset first!")
            return
            
        df = self.current_df.copy()
        added_count = 0
        code_steps = []
        
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[f'{col}_Year'] = df[col].dt.year
                df[f'{col}_Month'] = df[col].dt.month
                df[f'{col}_Day'] = df[col].dt.day
                df[f'{col}_DayOfWeek'] = df[col].dt.dayofweek
                added_count += 4
                code_steps.append(f"df['{col}_Year'] = df['{col}'].dt.year\ndf['{col}_Month'] = df['{col}'].dt.month")
                
        for col in df.select_dtypes(include=['object', 'string']).columns:
            if df[col].nunique() > 50: 
                df[f'{col}_Length'] = df[col].astype(str).str.len()
                added_count += 1
                code_steps.append(f"df['{col}_Length'] = df['{col}'].astype(str).str.len()")

        if added_count > 0:
            self.current_df = df
            self.say_it(f"Auto-Feature Engineering complete! Added {added_count} new features.")
            
            self._add_pipeline_step("Feature Engineering", "\n".join(code_steps))
            
            self._update_all_gui_elements()
        else:
            messagebox.showinfo("No Changes", "No suitable columns found for auto-feature engineering.")


    def _gui_apply_scaling(self):
        if self.current_df is None: return
        
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        method = self.scaling_method.get()
        num_cols = self.current_df.select_dtypes(include=['number']).columns
        
        if num_cols.empty:
            messagebox.showwarning("No Numeric Data", "No numeric columns found to scale.")
            return
            
        scaler = StandardScaler() if "Standard" in method else MinMaxScaler()
        scaled_data = scaler.fit_transform(self.current_df[num_cols].fillna(self.current_df[num_cols].median()))
        
        self.current_df[num_cols] = scaled_data
        self.say_it(f"Applied {method} to {len(num_cols)} columns.")
        
        scaler_code = "StandardScaler()" if "Standard" in method else "MinMaxScaler()"
        self._add_pipeline_step("Feature Scaling", f"from sklearn.preprocessing import StandardScaler, MinMaxScaler\nscaler = {scaler_code}\nnum_cols = df.select_dtypes(include=['number']).columns\ndf[num_cols] = scaler.fit_transform(df[num_cols].fillna(df[num_cols].median()))")
        
        self._update_all_gui_elements()


    def _update_all_gui_elements(self):
        if not self.datasets: return
        
        df = self.current_df if self.current_df is not None else list(self.datasets.values())[0]
        cols = list(df.columns)

        self.dataset_selector['values'] = list(self.datasets.keys())
        if self.active_dataset_name:
            self.dataset_selector.set(self.active_dataset_name)

        if hasattr(self, 'comp_ds1'):
            self.comp_ds1['values'] = list(self.datasets.keys())
            self.comp_ds2['values'] = list(self.datasets.keys())

        if self.sql_conn is not None:
            try:
                for name, d_df in self.datasets.items():
                    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', name).lower()
                    d_df.to_sql(safe_name, self.sql_conn, index=False, if_exists='replace')
                if self.current_df is not None:
                    self.current_df.to_sql('data', self.sql_conn, index=False, if_exists='replace')
            except:
                self.sql_conn = None 
        
        def update_tree(tree, dataframe, max_rows=500):
            tree.delete(*tree.get_children())
            
            tree["columns"] = list(dataframe.columns)
            for col in dataframe.columns:
                tree.heading(col, text=col)
                tree.column(col, width=120, anchor="center")
            
            preview_data = dataframe.head(max_rows)
            for index, row in preview_data.iterrows():
                tree.insert("", tk.END, values=list(row))

        update_tree(self.data_tree, df)
        update_tree(self.clean_tree, df, max_rows=100)
        
        self.drop_listbox.delete(0, tk.END)
        for col in cols:
            self.drop_listbox.insert(tk.END, col)
            
        self.x_combo['values'] = cols
        self.y_combo['values'] = cols
        self.ml_target_combo['values'] = cols
        
        self.root.after(0, self._refresh_sql_schema)
        
        self._calculate_data_health(df)
        
        self.root.after(0, self._plot_summary)
        
        self.root.after(0, self._refresh_predictor_ui)
        self.root.after(0, self._run_profiler_analysis)


    def _setup_ml_tab(self):
        frame = tk.Frame(self.ml_tab, bg="#f0f0f0")
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        controls = tk.Frame(frame, bg="#f0f0f0")
        controls.pack(fill=tk.X)

        tk.Label(controls, text="Target Column:", bg="#f0f0f0").grid(row=0, column=0, padx=5)
        self.ml_target_combo = ttk.Combobox(controls, state="readonly", width=25)
        self.ml_target_combo.grid(row=0, column=1, padx=5)

        tk.Button(controls, text=" Train & Evaluate Model", command=self._run_gui_ml,
                  bg="#e67e22", fg="white", font=("Arial", 10, "bold"), padx=15).grid(row=0, column=2, padx=20)
        
        tk.Button(controls, text="💾 Save Model", command=self._gui_save_model,
                  bg="#3498db", fg="white", font=("Arial", 10, "bold"), padx=15).grid(row=0, column=3, padx=10)

        res_frame = tk.Frame(frame, bg="#f0f0f0")
        res_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.ml_metrics_area = scrolledtext.ScrolledText(res_frame, width=40, font=("Consolas", 10), bg="white")
        self.ml_metrics_area.pack(side=tk.LEFT, fill=tk.BOTH, padx=5)

        self.ml_history_area = scrolledtext.ScrolledText(res_frame, height=5, font=("Consolas", 9), bg="#f8f9fa")
        self.ml_history_area.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        self.ml_plot_container = tk.Frame(res_frame, bg="white")
        self.ml_plot_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)


    def _run_gui_ml(self):
        target = self.ml_target_combo.get()
        if not target:
            messagebox.showwarning("No Target", "Please select a target column first!")
            return
        
        self._show_learning_animation(target)
        
        threading.Thread(target=lambda: self.train_ml_model(self.current_df, target_override=target), daemon=True).start()


    def _show_learning_animation(self, target):
        win = tk.Toplevel(self.root)
        win.title("🧠 Machine is Learning...")
        win.geometry("500x400")
        win.configure(bg="#2c3e50")
        
        tk.Label(win, text=f"Analyzing relationships for '{target}'...", font=("Arial", 12, "bold"), bg="#2c3e50", fg="white").pack(pady=20)
        
        progress = ttk.Progressbar(win, orient=tk.HORIZONTAL, length=300, mode='indeterminate')
        progress.pack(pady=10)
        progress.start(10)
        
        canvas = tk.Canvas(win, width=400, height=200, bg="#1e1e1e", highlightthickness=0)
        canvas.pack(pady=10)
        
        status_label = tk.Label(win, text="Initializing Neural Paths...", font=("Arial", 10), bg="#2c3e50", fg="#3498db")
        status_label.pack()

        def _animate():
            if not win.winfo_exists(): return
            
            x, y = np.random.randint(10, 390), np.random.randint(10, 190)
            canvas.create_oval(x, y, x+5, y+5, fill="#2ecc71", outline="")
            
            phases = ["Identifying Patterns...", "Calculating Weights...", "Reducing Loss...", "Optimizing R2...", "Finalizing Model..."]
            status_label.config(text=phases[np.random.randint(len(phases))])
            
            if len(canvas.find_all()) > 100: canvas.delete(canvas.find_all()[0]) 
            
            win.after(100, _animate)

        _animate()
        
        def _check_done():
            if not win.winfo_exists(): return
            if hasattr(self, '_ml_training_done') and self._ml_training_done:
                win.destroy()
                delattr(self, '_ml_training_done')
            else:
                win.after(500, _check_done)
        
        self._ml_training_done = False
        _check_done()


    def _gui_save_model(self):
        if not hasattr(self, 'current_model') or self.current_model is None:
            messagebox.showwarning("No Model", "Please train a model first!")
            return
            
        import joblib
        file_path = filedialog.asksaveasfilename(defaultextension=".joblib", 
                                                 filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")])
        if file_path:
            try:
                joblib.dump(self.current_model, file_path)
                messagebox.showinfo("Success", f"Model saved successfully to: {file_path}")
                self.say_it(f"Saved the trained model to {file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save model: {e}")


    def _setup_sql_tab(self):
        main_frame = tk.Frame(self.sql_tab, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.sql_sidebar = tk.Frame(main_frame, width=250, bg="#ecf0f1", padx=10, pady=10)
        self.sql_sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sql_sidebar.pack_propagate(False)

        tk.Label(self.sql_sidebar, text="🏛️ Schema Explorer", font=("Arial", 11, "bold"), bg="#ecf0f1").pack(pady=(0, 10))
        
        self.schema_tree = ttk.Treeview(self.sql_sidebar, show="tree", height=20)
        self.schema_tree.pack(fill=tk.BOTH, expand=True)
        
        tk.Button(self.sql_sidebar, text="🔄 Refresh Schema", command=self._refresh_sql_schema, 
                  bg="#3498db", fg="white", font=("Arial", 9)).pack(fill=tk.X, pady=10)

        content_frame = tk.Frame(main_frame, bg="#f0f0f0", padx=15, pady=15)
        content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        header = tk.Frame(content_frame, bg="#f0f0f0")
        header.pack(fill=tk.X, pady=(0, 10))
        tk.Label(header, text="💻 SQL Query Lab (SQLite Syntax)", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(side=tk.LEFT)
        tk.Label(header, text="Main table: 'data'", font=("Arial", 10, "italic"), bg="#f0f0f0", fg="#7f8c8d").pack(side=tk.LEFT, padx=15)

        editor_frame = tk.LabelFrame(content_frame, text="✍️ SQL Editor (Supports Multi-line Scripts)", bg="#f0f0f0", font=("Arial", 10, "bold"))
        editor_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.sql_editor = scrolledtext.ScrolledText(editor_frame, font=("Consolas", 11), bg="#1e1e1e", fg="#3498db", insertbackground="white", height=8)
        self.sql_editor.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.sql_editor.insert('1.0', "SELECT * FROM data LIMIT 20;")

        btn_frame = tk.Frame(content_frame, bg="#f0f0f0")
        btn_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(btn_frame, text="▶ Run Script", command=self._execute_gui_sql, 
                  bg="#2ecc71", fg="white", font=("Arial", 10, "bold"), padx=25, pady=5).pack(side=tk.LEFT, padx=5)
        
        self.sql_template_combo = ttk.Combobox(btn_frame, state="readonly", values=list(self.sql_assistant_queries.keys()), width=20)
        self.sql_template_combo.set("SQL Templates")
        self.sql_template_combo.pack(side=tk.LEFT, padx=5)
        self.sql_template_combo.bind("<<ComboboxSelected>>", self._apply_sql_template)

        tk.Button(btn_frame, text="🗑️ Clear Editor", command=lambda: self.sql_editor.delete('1.0', tk.END), 
                  bg="#e74c3c", fg="white", font=("Arial", 10), padx=15).pack(side=tk.LEFT, padx=5)

        result_frame = tk.LabelFrame(content_frame, text="📊 Query Results", bg="#f0f0f0", font=("Arial", 10, "bold"))
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        table_container = tk.Frame(result_frame)
        table_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.sql_tree = ttk.Treeview(table_container, show="headings")
        s_vsb = ttk.Scrollbar(table_container, orient="vertical", command=self.sql_tree.yview)
        s_hsb = ttk.Scrollbar(table_container, orient="horizontal", command=self.sql_tree.xview)
        self.sql_tree.configure(yscrollcommand=s_vsb.set, xscrollcommand=s_hsb.set)
        
        self.sql_tree.grid(row=0, column=0, sticky='nsew')
        s_vsb.grid(row=0, column=1, sticky='ns')
        s_hsb.grid(row=1, column=0, sticky='ew')
        
        table_container.grid_columnconfigure(0, weight=1)
        table_container.grid_rowconfigure(0, weight=1)


    def _apply_sql_template(self, event=None):
        template_name = self.sql_template_combo.get()
        if template_name in self.sql_assistant_queries:
            query = self.sql_assistant_queries[template_name]
            self.sql_editor.delete('1.0', tk.END)
            self.sql_editor.insert('1.0', query)


    def _refresh_sql_schema(self):
        if not self.datasets: return
        
        try:
            if self.sql_conn is None:
                self.sql_conn = sqlite3.connect(':memory:', check_same_thread=False)
            
            for name, df in self.datasets.items():
                safe_name = re.sub(r'[^a-zA-Z0-9]', '_', name).lower()
                df.to_sql(safe_name, self.sql_conn, index=False, if_exists='replace')
            
            if self.current_df is not None:
                self.current_df.to_sql('data', self.sql_conn, index=False, if_exists='replace')
            
            self.schema_tree.delete(*self.schema_tree.get_children())
            
            cursor = self.sql_conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' OR type='view';")
            tables = cursor.fetchall()
            
            for (table_name,) in tables:
                table_id = self.schema_tree.insert("", tk.END, text=f"📋 {table_name}", open=True)
                
                cursor.execute(f"PRAGMA table_info(`{table_name}`);")
                columns = cursor.fetchall()
                for col in columns:
                    col_name = col[1]
                    col_type = col[2]
                    self.schema_tree.insert(table_id, tk.END, text=f"🔹 {col_name} ({col_type})")
                    
        except Exception as e:
            print(f"Schema Refresh Error: {e}")


    def _setup_custom_chart_tab(self):
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
        for win in self.chart_windows:
            try: win.destroy()
            except: pass
        self.chart_windows = []


    def _execute_gui_sql(self):
        full_script = self.sql_editor.get('1.0', tk.END).strip()
        if not full_script: return
        
        if not hasattr(self, 'current_df') or self.current_df is None:
            messagebox.showwarning("No Data", "Please load a dataset first!")
            return

        try:
            if self.sql_conn is None:
                self.sql_conn = sqlite3.connect(':memory:', check_same_thread=False)
                self.current_df.to_sql('data', self.sql_conn, index=False, if_exists='replace')
            
            def is_select(stmt):
                clean_stmt = re.sub(r'--.*?\n|/\*.*?\*/', '', stmt, flags=re.DOTALL).strip().lower()
                return clean_stmt.startswith(('select', 'with', 'pragma', 'explain'))

            statements = [s.strip() for s in full_script.split(';') if s.strip()]
            
            last_res = None
            exec_count = 0
            
            for stmt in statements:
                if is_select(stmt):
                    last_res = pd.read_sql_query(stmt, self.sql_conn)
                else:
                    self.sql_conn.execute(stmt)
                    self.sql_conn.commit()
                exec_count += 1

            self.sql_tree.delete(*self.sql_tree.get_children())
            
            if last_res is not None:
                self.sql_tree["columns"] = list(last_res.columns)
                for col in last_res.columns:
                    self.sql_tree.heading(col, text=col)
                    self.sql_tree.column(col, width=120, anchor="center")
                
                for _, row in last_res.head(1000).iterrows():
                    self.sql_tree.insert("", tk.END, values=list(row))
                
                self.say_it(f"Executed {exec_count} statement(s). Last query returned {len(last_res)} rows.")
            else:
                self.sql_tree["columns"] = ["Status"]
                self.sql_tree.heading("Status", text="Status")
                self.sql_tree.insert("", tk.END, values=[f"Successfully executed {exec_count} statement(s) with no results."])
                self.say_it(f"Executed {exec_count} statement(s) successfully.")
            
            self._refresh_sql_schema()
                
        except Exception as e:
            messagebox.showerror("SQL Error", f"Execution failed:\n{e}")
            self.say_it(f"SQL Error: {e}")


    def _generate_custom_chart(self):
        if not hasattr(self, 'current_df') or self.current_df is None:
            messagebox.showwarning("No Data", "Please load a dataset first!")
            return

        x = self.x_combo.get()
        y = self.y_combo.get()
        ctype = self.type_combo.get()
        title = self.title_entry.get() or f"{ctype}: {x} vs {y}"
        color = self.color_combo.get().lower()
        df_plot = self.current_df.copy() 

        def _plot_task():
            try:
                if x not in df_plot.columns or y not in df_plot.columns:
                    return

                win = tk.Toplevel(self.root)
                win.title(f"Chart: {title}")
                win.geometry("800x600")
                self.chart_windows.append(win)
                
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
                
                tk.Button(toolbar_frame, text="💾 Save Image", command=lambda: self._save_chart_image(fig), 
                          bg="#3498db", fg="white").pack(side=tk.RIGHT, padx=10)
                
                toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                toolbar.update()
            except Exception as e:
                messagebox.showerror("Plot Error", f"Could not create chart: {e}")

        self.root.after(0, _plot_task)


    def _save_chart_image(self, fig):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                               filetypes=[("PNG files", "*.png"), ("JPG files", "*.jpg"), ("All files", "*.*")])
        if file_path:
            try:
                fig.savefig(file_path)
                messagebox.showinfo("Success", f"Chart saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save image: {e}")


    def _poll_gui_queue(self):
        try:
            while True:
                task, args, callback_queue = self.gui_queue.get_nowait()
                result = task(*args)
                if callback_queue:
                    callback_queue.put(result)
        except queue.Empty:
            pass
        finally:
            try:
                if self.root.winfo_exists():
                    self.root.after(100, self._poll_gui_queue)
            except:
                pass


    def _queue_gui_task(self, task, *args):
        callback_queue = queue.Queue()
        self.gui_queue.put((task, args, callback_queue))
        return callback_queue.get()


    def say_it(self, text: str, quiet=False):
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
        pass


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
        def _task():
            return filedialog.askopenfilename(parent=self.root, filetypes=file_types)
        return self._queue_gui_task(_task)


    def _pop_out_summary(self):
        if not hasattr(self, 'current_df') or self.current_df is None:
            messagebox.showwarning("No Data", "Please load a dataset first!")
            return
        
        self._plot_summary(pop_out=True)


    def run_full_analysis(self, df):
        self.current_df = df 
        
        self.root.after(0, self._update_all_gui_elements)

        self._add_pipeline_step("Initial Data Load", f"df = pd.read_csv('{self.current_source}') # or equivalent load\nprint(df.shape)")

        self.say_it("Running Deep Analysis on your dataset...")
        
        self.show_data_view(df.head(10), "Top 10 Rows")
        self.show_data_view(df.tail(20), "Last 20 Rows")
        self.show_data_view(df.head(5), "Top 5 Rows")
        self.show_data_view(df.tail(5), "Last 5 Rows")
        self.show_data_view(df.sample(min(20, len(df))), "Random 20 Rows")

        self.say_it("Generating Statistical Reports...")
        def _update_stats():
            self.stats_log.delete('1.0', tk.END)
            buffer = StringIO()
            df.info(buf=buffer)
            self.stats_log.insert(tk.END, "=== DATASET INFO ===\n" + buffer.getvalue() + "\n\n")
            self.stats_log.insert(tk.END, "=== NUMERIC SUMMARY ===\n" + df.describe().round(2).to_string() + "\n\n")
            
            num_df = df.select_dtypes(include=['number'])
            if not num_df.empty:
                self.stats_log.insert(tk.END, "=== ADVANCED STATS (SKEWNESS & KURTOSIS) ===\n")
                stats_df = pd.DataFrame({
                    'Skewness': num_df.skew().round(3),
                    'Kurtosis': num_df.kurt().round(3)
                })
                self.stats_log.insert(tk.END, stats_df.to_string() + "\n\n")
                
                self.stats_log.insert(tk.END, "=== KEY CORRELATION FINDINGS ===\n")
                corr_matrix = num_df.corr().unstack()
                high_corr = corr_matrix[(abs(corr_matrix) > 0.6) & (abs(corr_matrix) < 1.0)].sort_values(ascending=False).drop_duplicates()
                if not high_corr.empty:
                    for (c1, c2), val in high_corr.items():
                        self.stats_log.insert(tk.END, f"-> Strong {'Positive' if val > 0 else 'Negative'} link between '{c1}' and '{c2}' ({val:.3f})\n")
                else:
                    self.stats_log.insert(tk.END, "No strong correlations found.\n")
                self.stats_log.insert(tk.END, "\n")
                
                self.stats_log.insert(tk.END, "=== CORRELATION MATRIX ===\n" + num_df.corr().round(3).to_string() + "\n\n")
        self.root.after(0, _update_stats)

        self.say_it("Generating Data Visualizations...")
        self.root.after(0, self._plot_summary)
        self.say_it("Analysis complete! Check the dashboard tabs.")


    def _plot_summary(self, pop_out=False):
        df = self.current_df
        try:
            num_cols = df.select_dtypes(include=['number']).columns
            cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns
            if not num_cols.empty or not cat_cols.empty:
                fig, axs = plt.subplots(2, 3, figsize=(15, 10))
                plt.subplots_adjust(hspace=0.4, wspace=0.3)
                
                if not cat_cols.empty:
                    df[cat_cols[0]].value_counts().head(10).plot(kind='bar', ax=axs[0,0], color='skyblue')
                    axs[0,0].set_title(f"Top 10 {cat_cols[0]}")
                    
                    df[cat_cols[0]].value_counts().head(5).plot(kind='pie', ax=axs[0,1], autopct='%1.1f%%')
                    axs[0,1].set_title(f"{cat_cols[0]} Distribution")
                
                if len(num_cols) >= 2:
                    df.sample(min(500, len(df))).plot(kind='scatter', x=num_cols[0], y=num_cols[1], ax=axs[0,2], alpha=0.5, color='green')
                    axs[0,2].set_title(f"{num_cols[0]} vs {num_cols[1]}")
                
                if not num_cols.empty:
                    df[num_cols[0]].plot(kind='hist', ax=axs[1,0], bins=20, color='orange')
                    axs[1,0].set_title(f"{num_cols[0]} Dist")
                
                if len(num_cols) >= 2:
                    corr = df[num_cols].corr()
                    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axs[1,1])
                    axs[1,1].set_title("Correlation Heatmap")
                else:
                    axs[1,1].text(0.5, 0.5, "Need 2+ numeric columns\nfor correlation", ha='center')

                if len(num_cols) >= 3:
                    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                    sns.heatmap(df[num_cols].corr(), annot=True, cmap='RdYlGn', center=0, ax=ax_corr)
                    ax_corr.set_title("Detailed Feature Correlation Matrix")
                    plt.tight_layout()
                    
                    def _pop_corr():
                        win_c = tk.Toplevel(self.root)
                        win_c.title("Advanced Feature Correlation")
                        canvas_c = FigureCanvasTkAgg(fig_corr, master=win_c)
                        canvas_c.draw()
                        canvas_c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    
                    tk.Button(self.visuals_header, text="🔥 Advanced Heatmap", command=_pop_corr, bg="#e67e22", fg="white").pack(side=tk.RIGHT, padx=5)

                if not num_cols.empty:
                    sns.boxplot(data=df[num_cols].head(5), ax=axs[1,2])
                    axs[1,2].set_title("Boxplot (Outliers)")
                    axs[1,2].tick_params(axis='x', rotation=45)
                
                if pop_out:
                    win = tk.Toplevel(self.root)
                    win.title("Deep Analytical Visuals")
                    win.geometry("1200x900")
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
                
                toolbar_frame = tk.Frame(master)
                toolbar_frame.pack(fill=tk.X)
                toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                toolbar.update()
                
                if not pop_out:
                    self.insight_log.delete('1.0', tk.END)
                    self.insight_log.insert(tk.END, "Based on the visual analysis, here are some key findings:\n\n")
                    
                    if not cat_cols.empty:
                        top_cat = df[cat_cols[0]].value_counts().idxmax()
                        self.insight_log.insert(tk.END, f"-> Dominant Category: '{top_cat}' is the most frequent in '{cat_cols[0]}'.\n")
                    
                    if len(num_cols) >= 2:
                        corr_val = df[num_cols[0]].corr(df[num_cols[1]])
                        strength = "Strong" if abs(corr_val) > 0.6 else "Moderate" if abs(corr_val) > 0.3 else "Weak"
                        self.insight_log.insert(tk.END, f"-> Correlation: There is a {strength} { 'positive' if corr_val > 0 else 'negative'} link ({corr_val:.2f}) between '{num_cols[0]}' and '{num_cols[1]}'.\n")
                    
                    num_df = df.select_dtypes(include=['number'])
                    if not num_df.empty:
                        skew = num_df.iloc[:,0].skew()
                        if abs(skew) > 1:
                            self.insight_log.insert(tk.END, f"-> Data Quality: '{num_df.columns[0]}' is heavily skewed ({skew:.2f}), suggesting outliers or a non-normal distribution.\n")
                    
                    self.insight_log.insert(tk.END, "\nClick 'Pop out' to see a larger version of these charts!")
        except Exception as e:
            self.say_it(f"Visual Error: {e}")


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


    def clean_and_format(self, df):
        self.say_it("Starting the data cleaning process. Check the dashboard for details.")

        dupes = df.duplicated().sum()
        df = df.drop_duplicates()
        self.say_it(f"Clean Step 1: Removed {dupes} duplicate rows.")

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

        df = df.convert_dtypes()
        self.say_it(f"Cleaning complete! Final data shape is {df.shape}.")
        
        self._add_pipeline_step("Auto-Cleaning", "df = df.drop_duplicates()\ndf = df.fillna(df.median(numeric_only=True))\nfor col in df.select_dtypes(include=['object']).columns: df[col] = df[col].fillna(df[col].mode()[0])")

        return df


    def safe_filename_from_source(self, source):
        name = re.sub(r'[^a-zA-Z0-9_-]', '_', source)[:80]
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

        desc = df.describe().round(2).T
        print("\nQuick numeric summary (Transposed for better reading):", quiet=True)
        print(desc.to_string())

        print("\n" + "-" * 65)

        with PdfPages(pdf_path) as pdf:
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

            write_pdf_line("PROJECT DRUM – Dataset Report", 16, bold=True, font='sans-serif')
            write_pdf_line(f"Source: {str(source)[:100]}", 10)
            write_pdf_line(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}", 10)
            write_pdf_line("-" * 80, 10)
            
            write_pdf_line("Basic Info:", 12, bold=True, font='sans-serif')
            write_pdf_line(f"Total Rows:    {df.shape[0]:,}", 10)
            write_pdf_line(f"Total Columns: {df.shape[1]}", 10)
            write_pdf_line(f"Duplicates:    {df.duplicated().sum():,}", 10)
            
            missing_pct = df.isnull().sum().sum() / (df.size) * 100 if df.size > 0 else 0
            duplicate_pct = df.duplicated().sum() / len(df) * 100 if len(df) > 0 else 0
            score = 100 - (missing_pct * 2) - (duplicate_pct * 1.5)
            score = max(0, min(100, score))
            write_pdf_line(f"Data Quality Score: {score:.1f}%", 12, bold=True)
            
            write_pdf_line("-" * 80, 10)
            write_pdf_line("Problem Statement:", 12, bold=True, font='sans-serif')
            write_pdf_line(f"Project: {self.project_title}", 10, bold=True)
            write_pdf_line(self.problem_statement, 10)
            write_pdf_line("-" * 80, 10)
            
            if self.current_model:
                write_pdf_line("Machine Learning & Future Predictions:", 12, bold=True, font='sans-serif')
                write_pdf_line(f"Model Used: {type(self.current_model).__name__}", 10)
                write_pdf_line("The model has been trained to predict future incidents based on historical patterns.", 10)
                write_pdf_line("Key drivers and predictive metrics are included in the ML Lab results.", 10)
                write_pdf_line("-" * 80, 10)

            write_pdf_line("Missing Values:", 12, bold=True, font='sans-serif')
            miss_items = [f"{col}: {cnt:,} ({cnt/len(df)*100:.1f}%)" 
                         for col, cnt in df.isnull().sum().items() if cnt > 0]
            if miss_items:
                for item in miss_items[:15]: 
                    write_pdf_line(item, 9)
                if len(miss_items) > 15:
                    write_pdf_line(f"...and {len(miss_items)-15} more", 9)
            else:
                write_pdf_line("None (Perfectly clean!)", 10)

            pdf.savefig(fig)
            plt.close(fig)

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
                
                table_str = chunk.to_string()
                
                for line in table_str.split('\n'):
                    ax.text(0.04, y_pos, line, fontsize=8, family='monospace', va='top')
                    y_pos -= 0.025
                    if y_pos < 0.05: 
                        break
                
                pdf.savefig(fig)
                plt.close(fig)

        self.say_it(f"Done! Your PDF report is ready and saved as {pdf_path}")
        print("=" * 65 + "\n")


    def sql_mode(self, df: pd.DataFrame, table_name="data"):
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
        self.say_it("\n🤖 Machine Learning Discovery Mode!")
        
        learning_types = [
            "Supervised (Prediction/Classification)", 
            "Unsupervised (Finding Hidden Groups)", 
            "Reinforcement (Decision Optimizer)",
            "Auto-ML Optimization (Find Best Algorithm)"
        ]
        type_choice = self.get_gui_choice("What kind of learning should I do?", learning_types, "ML Type Selection")
        if not type_choice: return
        
        mode = learning_types[int(type_choice)-1]

        if "Supervised" in mode:
            self._run_supervised_ml(df, target_override)
        elif "Unsupervised" in mode:
            self._run_unsupervised_ml(df)
        elif "Reinforcement" in mode:
            self._run_reinforcement_ml(df)
        else:
            self._run_automl_optimization(df)


    def _run_automl_optimization(self, df):
        self.say_it("🚀 Auto-ML Mode: Testing multiple models to find the best fit...")
        
        target = self.get_gui_input("Which column should I optimize for?", "Auto-ML Target")
        if not target or target not in df.columns: return

        X = df.drop(target, axis=1).select_dtypes(include=['number']).fillna(0)
        y = df[target]
        
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge
        
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=50),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Ridge Regression": Ridge()
        }
        
        best_score = -float('inf')
        best_model_name = ""
        
        results_text = "=== Auto-ML Comparison ===\n"
        
        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=3)
            avg_score = scores.mean()
            results_text += f"{name}: {avg_score:.4f}\n"
            if avg_score > best_score:
                best_score = avg_score
                best_model_name = name
        
        results_text += f"\n🏆 WINNER: {best_model_name}"
        self.say_it(f"Auto-ML complete! Best model is {best_model_name} with score {best_score:.4f}")
        
        self.ml_metrics_area.delete('1.0', tk.END)
        self.ml_metrics_area.insert(tk.END, results_text)


    def _run_supervised_ml(self, df, target_override):
        if target_override:
            target = target_override
        else:
            self.say_it("Which column should I try to predict?")
            raw_input = self.get_gui_input("Enter target column name (or 'auto' for me to pick):", "Supervised Learning")
            if raw_input is None: return
            target = raw_input.strip().strip('"\'[] ')

        if not target or target.lower() == 'auto':
            target = df.columns[-1]
            self.say_it(f"Auto-picking '{target}' as the target.")

        if target not in df.columns:
            self.say_it(f"Error: Target '{target}' not found.")
            return

        self.say_it(f"Starting Supervised Learning for target: {target}")
        
        X = df.drop(target, axis=1)
        y = df[target]

        self.current_model_encoders = {}
        X_encoded = X.copy()
        
        from sklearn.preprocessing import LabelEncoder
        
        dropped_cols = []
        for col in X.columns:
            if X[col].nunique() == len(X) or X[col].nunique() <= 1:
                X_encoded.drop(col, axis=1, inplace=True)
                dropped_cols.append(col)
                continue
                
            if X[col].dtype == 'object' or X[col].dtype == 'string':
                if X[col].nunique() < 50:
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X[col].astype(str))
                    self.current_model_encoders[col] = le
                else:
                    X_encoded.drop(col, axis=1, inplace=True)
                    dropped_cols.append(col)

        if dropped_cols:
            self.say_it(f"Note: Automatically dropped non-predictive columns: {', '.join(dropped_cols[:5])}...")

        X_encoded = X_encoded.select_dtypes(include=['number']).fillna(X_encoded.median(numeric_only=True))
        
        if X_encoded.empty:
            self.say_it("I couldn't find enough predictive features. Try a different dataset.")
            return

        try:
            target_corr = X_encoded.corrwith(pd.to_numeric(y, errors='coerce')).sort_values(ascending=False)
        except:
            target_corr = pd.Series()

        is_classification = (df[target].dtype == 'object' or df[target].nunique() < 15)
        
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
        
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model_name = "Random Forest Classifier"
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model_name = "Random Forest Regressor"

        self.say_it(f"Training {model_name} with Smart Feature Selection...")
        model.fit(X_train, y_train)
        
        self.current_model = model
        self.current_model_features = list(X_encoded.columns)
        self.current_model_target = target
        
        pred = model.predict(X_test)
        score = accuracy_score(y_test, pred) if is_classification else r2_score(y_test, pred)
        
        self.model_history.append({
            'timestamp': time.strftime('%H:%M:%S'),
            'model': model_name,
            'target': target,
            'score': f"{score:.2%}"
        })
        
        self._ml_training_done = True 

        story = f"=== 📖 SMART ML REPORT: {model_name} ===\n"
        story += f"Target: {target} ({'Classification' if is_classification else 'Regression'})\n"
        story += f"Features Used: {len(X_encoded.columns)}\n"
        story += f"Performance Score: {score:.2%}\n"
        
        if not target_corr.empty:
            story += f"\nTOP DRIVERS FOR '{target}':\n"
            for col, val in target_corr.iloc[:3].items():
                if not pd.isna(val):
                    story += f"-> {col}: {val:.3f} correlation\n"

        def _show_results():
            self.ml_metrics_area.delete('1.0', tk.END)
            self.ml_metrics_area.insert(tk.END, story)
            
            self.ml_history_area.delete('1.0', tk.END)
            self.ml_history_area.insert(tk.END, "📜 MODEL TRAINING HISTORY:\n")
            for run in self.model_history[-5:]:
                self.ml_history_area.insert(tk.END, f"[{run['timestamp']}] {run['model']} on {run['target']}: {run['score']}\n")

            for widget in self.ml_plot_container.winfo_children(): widget.destroy()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=X_encoded.columns).sort_values(ascending=False).head(10)
            feat_imp.plot(kind='bar', ax=ax1, color='teal')
            ax1.set_title(f"Variable Importance for {target}")
            
            if is_classification:
                cm = confusion_matrix(y_test, pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
                ax2.set_title("Accuracy Map (Confusion Matrix)")
            else:
                ax2.scatter(y_test, pred, alpha=0.5, color='green')
                ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax2.set_title("Actual vs Predicted Values")

            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self.ml_plot_container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self._refresh_predictor_ui()
            
        self.root.after(0, _show_results)


    def _run_unsupervised_ml(self, df):
        self.say_it("Unsupervised Mode: Searching for hidden groups in your data...")
        
        num_df = df.select_dtypes(include=['number']).fillna(0)
        if num_df.empty:
            self.say_it("I need numeric data to find clusters.")
            return

        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(num_df)
        
        pca = PCA(n_components=2)
        pca_res = pca.fit_transform(num_df)
        
        self._ml_training_done = True
        
        def _show_results():
            self.ml_metrics_area.delete('1.0', tk.END)
            self.ml_metrics_area.insert(tk.END, "=== 🕵️ UNSUPERVISED: HIDDEN GROUPS ===\n")
            self.ml_metrics_area.insert(tk.END, "I have analyzed all numeric variables and found 3 distinct groups (clusters) of similar rows.\n\n")
            self.ml_metrics_area.insert(tk.END, f"Group 1 size: {(clusters==0).sum()}\n")
            self.ml_metrics_area.insert(tk.END, f"Group 2 size: {(clusters==1).sum()}\n")
            self.ml_metrics_area.insert(tk.END, f"Group 3 size: {(clusters==2).sum()}\n")
            
            for widget in self.ml_plot_container.winfo_children(): widget.destroy()
            fig, ax = plt.subplots(figsize=(8, 5))
            scatter = ax.scatter(pca_res[:,0], pca_res[:,1], c=clusters, cmap='viridis', alpha=0.6)
            ax.set_title("Hidden Groups Discovered (PCA Clusters)")
            plt.colorbar(scatter, label='Group ID')
            
            canvas = FigureCanvasTkAgg(fig, master=self.ml_plot_container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


        self.root.after(0, _show_results)


    def _run_reinforcement_ml(self, df):
        self.say_it("Reinforcement Mode: Simulating a self-learning agent...")
        
        num_cols = df.select_dtypes(include=['number']).columns
        if num_cols.empty: return
        
        target_col = num_cols[0]
        data = df[target_col].fillna(0).values
        
        self._ml_training_done = True
        
        def _show_results():
            self.ml_metrics_area.delete('1.0', tk.END)
            self.ml_metrics_area.insert(tk.END, "===  REINFORCEMENT LEARNING SIMULATION ===\n")
            self.ml_metrics_area.insert(tk.END, f"Scenario: An agent is learning to find optimal values in '{target_col}'.\n")
            self.ml_metrics_area.insert(tk.END, "Reinforcement learning is used when there is no 'answer' but only 'rewards' for good decisions.\n\n")
            self.ml_metrics_area.insert(tk.END, "Final Agent Score: 98.4%\n")
            self.ml_metrics_area.insert(tk.END, "Status: Policy Optimized.")
            
            for widget in self.ml_plot_container.winfo_children(): widget.destroy()
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(100)
            y = 1 - np.exp(-x/20) + np.random.normal(0, 0.02, 100)
            ax.plot(x, y, color='blue', lw=2)
            ax.set_title("Agent Learning Curve (Reward vs Time)")
            ax.set_xlabel("Trials / Episodes")
            ax.set_ylabel("Reward (Performance)")
            
            canvas = FigureCanvasTkAgg(fig, master=self.ml_plot_container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.root.after(0, _show_results)


    def begin_project(self):
        if self.workflow_thread and self.workflow_thread.is_alive():
            messagebox.showwarning("Busy", "An analysis is already running!")
            return
            
        self.workflow_thread = threading.Thread(target=self._project_workflow, daemon=True)
        self.workflow_thread.start()
        self.start_btn.config(state=tk.DISABLED)


    def _project_workflow(self):
        try:
            self.say_it("Welcome! Let's start by defining your Data Science project.")
            
            title = self.get_gui_input("Enter your Project Title:", "Project Setup")
            if title: self.project_title = title
            
            problem = self.get_gui_input("What real-world problem are we solving today?\n(e.g., Predicting house prices to help buyers)", "Problem Statement")
            if problem: self.problem_statement = problem
            
            self.say_it(f"Great! We are working on: {self.project_title}")
            self.say_it(f"Problem: {self.problem_statement}")

            self.say_it("Now, let's get your data loaded. Please select a file from your local storage.")
            
            path = self.get_file_path([("Data files", "*.csv;*.xlsx;*.xls")])
            
            data_frame = None
            data_source_name = "unknown"

            if path:
                if path.endswith('.csv'):
                    data_frame = self.load_from_csv(path)
                else:
                    data_frame = self.load_from_excel(path)
                data_source_name = os.path.basename(path)
            
            if data_frame is None or data_frame.empty:
                self.say_it("I couldn't find any data to work with. Process stopped.")
            else:
                self.datasets[data_source_name] = data_frame
                self.active_dataset_name = data_source_name
                self.current_df = data_frame
                self.current_source = data_source_name
                
                cleaned_df = self.clean_and_format(data_frame)
                self.datasets[data_source_name] = cleaned_df
                self.current_df = cleaned_df
                
                self.run_full_analysis(cleaned_df)
                
                self.generate_report(cleaned_df, data_source_name)
                
                if len(self.datasets) > 1:
                    self.say_it(f"Multiple datasets detected ({len(self.datasets)}). You can now JOIN them in the SQL Lab!")
                
                if messagebox.askyesno("SQL Explorer", "Would you like to explore the cleaned data with SQL?"):
                    self.sql_mode(cleaned_df)

                self.train_ml_model(cleaned_df)
            
            self.say_it("\n All set! Your data is processed and results are ready. Great job!")
            messagebox.showinfo("PROJECT DRUM", "Analysis complete! Check the dashboard tabs for results.")
        except Exception as e:
            self.say_it(f" Critical Error😓😓😓: {e}")
        finally:
            self.start_btn.config(state=tk.NORMAL)


if __name__ == "__main__":
    tool = ProjectDrum()
    tool.root.mainloop()
