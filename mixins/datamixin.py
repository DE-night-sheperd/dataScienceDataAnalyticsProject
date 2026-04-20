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

class DataMixin:
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
                   self.dl_target_combo, self.ts_time_combo, self.ts_val_combo, self.nlp_text_combo):
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

