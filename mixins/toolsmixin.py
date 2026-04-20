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

class ToolsMixin:
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

