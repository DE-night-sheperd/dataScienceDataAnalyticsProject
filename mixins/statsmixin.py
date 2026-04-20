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

class StatsMixin:
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

