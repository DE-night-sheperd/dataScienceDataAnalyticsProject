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

class ChartsMixin:
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

