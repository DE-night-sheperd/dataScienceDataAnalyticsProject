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

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False

try:
    from wordcloud import WordCloud
    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon', quiet=True)
    NLP_OK = True
except ImportError:
    NLP_OK = False

class ExtraLabMixin:
    # ══════════════════════════════════════════════════════════════════════════
    #  TIME SERIES LAB
    # ══════════════════════════════════════════════════════════════════════════
    def _parse_dates(self):
        if self.current_df is None: return
        t_col = self.ts_time_combo.get()
        if not t_col: messagebox.showwarning("Select", "Select a time column first."); return
        
        try:
            self.current_df[t_col] = pd.to_datetime(self.current_df[t_col], errors='coerce')
            self.say_it(f"Parsed '{t_col}' as datetime.")
            self._update_all_gui_elements()
        except Exception as e:
            messagebox.showerror("Error", f"Could not parse dates: {e}")

    def _plot_ts_trend(self):
        if self.current_df is None: return
        t_col = self.ts_time_combo.get()
        v_col = self.ts_val_combo.get()
        if not t_col or not v_col: messagebox.showwarning("Select", "Select time and value columns."); return
        
        df = self.current_df.dropna(subset=[t_col, v_col]).sort_values(t_col)
        if not pd.api.types.is_datetime64_any_dtype(df[t_col]):
            messagebox.showinfo("Format", "Time column must be datetime. Click 'Parse Dates' first.")
            return
            
        fig, ax = plt.subplots(figsize=(10, 5))
        dark_fig(fig)
        
        ax.plot(df[t_col], df[v_col], color=P["accent"], lw=2)
        ax.set_title(f"Trend over Time: {v_col}", color=P["fg"])
        ax.set_xlabel(t_col, color=P["fg"])
        ax.set_ylabel(v_col, color=P["fg"])
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        embed_fig(fig, self.ts_plot_container)
        self.ts_metrics_area.delete("1.0", tk.END)
        self.ts_metrics_area.insert(tk.END, f"Plotted trend for {len(df)} temporal points.\nMin date: {df[t_col].min()}\nMax date: {df[t_col].max()}\n")

    def _forecast_arima(self):
        if not STATSMODELS_OK:
            messagebox.showerror("Missing library", "pip install statsmodels")
            return
        if self.current_df is None: return
        t_col = self.ts_time_combo.get()
        v_col = self.ts_val_combo.get()
        if not t_col or not v_col: return
        
        df = self.current_df.dropna(subset=[t_col, v_col]).sort_values(t_col)
        df = df.set_index(t_col)
        
        try:
            self.ts_metrics_area.insert(tk.END, "\nFitting ARIMA(5,1,0)...\n")
            model = ARIMA(df[v_col].astype(float), order=(5,1,0))
            res = model.fit()
            
            steps = 30
            forecast = res.forecast(steps=steps)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            dark_fig(fig)
            ax.plot(df.index[-100:], df[v_col].iloc[-100:], color=P["accent"], label="Historical (last 100)")
            
            # Forecast index
            if pd.api.types.is_datetime64_any_dtype(df.index):
                freq = df.index.to_series().diff().median()
                fc_index = [df.index[-1] + freq * i for i in range(1, steps + 1)]
            else:
                fc_index = range(len(df), len(df) + steps)
                
            ax.plot(fc_index, forecast, color=P["warn"], lw=2, linestyle="--", label="Forecast")
            ax.set_title(f"ARIMA Forecast: {v_col}", color=P["fg"])
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            embed_fig(fig, self.ts_plot_container)
            self.ts_metrics_area.insert(tk.END, f"Forecast generated for {steps} steps ahead.\n")
        except Exception as e:
            self.ts_metrics_area.insert(tk.END, f"ARIMA failed: {e}\n")

    # ══════════════════════════════════════════════════════════════════════════
    #  NLP LAB
    # ══════════════════════════════════════════════════════════════════════════
    def _generate_wordcloud(self):
        if not NLP_OK:
            messagebox.showerror("Missing library", "pip install wordcloud nltk scikit-learn")
            return
        if self.current_df is None: return
        t_col = self.nlp_text_combo.get()
        if not t_col: return
        
        text = " ".join(self.current_df[t_col].dropna().astype(str))
        if len(text.strip()) == 0: return
        
        try:
            wc = WordCloud(width=800, height=400, background_color=P["surface"], 
                           colormap="plasma", max_words=150).generate(text)
                           
            fig, ax = plt.subplots(figsize=(10, 5))
            dark_fig(fig)
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            ax.set_title(f"WordCloud: {t_col}", color=P["fg"])
            plt.tight_layout()
            embed_fig(fig, self.nlp_plot_container)
            self.nlp_metrics_area.insert(tk.END, "WordCloud generated successfully.\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _analyze_sentiment(self):
        if not NLP_OK: return
        if self.current_df is None: return
        t_col = self.nlp_text_combo.get()
        if not t_col: return
        
        sia = SentimentIntensityAnalyzer()
        texts = self.current_df[t_col].dropna().astype(str).head(1000)
        
        sentiments = []
        for t in texts:
            score = sia.polarity_scores(t)
            sentiments.append(score['compound'])
            
        fig, ax = plt.subplots(figsize=(10, 5))
        dark_fig(fig)
        
        sns.histplot(sentiments, bins=30, kde=True, ax=ax, color=P["accent3"])
        ax.set_title("VADER Sentiment Distribution (Compound Score)", color=P["fg"])
        ax.set_xlabel("Sentiment Score (-1.0 to 1.0)")
        plt.tight_layout()
        
        embed_fig(fig, self.nlp_plot_container)
        
        avg_score = np.mean(sentiments)
        overall = "Positive" if avg_score > 0.05 else "Negative" if avg_score < -0.05 else "Neutral"
        self.nlp_metrics_area.insert(tk.END, f"Analyzed {len(texts)} rows.\nAverage Sentiment: {avg_score:.3f} ({overall})\n")

    def _extract_tfidf(self):
        if not NLP_OK: return
        if self.current_df is None: return
        t_col = self.nlp_text_combo.get()
        if not t_col: return
        
        texts = self.current_df[t_col].dropna().astype(str).head(2000)
        try:
            vec = TfidfVectorizer(stop_words='english', max_features=20)
            tfidf_matrix = vec.fit_transform(texts)
            feature_names = vec.get_feature_names_out()
            dense = tfidf_matrix.todense()
            denselist = dense.tolist()
            df_tfidf = pd.DataFrame(denselist, columns=feature_names)
            sums = df_tfidf.sum().sort_values(ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            dark_fig(fig)
            
            sums.plot(kind='barh', ax=ax, color=P["warn"])
            ax.set_title(f"Top 20 TF-IDF Words in {t_col}", color=P["fg"])
            plt.tight_layout()
            embed_fig(fig, self.nlp_plot_container)
            self.nlp_metrics_area.insert(tk.END, "TF-IDF extraction complete.\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))