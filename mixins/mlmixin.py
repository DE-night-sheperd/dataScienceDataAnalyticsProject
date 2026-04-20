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
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

try:
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
except ImportError:
    pass


try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    import lightgbm as lgb
    LGBM_OK = True
except ImportError:
    LGBM_OK = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_OK = True
except ImportError:
    SMOTE_OK = False

from sklearn.ensemble import VotingClassifier, VotingRegressor

class MLMixin:
    # ══════════════════════════════════════════════════════════════════════════
    #  MACHINE LEARNING — full visualisation
    # ══════════════════════════════════════════════════════════════════════════

    def _run_unsupervised(self):
        if not SKLEARN_OK or self.current_df is None: return
        algo = self.unsup_algo.get()
        param = self.unsup_k.get()
        
        def _task():
            try:
                df = self.current_df.copy()
                num_cols = df.select_dtypes(include="number").columns
                if len(num_cols) < 2:
                    self.root.after(0, lambda: messagebox.showwarning("Data", "Need at least 2 numeric columns."))
                    return
                
                X = df[num_cols].dropna()
                if X.empty: return
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                fig, ax = plt.subplots(figsize=(9, 6)); dark_fig(fig)
                text = f"=== 🧩 UNSUPERVISED: {algo} ===\n\nRows: {len(X)}\nFeatures: {len(num_cols)}\n"
                
                if algo in ["K-Means", "Agglomerative", "DBSCAN"]:
                    if algo == "K-Means":
                        k = int(param) if param.isdigit() else 3
                        model = KMeans(n_clusters=k, random_state=42)
                        text += f"Clusters (k): {k}\n"
                    elif algo == "Agglomerative":
                        k = int(param) if param.isdigit() else 3
                        model = AgglomerativeClustering(n_clusters=k)
                        text += f"Clusters (k): {k}\n"
                    elif algo == "DBSCAN":
                        eps = float(param) if param.replace('.','',1).isdigit() else 0.5
                        model = DBSCAN(eps=eps, min_samples=5)
                        text += f"Epsilon: {eps}, Min Samples: 5\n"
                    
                    labels = model.fit_predict(X_scaled)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    text += f"Found {n_clusters} clusters.\n"
                    
                    # Reduce to 2D for plotting if needed
                    if X_scaled.shape[1] > 2:
                        pca = PCA(n_components=2)
                        X_plot = pca.fit_transform(X_scaled)
                        text += f"PCA Variance Explained (2D): {sum(pca.explained_variance_ratio_):.2%}\n"
                    else:
                        X_plot = X_scaled
                        
                    scatter = ax.scatter(X_plot[:,0], X_plot[:,1], c=labels, cmap="plasma", alpha=0.7, s=20)
                    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                    ax.add_artist(legend1)
                    ax.set_title(f"{algo} Clustering", color=P["fg"])
                    
                elif algo in ["PCA (2D)", "t-SNE (2D)"]:
                    if algo == "PCA (2D)":
                        model = PCA(n_components=2, random_state=42)
                        X_plot = model.fit_transform(X_scaled)
                        text += f"Variance Explained: {sum(model.explained_variance_ratio_):.2%}\n"
                        ax.set_title("PCA Projection", color=P["fg"])
                    else:
                        perp = float(param) if param.replace('.','',1).isdigit() else 30.0
                        model = TSNE(n_components=2, perplexity=perp, random_state=42)
                        X_plot = model.fit_transform(X_scaled)
                        text += f"Perplexity: {perp}\n"
                        ax.set_title("t-SNE Projection", color=P["fg"])
                    
                    ax.scatter(X_plot[:,0], X_plot[:,1], color=CC[0], alpha=0.6, s=15)
                    
                plt.tight_layout()
                
                def _up():
                    self.unsup_metrics_area.delete("1.0", tk.END)
                    self.unsup_metrics_area.insert(tk.END, text)
                    for w in self.unsup_plot_container.winfo_children(): w.destroy()
                    canvas = FigureCanvasTkAgg(fig, master=self.unsup_plot_container)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                self.root.after(0, _up)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Unsupervised Error", str(e)))

        threading.Thread(target=_task, daemon=True).start()

    def _prepare_features(self, df, target):
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not in DataFrame")
        y = df[target].copy(); X = df.drop(columns=[target])
        encoders = {}; dropped = []
        for col in list(X.columns):
            n = X[col].nunique()
            if n <= 1 or n == len(X):
                X = X.drop(columns=[col]); dropped.append(col); continue
            if X[col].dtype == object or str(X[col].dtype) == "string":
                if n < 100:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str).fillna("__na__"))
                    encoders[col] = le
                else:
                    X = X.drop(columns=[col]); dropped.append(col)
        X = X.select_dtypes(include="number").fillna(X.select_dtypes(include="number").median())
        if X.empty:
            raise ValueError("No usable feature columns after preprocessing.")
        return X, y, encoders, dropped

    def _get_algorithm(self, algo_name, is_clf):
        mapping = {
            "Random Forest":                  (RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
                                               RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)),
            "Gradient Boosting":              (GradientBoostingClassifier(n_estimators=100, random_state=42),
                                               GradientBoostingRegressor(n_estimators=100, random_state=42)),
            "Logistic / Linear Regression":   (LogisticRegression(max_iter=500, random_state=42, n_jobs=-1),
                                               LinearRegression(n_jobs=-1)),
            "SVM":                            (SVC(kernel="rbf", probability=True, random_state=42),
                                               SVR(kernel="rbf")),
            "KNN":                            (KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
                                               KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),
            "Decision Tree":                  (DecisionTreeClassifier(max_depth=8, random_state=42),
                                               DecisionTreeRegressor(max_depth=8, random_state=42)),
            "AdaBoost":                       (AdaBoostClassifier(n_estimators=100, random_state=42),
                                               AdaBoostRegressor(n_estimators=100, random_state=42)),
            "Naive Bayes":                    (GaussianNB(), GaussianNB()),
            "ElasticNet":                     (LogisticRegression(penalty="elasticnet", solver="saga",
                                                                  l1_ratio=0.5, max_iter=500, n_jobs=-1),
                                               ElasticNet(alpha=0.1, random_state=42)),
        }
        if XGB_OK:
            mapping["XGBoost"] = (xgb.XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1, eval_metric='logloss'),
                                  xgb.XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1))
        if LGBM_OK:
            mapping["LightGBM"] = (lgb.LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1),
                                   lgb.LGBMRegressor(n_estimators=200, random_state=42, n_jobs=-1))
        
        estimators_clf = [('rf', RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)),
                          ('gb', GradientBoostingClassifier(n_estimators=150, random_state=42))]
        estimators_reg = [('rf', RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)),
                          ('gb', GradientBoostingRegressor(n_estimators=150, random_state=42))]
        if XGB_OK:
            estimators_clf.append(('xgb', xgb.XGBClassifier(n_estimators=150, random_state=42, n_jobs=-1, eval_metric='logloss')))
            estimators_reg.append(('xgb', xgb.XGBRegressor(n_estimators=150, random_state=42, n_jobs=-1)))
        if LGBM_OK:
            estimators_clf.append(('lgbm', lgb.LGBMClassifier(n_estimators=150, random_state=42, n_jobs=-1)))
            estimators_reg.append(('lgbm', lgb.LGBMRegressor(n_estimators=150, random_state=42, n_jobs=-1)))
            
        mapping["Voting Ensemble"] = (VotingClassifier(estimators=estimators_clf, voting='soft', n_jobs=-1),
                                      VotingRegressor(estimators=estimators_reg, n_jobs=-1))
                                      
        clf_model, reg_model = mapping.get(algo_name, mapping["Random Forest"])
        return clf_model if is_clf else reg_model

    def _run_gui_ml(self):
        if not SKLEARN_OK: messagebox.showerror("sklearn required","pip install scikit-learn"); return
        target = self.ml_target_combo.get()
        if not target: messagebox.showwarning("No Target","Select a target column!"); return
        if self.current_df is None: return
        self._start_learning_animation(target)
        threading.Thread(target=self._train_supervised,
                         args=(self.current_df.copy(), target), daemon=True).start()

    def _start_learning_animation(self, target):
        win = tk.Toplevel(self.root); win.title("🧠 Training…")
        win.geometry("420x260"); win.configure(bg=P["surface"]); win.resizable(False,False)
        tk.Label(win, text="Machine is Learning…", font=("Segoe UI",13,"bold"),
                 bg=P["surface"], fg=P["accent"]).pack(pady=14)
        tk.Label(win, text=f"Target: {target}", font=("Segoe UI",10),
                 bg=P["surface"], fg=P["fg_dim"]).pack()
        pb = ttk.Progressbar(win, mode="indeterminate", length=320); pb.pack(pady=12); pb.start(12)
        sl = tk.Label(win, text="Initialising…", font=("Segoe UI",9),
                      bg=P["surface"], fg=P["accent2"]); sl.pack()
        phases = ["Splitting data…","Fitting estimator…","Cross-validating…",
                  "Computing importances…","Building confusion matrix…","Finalising…"]
        self._ml_training_done = False
        def _tick():
            if not win.winfo_exists(): return
            sl.config(text=phases[np.random.randint(len(phases))])
            if self._ml_training_done: win.destroy(); self._ml_training_done = False
            else: win.after(600, _tick)
        _tick()

    def _train_supervised(self, df, target):
        try:
            X, y, encoders, dropped = self._prepare_features(df, target)
        except ValueError as e:
            self.root.after(0, lambda: (messagebox.showerror("Prep Error", str(e)),
                                        setattr(self, "_ml_training_done", True)))
            return

        is_clf = (y.dtype == object or y.nunique() <= 20)
        target_encoder = None
        if is_clf and y.dtype == object:
            target_encoder = LabelEncoder()
            y_enc = target_encoder.fit_transform(y.astype(str))
        else:
            y_enc = y.values.astype(float) if not is_clf else y.values

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_enc, test_size=0.2, random_state=42, stratify=(y_enc if is_clf else None))
        except Exception:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=0.2, random_state=42)

        algo_name = self.ml_algo_combo.get()
        model = self._get_algorithm(algo_name, is_clf)

        # Optimal scaling for distance/linear algorithms
        if algo_name in ["SVM", "KNN", "Logistic / Linear Regression", "ElasticNet"]:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            X_scaled = scaler.fit_transform(X) # For CV
            self._ml_scaler = scaler
        else:
            self._ml_scaler = None
            X_scaled = X

        try:
            model.fit(X_tr, y_tr)
        except Exception as e:
            self.root.after(0, lambda: (messagebox.showerror("Train Error", str(e)),
                                        setattr(self,"_ml_training_done",True)))
            return

        preds = model.predict(X_te)
        score = accuracy_score(y_te, preds) if is_clf else r2_score(y_te, preds)
        metric = "Accuracy" if is_clf else "R²"

        # CV score
        try:
            cv_scores = cross_val_score(model, X_scaled, y_enc, cv=5,
                                         scoring="accuracy" if is_clf else "r2", n_jobs=-1)
        except Exception:
            cv_scores = np.array([score])

        self.current_model           = model
        self.current_model_features  = list(X.columns)
        self.current_model_target    = target
        self.current_model_encoders  = encoders
        self._target_encoder         = target_encoder
        self._is_clf                 = is_clf
        self._y_test                 = y_te
        self._preds_test             = preds
        self._X_train                = X_tr
        self._X_test                 = X_te
        self._y_train                = y_tr

        self.model_history.append({
            "time": time.strftime("%H:%M:%S"), "algo": algo_name,
            "model": type(model).__name__, "target": target,
            "metric": f"{metric}={score:.4f}", "cv_mean": f"{cv_scores.mean():.4f}"
        })

        lines = [
            f"=== 🤖 ML REPORT: {algo_name} ===",
            f"Target      : {target}  ({'Classification' if is_clf else 'Regression'})",
            f"Features    : {len(X.columns)}",
            f"Train/Test  : {len(X_tr)} / {len(X_te)}",
            f"{metric}       : {score:.4f}",
            f"CV (5-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
        ]
        if is_clf:
            lines += ["", "Classification Report:",
                      classification_report(y_te, preds, zero_division=0)]
        else:
            rmse = math.sqrt(mean_squared_error(y_te, preds))
            mae  = mean_absolute_error(y_te, preds)
            lines += [f"RMSE        : {rmse:.4f}", f"MAE         : {mae:.4f}"]
        if dropped:
            lines += [f"\nDropped    : {', '.join(dropped[:8])}"]

        self._ml_training_done = True
        self.root.after(0, lambda: self._show_ml_results(model, X, y_te, preds,
                                                          score, metric, is_clf,
                                                          "\n".join(lines), cv_scores))
        self.say_it(f"Training done. {metric}: {score:.2%}")

    def _show_ml_results(self, model, X, y_test, preds, score, metric, is_clf, report_text, cv_scores):
        self.ml_metrics_area.delete("1.0",tk.END)
        self.ml_metrics_area.insert(tk.END, report_text)
        self.ml_history_area.delete("1.0",tk.END)
        self.ml_history_area.insert(tk.END,"📜 TRAINING HISTORY:\n")
        for r in self.model_history[-8:]:
            self.ml_history_area.insert(tk.END, f"[{r['time']}] {r['algo']} → {r['metric']} (CV={r['cv_mean']})\n")

        for w in self.ml_plot_container.winfo_children(): w.destroy()

        if is_clf:
            fig, axs = plt.subplots(2, 2, figsize=(13,10))
        else:
            fig, axs = plt.subplots(2, 2, figsize=(13,10))
        dark_fig(fig)

        # Feature importances
        if hasattr(model,"feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
            colors = [CC[i%len(CC)] for i in range(len(fi))]
            axs[0,0].barh(fi.index[::-1], fi.values[::-1], color=colors[::-1], edgecolor="none")
            axs[0,0].set_title("Feature Importances", color=P["fg"])
            axs[0,0].set_xlabel("Importance")
        elif hasattr(model,"coef_"):
            coefs = model.coef_.flatten()[:len(X.columns)]
            fi = pd.Series(np.abs(coefs), index=X.columns[:len(coefs)]).sort_values(ascending=False).head(15)
            axs[0,0].barh(fi.index[::-1], fi.values[::-1], color=CC[0], edgecolor="none")
            axs[0,0].set_title("Coefficient Magnitudes", color=P["fg"])
        else:
            axs[0,0].text(0.5,0.5,"Feature importances\nnot available", ha="center",
                          color=P["fg_dim"], transform=axs[0,0].transAxes)

        # CV scores
        axs[0,1].bar(range(len(cv_scores)), cv_scores, color=CC, edgecolor="none", alpha=0.85)
        axs[0,1].axhline(cv_scores.mean(), color=P["warn"], lw=2, linestyle="--",
                         label=f"Mean={cv_scores.mean():.3f}")
        axs[0,1].fill_between([-0.5,len(cv_scores)-0.5],
                               cv_scores.mean()-cv_scores.std(),
                               cv_scores.mean()+cv_scores.std(),
                               alpha=0.2, color=P["warn"])
        axs[0,1].set_title("5-Fold CV Scores", color=P["fg"])
        axs[0,1].set_xlabel("Fold"); axs[0,1].set_ylabel(metric)
        axs[0,1].legend(fontsize=9)
        axs[0,1].set_xticks(range(len(cv_scores)))
        axs[0,1].set_xticklabels([f"Fold {i+1}" for i in range(len(cv_scores))], rotation=20)

        if is_clf:
            # Confusion matrix
            cm = confusion_matrix(y_test, preds)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[1,0],
                        linewidths=0.5, linecolor=P["border"])
            axs[1,0].set_title("Confusion Matrix", color=P["fg"])
            axs[1,0].set_xlabel("Predicted"); axs[1,0].set_ylabel("Actual")

            # ROC curve (binary only)
            try:
                if len(np.unique(y_test)) == 2 and hasattr(model,"predict_proba"):
                    proba = model.predict_proba(self._X_test)[:,1]
                    fpr, tpr, _ = roc_curve(y_test, proba)
                    roc_auc = auc(fpr, tpr)
                    axs[1,1].plot(fpr, tpr, color=P["accent"], lw=2, label=f"AUC = {roc_auc:.3f}")
                    axs[1,1].plot([0,1],[0,1],"--", color=P["fg_dim"], lw=1)
                    axs[1,1].fill_between(fpr, tpr, alpha=0.1, color=P["accent"])
                    axs[1,1].set_xlabel("False Positive Rate"); axs[1,1].set_ylabel("True Positive Rate")
                    axs[1,1].set_title("ROC Curve", color=P["fg"])
                    axs[1,1].legend(fontsize=9)
                else:
                    # Multi-class: per-class bar
                    from sklearn.metrics import precision_recall_fscore_support
                    pr, rec, f1, sup = precision_recall_fscore_support(y_test, preds, zero_division=0)
                    x_labels = [str(c) for c in np.unique(y_test)][:len(pr)]
                    xp = np.arange(len(x_labels))
                    w = 0.25
                    axs[1,1].bar(xp-w, pr[:len(xp)],  w, label="Precision", color=CC[0], alpha=0.85)
                    axs[1,1].bar(xp,   rec[:len(xp)], w, label="Recall",    color=CC[1], alpha=0.85)
                    axs[1,1].bar(xp+w, f1[:len(xp)],  w, label="F1",        color=CC[2], alpha=0.85)
                    axs[1,1].set_xticks(xp); axs[1,1].set_xticklabels(x_labels, rotation=30)
                    axs[1,1].set_title("Per-Class Metrics", color=P["fg"])
                    axs[1,1].legend(fontsize=9)
            except Exception: pass
        else:
            # Actual vs Predicted
            axs[1,0].scatter(y_test, preds, alpha=0.35, color=CC[0], s=14)
            lim = [min(float(y_test.min()), float(preds.min())),
                   max(float(y_test.max()), float(preds.max()))]
            axs[1,0].plot(lim, lim, "--", color=P["warn"], lw=2, label="Perfect fit")
            # Regression line through scatter
            m, b, r, _, _ = stats.linregress(y_test, preds)
            xr = np.linspace(lim[0], lim[1], 100)
            axs[1,0].plot(xr, m*xr+b, color=P["danger"], lw=1.5, alpha=0.7, label=f"Fit line r={r:.2f}")
            axs[1,0].set_xlabel("Actual"); axs[1,0].set_ylabel("Predicted")
            axs[1,0].set_title("Actual vs Predicted", color=P["fg"])
            axs[1,0].legend(fontsize=9)

            # Residuals
            residuals = preds - y_test
            axs[1,1].scatter(preds, residuals, alpha=0.35, s=12, color=CC[3])
            axs[1,1].axhline(0, color=P["warn"], lw=1.5, linestyle="--")
            axs[1,1].axhline(residuals.std(),  color=P["danger"], lw=1, linestyle=":", alpha=0.7)
            axs[1,1].axhline(-residuals.std(), color=P["danger"], lw=1, linestyle=":", alpha=0.7)
            axs[1,1].set_xlabel("Predicted"); axs[1,1].set_ylabel("Residual")
            axs[1,1].set_title("Residual Plot", color=P["fg"])

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.ml_plot_container)
        canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        tf = tk.Frame(self.ml_plot_container, bg=P["surface"]); tf.pack(fill=tk.X)
        NavigationToolbar2Tk(canvas, tf)
        self._refresh_predictor_ui()

    
    def _run_hyperparameter_tuning(self):
        if not SKLEARN_OK or self.current_df is None: return
        target = self.ml_target_combo.get()
        if not target: messagebox.showwarning("No Target", "Select target."); return
        algo_name = self.ml_algo_combo.get()
        
        def _task():
            try:
                X, y, encoders, dropped = self._prepare_features(self.current_df.copy(), target)
                is_clf = (y.dtype == object or y.nunique() <= 20)
                if is_clf and y.dtype == object:
                    le = LabelEncoder()
                    y = pd.Series(le.fit_transform(y.astype(str)))
                else:
                    y = y.fillna(0)
                
                model = self._get_algorithm(algo_name, is_clf)
                
                # Define simple param grids
                param_grid = {}
                if "Random Forest" in algo_name:
                    param_grid = {'n_estimators': [100, 300, 500], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'bootstrap': [True, False]}
                elif "Gradient Boosting" in algo_name:
                    param_grid = {'n_estimators': [100, 300, 500], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [3, 5, 7, 9], 'subsample': [0.8, 1.0]}
                elif "XGBoost" in algo_name:
                    param_grid = {'n_estimators': [100, 300, 500], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7, 9], 'colsample_bytree': [0.8, 1.0]}
                elif "LightGBM" in algo_name:
                    param_grid = {'n_estimators': [100, 300, 500], 'learning_rate': [0.01, 0.05, 0.1], 'num_leaves': [31, 50, 100]}
                elif "Decision Tree" in algo_name:
                    param_grid = {'max_depth': [None, 5, 10, 20], 'min_samples_split': [2, 5, 10]}
                elif "SVM" in algo_name:
                    param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
                elif "KNN" in algo_name:
                    param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
                elif "Logistic / Linear" in algo_name and is_clf:
                    param_grid = {'C': [0.1, 1.0, 10.0]}
                else:
                    self.root.after(0, lambda: messagebox.showinfo("Info", f"No tuning grid defined for {algo_name}."))
                    self.root.after(0, lambda: setattr(self, "_ml_training_done", True))
                    return

                scoring = "accuracy" if is_clf else "r2"
                search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=25, cv=5, scoring=scoring, n_jobs=-1, random_state=42)
                search.fit(X, y)
                
                best_model = search.best_estimator_
                best_score = search.best_score_
                best_params = search.best_params_
                
                text = f"=== ⚙️ HYPERPARAMETER TUNING ===\n\nAlgorithm: {algo_name}\n"
                text += f"Best {scoring.upper()}: {best_score:.4f}\n"
                text += f"Best Params: {best_params}\n\n"
                text += "You can manually set these parameters in the code or keep the best model in memory.\n"
                
                def _up():
                    self.ml_metrics_area.delete("1.0", tk.END)
                    self.ml_metrics_area.insert(tk.END, text)
                    self.current_model = best_model  # Update current model
                    self._ml_training_done = True
                    self.say_it(f"Tuning complete. Best score: {best_score:.2f}")
                self.root.after(0, _up)
                
            except Exception as e:
                self.root.after(0, lambda: (messagebox.showerror("Tuning Error", str(e)), setattr(self,"_ml_training_done",True)))

        self._start_learning_animation(target)
        threading.Thread(target=_task, daemon=True).start()

    def _plot_learning_curve(self):
        if not SKLEARN_OK: return
        target = self.ml_target_combo.get()
        if not target or self.current_df is None: messagebox.showwarning("No Target","Select target + load data."); return
        def _task():
            try:
                X, y, _, _ = self._prepare_features(self.current_df.copy(), target)
                is_clf = (y.dtype == object or y.nunique() <= 20)
                if is_clf and y.dtype == object:
                    le = LabelEncoder(); y = pd.Series(le.fit_transform(y.astype(str)))
                else:
                    y = y.fillna(0)
                model = self._get_algorithm(self.ml_algo_combo.get(), is_clf)
                train_sizes, train_scores, val_scores = learning_curve(
                    model, X, y, cv=5, train_sizes=np.linspace(0.1,1.0,10),
                    scoring="accuracy" if is_clf else "r2", n_jobs=-1)
                def _draw():
                    win = tk.Toplevel(self.root); win.title("Learning Curve"); win.configure(bg=P["surface"])
                    fig, ax = plt.subplots(figsize=(9,6)); dark_fig(fig)
                    tr_mean = train_scores.mean(axis=1); tr_std = train_scores.std(axis=1)
                    va_mean = val_scores.mean(axis=1);   va_std = val_scores.std(axis=1)
                    ax.plot(train_sizes, tr_mean, "o-", color=CC[0], lw=2, label="Training score")
                    ax.fill_between(train_sizes, tr_mean-tr_std, tr_mean+tr_std, alpha=0.15, color=CC[0])
                    ax.plot(train_sizes, va_mean, "s-", color=CC[1], lw=2, label="CV score")
                    ax.fill_between(train_sizes, va_mean-va_std, va_mean+va_std, alpha=0.15, color=CC[1])
                    ax.set_xlabel("Training examples"); ax.set_ylabel("Score")
                    ax.set_title(f"Learning Curve — {self.ml_algo_combo.get()}", color=P["fg"])
                    ax.legend(fontsize=10); plt.tight_layout()
                    embed_fig(fig, tk.Frame(win, bg=P["surface"]))
                    canvas = FigureCanvasTkAgg(fig, master=win)
                    canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                self.root.after(0, _draw)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        threading.Thread(target=_task, daemon=True).start()

    def _plot_cv_scores(self):
        if not SKLEARN_OK or self.current_df is None: return
        target = self.ml_target_combo.get()
        if not target: messagebox.showwarning("No Target","Select target first."); return
        def _task():
            try:
                X, y, _, _ = self._prepare_features(self.current_df.copy(), target)
                is_clf = (y.dtype == object or y.nunique() <= 20)
                if is_clf and y.dtype == object:
                    le = LabelEncoder(); y = pd.Series(le.fit_transform(y.astype(str)))
                else:
                    y = y.fillna(0)
                model = self._get_algorithm(self.ml_algo_combo.get(), is_clf)
                all_scores = cross_val_score(model, X, y, cv=10,
                                              scoring="accuracy" if is_clf else "r2")
                def _draw():
                    win = tk.Toplevel(self.root); win.title("10-Fold CV Scores"); win.configure(bg=P["surface"])
                    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,5)); dark_fig(fig)
                    ax1.bar(range(1,11), all_scores, color=[CC[i%len(CC)] for i in range(10)], edgecolor="none")
                    ax1.axhline(all_scores.mean(), color=P["warn"], lw=2, linestyle="--", label=f"μ={all_scores.mean():.3f}")
                    ax1.fill_between([-0.5,10.5], all_scores.mean()-all_scores.std(), all_scores.mean()+all_scores.std(),
                                     alpha=0.15, color=P["warn"])
                    ax1.set_title("10-Fold CV Scores", color=P["fg"]); ax1.legend()
                    ax1.set_xlabel("Fold"); ax1.set_ylabel("Score")
                    ax2.hist(all_scores, bins=6, color=P["accent3"], edgecolor=P["surface"])
                    ax2.axvline(all_scores.mean(), color=P["warn"], lw=2, linestyle="--")
                    ax2.set_title("Score Distribution", color=P["fg"])
                    ax2.set_xlabel("Score"); ax2.set_ylabel("Count")
                    plt.tight_layout()
                    canvas = FigureCanvasTkAgg(fig, master=win)
                    canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                self.root.after(0, _draw)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("CV Error", str(e)))
        threading.Thread(target=_task, daemon=True).start()

    def _run_automl_optimization(self, df):
        if not SKLEARN_OK or df is None: return
        target = self.ml_target_combo.get()
        if not target: messagebox.showwarning("No Target","Select target."); return
        def _task():
            try:
                X, y, _, _ = self._prepare_features(df.copy(), target)
                is_clf = (y.dtype == object or y.nunique() <= 20)
                if is_clf and y.dtype == object:
                    le = LabelEncoder(); y = pd.Series(le.fit_transform(y.astype(str)))
                else:
                    y = y.fillna(0)
                scoring = "accuracy" if is_clf else "r2"
                models = {
                    "RandomForest":     (RandomForestClassifier if is_clf else RandomForestRegressor)(n_estimators=100, random_state=42),
                    "GradientBoosting": (GradientBoostingClassifier if is_clf else GradientBoostingRegressor)(n_estimators=100, random_state=42),
                    "Ridge/LogReg":     (LogisticRegression(max_iter=400) if is_clf else Ridge()),
                    "KNN":              (KNeighborsClassifier() if is_clf else KNeighborsRegressor()),
                    "DecisionTree":     (DecisionTreeClassifier(random_state=42) if is_clf else DecisionTreeRegressor(random_state=42)),
                }
                results = {}
                for name, mdl in models.items():
                    try:
                        cv = cross_val_score(mdl, X, y, cv=5, scoring=scoring, n_jobs=-1).mean()
                        results[name] = round(cv, 4)
                    except Exception: results[name] = 0.0
                best = max(results, key=results.get)
                text = "=== ⚗️ AUTO-ML BENCHMARK ===\n\n"
                for k, v in sorted(results.items(), key=lambda x: -x[1]):
                    bar = "█" * int(v * 30) + "░" * (30 - int(v * 30))
                    flag = " 🏆 WINNER" if k == best else ""
                    text += f"  {k:22}: {v:.4f}  {bar}{flag}\n"
                text += f"\n✅ Best model: {best} ({results[best]:.4f})\n"
                text += f"Recommendation: Use {best} for '{target}' prediction.\n"
                def _up():
                    self.ml_metrics_area.delete("1.0",tk.END)
                    self.ml_metrics_area.insert(tk.END, text)
                    self._ml_training_done = True
                self.root.after(0, _up)
                self.say_it(f"Auto-ML done. Best: {best} ({results[best]:.4f})")
            except Exception as e:
                self.root.after(0, lambda: (messagebox.showerror("Auto-ML Error", str(e)),
                                            setattr(self,"_ml_training_done",True)))
        self._start_learning_animation(target)
        threading.Thread(target=_task, daemon=True).start()

    def _gui_save_model(self):
        if self.current_model is None: messagebox.showwarning("No Model","Train first!"); return
        path = filedialog.asksaveasfilename(defaultextension=".joblib",
                                            filetypes=[("Joblib","*.joblib"),("Pickle","*.pkl")])
        if path:
            if JOBLIB_OK: joblib.dump(self.current_model, path)
            else:
                with open(path,"wb") as f: pickle.dump(self.current_model, f)
            messagebox.showinfo("Saved", f"Model saved to: {path}")

    # ══════════════════════════════════════════════════════════════════════════
    #  DEEP LEARNING
    # ══════════════════════════════════════════════════════════════════════════
    def _train_deep_learning(self):
        target = self.dl_target_combo.get()
        if not target:
            messagebox.showwarning("No Target","Select a target column!"); return
        if self.current_df is None: return
        if not TF_OK and not TORCH_OK:
            messagebox.showerror("Deep Learning","Neither TensorFlow nor PyTorch is installed.\n"
                                 "pip install tensorflow   OR   pip install torch"); return
        framework = self.dl_framework.get()
        self.dl_metrics_area.delete("1.0",tk.END)
        self.dl_metrics_area.insert(tk.END, f"🧠 Training {framework} neural network...\n")
        self._start_learning_animation(target)
        threading.Thread(target=self._dl_training_thread,
                         args=(self.current_df.copy(), target, framework), daemon=True).start()

    def _dl_training_thread(self, df, target, framework):
        try:
            X, y, encoders, _ = self._prepare_features(df, target)
            is_clf = (y.dtype == object or y.nunique() <= 20)
            self.dl_is_clf  = is_clf
            self.dl_features = list(X.columns)
            self.dl_target   = target

            target_le = None
            if is_clf and y.dtype == object:
                target_le = LabelEncoder()
                y = pd.Series(target_le.fit_transform(y.astype(str)))
            self.dl_le = target_le

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.fillna(0))
            X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y.values,
                                                        test_size=0.2, random_state=42)

            try:
                layer_sizes = [int(x.strip()) for x in self.dl_layers_entry.get().split(",") if x.strip()]
            except Exception:
                layer_sizes = [128, 64, 32]
            try:
                epochs = int(self.dl_epochs_entry.get())
            except Exception:
                epochs = 50
            try:
                lr = float(self.dl_lr_entry.get())
            except Exception:
                lr = 0.001
            try:
                dropout = float(self.dl_dropout_entry.get())
            except Exception:
                dropout = 0.2

            n_classes = len(np.unique(y.values)) if is_clf else 1
            history_data = {"loss":[], "val_loss":[], "acc":[], "val_acc":[]}

            if framework == "TensorFlow/Keras" and TF_OK:
                # Build Keras model
                model_layers = [layers.Dense(layer_sizes[0], activation="relu",
                                             input_shape=(X_tr.shape[1],))]
                if dropout > 0: model_layers.append(layers.Dropout(dropout))
                for sz in layer_sizes[1:]:
                    model_layers.append(layers.Dense(sz, activation="relu"))
                    model_layers.append(layers.BatchNormalization())
                    if dropout > 0: model_layers.append(layers.Dropout(dropout))
                if is_clf:
                    if n_classes == 2:
                        model_layers.append(layers.Dense(1, activation="sigmoid"))
                        loss = "binary_crossentropy"; metrics_list = ["accuracy"]
                    else:
                        model_layers.append(layers.Dense(n_classes, activation="softmax"))
                        loss = "sparse_categorical_crossentropy"; metrics_list = ["accuracy"]
                else:
                    model_layers.append(layers.Dense(1))
                    loss = "mse"; metrics_list = ["mae"]

                nn = keras.Sequential(model_layers)
                nn.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                           loss=loss, metrics=metrics_list)

                class HistoryCallback(keras.callbacks.Callback):
                    def __init__(self_cb):
                        super().__init__()
                    def on_epoch_end(self_cb, epoch, logs=None):
                        if logs:
                            history_data["loss"].append(logs.get("loss",0))
                            history_data["val_loss"].append(logs.get("val_loss",0))
                            if is_clf:
                                history_data["acc"].append(logs.get("accuracy",0))
                                history_data["val_acc"].append(logs.get("val_accuracy",0))

                early_stop = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                reduce_lr = callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)

                nn.fit(X_tr, y_tr, epochs=epochs, batch_size=32,
                       validation_split=0.15, callbacks=[HistoryCallback(), early_stop, reduce_lr], verbose=0)

                y_pred_raw = nn.predict(X_te, verbose=0)
                if is_clf:
                    if n_classes == 2:
                        y_pred = (y_pred_raw.flatten() > 0.5).astype(int)
                    else:
                        y_pred = np.argmax(y_pred_raw, axis=1)
                else:
                    y_pred = y_pred_raw.flatten()

                self.dl_model = nn
                self._dl_scaler = scaler

            elif TORCH_OK:
                # PyTorch manual with GPU support
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                X_tr_t = torch.tensor(X_tr.astype(np.float32)).to(device)
                X_te_t = torch.tensor(X_te.astype(np.float32)).to(device)
                if is_clf:
                    y_tr_t = torch.tensor(y_tr.astype(np.int64)).to(device)
                    y_te_t = torch.tensor(y_te.astype(np.int64)).to(device)
                else:
                    y_tr_t = torch.tensor(y_tr.astype(np.float32)).unsqueeze(1).to(device)
                    y_te_t = torch.tensor(y_te.astype(np.float32)).unsqueeze(1).to(device)

                class MLP(nn.Module):
                    def __init__(self_m, in_dim, layer_sizes, n_out, clf):
                        super().__init__()
                        layers_list = []
                        prev = in_dim
                        for sz in layer_sizes:
                            layers_list += [nn.Linear(prev, sz), nn.ReLU(), nn.BatchNorm1d(sz)]
                            if dropout > 0: layers_list.append(nn.Dropout(dropout))
                            prev = sz
                        layers_list.append(nn.Linear(prev, n_out))
                        if clf and n_out == 1: layers_list.append(nn.Sigmoid())
                        elif clf: layers_list.append(nn.Softmax(dim=1))
                        self_m.net = nn.Sequential(*layers_list)
                    def forward(self_m, x): return self_m.net(x)

                n_out = n_classes if is_clf else 1
                net = MLP(X_tr.shape[1], layer_sizes, n_out if n_classes > 2 else 1, is_clf).to(device)
                optimizer = optim.Adam(net.parameters(), lr=lr)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
                criterion = (nn.BCELoss() if (is_clf and n_classes == 2) else
                             nn.CrossEntropyLoss() if is_clf else nn.MSELoss())

                dataset = TensorDataset(X_tr_t, y_tr_t)
                loader  = DataLoader(dataset, batch_size=32, shuffle=True)
                best_val = float("inf"); patience_cnt = 0

                for epoch in range(epochs):
                    net.train(); ep_loss = 0
                    for xb, yb in loader:
                        optimizer.zero_grad()
                        out = net(xb)
                        if is_clf and n_classes > 2:
                            loss = criterion(out, yb)
                        else:
                            loss = criterion(out, yb.float() if not is_clf else yb.unsqueeze(1).float())
                        loss.backward(); optimizer.step()
                        ep_loss += loss.item()
                    ep_loss /= len(loader)
                    history_data["loss"].append(ep_loss)
                    # Simple val
                    net.eval()
                    with torch.no_grad():
                        val_out = net(X_te_t)
                        if is_clf and n_classes > 2:
                            val_loss = criterion(val_out, y_te_t).item()
                        else:
                            val_loss = criterion(val_out, y_te_t if is_clf and n_classes > 2 else y_te_t.float() if not is_clf else y_te_t.unsqueeze(1).float()).item()
                    history_data["val_loss"].append(val_loss)
                    if val_loss < best_val - 1e-4:
                        best_val = val_loss; patience_cnt = 0
                    else:
                        patience_cnt += 1
                    if patience_cnt >= 10: break
                    scheduler.step(val_loss)

                net.eval()
                with torch.no_grad():
                    raw = net(X_te_t).cpu()
                if is_clf:
                    if n_classes == 2: y_pred = (raw.numpy().flatten() > 0.5).astype(int)
                    else: y_pred = torch.argmax(raw, dim=1).numpy()
                else:
                    y_pred = raw.numpy().flatten()

                self.dl_model   = net
                self._dl_scaler = scaler
            else:
                raise RuntimeError("No DL framework available.")

            # Metrics
            if is_clf:
                score = accuracy_score(y_te, y_pred)
                report = classification_report(y_te, y_pred, zero_division=0)
            else:
                score = r2_score(y_te, y_pred)
                report = f"R²={score:.4f}  RMSE={math.sqrt(mean_squared_error(y_te,y_pred)):.4f}"

            arch_str = " → ".join([str(X_tr.shape[1])] + [str(s) for s in layer_sizes] + [str(n_classes if is_clf else 1)])
            result_text = (f"=== 🧠 DEEP LEARNING REPORT ===\n"
                           f"Framework : {framework}\n"
                           f"Target    : {target} ({'Classification' if is_clf else 'Regression'})\n"
                           f"Architecture: {arch_str}\n"
                           f"Epochs run: {len(history_data['loss'])}\n"
                           f"{'Accuracy' if is_clf else 'R²'}: {score:.4f}\n\n"
                           f"{report if is_clf else ''}")

            self._ml_training_done = True
            self.root.after(0, lambda: self._show_dl_results(result_text, history_data, y_te, y_pred, is_clf))
            self.say_it(f"Deep learning done. Score: {score:.4f}")

        except Exception as e:
            tb = traceback.format_exc()
            self._ml_training_done = True
            self.root.after(0, lambda: (messagebox.showerror("DL Error", f"{e}\n\n{tb[-500:]}"),))

    def _show_dl_results(self, text, history, y_test, y_pred, is_clf):
        self.dl_metrics_area.delete("1.0",tk.END)
        self.dl_metrics_area.insert(tk.END, text)
        for w in self.dl_plot_container.winfo_children(): w.destroy()

        fig, axs = plt.subplots(2, 2, figsize=(13,9)); dark_fig(fig)

        # Loss curves
        axs[0,0].plot(history["loss"],     color=CC[0], lw=2, label="Train Loss")
        axs[0,0].plot(history["val_loss"], color=CC[1], lw=2, linestyle="--", label="Val Loss")
        axs[0,0].set_title("Loss Curve", color=P["fg"]); axs[0,0].legend()
        axs[0,0].set_xlabel("Epoch"); axs[0,0].set_ylabel("Loss")

        if is_clf and history["acc"]:
            axs[0,1].plot(history["acc"],     color=CC[2], lw=2, label="Train Acc")
            axs[0,1].plot(history["val_acc"], color=CC[3], lw=2, linestyle="--", label="Val Acc")
            axs[0,1].set_title("Accuracy Curve", color=P["fg"]); axs[0,1].legend()
            axs[0,1].set_xlabel("Epoch"); axs[0,1].set_ylabel("Accuracy")
        else:
            # Loss improvement
            axs[0,1].plot(np.array(history["loss"]) - np.array(history["val_loss"]),
                          color=CC[4], lw=1.5)
            axs[0,1].axhline(0, color=P["warn"], lw=1, linestyle="--")
            axs[0,1].set_title("Train−Val Loss Gap", color=P["fg"])
            axs[0,1].set_xlabel("Epoch"); axs[0,1].set_ylabel("Gap")

        if is_clf:
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[1,0],
                        linewidths=0.5, linecolor=P["border"])
            axs[1,0].set_title("Confusion Matrix", color=P["fg"])
            axs[1,0].set_xlabel("Predicted"); axs[1,0].set_ylabel("Actual")
            axs[1,1].bar(range(len(np.unique(y_test))),
                         [(y_pred==c).sum() for c in np.unique(y_test)],
                         color=CC); axs[1,1].set_title("Predictions per Class")
        else:
            axs[1,0].scatter(y_test, y_pred, alpha=0.3, s=12, color=CC[0])
            lim = [min(float(y_test.min()),float(y_pred.min())),
                   max(float(y_test.max()),float(y_pred.max()))]
            axs[1,0].plot(lim, lim, "--", color=P["warn"], lw=2)
            axs[1,0].set_title("Actual vs Predicted", color=P["fg"])
            axs[1,0].set_xlabel("Actual"); axs[1,0].set_ylabel("Predicted")
            residuals = y_pred - y_test
            axs[1,1].scatter(y_pred, residuals, alpha=0.3, s=12, color=CC[3])
            axs[1,1].axhline(0, color=P["warn"], lw=1.5, linestyle="--")
            axs[1,1].set_title("Residuals", color=P["fg"])
            axs[1,1].set_xlabel("Predicted"); axs[1,1].set_ylabel("Residual")

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.dl_plot_container)
        canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        tf = tk.Frame(self.dl_plot_container, bg=P["surface"]); tf.pack(fill=tk.X)
        NavigationToolbar2Tk(canvas, tf)

    def _dl_predict_tab(self):
        if self.dl_model is None: messagebox.showwarning("No DL Model","Train a DL model first!"); return
        val = simpledialog.askstring("DL Predict",
                                     f"Enter comma-separated values for features:\n{', '.join(self.dl_features[:5])}...")
        if not val: return
        try:
            vals = [float(v.strip()) for v in val.split(",")]
            if len(vals) != len(self.dl_features):
                messagebox.showerror("Input Error", f"Expected {len(self.dl_features)} values, got {len(vals)}")
                return
            arr = self._dl_scaler.transform(np.array([vals]))
            if TF_OK and hasattr(self.dl_model, "predict"):
                raw = self.dl_model.predict(arr, verbose=0)
                if self.dl_is_clf:
                    pred = int(np.argmax(raw) if raw.shape[-1] > 1 else int(raw.flatten()[0] > 0.5))
                else:
                    pred = float(raw.flatten()[0])
            else:
                import torch as _torch
                t = _torch.tensor(arr.astype(np.float32))
                with _torch.no_grad(): raw = self.dl_model(t).numpy()
                pred = int(np.argmax(raw)) if self.dl_is_clf else float(raw.flatten()[0])
            if self.dl_le is not None:
                try: pred = self.dl_le.inverse_transform([pred])[0]
                except: pass
            messagebox.showinfo("DL Prediction", f"Predicted {self.dl_target}: {pred}")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    # ══════════════════════════════════════════════════════════════════════════
    #  REINFORCEMENT LEARNING
    # ══════════════════════════════════════════════════════════════════════════
    def _run_rl_training(self):
        mode = self.rl_mode.get()
        threading.Thread(target=self._rl_task, args=(mode,), daemon=True).start()

    def _rl_task(self, mode):
        try:
            n       = int(self.rl_grid_entry.get())
            eps_tot = int(self.rl_episodes_entry.get())
            eps_start = float(self.rl_eps_entry.get())
        except Exception:
            n = 6; eps_tot = 2000; eps_start = 1.0

        if mode == "Q-Learning Grid-World":
            self._run_qlearning_gridworld(n, eps_tot, eps_start)
        elif mode == "Tabular Cliff Walking":
            self._run_cliff_walking(eps_tot, eps_start)
        elif mode == "Multi-Armed Bandit":
            self._run_multi_armed_bandit(eps_tot)

    def _run_qlearning_gridworld(self, n, episodes, eps_start):
        """Grid-World: agent navigates n×n grid from (0,0) to (n-1,n-1). Walls random."""
        rng = np.random.default_rng(42)
        # Place walls
        walls = set()
        for _ in range(int(n*n*0.1)):
            r, c = rng.integers(0,n), rng.integers(0,n)
            if (r,c) not in [(0,0),(n-1,n-1)]:
                walls.add((r,c))

        n_states = n*n; n_actions = 4  # up,down,left,right
        Q = np.zeros((n_states, n_actions))
        alpha=0.1; gamma=0.95; epsilon=eps_start
        rewards_ep = []
        deltas      = []

        def state_idx(r,c): return r*n + c
        def step(r,c,a):
            nr,nc = r,c
            if   a==0: nr=r-1
            elif a==1: nr=r+1
            elif a==2: nc=c-1
            elif a==3: nc=c+1
            if nr<0 or nr>=n or nc<0 or nc>=n or (nr,nc) in walls:
                return r,c,-1,False
            if nr==n-1 and nc==n-1:
                return nr,nc,10,True
            return nr,nc,-0.05,False

        for ep in range(episodes):
            r,c = 0,0; total_rew = 0; done = False; steps = 0
            while not done and steps < n*n*4:
                s = state_idx(r,c)
                if np.random.rand() < epsilon: a = np.random.randint(n_actions)
                else: a = np.argmax(Q[s])
                nr,nc,rew,done = step(r,c,a)
                ns = state_idx(nr,nc)
                old_q = Q[s,a]
                Q[s,a] += alpha*(rew + gamma*np.max(Q[ns]) - Q[s,a])
                deltas.append(abs(Q[s,a] - old_q))
                r,c = nr,nc; total_rew += rew; steps += 1
            rewards_ep.append(total_rew)
            epsilon = max(0.01, eps_start * (1 - ep/episodes))

        self.rl_Q    = Q
        self.rl_grid_n = n
        self.rl_walls  = walls
        self.rl_rewards = rewards_ep

        # Smooth rewards
        window = max(1, episodes//50)
        smooth = np.convolve(rewards_ep, np.ones(window)/window, mode="valid")

        text = (f"=== 🎮 Q-LEARNING GRID-WORLD ===\n"
                f"Grid size  : {n}×{n}  ({n*n} states)\n"
                f"Episodes   : {episodes}\n"
                f"Walls      : {len(walls)}\n"
                f"Final ε    : {epsilon:.3f}\n"
                f"Last 100 avg reward: {np.mean(rewards_ep[-100:]):.2f}\n"
                f"Best episode reward: {max(rewards_ep):.2f}\n"
                f"Q-table converged  : {'Yes ✅' if np.mean(deltas[-1000:]) < 0.001 else 'Partial ⚠'}\n\n"
                f"Greedy policy arrows shown in 'Show Policy' button.")

        def _draw():
            self.rl_metrics_area.delete("1.0",tk.END)
            self.rl_metrics_area.insert(tk.END, text)
            for w in self.rl_plot_container.winfo_children(): w.destroy()
            fig, axs = plt.subplots(1, 2, figsize=(13,5)); dark_fig(fig)

            axs[0].plot(rewards_ep, color=P["fg_dim"], lw=0.5, alpha=0.4, label="Episode reward")
            axs[0].plot(np.arange(len(smooth))+(window//2), smooth, color=CC[1], lw=2, label=f"Smooth ({window}ep)")
            axs[0].set_title("Episode Rewards", color=P["fg"])
            axs[0].set_xlabel("Episode"); axs[0].set_ylabel("Total Reward")
            axs[0].legend()

            # Q-value max heatmap
            qmax = Q.max(axis=1).reshape(n,n)
            sns.heatmap(qmax, ax=axs[1], cmap="plasma", annot=n<=8,
                        fmt=".1f", linewidths=0.3, linecolor=P["border"])
            # Mark walls, start, goal
            for (wr,wc) in walls:
                axs[1].add_patch(mpatches.Rectangle((wc,wr),1,1,color=P["danger"],alpha=0.7))
            axs[1].add_patch(mpatches.Rectangle((0,0),1,1,color=P["accent2"],alpha=0.5))
            axs[1].add_patch(mpatches.Rectangle((n-1,n-1),1,1,color=P["warn"],alpha=0.8))
            axs[1].set_title("Max Q-Values (yellow=goal)", color=P["fg"])

            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self.rl_plot_container)
            canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.root.after(0, _draw)
        self.say_it(f"Q-Learning done. Avg last-100 reward: {np.mean(rewards_ep[-100:]):.2f}")

    def _run_cliff_walking(self, episodes, eps_start):
        """4×12 Cliff Walking (Sutton & Barto)."""
        rows,cols = 4,12; n_actions=4
        Q = np.zeros((rows*cols, n_actions))
        alpha=0.1; gamma=0.99; eps=eps_start
        rewards_ep = []

        def state(r,c): return r*cols+c
        def step(r,c,a):
            nr,nc = r,c
            if a==0: nr=r-1
            elif a==1: nr=r+1
            elif a==2: nc=c-1
            elif a==3: nc=c+1
            nr = np.clip(nr,0,rows-1); nc = np.clip(nc,0,cols-1)
            # Cliff
            if nr==rows-1 and 0<nc<cols-1:
                return rows-1,0,-100,False
            if nr==rows-1 and nc==cols-1:
                return nr,nc,0,True
            return nr,nc,-1,False

        for ep in range(episodes):
            r,c=rows-1,0; tot=0; done=False; steps=0
            while not done and steps<200:
                s=state(r,c)
                a=np.random.randint(n_actions) if np.random.rand()<eps else np.argmax(Q[s])
                nr,nc,rew,done=step(r,c,a)
                ns=state(nr,nc)
                Q[s,a]+=alpha*(rew+gamma*np.max(Q[ns])-Q[s,a])
                r,c=nr,nc; tot+=rew; steps+=1
            rewards_ep.append(tot)
            eps=max(0.01,eps_start*(1-ep/episodes))

        self.rl_rewards = rewards_ep
        smooth=np.convolve(rewards_ep,np.ones(50)/50,mode="valid")
        def _draw():
            self.rl_metrics_area.delete("1.0",tk.END)
            self.rl_metrics_area.insert(tk.END,
                f"=== CLIFF WALKING (4×12) ===\nEpisodes: {episodes}\n"
                f"Best: {max(rewards_ep):.0f}  Last-100 avg: {np.mean(rewards_ep[-100:]):.1f}\n")
            for w in self.rl_plot_container.winfo_children(): w.destroy()
            fig,ax=plt.subplots(figsize=(12,5)); dark_fig(fig)
            ax.plot(rewards_ep, color=P["fg_dim"],lw=0.5,alpha=0.3)
            ax.plot(np.arange(len(smooth))+25,smooth,color=CC[0],lw=2,label="Smoothed reward")
            ax.set_title("Cliff Walking — Episode Rewards",color=P["fg"])
            ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward"); ax.legend()
            plt.tight_layout()
            canvas=FigureCanvasTkAgg(fig,master=self.rl_plot_container)
            canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)
        self.root.after(0, _draw)

    def _run_multi_armed_bandit(self, episodes):
        """10-arm bandit, epsilon-greedy vs UCB comparison."""
        n_arms = 10; true_means = np.random.randn(n_arms)
        results = {}
        for method, eps in [("ε-greedy ε=0.1",0.1),("ε-greedy ε=0.01",0.01)]:
            Q_est=np.zeros(n_arms); N=np.zeros(n_arms); rews=[]
            for t in range(episodes):
                a = np.random.randint(n_arms) if np.random.rand()<eps else np.argmax(Q_est)
                r = true_means[a]+np.random.randn()
                N[a]+=1; Q_est[a]+=(r-Q_est[a])/N[a]; rews.append(r)
            results[method] = np.cumsum(rews)/(np.arange(episodes)+1)
        # UCB
        Q_est=np.zeros(n_arms); N=np.ones(n_arms)*0.001; rews=[]
        for t in range(episodes):
            ucb = Q_est + np.sqrt(2*np.log(t+1)/(N))
            a = np.argmax(ucb); r = true_means[a]+np.random.randn()
            N[a]+=1; Q_est[a]+=(r-Q_est[a])/N[a]; rews.append(r)
        results["UCB c=2"] = np.cumsum(rews)/(np.arange(episodes)+1)

        def _draw():
            self.rl_metrics_area.delete("1.0",tk.END)
            self.rl_metrics_area.insert(tk.END,
                f"=== MULTI-ARMED BANDIT ===\nArms: {n_arms}  Pulls: {episodes}\n"
                f"Optimal arm: #{np.argmax(true_means)} (mean={max(true_means):.2f})\n\n"
                f"Average final reward:\n"
                + "\n".join([f"  {k}: {v[-1]:.4f}" for k,v in results.items()]))
            for w in self.rl_plot_container.winfo_children(): w.destroy()
            fig,ax=plt.subplots(figsize=(12,5)); dark_fig(fig)
            for i,(method,cumavg) in enumerate(results.items()):
                ax.plot(cumavg,color=CC[i],lw=2,label=method)
            ax.set_title("Multi-Armed Bandit — Average Reward",color=P["fg"])
            ax.set_xlabel("Episode"); ax.set_ylabel("Avg Reward"); ax.legend()
            plt.tight_layout()
            canvas=FigureCanvasTkAgg(fig,master=self.rl_plot_container)
            canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)
        self.root.after(0, _draw)

    def _show_rl_policy(self):
        if not hasattr(self,"rl_Q") or self.rl_Q is None:
            messagebox.showwarning("No RL Model","Run Q-Learning first!"); return
        n = self.rl_grid_n; Q = self.rl_Q
        arrows = ["↑","↓","←","→"]
        policy = Q.argmax(axis=1).reshape(n,n)
        fig,ax = plt.subplots(figsize=(max(6,n),max(6,n))); dark_fig(fig)
        ax.set_xlim(0,n); ax.set_ylim(0,n); ax.set_aspect("equal")
        for r in range(n):
            for c in range(n):
                if (r,c) in self.rl_walls:
                    ax.add_patch(mpatches.Rectangle((c,n-1-r),1,1,color=P["danger"],alpha=0.8))
                elif r==n-1 and c==n-1:
                    ax.add_patch(mpatches.Rectangle((c,n-1-r),1,1,color=P["warn"],alpha=0.8))
                    ax.text(c+0.5,n-1-r+0.5,"🏆",ha="center",va="center",fontsize=12)
                elif r==0 and c==0:
                    ax.add_patch(mpatches.Rectangle((c,n-1-r),1,1,color=P["accent2"],alpha=0.5))
                    ax.text(c+0.5,n-1-r+0.5,"S",ha="center",va="center",color="#000",fontsize=10,fontweight="bold")
                else:
                    ax.text(c+0.5,n-1-r+0.5,arrows[policy[r,c]],
                            ha="center",va="center",color=P["fg"],fontsize=14)
        for i in range(n+1):
            ax.axhline(i,color=P["border"],lw=0.5)
            ax.axvline(i,color=P["border"],lw=0.5)
        ax.set_title("Greedy Policy  (🏆=Goal  S=Start  🟥=Wall)",color=P["fg"],pad=12)
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        win=tk.Toplevel(self.root); win.title("RL Policy Map"); win.configure(bg=P["surface"])
        canvas=FigureCanvasTkAgg(fig,master=win); canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  PREDICTOR
    # ══════════════════════════════════════════════════════════════════════════
    def _refresh_predictor_ui(self):
        for w in self.predictor_input_container.winfo_children(): w.destroy()
        if not self.current_model:
            self.predictor_status.config(text="No model trained yet.", fg=P["danger"]); return
        self.predictor_status.config(
            text=f"Model: {type(self.current_model).__name__}  |  Target: {self.current_model_target}",
            fg=P["accent2"])
        self.predictor_entries = {}
        gf = tk.Frame(self.predictor_input_container, bg=P["bg"]); gf.pack(pady=8)
        for i, feat in enumerate(self.current_model_features):
            r, c = divmod(i, 3)
            tk.Label(gf, text=f"{feat}:", bg=P["bg"], fg=P["fg"],
                     font=("Segoe UI",9)).grid(row=r, column=c*2, padx=8, pady=4, sticky="e")
            if feat in self.current_model_encoders:
                opts = list(self.current_model_encoders[feat].classes_)
                ent  = ttk.Combobox(gf, values=opts, width=14, state="readonly")
                if opts: ent.set(opts[0])
            else:
                ent = tk.Entry(gf, width=16, bg=P["surface2"], fg=P["fg"], insertbackground=P["fg"])
            ent.grid(row=r, column=c*2+1, padx=8, pady=4, sticky="w")
            self.predictor_entries[feat] = ent

    def _run_live_prediction(self):
        if not self.current_model: messagebox.showwarning("No Model","Train a model first!"); return
        try:
            vals = []
            for feat in self.current_model_features:
                raw = self.predictor_entries[feat].get()
                if raw == "": messagebox.showerror("Input",f"Enter value for '{feat}'"); return
                if feat in self.current_model_encoders:
                    try: val = int(self.current_model_encoders[feat].transform([raw])[0])
                    except: val = 0
                else:
                    val = float(raw)
                vals.append(val)
            inp  = pd.DataFrame([vals], columns=self.current_model_features)
            if getattr(self, "_ml_scaler", None) is not None:
                inp_scaled = self._ml_scaler.transform(inp)
            else:
                inp_scaled = inp
            pred = self.current_model.predict(inp_scaled)[0]
            self.last_prediction_input = inp
            if self._is_clf and self._target_encoder is not None:
                try: pred = self._target_encoder.inverse_transform([int(pred)])[0]
                except: pass
            if isinstance(pred, float):
                res = f"PREDICTED {self.current_model_target.upper()}: {pred:.4f}"
            else:
                res = f"PREDICTED {self.current_model_target.upper()}: {pred}"
            self.prediction_result_label.config(text=res, fg=P["accent2"])
            self.say_it(f"Prediction: {pred}")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def _explain_prediction(self):
        if self.last_prediction_input is None or self.current_model is None:
            messagebox.showwarning("No Prediction","Make a prediction first!"); return
        
        # Check if SHAP is available
        if SHAP_OK and hasattr(self.current_model, "feature_importances_"):
            try:
                win = tk.Toplevel(self.root); win.title("💡 SHAP Explanation"); win.configure(bg=P["surface"])
                fig = plt.figure(figsize=(10, 6)); dark_fig(fig)
                
                explainer = shap.TreeExplainer(self.current_model)
                shap_values = explainer.shap_values(self.last_prediction_input)
                
                # If binary classification, shap_values is a list. Take the positive class.
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                                     base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value, 
                                                     data=self.last_prediction_input.iloc[0],  
                                                     feature_names=self.current_model_features), show=False)
                
                plt.tight_layout()
                canvas = FigureCanvasTkAgg(fig, master=win); canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                return
            except Exception as e:
                self.say_it(f"SHAP Error: {e}. Falling back to basic XAI.")

        # Fallback basic explanation
        if not hasattr(self.current_model,"feature_importances_"):
            messagebox.showinfo("XAI","Feature importances not available for this model type."); return
        fi  = self.current_model.feature_importances_
        ser = pd.Series(fi, index=self.current_model_features).sort_values(ascending=False)
        text = "=== 💡 XAI — Why this prediction? ===\n\n"
        for i,(feat,imp) in enumerate(ser.head(8).items(),1):
            raw = self.last_prediction_input[feat].values[0]
            if feat in self.current_model_encoders:
                try: raw = self.current_model_encoders[feat].inverse_transform([int(raw)])[0]
                except: pass
            text += f"{i}. {feat.upper()}  =  {raw}\n   Importance: {imp:.2%}  {'█'*int(imp*40)}\n\n"
        win = tk.Toplevel(self.root); win.title("Prediction Explanation"); win.configure(bg=P["surface"])
        t = scrolledtext.ScrolledText(win, font=("Segoe UI",10), bg=P["surface"], fg=P["fg"],
                                       width=52, height=20, bd=0)
        t.pack(padx=12, pady=12); t.insert(tk.END, text); t.config(state=tk.DISABLED)

    def _plot_partial_dependence(self):
        if self.current_model is None or self.current_df is None: return
        if not hasattr(self.current_model,"feature_importances_"):
            messagebox.showinfo("PDP","Partial dependence requires a tree-based model."); return
        try:
            from sklearn.inspection import PartialDependenceDisplay
            X, y, _, _ = self._prepare_features(self.current_df.copy(), self.current_model_target)
            top_feat = pd.Series(self.current_model.feature_importances_,
                                  index=X.columns).sort_values(ascending=False).head(3).index.tolist()
            win = tk.Toplevel(self.root); win.title("Partial Dependence Plots")
            win.configure(bg=P["surface"])
            fig,ax = plt.subplots(1,len(top_feat),figsize=(14,5)); dark_fig(fig)
            if len(top_feat)==1: ax=[ax]
            for i,feat in enumerate(top_feat):
                feat_idx = list(X.columns).index(feat)
                vals = np.linspace(X[feat].min(), X[feat].max(), 50)
                X_mod = X.copy(); preds = []
                for v in vals:
                    X_mod[feat] = v
                    p = self.current_model.predict(X_mod)
                    preds.append(p.mean())
                ax[i].plot(vals, preds, color=CC[i], lw=2)
                ax[i].fill_between(vals, preds, alpha=0.15, color=CC[i])
                ax[i].set_title(f"PDP: {feat}",color=P["fg"])
                ax[i].set_xlabel(feat); ax[i].set_ylabel("Avg prediction")
            plt.tight_layout()
            canvas=FigureCanvasTkAgg(fig,master=win); canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("PDP Error", str(e))

