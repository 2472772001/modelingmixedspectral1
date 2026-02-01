# -*- coding: utf-8 -*-
"""
Experiments for:
(1) Factor effect analysis
(2) Spectral indicator regression & classification
Using processed datasets:
- cleaned_dataset_peaksorted_ratios.xlsx  (360 rows, per-measurement)
- conditions_mean_120rows.xlsx            (120 rows, mean of 3 repeats per condition)

Outputs:
- paper_outputs/fig_*.png, fig_*.pdf
- paper_outputs/table_*.csv
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)

# -----------------------------
# 0) Paths & basic settings
# -----------------------------
DATA_DIR = "."  # <-- if your script is not in same folder, change this
PATH_360 = os.path.join(DATA_DIR, "cleaned_dataset_peaksorted_ratios.xlsx")
PATH_120 = os.path.join(DATA_DIR, "conditions_mean_120rows.xlsx")

OUT_DIR = os.path.join(DATA_DIR, "paper_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# -----------------------------
# 1) Utilities
# -----------------------------
def save_fig(fig, name: str):
    """Save figure to both PNG and PDF for paper usage."""
    png_path = os.path.join(OUT_DIR, f"{name}.png")
    pdf_path = os.path.join(OUT_DIR, f"{name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def safe_div(a, b):
    return np.where(b == 0, np.nan, a / b)

def weighted_std(values: np.ndarray, weights: np.ndarray):
    """
    Weighted standard deviation.
    values: shape (n,)
    weights: shape (n,)
    """
    w_sum = np.sum(weights)
    if w_sum <= 0:
        return np.nan
    mean = np.sum(weights * values) / w_sum
    var = np.sum(weights * (values - mean) ** 2) / w_sum
    return math.sqrt(var)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create paper-friendly spectral indicators from 3-peak parameters.
    - I_total: total intensity
    - lambda_c: intensity-weighted centroid wavelength
    - lambda_spread: intensity-weighted wavelength spread (std)
    - rsd_mean: mean of peak widths (a simple stability proxy)
    """
    out = df.copy()
    A1, A2, A3 = out["gs1amp"].to_numpy(), out["gs2amp"].to_numpy(), out["gs3amp"].to_numpy()
    mu1, mu2, mu3 = out["gs1avg"].to_numpy(), out["gs2avg"].to_numpy(), out["gs3avg"].to_numpy()

    I = A1 + A2 + A3
    out["I_total"] = I

    # weighted centroid
    lam_c = safe_div(A1 * mu1 + A2 * mu2 + A3 * mu3, I)
    out["lambda_c"] = lam_c

    # weighted spread (std)
    spread = []
    for i in range(len(out)):
        weights = np.array([A1[i], A2[i], A3[i]], dtype=float)
        values = np.array([mu1[i], mu2[i], mu3[i]], dtype=float)
        spread.append(weighted_std(values, weights))
    out["lambda_spread"] = np.array(spread)

    out["rsd_mean"] = (out["gs1rsd"] + out["gs2rsd"] + out["gs3rsd"]) / 3.0
    return out

def make_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Two feature sets for comparison:
    - absolute: fp1, fp2, fp3, fjr, v
    - ratio: p1, p2, fp_total, fjr, v
      NOTE: drop p3 to avoid perfect collinearity (p1+p2+p3=1).
    """
    feats = {}
    feats["absolute"] = ["fp1", "fp2", "fp3", "fjr", "v"]
    feats["ratio"] = ["p1", "p2", "fp_total", "fjr", "v"]
    return feats

def add_interactions(df: pd.DataFrame, base_cols: List[str], interaction_pairs: List[Tuple[str, str]]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add interaction terms colA*colB. Returns (new_df, new_feature_cols).
    """
    out = df.copy()
    new_cols = base_cols.copy()
    for a, b in interaction_pairs:
        name = f"{a}_x_{b}"
        out[name] = out[a] * out[b]
        new_cols.append(name)
    return out, new_cols

# -----------------------------
# 2) Load data
# -----------------------------
df360 = pd.read_excel(PATH_360)
df120 = pd.read_excel(PATH_120)

# Sanity: these should exist
required_cols = ["fp1","fp2","fp3","fjr","v","fp_total","p1","p2","p3",
                 "gs1amp","gs1avg","gs1rsd","gs2amp","gs2avg","gs2rsd","gs3amp","gs3avg","gs3rsd"]
missing = [c for c in required_cols if c not in df360.columns]
if missing:
    raise ValueError(f"Missing columns in df360: {missing}")

# add indicators
df360 = add_indicators(df360)
df120 = add_indicators(df120)

# build groups for leakage-free splits
# df360 has group_key; df120 has group_key too (from earlier preparation)
if "group_key" not in df360.columns:
    # fallback: create it
    df360["group_key"] = df360[["fp1","fp2","fp3","fjr","v"]].astype(str).agg("|".join, axis=1)
if "group_key" not in df120.columns:
    df120["group_key"] = df120[["fp1","fp2","fp3","fjr","v"]].astype(str).agg("|".join, axis=1)

# -----------------------------
# 3) Part A: Factor effect analysis (Direction 6)
# -----------------------------
print("\n[Part A] Factor effect analysis...")

# 3.1 Trend plot: v vs gs1amp/gs2amp/gs3amp with error bars (mean ± std across conditions)
def plot_v_vs_amp(df_mean: pd.DataFrame):
    v_levels = sorted(df_mean["v"].dropna().unique())
    amp_cols = ["gs1amp", "gs2amp", "gs3amp"]

    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)

    for col in amp_cols:
        means = []
        stds = []
        for v in v_levels:
            subset = df_mean[df_mean["v"] == v][col].dropna()
            means.append(subset.mean())
            stds.append(subset.std(ddof=1))
        ax.errorbar(v_levels, means, yerr=stds, marker="o", linewidth=1.5, capsize=3, label=col)

    ax.set_xlabel("Current v (mA)")
    ax.set_ylabel("Peak amplitude (amp)")
    ax.set_title("Effect of current on peak amplitudes (condition-mean, with std)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

fig = plot_v_vs_amp(df120)
save_fig(fig, "fig_A1_v_vs_peak_amp")

# 3.2 Scatter: fjr vs gs1rsd/gs2rsd/gs3rsd
def plot_fjr_vs_rsd(df_mean: pd.DataFrame):
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    for col in ["gs1rsd","gs2rsd","gs3rsd"]:
        ax.scatter(df_mean["fjr"], df_mean[col], s=18, alpha=0.7, label=col)
    ax.set_xlabel("Powder-to-binder ratio fjr")
    ax.set_ylabel("Peak width / dispersion (rsd)")
    ax.set_title("Effect of fjr on peak rsd (condition-mean)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

fig = plot_fjr_vs_rsd(df120)
save_fig(fig, "fig_A2_fjr_vs_peak_rsd")

# 3.3 Scatter: p1/p2/p3 vs lambda_c (main color position)
def plot_ratio_vs_lambda_c(df_mean: pd.DataFrame):
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    ax.scatter(df_mean["p1"], df_mean["lambda_c"], s=18, alpha=0.7, label="p1")
    ax.scatter(df_mean["p2"], df_mean["lambda_c"], s=18, alpha=0.7, label="p2")
    ax.scatter(df_mean["p3"], df_mean["lambda_c"], s=18, alpha=0.7, label="p3")
    ax.set_xlabel("Powder proportion (p1/p2/p3)")
    ax.set_ylabel("lambda_c (nm)")
    ax.set_title("Powder proportions vs color centroid (lambda_c)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

fig = plot_ratio_vs_lambda_c(df120)
save_fig(fig, "fig_A3_ratios_vs_lambda_c")

# 3.4 Quantify effect sizes with standardized linear regression (with / without interactions)
def standardized_linear_effects(df_mean: pd.DataFrame, y_col: str, feature_cols: List[str]) -> pd.DataFrame:
    """
    Fit y ~ X with standardized X and y, return standardized coefficients.
    """
    data = df_mean.dropna(subset=[y_col] + feature_cols).copy()
    X = data[feature_cols].to_numpy()
    y = data[y_col].to_numpy().reshape(-1, 1)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(X)
    ys = y_scaler.fit_transform(y).ravel()

    lr = LinearRegression()
    lr.fit(Xs, ys)

    coef = lr.coef_.ravel()
    out = pd.DataFrame({
        "target": y_col,
        "feature": feature_cols,
        "std_coef": coef,
        "abs_std_coef": np.abs(coef)
    }).sort_values("abs_std_coef", ascending=False)
    return out

# Compare: absolute vs ratio; with interactions (v*powder terms)
feature_sets = make_feature_sets(df120)

# For ratio set, add interactions v*p1 and v*p2 (p3 is redundant)
df120_ratio_int, ratio_cols_int = add_interactions(
    df120, feature_sets["ratio"], interaction_pairs=[("v","p1"), ("v","p2")]
)
df120_abs_int, abs_cols_int = add_interactions(
    df120, feature_sets["absolute"], interaction_pairs=[("v","fp1"), ("v","fp2"), ("v","fp3")]
)

targets_effect = ["I_total", "lambda_c", "lambda_spread", "rsd_mean",
                  "gs1amp","gs2amp","gs3amp","gs1avg","gs2avg","gs3avg","gs1rsd","gs2rsd","gs3rsd"]

tables = []
for y in targets_effect:
    # main effects
    t_abs = standardized_linear_effects(df120, y, feature_sets["absolute"])
    t_abs["feature_set"] = "absolute_main"
    tables.append(t_abs)

    t_ratio = standardized_linear_effects(df120, y, feature_sets["ratio"])
    t_ratio["feature_set"] = "ratio_main"
    tables.append(t_ratio)

    # with interactions
    t_abs_i = standardized_linear_effects(df120_abs_int, y, abs_cols_int)
    t_abs_i["feature_set"] = "absolute_with_interactions"
    tables.append(t_abs_i)

    t_ratio_i = standardized_linear_effects(df120_ratio_int, y, ratio_cols_int)
    t_ratio_i["feature_set"] = "ratio_with_interactions"
    tables.append(t_ratio_i)

effect_table = pd.concat(tables, ignore_index=True)
effect_path = os.path.join(OUT_DIR, "table_A_effect_sizes_stdcoef.csv")
effect_table.to_csv(effect_path, index=False, encoding="utf-8-sig")
print(f"Saved effect size table: {effect_path}")

# -----------------------------
# 4) Part B: Indicator regression & classification (Direction 5)
# -----------------------------
print("\n[Part B] Indicator regression & classification...")

# 4.1 Build regression evaluation with GroupKFold (no leakage)
@dataclass
class RegResult:
    feature_set: str
    model_name: str
    target: str
    cv_type: str
    mae: float
    rmse: float
    r2: float

def eval_regression_groupkfold(df: pd.DataFrame, feature_cols: List[str], target_col: str, groups: np.ndarray, model, n_splits=5) -> RegResult:
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        y_true_all.append(yte)
        y_pred_all.append(pred)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    return y_true_all, y_pred_all

def summarize_reg(y_true, y_pred) -> Tuple[float, float, float]:
    return (
        mean_absolute_error(y_true, y_pred),
        rmse(y_true, y_pred),
        r2_score(y_true, y_pred),
    )

# 4.2 Leave-one-current-out for regression (cross-condition generalization across v)
def eval_regression_leave_one_current(df: pd.DataFrame, feature_cols: List[str], target_col: str, model) -> Tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    v_levels = sorted(df["v"].dropna().unique())

    y_true_all, y_pred_all = [], []
    for v0 in v_levels:
        train_mask = df["v"] != v0
        test_mask = df["v"] == v0
        Xtr, ytr = X[train_mask], y[train_mask]
        Xte, yte = X[test_mask], y[test_mask]
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        y_true_all.append(yte)
        y_pred_all.append(pred)

    return np.concatenate(y_true_all), np.concatenate(y_pred_all)

# Models for regression (compare linear vs tree)
models_reg = {
    "Ridge(standardized)": Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE))
    ]),
    "RandomForest": RandomForestRegressor(
        n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1,
        max_depth=None, min_samples_leaf=2
    ),
}

feature_sets_360 = make_feature_sets(df360)

reg_targets = ["I_total", "lambda_c", "lambda_spread", "rsd_mean"]
groups_360 = df360["group_key"].to_numpy()

reg_results = []

# For fair comparison, drop NaNs
df360_reg = df360.dropna(subset=reg_targets + feature_sets_360["absolute"] + feature_sets_360["ratio"]).copy()

for feat_name, feat_cols in feature_sets_360.items():
    for model_name, model in models_reg.items():
        for target in reg_targets:
            # GroupKFold CV
            y_true, y_pred = eval_regression_groupkfold(
                df360_reg, feat_cols, target, groups=df360_reg["group_key"].to_numpy(), model=model, n_splits=5
            )
            mae_v, rmse_v, r2_v = summarize_reg(y_true, y_pred)
            reg_results.append(RegResult(feat_name, model_name, target, "GroupKFold(5)", mae_v, rmse_v, r2_v))

            # Leave-one-current-out
            y_true2, y_pred2 = eval_regression_leave_one_current(df360_reg, feat_cols, target, model=model)
            mae2, rmse2, r2_2 = summarize_reg(y_true2, y_pred2)
            reg_results.append(RegResult(feat_name, model_name, target, "LeaveOneCurrentOut(8)", mae2, rmse2, r2_2))

reg_table = pd.DataFrame([r.__dict__ for r in reg_results])
reg_path = os.path.join(OUT_DIR, "table_B_regression_results.csv")
reg_table.to_csv(reg_path, index=False, encoding="utf-8-sig")
print(f"Saved regression results: {reg_path}")

# 4.3 Save prediction scatter plots for best configurations (optional: pick best by RMSE)
def plot_pred_scatter(y_true, y_pred, title, xlabel="True", ylabel="Predicted"):
    fig = plt.figure(figsize=(5.5, 5.5))
    ax = fig.add_subplot(111)
    ax.scatter(y_true, y_pred, s=18, alpha=0.7)
    # 45-degree line
    lo = min(np.nanmin(y_true), np.nanmin(y_pred))
    hi = max(np.nanmax(y_true), np.nanmax(y_pred))
    ax.plot([lo, hi], [lo, hi], linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig

# Pick one "paper-ready" best config per target based on LeaveOneCurrentOut RMSE
best_rows = []
for t in reg_targets:
    tmp = reg_table[(reg_table["target"] == t) & (reg_table["cv_type"] == "LeaveOneCurrentOut(8)")]
    best = tmp.sort_values("rmse").iloc[0]
    best_rows.append(best)
best_df = pd.DataFrame(best_rows)
best_df.to_csv(os.path.join(OUT_DIR, "table_B_best_configs_for_scatter.csv"), index=False, encoding="utf-8-sig")

for _, row in best_df.iterrows():
    feat_name = row["feature_set"]
    model_name = row["model_name"]
    target = row["target"]

    model = models_reg[model_name]
    feat_cols = feature_sets_360[feat_name]

    y_true2, y_pred2 = eval_regression_leave_one_current(df360_reg, feat_cols, target, model=model)
    fig = plot_pred_scatter(
        y_true2, y_pred2,
        title=f"Regression ({target}) - {model_name} + {feat_name} (Leave-one-current-out)"
    )
    save_fig(fig, f"pred_scatter_{target}_{model_name.replace('(','').replace(')','').replace('/','_')}_{feat_name}")

# -----------------------------
# 5) Classification (3 classes by lambda_c quantiles)
# -----------------------------
print("\n[Part B2] Classification experiments...")

# Build labels based on CONDITION-MEAN (df120), then merge label back to df360 by group_key
# This avoids label instability from measurement noise.
df120_lab = df120.copy()
q1, q2 = df120_lab["lambda_c"].quantile([1/3, 2/3]).to_list()

def label_lambda_c(x):
    if x <= q1:
        return 0  # "blue-ish"
    elif x <= q2:
        return 1  # "mid"
    else:
        return 2  # "red-ish"

df120_lab["class_color"] = df120_lab["lambda_c"].apply(label_lambda_c)

label_map = df120_lab[["group_key", "class_color"]].drop_duplicates()
df360_cls = df360.merge(label_map, on="group_key", how="left")

# models for classification
models_cls = {
    "LogReg(standardized)": Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            multi_class="multinomial", max_iter=2000, random_state=RANDOM_STATE
        ))
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=600, random_state=RANDOM_STATE, n_jobs=-1,
        max_depth=None, min_samples_leaf=2
    ),
}

@dataclass
class ClsResult:
    feature_set: str
    model_name: str
    cv_type: str
    acc: float
    macro_f1: float

def eval_classification_groupkfold(df: pd.DataFrame, feature_cols: List[str], label_col: str, groups: np.ndarray, model, n_splits=5):
    data = df.dropna(subset=feature_cols + [label_col]).copy()
    X = data[feature_cols].to_numpy()
    y = data[label_col].to_numpy().astype(int)
    g = data["group_key"].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    y_true_all, y_pred_all = [], []
    for tr, te in gkf.split(X, y, groups=g):
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        y_true_all.append(y[te])
        y_pred_all.append(pred)
    return np.concatenate(y_true_all), np.concatenate(y_pred_all)

def eval_classification_leave_one_current(df: pd.DataFrame, feature_cols: List[str], label_col: str, model):
    data = df.dropna(subset=feature_cols + [label_col]).copy()
    X = data[feature_cols].to_numpy()
    y = data[label_col].to_numpy().astype(int)

    y_true_all, y_pred_all = [], []
    v_levels = sorted(data["v"].dropna().unique())
    for v0 in v_levels:
        tr_mask = data["v"] != v0
        te_mask = data["v"] == v0
        Xtr, ytr = X[tr_mask], y[tr_mask]
        Xte, yte = X[te_mask], y[te_mask]
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        y_true_all.append(yte)
        y_pred_all.append(pred)
    return np.concatenate(y_true_all), np.concatenate(y_pred_all)

def summarize_cls(y_true, y_pred) -> Tuple[float, float]:
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")

def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    fig = plt.figure(figsize=(5.2, 4.6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
    ax.set_xticklabels(["blue-ish","mid","red-ish"])
    ax.set_yticklabels(["blue-ish","mid","red-ish"])

    # annotate
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig

feature_sets_cls = make_feature_sets(df360_cls)
cls_results = []

df360_cls2 = df360_cls.dropna(subset=feature_sets_cls["absolute"] + feature_sets_cls["ratio"] + ["class_color"]).copy()

for feat_name, feat_cols in feature_sets_cls.items():
    for model_name, model in models_cls.items():
        # GroupKFold
        yt, yp = eval_classification_groupkfold(df360_cls2, feat_cols, "class_color",
                                               groups=df360_cls2["group_key"].to_numpy(),
                                               model=model, n_splits=5)
        acc, mf1 = summarize_cls(yt, yp)
        cls_results.append(ClsResult(feat_name, model_name, "GroupKFold(5)", acc, mf1))

        # Leave-one-current-out
        yt2, yp2 = eval_classification_leave_one_current(df360_cls2, feat_cols, "class_color", model=model)
        acc2, mf12 = summarize_cls(yt2, yp2)
        cls_results.append(ClsResult(feat_name, model_name, "LeaveOneCurrentOut(8)", acc2, mf12))

cls_table = pd.DataFrame([r.__dict__ for r in cls_results])
cls_path = os.path.join(OUT_DIR, "table_C_classification_results.csv")
cls_table.to_csv(cls_path, index=False, encoding="utf-8-sig")
print(f"Saved classification results: {cls_path}")

# Save confusion matrix for the best LeaveOneCurrentOut config (macro_f1 highest)
best_cls = cls_table[cls_table["cv_type"] == "LeaveOneCurrentOut(8)"].sort_values("macro_f1", ascending=False).iloc[0]
best_feat = best_cls["feature_set"]
best_model_name = best_cls["model_name"]
best_model = models_cls[best_model_name]
best_cols = feature_sets_cls[best_feat]

yt2, yp2 = eval_classification_leave_one_current(df360_cls2, best_cols, "class_color", model=best_model)
fig = plot_confusion(yt2, yp2, title=f"Best classifier (Leave-one-current-out): {best_model_name}+{best_feat}")
save_fig(fig, f"confusion_best_{best_model_name.replace('(','').replace(')','').replace('/','_')}_{best_feat}")

# -----------------------------
# 6) Extra: export summary tables for writing results
# -----------------------------
# Summaries (pivot tables) that look nice in paper
reg_pivot = reg_table.pivot_table(index=["target"], columns=["cv_type","feature_set","model_name"], values=["rmse","r2","mae"])
reg_pivot.to_csv(os.path.join(OUT_DIR, "table_B_regression_pivot.csv"), encoding="utf-8-sig")

cls_pivot = cls_table.pivot_table(index=[], columns=["cv_type","feature_set","model_name"], values=["acc","macro_f1"])
cls_pivot.to_csv(os.path.join(OUT_DIR, "table_C_classification_pivot.csv"), encoding="utf-8-sig")

print("\nAll done. Outputs saved to:", OUT_DIR)
print("Key tables:")
print("- table_A_effect_sizes_stdcoef.csv")
print("- table_B_regression_results.csv (+ best config scatter plots)")
print("- table_C_classification_results.csv (+ best confusion matrix)")
print("Key figures:")
print("- fig_A1_v_vs_peak_amp, fig_A2_fjr_vs_peak_rsd, fig_A3_ratios_vs_lambda_c")
