# -*- coding: utf-8 -*-
"""
Reproducible analysis code for a two-center breast cancer LVI prediction study.
Center 1 is the development cohort; Center 2 is the locked external validation cohort.

Installation:
    pip install -r requirements.txt

Input data are not distributed with this repository. Set the LVI_DATA_DIR environment
variable to the local directory containing Data1.xlsx, Data2.xlsx, and agreement data.xlsx.
If LVI_DATA_DIR is not set, the script searches beside this file.

Run:
    python LVI_prediction_analysis.py

Before running, verify that the workbook column names and coding match the study data.
See README.md for input requirements and reproducibility notes.

Important:
- Center 1 only: nested CV, hyperparameter selection, final model fitting and threshold
  specification. Center 2 is used only for locked external validation.
- M3 (M2 + BPS) is the prespecified primary model. FBE and AVS are not entered as a separate model because the five observed FBE–AVS combinations map one-to-one to the five BPS categories.
- Current CV is row-level stratified CV. Confirm one row per patient; if multiple lesions per patient, use a verified patient ID and group-aware CV before publication. The N column is not assumed to be a patient ID.
- Thresholds are NOT automatically optimized from the data. They must be justified by
  clinical consequences/costs and locked before external validation.
- DFS analysis uses confirmed DFS_time (days) and DFS_status (1=recurrence/death, 0=censored); blanks are excluded.
- Reader workflow: two junior readers independently assess; senior reader adjudicates discordance only.
- Interreader agreement uses the two original junior ratings; intrareader requires a blinded repeat read.
"""

from pathlib import Path
import os
import json
import warnings
import gc
import matplotlib
# Use a non-interactive backend: avoids notebook display buffers during high-resolution export.
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tifffile

from scipy.special import logit
from scipy.stats import chi2_contingency, fisher_exact, ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, roc_curve,
    confusion_matrix
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# =============================================================================
# 1. USER SETTINGS - CHECK THESE BEFORE RUNNING
# =============================================================================
# Set LVI_DATA_DIR to the directory containing the private input workbooks.
# If unset, the script looks for input files beside this .py file.
BASE_DIR = Path(os.environ.get("LVI_DATA_DIR", Path(__file__).resolve().parent)).expanduser().resolve()
CENTER1_FILE = "Data1.xlsx"
CENTER2_FILE = "Data2.xlsx"
OUT_DIR = BASE_DIR / "LVI_analysis_results"

TARGET = "Group"
POSITIVE_LABELS = {"1", "1.0", "positive", "pos", "yes", "是", "有"}
NEGATIVE_LABELS = {"0", "0.0", "negative", "neg", "no", "否", "无"}

CLINICAL_CONT = ["Age"]
CLINICAL_CAT = ["Marital Status", "family history", "ER", "PR", "HER2", "ki67"]
MRI_CONT = ["Diameter", "ADC ratio"]
# Categorical conventional MRI features; the workbook column must be named exactly
# The source workbook column must be named exactly "Internal enhancement".
MRI_CAT = ["Shape", "Lobulated", "Angular", "Spiculated", "Internal enhancement", "TIC"]
FBE = ["FBE"]
AVS = ["AVS"]
BPS = ["BPS"]

MODEL_FEATURES = {
    "M1_Clinical": CLINICAL_CONT + CLINICAL_CAT,
    "M2_Clinical_plus_conventional_MRI":
        CLINICAL_CONT + CLINICAL_CAT + MRI_CONT + MRI_CAT,
    "M3_Primary_combined_plus_BPS":
        CLINICAL_CONT + CLINICAL_CAT + MRI_CONT + MRI_CAT + BPS,
}
PRIMARY_MODEL = "M3_Primary_combined_plus_BPS"

OUTER_FOLDS = 5
INNER_FOLDS = 4
RANDOM_SEED = 2026
N_BOOT = 2000
N_JOBS = 1  # Memory-safe: avoid spawning all CPU workers during nested CV.

# Strict provenance guard: the corrected analysis dataset discussed in this project
# contains 634/141 LVI-positive cases in Center 1 and 156/34 in Center 2.
# The script stops if older, differently coded source files are selected.
ENFORCE_EXPECTED_COHORT_COUNTS = True
EXPECTED_COHORT_COUNTS = {"Center 1": (634, 141), "Center 2": (156, 34)}

# Clinical decision threshold: leave None until clinically justified.
# Do NOT select it by maximizing Youden's index or by looking at Center 2.
# Once the clinical team has prespecified it, enter e.g. 0.20 (illustration only).
FIXED_THRESHOLD = None

# DCA display settings requested: retain x-axis, zoom y-axis.
DCA_XLIM = (0.05, 0.80)
DCA_YLIM = (-0.05, 0.20)
FIG_DPI = 600
FIG_FORMAT = "tiff"
SAVE_INDIVIDUAL_DFS_KM = False  # Keep only requested composite Figure 5/6 by default.

# Publication-wide typography: readable at journal column width.
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 11,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8, "figure.titlesize": 13,
    "axes.linewidth": 0.8, "savefig.dpi": FIG_DPI,
})

# DFS settings confirmed for this study. DFS_time is recorded in days;
# DFS_status: 1=recurrence/death, 0=censored; blank means no follow-up record.
DFS_TIME_COL = "DFS_time"
DFS_EVENT_COL = "DFS_status"
# Postoperative pathological nodal status: confirmed binary coding in the source data.
# 0 = node-negative; 1 = node-positive. N is deliberately excluded from MODEL_FEATURES.
N_COL = "N"
N_NEGATIVE_VALUE = 0
N_POSITIVE_VALUE = 1
DFS_TIME_UNIT = "days"
DAYS_PER_MONTH = 30.4375
# Do not derive an optimal DFS threshold. If a clinical LVI probability threshold
# is prespecified independently of DFS, set FIXED_THRESHOLD above; otherwise
# predicted-probability analyses remain continuous.
DFS_BOOTSTRAP = 1000

# Reader reproducibility analysis:
# Two junior readers independently assessed cases. The senior reader is an
# adjudicator ONLY for discordant cases, so senior decisions are NOT treated as
# an independent third reader in agreement statistics.
AGREEMENT_FILE = BASE_DIR / "agreement data.xlsx"  # 622 cases; unsuffixed=R1, suffix 1=R2, suffix 2=senior adjudicator
READER_CASE_COL = "case_id"
READER_ID_COL = "reader_id"
READER_SESSION_COL = "session"
READER_FEATURE_COL = "feature"
READER_VALUE_COL = "rating"
JUNIOR_READERS = ["Junior1", "Junior2"]
SENIOR_READER_LABELS = ["Senior", "Reader3", "R3", "Adjudicator"]
RELIABILITY_N_CASES = 100
RELIABILITY_SEED = 2026
RELIABILITY_BOOTSTRAP = 2000
RELIABILITY_CONTINUOUS = ["Diameter", "ADC ratio"]
RELIABILITY_CATEGORICAL = [
    "Shape", "Lobulated", "Angular", "Spiculated",
    "Internal enhancement", "TIC", "FBE", "AVS", "BPS"
]
# Input must be long format with one row per case-reader-session-feature:
# case_id | reader_id | session | feature | rating
# Session 1: original independent junior ratings.
# Session 2: repeat read by the SAME junior reader after a washout period
# (recommended approximately 4 weeks); enter these ratings after rereading.

# =============================================================================
# 2. DATA AND CODING
# =============================================================================
def read_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}. Check BASE_DIR and Excel filenames.")
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def map_binary_outcome(series, cohort):
    values = []
    for value in series:
        if pd.isna(value):
            values.append(np.nan)
            continue
        z = str(value).strip().lower()
        if z in POSITIVE_LABELS:
            values.append(1)
        elif z in NEGATIVE_LABELS:
            values.append(0)
        else:
            try:
                f = float(z)
                values.append(int(f) if f in (0, 1) else np.nan)
            except Exception:
                values.append(np.nan)
    arr = pd.Series(values, index=series.index)
    if arr.isna().any():
        bad = series[arr.isna()].astype(str).unique().tolist()
        raise ValueError(f"{cohort}: unrecognized/missing outcome values in {TARGET}: {bad}")
    y = arr.astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError(f"{cohort}: outcome must contain both 0 and 1.")
    return y


def prepare_data(df, cohort):
    required = [TARGET] + sorted(set(sum(MODEL_FEATURES.values(), [])))
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{cohort} is missing required columns: {missing}")
    df = df.copy()
    y = map_binary_outcome(df[TARGET], cohort)
    for col in CLINICAL_CONT + MRI_CONT:
        before = df[col].notna()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if (before & df[col].isna()).any():
            raise ValueError(f"{cohort}: nonnumeric nonmissing values in {col}; inspect source.")
    for col in CLINICAL_CAT + MRI_CAT + FBE + AVS + BPS:
        df[col] = df[col].map(lambda x: str(x).strip() if pd.notna(x) else np.nan)
    return df, y



# =============================================================================
# 3. PREPROCESSING + NESTED CROSS-VALIDATION
# =============================================================================
def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_pipeline(features):
    num_cols = [c for c in CLINICAL_CONT + MRI_CONT if c in features]
    cat_cols = [c for c in CLINICAL_CAT + MRI_CAT + FBE + AVS + BPS if c in features]
    transformers = []
    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler())
        ])
        transformers.append(("numeric", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot())
        ])
        transformers.append(("categorical", cat_pipe, cat_cols))
    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([
        ("preprocess", pre),
        ("model", LogisticRegression(max_iter=20000, random_state=RANDOM_SEED))
    ])


def parameter_grid():
    return [
        {"model__penalty": ["l2"], "model__solver": ["lbfgs"], "model__C": [0.1, 1, 10]},
        {"model__penalty": ["l1"], "model__solver": ["liblinear"], "model__C": [0.1, 1, 10]},
        {"model__penalty": ["elasticnet"], "model__solver": ["saga"],
         "model__l1_ratio": [0.5], "model__C": [0.1, 1, 10]}
    ]


def nested_cv(X, y, features, model_name):
    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof = np.full(len(y), np.nan)
    fold_rows = []
    for fold, (tr, va) in enumerate(outer.split(X, y), start=1):
        inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED + fold)
        search = GridSearchCV(make_pipeline(features), parameter_grid(),
                              scoring="roc_auc", cv=inner, n_jobs=N_JOBS, pre_dispatch=1,
                              refit=True, error_score="raise")
        search.fit(X.iloc[tr], y[tr])
        oof[va] = search.predict_proba(X.iloc[va])[:, 1]
        fold_rows.append({"model": model_name, "outer_fold": fold,
                          "inner_best_auc": search.best_score_,
                          "best_params": json.dumps(search.best_params_)})
        print(f"{model_name}: completed outer fold {fold}/{OUTER_FOLDS}")
    if np.isnan(oof).any():
        raise RuntimeError(f"{model_name}: incomplete OOF predictions.")
    return oof, fold_rows


def fit_locked_model(X, y, features):
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    search = GridSearchCV(make_pipeline(features), parameter_grid(),
                          scoring="roc_auc", cv=inner, n_jobs=N_JOBS, pre_dispatch=1,
                          refit=True, error_score="raise")
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


# =============================================================================
# 4. METRICS AND PLOTS
# =============================================================================
def auc_bootstrap_ci(y, p, n_boot=N_BOOT, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y, int), np.asarray(p, float)
    vals = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(y), len(y))
        if np.unique(y[ix]).size == 2:
            vals.append(roc_auc_score(y[ix], p[ix]))
    if len(vals) < 100:
        return np.nan, np.nan
    return np.percentile(vals, [2.5, 97.5])


def calibration_parameters(y, p):
    try:
        import statsmodels.api as sm
        y = np.asarray(y, int)
        lp = logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))
        fit_int = sm.GLM(y, np.ones((len(y), 1)),
                         family=sm.families.Binomial(), offset=lp).fit()
        fit_slope = sm.GLM(y, sm.add_constant(lp),
                           family=sm.families.Binomial()).fit()
        return float(fit_int.params[0]), float(fit_slope.params[1])
    except Exception:
        return np.nan, np.nan


def metric_row(y, p, cohort, model):
    lo, hi = auc_bootstrap_ci(y, p)
    ci, slope = calibration_parameters(y, p)
    row = {"cohort": cohort, "model": model, "n": len(y), "events": int(np.sum(y)),
           "prevalence": float(np.mean(y)), "AUC": roc_auc_score(y, p),
           "AUC_95CI_low": lo, "AUC_95CI_high": hi,
           "PR_AUC_average_precision": average_precision_score(y, p),
           "Brier": brier_score_loss(y, p),
           "calibration_intercept": ci, "calibration_slope": slope}
    if FIXED_THRESHOLD is not None:
        tn, fp, fn, tp = confusion_matrix(y, np.asarray(p) >= FIXED_THRESHOLD,
                                          labels=[0, 1]).ravel()
        row.update({"threshold": FIXED_THRESHOLD,
                    "sensitivity": tp/(tp+fn) if tp+fn else np.nan,
                    "specificity": tn/(tn+fp) if tn+fp else np.nan,
                    "PPV": tp/(tp+fp) if tp+fp else np.nan,
                    "NPV": tn/(tn+fn) if tn+fn else np.nan})
    return row


def save_tiff_publication_dpi(fig, path):
    """Low-peak-memory RGB TIFF export at the configured publication DPI."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.set_dpi(FIG_DPI)
    print(f"    rendering {path.name} at {FIG_DPI} dpi ...", flush=True)
    fig.canvas.draw()
    print(f"    encoding {path.name} as RGB TIFF ...", flush=True)
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    tifffile.imwrite(str(path), rgb, photometric="rgb", compression="deflate",
                     resolution=(FIG_DPI, FIG_DPI), resolutionunit="INCH", metadata=None,
                     maxworkers=1)
    del rgb, rgba
    size = fig.get_size_inches().copy()
    plt.close(fig)
    gc.collect()
    print(f"    saved {path.name}: {size[0]:.1f} x {size[1]:.1f} in, {FIG_DPI} dpi", flush=True)


def _panel_tag(ax, tag):
    """Place panel letter just outside the upper-left corner of the axes."""
    ax.text(-0.075, 1.075, tag, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=12, fontweight="bold",
            clip_on=False)


def _model_label(name):
    labels = {
        "M1_Clinical": "M1: Clinical",
        "M2_Clinical_plus_conventional_MRI": "M2: Clinical + conventional MRI",
        "M3_Primary_combined_plus_BPS": "M3: Clinical + MRI + BPS",
    }
    return labels.get(name, name.replace("_", " "))


def _format_two_panel_figure(fig, axes):
    """Wide two-panel publication layout with space for outside panel letters."""
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.135, top=0.865,
                        wspace=0.24)


def _add_roc_legend(ax):
    """Compact lower-right legend, kept inside the ROC axes as requested."""
    ax.legend(loc="lower right", ncol=1, frameon=True, framealpha=0.96,
              facecolor="white", edgecolor="0.75", fontsize=6.1,
              handlelength=1.45, labelspacing=0.24,
              handletextpad=0.42, borderpad=0.38, borderaxespad=0.32)


def _add_calibration_legend(ax):
    """Calibration legend in the upper-left position of the example."""
    ax.legend(loc="upper left", ncol=1, frameon=True, framealpha=0.92,
              facecolor="white", edgecolor="0.75", fontsize=6.5,
              handlelength=1.45, labelspacing=0.28,
              handletextpad=0.42, borderpad=0.38, borderaxespad=0.32)


def _add_dca_legend(ax):
    """DCA legend in the upper-right position of the example."""
    ax.legend(loc="upper right", ncol=1, frameon=True, framealpha=0.92,
              facecolor="white", edgecolor="0.75", fontsize=6.5,
              handlelength=1.45, labelspacing=0.28,
              handletextpad=0.42, borderpad=0.38, borderaxespad=0.32)


def _short_model_label(name):
    labels = {
        "M1_Clinical": "M1 Clinical",
        "M2_Clinical_plus_conventional_MRI": "M2 Conv. MRI",
        "M3_Primary_combined_plus_BPS": "M3 + BPS",
    }
    return labels.get(name, name.replace("_", " "))


def plot_roc_panels(y_a, pred_a, y_b, pred_b, path):
    """Wide paired ROC figure with bootstrap AUC confidence intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharex=True, sharey=True)
    for ax, y, preds, title, tag in [
        (axes[0], y_a, pred_a, "Internal cohort", "A"),
        (axes[1], y_b, pred_b, "External cohort", "B"),
    ]:
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.65, label="_nolegend_")
        for j, (name, prob) in enumerate(preds.items()):
            fpr, tpr, _ = roc_curve(y, prob)
            auc_value = roc_auc_score(y, prob)
            # Use the same bootstrap seed as the original performance table.
            ci_low, ci_high = auc_bootstrap_ci(
                y, prob, n_boot=N_BOOT, seed=RANDOM_SEED
            )
            label = (f"{_short_model_label(name)}: {auc_value:.3f} "
                     f"({ci_low:.3f}–{ci_high:.3f})")
            ax.plot(fpr, tpr, linewidth=1.05, label=label)
        ax.set_title(title, pad=7, fontsize=9.0)
        ax.set_xlabel("1 - Specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_aspect("equal", adjustable="box")
        _add_roc_legend(ax)
        _panel_tag(ax, tag)
    _format_two_panel_figure(fig, axes)
    save_tiff_publication_dpi(fig, path)


def plot_calibration_panels(y_a, pred_a, y_b, pred_b, path):
    """Paired calibration figure; ideal line retained without a legend label."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharex=True, sharey=True)
    for ax, y, preds, title, tag in [
        (axes[0], y_a, pred_a, "Internal cohort", "A"),
        (axes[1], y_b, pred_b, "External cohort", "B"),
    ]:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray",
                linewidth=0.65, label="_nolegend_", zorder=1)
        for name, prob in preds.items():
            observed, predicted = calibration_curve(y, prob, n_bins=6, strategy="quantile")
            ax.plot(predicted, observed, marker="o", markersize=2.6, linewidth=1.05,
                    label=_short_model_label(name), zorder=2)
        ax.set_title(title, pad=7, fontsize=9.0)
        ax.set_xlabel("Mean predicted probability of LVI")
        ax.set_ylabel("Observed LVI proportion")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        _add_calibration_legend(ax)
        _panel_tag(ax, tag)
    _format_two_panel_figure(fig, axes)
    save_tiff_publication_dpi(fig, path)


def net_benefit(y, p, threshold):
    y, p = np.asarray(y, int), np.asarray(p, float)
    pred = p >= threshold
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    return tp/len(y) - fp/len(y) * threshold/(1-threshold)


def plot_dca_panels(y_a, pred_a, y_b, pred_b, path):
    """Two-panel decision curves; legend remains upper-right inside each axes."""
    thresholds = np.arange(DCA_XLIM[0], DCA_XLIM[1] + 0.001, 0.01)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharex=True, sharey=True)
    for ax, y, preds, title, tag in [
        (axes[0], y_a, pred_a, "Internal cohort", "A"),
        (axes[1], y_b, pred_b, "External cohort", "B"),
    ]:
        prevalence = np.mean(y)
        ax.plot(thresholds, np.zeros_like(thresholds), "--", linewidth=0.65,
                color="black", label="Treat none")
        treat_all = prevalence - (1-prevalence)*thresholds/(1-thresholds)
        ax.plot(thresholds, treat_all, ":", linewidth=0.70,
                color="dimgray", label="Treat all")
        for name, prob in preds.items():
            ax.plot(thresholds, [net_benefit(y, prob, t) for t in thresholds],
                    linewidth=1.05, label=_short_model_label(name))
        ax.set_title(title, pad=7, fontsize=9.0)
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.set_xlim(DCA_XLIM); ax.set_ylim(DCA_YLIM)
        _add_dca_legend(ax)
        _panel_tag(ax, tag)
    _format_two_panel_figure(fig, axes)
    save_tiff_publication_dpi(fig, path)




# =============================================================================
# 4B. FLOWCHART AND INTEROBSERVER AGREEMENT SUPPLEMENT
# =============================================================================
def plot_patient_flowchart(path):
    """Create corrected two-center patient flow diagram (600-dpi TIFF)."""
    from matplotlib.patches import Rectangle, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")

    def box(x, y, w, h, text, fontsize=9.0, lw=0.85):
        rect = Rectangle((x, y), w, h, fill=False, linewidth=lw, edgecolor="black")
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                fontsize=fontsize, wrap=True, color="black")

    def arrow(x1, y1, x2, y2, lw=0.8):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle="-|>",
                                     mutation_scale=9, linewidth=lw, color="black"))

    # Center 1 column
    box(0.35, 5.75, 4.25, 0.95,
        "1,116 patients with pathologically confirmed invasive breast cancer\n"
        "undergoing preoperative MRI at Center 1\nApril 2008–December 2025", 8.8)
    box(0.65, 4.15, 3.65, 1.10,
        "Exclusion criteria\n"
        "• Inadequate image quality\n"
        "• Incomplete clinical-pathological data\n"
        "• Neoadjuvant therapy before MRI", 8.3)
    box(1.05, 2.85, 2.85, 0.65, "634 patients included", 9.2)
    box(0.35, 0.55, 1.95, 0.78, "LVI-positive\n(n = 141)", 9.0)
    box(2.65, 0.55, 1.95, 0.78, "LVI-negative\n(n = 493)", 9.0)

    # Center 2 column
    box(5.40, 5.75, 4.25, 0.95,
        "302 patients with pathologically confirmed invasive breast cancer\n"
        "undergoing preoperative MRI at Center 2\nJanuary 2020–March 2026", 8.8)
    box(5.70, 4.15, 3.65, 1.10,
        "Exclusion criteria\n"
        "• Inadequate image quality\n"
        "• Incomplete clinical-pathological data\n"
        "• Neoadjuvant therapy before MRI", 8.3)
    box(6.10, 2.85, 2.85, 0.65, "156 patients included", 9.2)
    box(5.40, 0.55, 1.95, 0.78, "LVI-positive\n(n = 34)", 9.0)
    box(7.70, 0.55, 1.95, 0.78, "LVI-negative\n(n = 122)", 9.0)

    # Flow connectors
    arrow(2.475, 5.75, 2.475, 5.25)
    arrow(2.475, 4.15, 2.475, 3.50)
    arrow(7.525, 5.75, 7.525, 5.25)
    arrow(7.525, 4.15, 7.525, 3.50)

    # Outcome splits
    ax.plot([2.475, 2.475], [2.85, 2.15], color="black", lw=0.8)
    ax.plot([1.325, 3.625], [2.15, 2.15], color="black", lw=0.8)
    arrow(1.325, 2.15, 1.325, 1.33)
    arrow(3.625, 2.15, 3.625, 1.33)
    ax.plot([7.525, 7.525], [2.85, 2.15], color="black", lw=0.8)
    ax.plot([6.375, 8.675], [2.15, 2.15], color="black", lw=0.8)
    arrow(6.375, 2.15, 6.375, 1.33)
    arrow(8.675, 2.15, 8.675, 1.33)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.04)
    save_tiff_publication_dpi(fig, path)


def paired_bootstrap_kappa_ci(rater1, rater2, n_boot=2000, seed=2026):
    """Paired-case nonparametric bootstrap CI for unweighted Cohen's kappa.

    Pass the two readers' ratings for one feature. Missing pairs are removed
    before resampling. Returns (n_paired, kappa, lower_95CI, upper_95CI).
    This helper is ready for use once the original paired reader-level ratings
    and their exact column mapping are supplied.
    """
    from sklearn.metrics import cohen_kappa_score
    pair = pd.DataFrame({"r1": rater1, "r2": rater2}).dropna()
    n = len(pair)
    if n < 2:
        return n, np.nan, np.nan, np.nan
    a = pair["r1"].to_numpy()
    b = pair["r2"].to_numpy()
    try:
        estimate = float(cohen_kappa_score(a, b))
    except Exception:
        estimate = np.nan
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        try:
            value = cohen_kappa_score(a[ix], b[ix])
            if np.isfinite(value):
                boot.append(float(value))
        except Exception:
            continue
    if len(boot) < max(100, int(0.5*n_boot)):
        return n, estimate, np.nan, np.nan
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return n, estimate, float(lo), float(hi)


def interobserver_agreement_table():
    """Return the corrected 622-case Reader 1 vs Reader 2 agreement results."""
    result_path = OUT_DIR / "06_Reader_Agreement" / "Interreader_agreement_summary_corrected.csv"
    if result_path.exists():
        frame = pd.read_csv(result_path)
        frame["reader_or_pair"] = "Reader 1 (unsuffixed) vs Reader 2 (suffix 1)"
        return frame
    return pd.DataFrame([{
        "Variable": "Pending",
        "Variable_type": "Not calculated",
        "Agreement_statistic": "Not calculated",
        "N_paired": 0,
        "Estimate": np.nan,
        "95% CI lower": np.nan,
        "95% CI upper": np.nan,
        "reader_or_pair": "Awaiting agreement data.xlsx"
    }])


# =============================================================================
# 5. PUBLICATION TABLES
# =============================================================================
def fmt_mean_sd(x):
    """Publication format for continuous variables: mean ± sample SD."""
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if len(x) == 0:
        return "—"
    sd = x.std(ddof=1) if len(x) > 1 else np.nan
    return f"{x.mean():.2f} ± {sd:.2f}" if pd.notna(sd) else f"{x.mean():.2f} ± —"


def p_fmt(p):
    if pd.isna(p):
        return "—"
    return "<0.001" if p < .001 else f"{p:.3f}"


def continuous_p_welch(x1, x2):
    x1 = pd.to_numeric(pd.Series(x1), errors="coerce").dropna()
    x2 = pd.to_numeric(pd.Series(x2), errors="coerce").dropna()
    if len(x1) < 2 or len(x2) < 2:
        return np.nan
    return float(ttest_ind(x1, x2, equal_var=False, nan_policy="omit").pvalue)


def categorical_p(x, group):
    """Omnibus Pearson chi-square or Fisher exact test for a 2x2 table."""
    tmp = pd.DataFrame({"x": x, "group": group}).dropna()
    tab = pd.crosstab(tmp["x"].astype(str), tmp["group"].astype(str))
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    try:
        if tab.shape == (2, 2) and (chi2_contingency(tab, correction=False)[3] < 5).any():
            return float(fisher_exact(tab.values)[1])
        return float(chi2_contingency(tab, correction=False)[1])
    except Exception:
        return np.nan


def ordered_levels(series):
    levels = series.dropna().astype(str).unique().tolist()
    return sorted(levels, key=lambda z: (0, float(z)) if z.replace('.', '', 1).isdigit() else (1, z))


def make_publication_table(df, group_col, group_levels, group_labels, numeric, categorical):
    """Create a compact journal-style table: characteristic, group columns, P value.

    Categorical variables are displayed as a variable heading followed by indented
    category rows. No missing-value columns are added because the analysis dataset
    has been confirmed complete by the study team.
    """
    rows = []
    group_values = df[group_col].astype(str)
    for var in numeric:
        row = {"Characteristic": var, "P value": p_fmt(continuous_p_welch(
            df.loc[group_values == str(group_levels[0]), var],
            df.loc[group_values == str(group_levels[1]), var]))}
        for g in group_levels:
            row[group_labels[g]] = fmt_mean_sd(df.loc[group_values == str(g), var])
        rows.append(row)

    for var in categorical:
        p = categorical_p(df[var], group_values)
        heading = {"Characteristic": var, "P value": p_fmt(p)}
        for g in group_levels:
            heading[group_labels[g]] = ""
        rows.append(heading)
        for level in ordered_levels(df[var]):
            row = {"Characteristic": f"    {level}", "P value": ""}
            for g in group_levels:
                denom = int((group_values == str(g)).sum())
                count = int(((group_values == str(g)) & (df[var].astype(str) == level)).sum())
                row[group_labels[g]] = f"{count} ({100 * count / denom:.1f}%)" if denom else "—"
            rows.append(row)
    return pd.DataFrame(rows, columns=["Characteristic"] + [group_labels[g] for g in group_levels] + ["P value"])


def between_center_table(a, ya, b, yb):
    """Main Table 1: direct comparison between Center 1 and Center 2."""
    da, db = a.copy(), b.copy()
    da["Center"] = "Center 1"
    db["Center"] = "Center 2"
    da["LVI_status_display"] = np.where(np.asarray(ya, int) == 1, "LVI-positive", "LVI-negative")
    db["LVI_status_display"] = np.where(np.asarray(yb, int) == 1, "LVI-positive", "LVI-negative")
    both = pd.concat([da, db], ignore_index=True)
    numeric = CLINICAL_CONT + MRI_CONT
    categorical = CLINICAL_CAT + MRI_CAT + FBE + AVS + BPS + [N_COL, "LVI_status_display"]
    result = make_publication_table(
        both, "Center", ["Center 1", "Center 2"],
        {"Center 1": f"Center 1 (n={len(a)})", "Center 2": f"Center 2 (n={len(b)})"},
        numeric, categorical)
    result["Characteristic"] = result["Characteristic"].replace({"LVI_status_display": "LVI status"})
    return result


def within_center_lvi_table(df, y, cohort):
    """Publication-style within-center LVI− versus LVI+ table."""
    d = df.copy()
    d["LVI_group_display"] = np.where(np.asarray(y, int) == 1, "LVI-positive", "LVI-negative")
    numeric = CLINICAL_CONT + MRI_CONT
    categorical = CLINICAL_CAT + MRI_CAT + FBE + AVS + BPS + [N_COL]
    n0 = int(np.sum(np.asarray(y) == 0))
    n1 = int(np.sum(np.asarray(y) == 1))
    result = make_publication_table(
        d, "LVI_group_display", ["LVI-negative", "LVI-positive"],
        {"LVI-negative": f"LVI-negative (n={n0})", "LVI-positive": f"LVI-positive (n={n1})"},
        numeric, categorical)
    result.insert(0, "Center", cohort)
    return result


def paired_auc_bootstrap_with_replicates(y, p_ref, p_new, n_boot=N_BOOT, seed=2026):
    """Paired patient-level bootstrap; return point estimate, percentile CI and replicates."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int)
    p_ref, p_new = np.asarray(p_ref), np.asarray(p_new)
    diffs = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(y), len(y))
        if np.unique(y[ix]).size == 2:
            diffs.append(roc_auc_score(y[ix], p_new[ix]) - roc_auc_score(y[ix], p_ref[ix]))
    point = roc_auc_score(y, p_new) - roc_auc_score(y, p_ref)
    if len(diffs) < 100:
        return point, np.nan, np.nan, diffs
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, lo, hi, diffs


def make_auc_comparison(y, predictions, cohort, replicate_rows):
    comparisons = [
        ("M1_Clinical", "M2_Clinical_plus_conventional_MRI"),
        ("M2_Clinical_plus_conventional_MRI", "M3_Primary_combined_plus_BPS"),
    ]
    rows = []
    for i, (ref, new) in enumerate(comparisons):
        seed = RANDOM_SEED + i
        d, lo, hi, reps = paired_auc_bootstrap_with_replicates(
            y, predictions[ref], predictions[new], seed=seed)
        rows.append({"cohort": cohort, "reference_model": ref, "comparison_model": new,
                     "AUC_difference_new_minus_reference": d,
                     "bootstrap_95CI_low": lo, "bootstrap_95CI_high": hi,
                     "bootstrap_requested": N_BOOT, "bootstrap_valid_replicates": len(reps),
                     "bootstrap_seed": seed,
                     "CI_interpretation": "Paired patient-level bootstrap percentile 95% CI; no p-value"})
        for j, val in enumerate(reps, start=1):
            replicate_rows.append({"cohort": cohort, "reference_model": ref,
                                   "comparison_model": new, "bootstrap_replicate": j,
                                   "AUC_difference_new_minus_reference": val})
    return pd.DataFrame(rows)

def coefficient_table(fitted_models):
    """Penalized logistic coefficients; ORs are descriptive, no naive Wald P/CI."""
    rows = []
    for model_name, pipe in fitted_models.items():
        pre = pipe.named_steps["preprocess"]
        clf = pipe.named_steps["model"]
        try:
            names = pre.get_feature_names_out()
        except Exception:
            names = [f"feature_{i}" for i in range(clf.coef_.shape[1])]
        for term, beta in zip(names, clf.coef_[0]):
            rows.append({"model": model_name, "term": term,
                         "coefficient": float(beta), "odds_ratio": float(np.exp(beta)),
                         "note": "Penalized model; continuous predictors standardized; no Wald P/CI"})
        rows.append({"model": model_name, "term": "Intercept",
                     "coefficient": float(clf.intercept_[0]),
                     "odds_ratio": np.nan,
                     "note": "Penalized model; intercept"})
    return pd.DataFrame(rows)


def threshold_metrics(y, p, threshold):
    tn, fp, fn, tp = confusion_matrix(y, np.asarray(p) >= threshold,
                                      labels=[0, 1]).ravel()
    return {"threshold": threshold, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "sensitivity": tp/(tp+fn) if tp+fn else np.nan,
            "specificity": tn/(tn+fp) if tn+fp else np.nan,
            "PPV": tp/(tp+fp) if tp+fp else np.nan,
            "NPV": tn/(tn+fn) if tn+fn else np.nan,
            "net_benefit": net_benefit(y, p, threshold)}


def threshold_analysis(y, p, cohort):
    # Descriptive threshold grid only. Does not select a "best" threshold.
    rows = []
    for t in np.arange(0.05, 0.801, 0.01):
        row = threshold_metrics(y, p, float(round(t, 2)))
        row["cohort"] = cohort
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 6. DFS EXPLORATORY PROGNOSTIC ANALYSIS
# =============================================================================
def _reverse_km_median_followup(time_months, event):
    """Reverse Kaplan-Meier estimate of median follow-up."""
    from lifelines import KaplanMeierFitter
    # Reverse KM: original censoring observations become events.
    km = KaplanMeierFitter()
    km.fit(time_months, event_observed=1 - np.asarray(event, dtype=int))
    med = km.median_survival_time_
    return float(med) if np.isfinite(med) else np.nan


def _cox_one_predictor(df, predictor, label, stratify_center=False):
    """One-predictor Cox model; intentionally avoids multivariable overfitting."""
    from lifelines import CoxPHFitter
    cols = ["time_months", "event", predictor]
    if stratify_center:
        cols.append("center")
    z = df[cols].copy()
    z[predictor] = pd.to_numeric(z[predictor], errors="coerce")
    z = z.dropna()
    if z.empty or z["event"].sum() < 2 or z[predictor].nunique() < 2:
        return {"analysis": label, "n": len(z), "events": int(z.event.sum()) if len(z) else 0,
                "HR": np.nan, "CI95_low": np.nan, "CI95_high": np.nan,
                "p_value": np.nan, "note": "Insufficient events or predictor variation"}
    try:
        cph = CoxPHFitter(penalizer=0.0)
        if stratify_center:
            cph.fit(z, duration_col="time_months", event_col="event", strata=["center"])
        else:
            cph.fit(z, duration_col="time_months", event_col="event")
        r = cph.summary.loc[predictor]
        return {"analysis": label, "n": len(z), "events": int(z.event.sum()),
                "HR": float(r["exp(coef)"]),
                "CI95_low": float(r["exp(coef) lower 95%"]),
                "CI95_high": float(r["exp(coef) upper 95%"]),
                "p_value": float(r["p"]),
                "note": "Exploratory one-predictor Cox model" + ("; stratified by center" if stratify_center else "")}
    except Exception as e:
        return {"analysis": label, "n": len(z), "events": int(z.event.sum()),
                "HR": np.nan, "CI95_low": np.nan, "CI95_high": np.nan,
                "p_value": np.nan, "note": f"Cox model not estimable: {e}"}


def _bootstrap_cindex(time, event, risk, n_boot=DFS_BOOTSTRAP, seed=RANDOM_SEED):
    from lifelines.utils import concordance_index
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    risk = np.asarray(risk, float)
    try:
        point = concordance_index(time, -risk, event)
    except Exception:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(time), len(time))
        if event[ix].sum() == 0:
            continue
        try:
            vals.append(concordance_index(time[ix], -risk[ix], event[ix]))
        except Exception:
            pass
    if len(vals) < 100:
        return float(point), np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def run_dfs_analysis(a_raw, b_raw, a_oof_primary, b_locked_primary):
    """Exploratory DFS: actual LVI, M3-predicted LVI groups, and N strata."""
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
        from lifelines.plotting import add_at_risk_counts
    except ImportError:
        raise ImportError("DFS analysis requires lifelines. Install with: pip install lifelines")

    out = OUT_DIR / "05_DFS_exploratory_analysis"
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for cohort, raw, pred in [("Center_1", a_raw, a_oof_primary),
                              ("Center_2", b_raw, b_locked_primary)]:
        if DFS_TIME_COL not in raw.columns or DFS_EVENT_COL not in raw.columns:
            raise KeyError(f"{cohort} must contain {DFS_TIME_COL} and {DFS_EVENT_COL}.")
        if len(raw) != len(pred):
            raise ValueError(f"{cohort}: prediction length does not match raw data length.")
        z = pd.DataFrame(index=np.arange(len(raw)))
        z["center"] = cohort
        z["LVI_actual"] = map_binary_outcome(raw[TARGET], cohort)
        z["DFS_time_days"] = pd.to_numeric(raw[DFS_TIME_COL], errors="coerce").to_numpy()
        z["event_raw"] = pd.to_numeric(raw[DFS_EVENT_COL], errors="coerce").to_numpy()
        z["M3_predicted_LVI_probability"] = np.asarray(pred, dtype=float)
        if N_COL in raw.columns:
            n_numeric = pd.to_numeric(raw[N_COL], errors="coerce")
            z["N_raw"] = raw[N_COL].to_numpy()
            z["N_binary"] = np.where(n_numeric.eq(N_NEGATIVE_VALUE), 0,
                                      np.where(n_numeric.eq(N_POSITIVE_VALUE), 1, np.nan))
        else:
            z["N_raw"], z["N_binary"] = np.nan, np.nan
        valid = (z.DFS_time_days.notna() & (z.DFS_time_days > 0) &
                 z.event_raw.isin([0, 1]) & z.M3_predicted_LVI_probability.notna() &
                 z.LVI_actual.notna())
        z["valid_DFS"] = valid
        z["exclusion_reason"] = ""
        z.loc[z.DFS_time_days.isna(), "exclusion_reason"] = "Missing DFS_time"
        z.loc[z.DFS_time_days.notna() & (z.DFS_time_days <= 0), "exclusion_reason"] = "DFS_time <= 0"
        z.loc[z.event_raw.isna(), "exclusion_reason"] = "Missing DFS_status"
        z.loc[z.event_raw.notna() & ~z.event_raw.isin([0, 1]), "exclusion_reason"] = "DFS_status not 0/1"
        z.loc[z.M3_predicted_LVI_probability.isna(), "exclusion_reason"] = "Missing M3 prediction"
        z["time_months"] = z.DFS_time_days / DAYS_PER_MONTH
        z["event"] = z.event_raw
        z.loc[~valid, "event"] = np.nan
        z["row_number_excel_approx"] = np.arange(len(raw)) + 2
        z.to_csv(out / f"DFS_followup_record_review_{cohort}.csv", index=False)
        zz = z.loc[valid].copy()
        zz["event"] = zz.event.astype(int)
        frames.append(zz)

    surv = pd.concat(frames, ignore_index=True)
    if surv.empty:
        raise ValueError("No valid DFS records found in either cohort.")
    surv["M3_probability_per_0_1"] = surv.M3_predicted_LVI_probability / 0.1

    # A-only OOF median split, frozen and applied to both centers. This is an
    # exploratory display grouping, NOT a clinical cutoff and NOT DFS-optimized.
    a_probs = np.asarray(a_oof_primary, dtype=float)
    a_probs = a_probs[np.isfinite(a_probs)]
    if len(a_probs) == 0:
        raise ValueError("No finite Center 1 OOF M3 predictions available for DFS grouping.")
    predicted_group_cutoff = float(np.median(a_probs))
    surv["M3_predicted_LVI_group"] = (
        surv.M3_predicted_LVI_probability >= predicted_group_cutoff).astype(int)
    surv.to_csv(out / "DFS_analysis_dataset.csv", index=False)
    pd.DataFrame([{
        "grouping": "Center 1 OOF prediction median; applied unchanged to Center 2",
        "cutoff": predicted_group_cutoff,
        "A_OOF_prediction_n": len(a_probs),
        "note": "Exploratory dichotomization for KM visualization only; not a clinical threshold and not selected using DFS outcomes"
    }]).to_csv(out / "DFS_M3_predicted_LVI_group_cutoff.csv", index=False)

    # KM plotting helper: survival curves in upper panel, number-at-risk table below.
    def save_km_with_risk_table(data, group_col, groups, labels, title, filename, logrank_file=None):
        fig, ax = plt.subplots(figsize=(7.0, 5.8))
        fitters = []
        present = []
        for value, label in zip(groups, labels):
            g = data[data[group_col] == value].copy()
            if len(g) == 0:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(g.time_months, g.event,
                    label=f"{label} (n={len(g)}, events={int(g.event.sum())})")
            kmf.plot_survival_function(ax=ax, ci_show=True, censor_styles={"marker": "+", "ms": 5})
            fitters.append(kmf)
            present.append((value, g))
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Time after surgery (months)")
        ax.set_ylabel("Disease-free survival probability")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        if fitters:
            add_at_risk_counts(*fitters, ax=ax, rows_to_show=["At risk"],
                               labels=[f"{labels[groups.index(v)]}" for v, _ in present])
        fig.subplots_adjust(left=0.12, right=0.97, top=0.91, bottom=0.28)
        if SAVE_INDIVIDUAL_DFS_KM:
            save_tiff_publication_dpi(fig, out / Path(filename).with_suffix(".tiff"))
        if logrank_file is not None and len(present) == 2:
            g0, g1 = present[0][1], present[1][1]
            lr = logrank_test(g0.time_months, g1.time_months,
                              event_observed_A=g0.event, event_observed_B=g1.event)
            pd.DataFrame([{
                "comparison": f"{labels[groups.index(present[1][0])]} vs {labels[groups.index(present[0][0]) ]}",
                "n": len(data), "events": int(data.event.sum()),
                "logrank_chi_square": float(lr.test_statistic),
                "logrank_p_exploratory": float(lr.p_value),
                "note": "Exploratory only; sparse DFS events; no equivalence/superiority inference"
            }]).to_csv(out / logrank_file, index=False)

    # Follow-up summary.
    summary_rows = []
    for center in ["Center_1", "Center_2", "Combined"]:
        z = surv if center == "Combined" else surv[surv.center == center]
        summary_rows.append({
            "cohort": center, "valid_followup_n": len(z), "DFS_events": int(z.event.sum()),
            "censored": int((z.event == 0).sum()), "event_percent": 100*z.event.mean(),
            "median_followup_months_reverse_KM": _reverse_km_median_followup(z.time_months, z.event),
            "followup_months_median_IQR": f"{z.time_months.median():.2f} ({z.time_months.quantile(.25):.2f}, {z.time_months.quantile(.75):.2f})"
        })
    pd.DataFrame(summary_rows).to_csv(out / "DFS_followup_summary.csv", index=False)

    # Overall prognostic KM by pathological LVI and model-predicted LVI group.
    save_km_with_risk_table(surv, "LVI_actual", [0, 1],
        ["Pathological LVI negative", "Pathological LVI positive"],
        "DFS by pathological LVI status", "KM_DFS_by_pathological_LVI_combined.png",
        "DFS_logrank_pathological_LVI.csv")
    save_km_with_risk_table(surv, "M3_predicted_LVI_group", [0, 1],
        ["M3-predicted LVI lower group", "M3-predicted LVI higher group"],
        "DFS by M3-predicted LVI group (exploratory median split)",
        "KM_DFS_by_M3_predicted_LVI_group.png", "DFS_logrank_M3_predicted_LVI_group.csv")

    # N subgroup counts and KM for model-predicted LVI groups within each N stratum.
    n_valid = surv[surv.N_binary.isin([0, 1])].copy()
    n_summary_rows, predicted_n_logrank_rows = [], []
    for center in ["Center_1", "Center_2", "Combined"]:
        zc = n_valid if center == "Combined" else n_valid[n_valid.center == center]
        for nv, nl in [(0, "N0_node_negative"), (1, "N_positive")]:
            zn = zc[zc.N_binary == nv]
            n_summary_rows.append({
                "cohort": center, "N_group": nl, "n": len(zn),
                "DFS_events": int(zn.event.sum()), "censored": int((zn.event == 0).sum()),
                "M3_predicted_LVI_lower_n": int((zn.M3_predicted_LVI_group == 0).sum()),
                "M3_predicted_LVI_lower_events": int(zn.loc[zn.M3_predicted_LVI_group == 0, "event"].sum()),
                "M3_predicted_LVI_higher_n": int((zn.M3_predicted_LVI_group == 1).sum()),
                "M3_predicted_LVI_higher_events": int(zn.loc[zn.M3_predicted_LVI_group == 1, "event"].sum()),
                "N_missing_or_invalid_excluded": int((surv.center == center).sum()-len(zc)) if center != "Combined" else int(len(surv)-len(n_valid))
            })
    pd.DataFrame(n_summary_rows).to_csv(out / "DFS_N_by_M3_predicted_LVI_subgroup_summary.csv", index=False)

    # N0/N+ curves compare M3-predicted LVI groups using the SAME A-derived cutoff.
    for nv, nl in [(0, "N0_node_negative"), (1, "N_positive")]:
        zn = n_valid[n_valid.N_binary == nv].copy()
        if zn.empty:
            continue
        save_km_with_risk_table(zn, "M3_predicted_LVI_group", [0, 1],
            ["M3-predicted LVI lower group", "M3-predicted LVI higher group"],
            f"DFS by M3-predicted LVI group within {nl.replace('_', ' ')}",
            f"KM_DFS_M3_predicted_LVI_within_{nl}.png",
            f"DFS_logrank_M3_predicted_LVI_within_{nl}.csv")

    # Also retain the N0 vs N+ overall KM, and actual pathological LVI within N strata.
    if n_valid.N_binary.nunique() == 2:
        save_km_with_risk_table(n_valid, "N_binary", [0, 1],
            ["N0 (node-negative)", "N+ (node-positive)"],
            "DFS by pathological nodal status", "KM_DFS_by_pathological_N_status.png",
            "DFS_logrank_N_status.csv")
    actual_n_logrank = []
    for nv, nl in [(0, "N0_node_negative"), (1, "N_positive")]:
        zn = n_valid[n_valid.N_binary == nv]
        if len(zn):
            save_km_with_risk_table(zn, "LVI_actual", [0, 1],
                ["Pathological LVI negative", "Pathological LVI positive"],
                f"DFS by pathological LVI within {nl.replace('_', ' ')}",
                f"KM_DFS_pathological_LVI_within_{nl}.png")
            g0, g1 = zn[zn.LVI_actual == 0], zn[zn.LVI_actual == 1]
            if len(g0) and len(g1):
                lr = logrank_test(g0.time_months, g1.time_months,
                                  event_observed_A=g0.event, event_observed_B=g1.event)
                pval = float(lr.p_value)
            else:
                pval = np.nan
            actual_n_logrank.append({"N_group": nl, "n": len(zn), "events": int(zn.event.sum()),
                "LVI_negative_n": len(g0), "LVI_negative_events": int(g0.event.sum()),
                "LVI_positive_n": len(g1), "LVI_positive_events": int(g1.event.sum()),
                "logrank_p_exploratory": pval})
    pd.DataFrame(actual_n_logrank).to_csv(out / "DFS_logrank_pathological_LVI_within_N_strata.csv", index=False)

    # Publication Figures 5 and 6:
    # Each panel has a dedicated survival-curve axis and a separate risk-table axis.
    # This avoids the clipping/overlap caused by lifelines.add_at_risk_counts().
    def draw_km_panel_with_separate_risk_table(
            fig, outer_spec, data, group_col, group_values, group_labels,
            panel_title, panel_tag, x_max=None, y_label=True,
            risk_axis_label="Group", curve_colors=None):
        """
        KM panel with exactly aligned time coordinates:
        the upper KM axis and lower number-at-risk axis occupy the SAME
        left/right plotting boundaries. Therefore x=0, 12, 24... have
        identical horizontal positions in the curve and risk table.
        Group names are y tick labels to the left of the risk-table axis;
        they do not consume a separate horizontal plotting column.
        """
        d = data.dropna(subset=["time_months", "event", group_col]).copy()
        d["time_months"] = pd.to_numeric(d["time_months"], errors="coerce")
        d["event"] = pd.to_numeric(d["event"], errors="coerce").astype(int)
        d = d.dropna(subset=["time_months", "event"])
        if x_max is None:
            x_max = max(12.0, float(np.ceil(d["time_months"].max() / 12.0) * 12.0))
        ticks = np.arange(0, x_max + 0.1, 12.0)

        # Important: both axes are created in the same single-column GridSpec.
        # Do NOT create a separate label-column subplot; that caused the x=0
        # position of the risk table to shift right relative to the KM curve.
        sub = outer_spec.subgridspec(2, 1, height_ratios=[4.8, 1.55], hspace=0.28)
        ax = fig.add_subplot(sub[0])
        tax = fig.add_subplot(sub[1], sharex=ax)

        if curve_colors is None:
            curve_colors = ["#1f77b4", "#d62728"]
        rows = []
        km_groups = []
        for idx, (value, label) in enumerate(zip(group_values, group_labels)):
            g = d[d[group_col] == value].copy()
            if g.empty:
                continue
            color = curve_colors[idx % len(curve_colors)]
            kmf = KaplanMeierFitter()
            kmf.fit(g["time_months"].astype(float), g["event"].astype(int), label=label)
            sf = kmf.survival_function_
            ci = kmf.confidence_interval_survival_function_
            ax.step(sf.index, sf.iloc[:, 0], where="post",
                    linewidth=1.8, color=color, label=label)
            ax.fill_between(
                ci.index.to_numpy(dtype=float),
                ci.iloc[:, 0].to_numpy(dtype=float),
                ci.iloc[:, 1].to_numpy(dtype=float),
                step="post", color=color, alpha=0.10, linewidth=0
            )
            censored = g.loc[g["event"] == 0, "time_months"].to_numpy(dtype=float)
            if len(censored):
                sf_t = sf.index.to_numpy(dtype=float)
                sf_y = sf.iloc[:, 0].to_numpy(dtype=float)
                censor_y = []
                for ct in censored:
                    pos = np.searchsorted(sf_t, ct, side="right") - 1
                    censor_y.append(sf_y[max(0, pos)])
                ax.plot(censored, censor_y, linestyle="none", marker="+",
                        markersize=4, markeredgewidth=0.8, color=color)
            counts = [int((g["time_months"] >= t).sum()) for t in ticks]
            rows.append((label, counts, color))
            km_groups.append(g)

        p_value = np.nan
        if len(km_groups) == 2:
            lr = logrank_test(
                km_groups[0]["time_months"], km_groups[1]["time_months"],
                event_observed_A=km_groups[0]["event"],
                event_observed_B=km_groups[1]["event"]
            )
            p_value = float(lr.p_value)

        # Upper KM plot.
        # Reserve a visible 6-month blank gutter before month 0. Because the risk-table
        # axis shares x with the KM axis, this padding is identical in both
        # panels. The month-0 tick and its risk count therefore sit slightly
        # inside the plotting area, leaving room for the colored group labels
        # outside the left spine (as in the user's reference figure).
        left_gutter_months = min(6.0, max(4.0, x_max * 0.05))
        ax.set_xlim(-left_gutter_months, x_max)
        ax.set_ylim(0.70, 1.02)
        ax.set_yticks([0.7, 0.8, 0.9, 1.0])
        ax.set_xticks(ticks)
        ax.grid(axis="y", alpha=0.18, linewidth=0.55)
        ax.tick_params(axis="x", labelbottom=False, length=3)
        ax.set_ylabel("DFS Probability", fontsize=8.5, fontweight="bold")
        if not y_label:
            ax.tick_params(axis="y", labelleft=False)
            ax.set_ylabel("")
        ax.set_title(panel_title, fontsize=10.0, fontweight="bold", pad=24)
        handles, legend_labels = ax.get_legend_handles_labels()
        ax.legend(handles, legend_labels, loc="lower center",
                  bbox_to_anchor=(0.5, 1.025), ncol=2, frameon=False,
                  fontsize=7.0, handlelength=1.7, columnspacing=1.2,
                  borderaxespad=0.0)
        ax.text(-0.12, 1.13, panel_tag, transform=ax.transAxes,
                fontsize=11, fontweight="bold", ha="left", va="bottom",
                clip_on=False)
        p_text = ("p = not estimable" if not np.isfinite(p_value) else
                  (f"p = {p_value:.4f}" if p_value >= 0.0001 else "p < 0.0001"))
        ax.text(0.5, 0.075, p_text, transform=ax.transAxes,
                ha="center", va="center", fontsize=8.5, fontweight="bold")

        # Lower risk table: same x-axis geometry as the KM plot above.
        y_positions = [0.68, 0.30]
        tax.set_xlim(-left_gutter_months, x_max)  # MUST match the upper KM axis; do not reset to 0.
        tax.set_xticks(ticks)
        tax.set_ylim(0.05, 0.95)
        tax.set_yticks(y_positions[:len(rows)])
        tax.set_yticklabels([r[0] for r in rows], fontsize=6.8, fontweight="bold")
        for tick_label, row in zip(tax.get_yticklabels(), rows):
            tick_label.set_color(row[2])
        tax.set_ylabel(risk_axis_label, fontsize=7.0, fontweight="bold",
                       rotation=90, labelpad=16)
        tax.set_xlabel("Time (Months)", fontsize=8.0, fontweight="bold", labelpad=3)
        tax.tick_params(axis="x", labelsize=7.2, length=3, pad=2)
        # Extra separation between the y-axis group labels and the first
        # risk-count column; month-0 itself is offset by the shared left gutter.
        tax.tick_params(axis="y", length=0, pad=8)
        tax.spines["left"].set_visible(True)
        tax.spines["bottom"].set_visible(True)
        tax.spines["left"].set_linewidth(0.7)
        tax.spines["bottom"].set_linewidth(0.7)
        for spine in ["top", "right"]:
            tax.spines[spine].set_visible(False)
        tax.grid(axis="x", alpha=0.14, linewidth=0.5)
        tax.set_axisbelow(True)
        tax.text(0.02, 1.04, "Number at risk", transform=tax.transAxes,
                 fontsize=7.2, fontweight="bold", ha="left", va="bottom")
        for i, (_label, counts, color) in enumerate(rows):
            for xx, count in zip(ticks, counts):
                tax.text(xx, y_positions[i], str(count), fontsize=7.0,
                         ha="center", va="center", color=color)
        return ax, tax

    # -------------------------------------------------------------------------
    # FIGURES 5-6: composite DFS Kaplan-Meier plots with aligned risk-table axes
    # -------------------------------------------------------------------------
    print("[FIGURE 5/6] Creating composite DFS KM figures ...", flush=True)
    common_xmax = float(np.ceil(surv["time_months"].max() / 12.0) * 12.0)
    common_xmax = max(12.0, common_xmax)

    # Figure 5: pooled cohort, pathological LVI and model-predicted risk groups.
    fig5 = plt.figure(figsize=(10.8, 6.2))
    outer5 = fig5.add_gridspec(
        1, 2, left=0.105, right=0.985, top=0.88, bottom=0.12, wspace=0.42
    )
    draw_km_panel_with_separate_risk_table(
        fig5, outer5[0], surv, "LVI_actual", [0, 1],
        ["LVI (-)", "LVI (+)"],
        "A: DFS by Pathological LVI Status", "A",
        x_max=common_xmax, y_label=True, risk_axis_label="Pathology",
        curve_colors=["#2586c8", "#ed1c24"]
    )
    draw_km_panel_with_separate_risk_table(
        fig5, outer5[1], surv, "M3_predicted_LVI_group", [0, 1],
        ["Low Risk (Pred)", "High Risk (Pred)"],
        "B: DFS by Model-Predicted Risk Group", "B",
        x_max=common_xmax, y_label=True, risk_axis_label="Model Prediction",
        curve_colors=["#20a52b", "#ff7900"]
    )
    fig5_png = out / "Figure5_DFS_overall.png"
    fig5.savefig(fig5_png, dpi=300, bbox_inches="tight")
    save_tiff_publication_dpi(fig5, out / "Figure5_DFS_overall.tiff")
    print(f"[FIGURE 5/6] Saved: {fig5_png}", flush=True)

    # Figure 6: M3-predicted risk groups within postoperative N0 and N+ strata.
    fig6 = plt.figure(figsize=(10.8, 6.2))
    outer6 = fig6.add_gridspec(
        1, 2, left=0.105, right=0.985, top=0.88, bottom=0.12, wspace=0.42
    )
    # Panel A (N0): blue/red palette, matching Figure 5A.
    z_n0 = n_valid[n_valid["N_binary"] == 0].copy()
    draw_km_panel_with_separate_risk_table(
        fig6, outer6[0], z_n0, "M3_predicted_LVI_group", [0, 1],
        ["Low Risk (Pred)", "High Risk (Pred)"],
        "A: DFS in N0 (Node-negative)", "A",
        x_max=common_xmax, y_label=True,
        risk_axis_label="Model Prediction",
        curve_colors=["#2586c8", "#ed1c24"]
    )

    # Panel B (N+): green/yellow-orange palette, matching Figure 5B.
    z_nplus = n_valid[n_valid["N_binary"] == 1].copy()
    draw_km_panel_with_separate_risk_table(
        fig6, outer6[1], z_nplus, "M3_predicted_LVI_group", [0, 1],
        ["Low Risk (Pred)", "High Risk (Pred)"],
        "B: DFS in N+ (Node-positive)", "B",
        x_max=common_xmax, y_label=True,
        risk_axis_label="Model Prediction",
        curve_colors=["#20a52b", "#ff7900"]
    )
    fig6_png = out / "Figure6_DFS_by_N_strata.png"
    fig6.savefig(fig6_png, dpi=300, bbox_inches="tight")
    save_tiff_publication_dpi(fig6, out / "Figure6_DFS_by_N_strata.tiff")
    print(f"[FIGURE 5/6] Saved: {fig6_png}", flush=True)

    # Exploratory single-predictor Cox analyses only; no N-adjusted or interaction model.
    cox_rows = [
        _cox_one_predictor(surv, "LVI_actual", "Pathological LVI (positive vs negative)"),
        _cox_one_predictor(surv, "M3_probability_per_0_1", "M3 predicted LVI probability (per 0.1 increase)"),
        _cox_one_predictor(surv, "LVI_actual", "Pathological LVI; center-stratified", stratify_center=True),
        _cox_one_predictor(surv, "M3_probability_per_0_1", "M3 probability per 0.1; center-stratified", stratify_center=True)
    ]
    pd.DataFrame(cox_rows).to_csv(out / "DFS_exploratory_Cox_models.csv", index=False)
    cindex_rows = []
    for marker, label in [("LVI_actual", "Pathological LVI"),
                          ("M3_predicted_LVI_probability", "M3 predicted LVI probability")]:
        point, lo, hi = _bootstrap_cindex(surv.time_months, surv.event, surv[marker])
        cindex_rows.append({"marker": label, "n": len(surv), "events": int(surv.event.sum()),
            "Harrell_C_index": point, "bootstrap_95CI_low": lo, "bootstrap_95CI_high": hi,
            "note": "Exploratory descriptive comparison; no superiority claim"})
    pd.DataFrame(cindex_rows).to_csv(out / "DFS_exploratory_C_index.csv", index=False)

    bsum = surv[surv.center == "Center_2"]
    pd.DataFrame([{"cohort": "Center_2", "n_with_valid_DFS": len(bsum),
        "DFS_events": int(bsum.event.sum()), "standalone_Cox_performed": False,
        "reason": "Only two DFS events; insufficient for reliable standalone survival modelling"}]).to_csv(
            out / "Center2_DFS_limitation.csv", index=False)
    print(f"DFS analysis completed: combined valid n={len(surv)}, events={int(surv.event.sum())}.")
    print(f"M3 predicted-LVI KM cutoff: Center 1 OOF median = {predicted_group_cutoff:.6f}; applied unchanged to Center 2.")
    print("All KM figures now include an at-risk table below the survival curves.")
    print("N is postoperative pathology: excluded from LVI prediction models; used only for exploratory DFS subgroup analyses.")
    print("Only 16 events: interpret subgroup KM/log-rank and Cox estimates cautiously; no interaction or multivariable Cox model.")
    return surv



# =============================================================================
# 6B. READER AGREEMENT: 622 CASES, TWO INDEPENDENT READERS + SENIOR ADJUDICATION
# =============================================================================
AGREEMENT_CONTINUOUS = ["Diameter", "ADC ratio"]
AGREEMENT_CATEGORICAL = [
    "Shape", "Lobulated", "Angular", "Spiculated",
    "FBE", "AVS", "BPS", "TIC", "Internal enhancement"
]
AGREEMENT_BOOTSTRAP = 5000
AGREEMENT_SEED = 2026


def _icc21_absolute_agreement(x, y):
    """ICC(2,1): two-way random-effects, single-measure, absolute agreement."""
    arr = np.column_stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
    n, k = arr.shape
    if n < 2 or k != 2:
        return np.nan
    grand = arr.mean()
    row_means = arr.mean(axis=1)
    col_means = arr.mean(axis=0)
    ms_rows = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_cols = n * np.sum((col_means - grand) ** 2) / (k - 1)
    residual = arr - row_means[:, None] - col_means[None, :] + grand
    ms_error = np.sum(residual ** 2) / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if denominator else np.nan


def _fast_unweighted_kappa(x, y):
    """Unweighted Cohen kappa; fast contingency-table implementation."""
    x = np.asarray(x).astype(str)
    y = np.asarray(y).astype(str)
    _, codes = np.unique(np.concatenate((x, y)), return_inverse=True)
    m = int(codes.max()) + 1
    a, b = codes[:len(x)], codes[len(x):]
    table = np.bincount(a * m + b, minlength=m * m).reshape(m, m)
    n = len(x)
    observed = np.trace(table) / n
    expected = (table.sum(axis=0) @ table.sum(axis=1)) / (n * n)
    return float((observed - expected) / (1 - expected)) if expected < 1 else np.nan


def run_reader_agreement_analysis():
    """
    Corrected reader mapping:
      - no suffix: Reader 1
      - suffix '1': Reader 2
      - suffix '2': senior adjudicator, excluded from agreement statistics
    All 622 cases are used; no sampling and no intrareader analysis.
    """
    out = OUT_DIR / "06_Reader_Agreement"
    out.mkdir(parents=True, exist_ok=True)
    if not Path(AGREEMENT_FILE).exists():
        raise FileNotFoundError(
            f"Cannot find {AGREEMENT_FILE}. Place 'agreement data.xlsx' beside this script."
        )
    raw = pd.read_excel(AGREEMENT_FILE, sheet_name="Sheet1")
    raw.columns = [str(c).strip() for c in raw.columns]
    variables = [(v, "continuous") for v in AGREEMENT_CONTINUOUS] + [
        (v, "categorical") for v in AGREEMENT_CATEGORICAL
    ]
    required = []
    for variable, _kind in variables:
        required.extend([variable, f"{variable}1", f"{variable}2"])
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise KeyError(f"agreement data.xlsx is missing expected columns: {missing}")

    rng = np.random.default_rng(AGREEMENT_SEED)
    summary_rows, audit_rows, bootstrap_rows = [], [], []

    for variable, kind in variables:
        # ONLY Reader 1 and Reader 2 enter interreader agreement.
        pair = pd.DataFrame({
            "Reader1": raw[variable],
            "Reader2": raw[f"{variable}1"]
        }).replace(r"^\\s*$", np.nan, regex=True).dropna()
        if kind == "continuous":
            pair["Reader1"] = pd.to_numeric(pair["Reader1"], errors="coerce")
            pair["Reader2"] = pd.to_numeric(pair["Reader2"], errors="coerce")
            pair = pair.dropna()
        pair = pair.reset_index(drop=True)
        x, y, n = pair["Reader1"].to_numpy(), pair["Reader2"].to_numpy(), len(pair)

        if kind == "continuous":
            estimate = _icc21_absolute_agreement(x, y) if n >= 2 else np.nan
            exact = np.nan
            method = "ICC(2,1), absolute agreement"
        else:
            estimate = _fast_unweighted_kappa(x, y) if n else np.nan
            exact = float(np.mean(x.astype(str) == y.astype(str))) if n else np.nan
            method = "Unweighted Cohen's kappa"

        boot_values = []
        for iteration in range(AGREEMENT_BOOTSTRAP):
            if n == 0:
                break
            index = rng.integers(0, n, size=n)
            xb, yb = x[index], y[index]
            value = (_icc21_absolute_agreement(xb, yb)
                     if kind == "continuous"
                     else _fast_unweighted_kappa(xb, yb))
            if np.isfinite(value):
                boot_values.append(float(value))
        ci_low, ci_high = (
            np.quantile(boot_values, [0.025, 0.975])
            if boot_values else (np.nan, np.nan)
        )
        summary_rows.append({
            "Variable": variable,
            "Variable_type": kind,
            "Reader_1_column": variable,
            "Reader_2_column": f"{variable}1",
            "Senior_adjudicator_column_excluded": f"{variable}2",
            "N_paired": n,
            "Agreement_statistic": method,
            "Estimate": estimate,
            "95% CI lower": ci_low,
            "95% CI upper": ci_high,
            "Exact_agreement": exact,
            "Bootstrap_replicates_valid": len(boot_values)
        })
        audit_rows.append({
            "Variable": variable,
            "Reader_1_column": variable,
            "Reader_2_column": f"{variable}1",
            "Senior_adjudicator_column_excluded": f"{variable}2",
            "N_total": len(raw),
            "N_complete_R1_R2": n,
            "N_missing_or_invalid_pair": len(raw) - n
        })
        bootstrap_rows.extend({
            "Variable": variable,
            "Bootstrap_iteration_valid_order": i + 1,
            "Estimate": value
        } for i, value in enumerate(boot_values))

    summary = pd.DataFrame(summary_rows)
    audit = pd.DataFrame(audit_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    summary.to_csv(out / "Interreader_agreement_summary_corrected.csv",
                   index=False, encoding="utf-8-sig")
    audit.to_csv(out / "Reader_pair_data_audit_corrected.csv",
                 index=False, encoding="utf-8-sig")
    bootstrap.to_csv(out / "Bootstrap_replicates_corrected.csv",
                     index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out / "Interreader_agreement_summary_corrected.xlsx",
                        engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Agreement_summary", index=False)
        audit.to_excel(writer, sheet_name="Data_audit", index=False)
        pair_data = pd.DataFrame({"case_row_excel": np.arange(2, len(raw) + 2)})
        for variable, _kind in variables:
            pair_data[f"{variable}_Reader1"] = raw[variable]
            pair_data[f"{variable}_Reader2"] = raw[f"{variable}1"]
        pair_data.to_excel(writer, sheet_name="Reader1_vs_Reader2_data", index=False)

    with open(out / "READ_ME_reader_agreement.txt", "w", encoding="utf-8") as handle:
        handle.write(
            "Reader mapping: unsuffixed columns = Reader 1; suffix '1' = Reader 2; "
            "suffix '2' = senior adjudicator. Agreement statistics use only Reader 1 "
            "and Reader 2 ratings across all available paired cases. Senior adjudication "
            "is excluded because it is not an independent third reading. Continuous "
            "variables use ICC(2,1) absolute agreement; categorical variables use "
            "unweighted Cohen kappa. 95% CIs use 5,000 paired case-level bootstrap "
            "resamples. No intrareader agreement is calculated because repeat reads "
            "are not available."
        )
    print("[READER AGREEMENT] Completed corrected Reader 1 vs Reader 2 analysis.")
    print(summary.to_string(index=False))


# =============================================================================
# 7. MAIN
# =============================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a_raw = read_data(BASE_DIR / CENTER1_FILE)
    b_raw = read_data(BASE_DIR / CENTER2_FILE)
    a, ya = prepare_data(a_raw, "Center 1")
    b, yb = prepare_data(b_raw, "Center 2")

    if ENFORCE_EXPECTED_COHORT_COUNTS:
        observed = {"Center 1": (len(a), int(ya.sum())), "Center 2": (len(b), int(yb.sum()))}
        if observed != EXPECTED_COHORT_COUNTS:
            raise ValueError(
                "Dataset version/coding mismatch. Expected confirmed cohort counts "
                f"{EXPECTED_COHORT_COUNTS}, but loaded {observed}. Do not proceed with modeling; "
                "select the corrected Data1.xlsx/Data2.xlsx used for the current Publication_Tables workbook."
            )

    print(f"Center 1: n={len(a)}, LVI+={int(ya.sum())}, prevalence={ya.mean():.4f}")
    print(f"Center 2: n={len(b)}, LVI+={int(yb.sum())}, prevalence={yb.mean():.4f}")

    internal_predictions, external_predictions = {}, {}
    performance, fold_rows, locked_rows = [], [], []
    fitted_models = {}

    for model_name, features in MODEL_FEATURES.items():
        print(f"\n===== {model_name} =====")
        Xa, Xb = a[features].copy(), b[features].copy()

        # Center 1 nested-CV OOF predictions
        oof, fold_info = nested_cv(Xa, ya, features, model_name)
        fold_rows.extend(fold_info)
        internal_predictions[model_name] = oof
        performance.append(metric_row(ya, oof, "Center 1 nested-CV OOF", model_name))
        pd.DataFrame({"observed_LVI": ya, "OOF_probability": oof}).to_csv(
            OUT_DIR / f"OOF_predictions_{model_name}.csv", index=False)

        # Final model fitted using Center 1 only; apply to B once
        frozen, best_params, inner_auc = fit_locked_model(Xa, ya, features)
        fitted_models[model_name] = frozen
        p_b = frozen.predict_proba(Xb)[:, 1]
        external_predictions[model_name] = p_b
        performance.append(metric_row(yb, p_b, "Center 2 locked external", model_name))
        joblib.dump(frozen, OUT_DIR / f"locked_model_{model_name}.joblib")
        pd.DataFrame({"observed_LVI": yb, "predicted_probability": p_b}).to_csv(
            OUT_DIR / f"external_predictions_{model_name}.csv", index=False)
        gc.collect()
        locked_rows.append({
            "model": model_name, "features": json.dumps(features, ensure_ascii=False),
            "selected_params_Center1_only": json.dumps(best_params, ensure_ascii=False),
            "inner_CV_best_AUC_Center1": inner_auc,
            "Center2_used_for_tuning_or_selection": False
        })

    perf_df = pd.DataFrame(performance)
    perf_df.to_csv(OUT_DIR / "performance_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "nested_CV_fold_details.csv", index=False)
    pd.DataFrame(locked_rows).to_csv(OUT_DIR / "locked_model_specifications.csv", index=False)
    pd.DataFrame({"observed_LVI": yb, **{
        f"{k}_predicted_probability": v for k, v in external_predictions.items()
    }}).to_csv(OUT_DIR / "all_external_predictions.csv", index=False)

    # Publication Figures 2-4: paired internal/external panels.
    # Center 1 uses nested-CV OOF predictions; Center 2 uses locked A-trained models.
    print("[FIGURE 1] Exporting corrected two-center patient flowchart ...", flush=True)
    plot_patient_flowchart(OUT_DIR / "Figure1_Patient_flowchart.tiff")

    print("[FIGURE 2/6] Exporting paired ROC TIFF ...", flush=True)
    plot_roc_panels(ya, internal_predictions, yb, external_predictions,
                    OUT_DIR / "Figure2_ROC_internal_external.tiff")
    print("[FIGURE 3/6] Exporting paired calibration TIFF ...", flush=True)
    plot_calibration_panels(ya, internal_predictions, yb, external_predictions,
                            OUT_DIR / "Figure3_Calibration_internal_external.tiff")
    print("[FIGURE 4/6] Exporting paired DCA TIFF ...", flush=True)
    plot_dca_panels(ya, internal_predictions, yb, external_predictions,
                    OUT_DIR / "Figure4_DCA_internal_external.tiff")

    print("[FIGURE S1] Exporting M3 probability-distribution TIFF ...", flush=True)
    # Supplementary Figure S1: distribution of M3 predicted probabilities by observed
    # pathological LVI status, shown separately for internal and external cohorts.
    fig_s1, axs1 = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
    for ax, y, pred, title, tag in [
        (axs1[0], ya, internal_predictions[PRIMARY_MODEL], "Internal cohort", "A"),
        (axs1[1], yb, external_predictions[PRIMARY_MODEL], "External cohort", "B"),
    ]:
        values = [np.asarray(pred)[np.asarray(y) == 0], np.asarray(pred)[np.asarray(y) == 1]]
        violin = ax.violinplot(values, positions=[0, 1], widths=0.72,
                               showmeans=False, showmedians=True, showextrema=False)
        for body in violin["bodies"]:
            body.set_alpha(0.28)
        for xpos, vals in enumerate(values):
            if len(vals):
                ax.boxplot(vals, positions=[xpos], widths=0.22, showfliers=False,
                           medianprops={"linewidth": 1.5})
        ax.set_xticks([0, 1], ["LVI negative", "LVI positive"])
        ax.set_title(title, pad=10)
        ax.set_xlabel("Observed pathological LVI status")
        ax.set_ylabel("M3-predicted probability of LVI")
        ax.set_ylim(0, 1)
        _panel_tag(ax, tag)
    fig_s1.tight_layout(w_pad=2.0)
    save_tiff_publication_dpi(fig_s1, OUT_DIR / "FigureS1_M3_probability_distribution.tiff")

    # Main Table 1: direct comparison between Center 1 and Center 2.
    # Continuous variables are displayed as mean ± SD; Welch's t-test is used.
    t1 = between_center_table(a, ya, b, yb)

    # Supplementary Table S1: within-center LVI-negative vs LVI-positive comparisons.
    s1a = within_center_lvi_table(a, ya, "Center 1")
    s1b = within_center_lvi_table(b, yb, "Center 2")
    s1_within = pd.concat([s1a, s1b], ignore_index=True)

    # Table 2: model performance.
    t2 = perf_df.copy()
    t2["LVI events, n (%)"] = t2.apply(
        lambda r: f'{int(r["events"])} ({100 * r["prevalence"]:.1f}%)', axis=1)
    t2["AUC (95% CI)"] = t2.apply(
        lambda r: f'{r["AUC"]:.3f} ({r["AUC_95CI_low"]:.3f}-{r["AUC_95CI_high"]:.3f})', axis=1)
    t2["PR-AUC"] = t2["PR_AUC_average_precision"].map(lambda x: f"{x:.3f}")
    t2["Brier score"] = t2["Brier"].map(lambda x: f"{x:.3f}")
    t2["Calibration intercept"] = t2["calibration_intercept"].map(lambda x: f"{x:.3f}")
    t2["Calibration slope"] = t2["calibration_slope"].map(lambda x: f"{x:.3f}")
    t2 = t2.rename(columns={"cohort": "Cohort", "model": "Model", "n": "N"})
    t2 = t2[["Cohort", "Model", "N", "LVI events, n (%)", "AUC (95% CI)",
             "PR-AUC", "Brier score", "Calibration intercept", "Calibration slope"]]

    # Table 3 and Supplementary Table S4: paired bootstrap AUC differences and replicates.
    bootstrap_replicates = []
    t3a = make_auc_comparison(ya, internal_predictions, "Center 1 nested-CV OOF", bootstrap_replicates)
    t3b = make_auc_comparison(yb, external_predictions, "Center 2 external", bootstrap_replicates)
    t3 = pd.concat([t3a, t3b], ignore_index=True)
    s4_bootstrap = pd.DataFrame(bootstrap_replicates)

    # Table 4: penalized coefficients/ORs, descriptive only (no Wald P values or CIs).
    t4 = coefficient_table(fitted_models)

    # Supplementary Table S2: nested-CV fold results and locked Center-1-only specifications.
    s3_folds = pd.DataFrame(fold_rows)
    s3_specs = pd.DataFrame(locked_rows)

    # Supplementary Table S3: M3 threshold grid. Provenance is explicit in code:
    # A = nested-CV OOF M3 probabilities; B = locked A-trained M3 probabilities.
    tgrid_a = threshold_analysis(ya, internal_predictions[PRIMARY_MODEL], "Center 1 nested-CV OOF")
    tgrid_b = threshold_analysis(yb, external_predictions[PRIMARY_MODEL], "Center 2 external validation")
    tgrid = pd.concat([tgrid_a, tgrid_b], ignore_index=True)
    tgrid.insert(0, "model", PRIMARY_MODEL)
    tgrid.to_csv(OUT_DIR / "threshold_analysis_M3.csv", index=False)

    # If a clinical threshold was prespecified independently, provide its locked metrics.
    threshold_locked = pd.DataFrame()
    if FIXED_THRESHOLD is not None:
        threshold_locked = pd.DataFrame([
            {"model": PRIMARY_MODEL, "cohort": "Center 1 nested-CV OOF", **threshold_metrics(
                ya, internal_predictions[PRIMARY_MODEL], FIXED_THRESHOLD)},
            {"model": PRIMARY_MODEL, "cohort": "Center 2 external validation", **threshold_metrics(
                yb, external_predictions[PRIMARY_MODEL], FIXED_THRESHOLD)}
        ])
        threshold_locked.to_csv(OUT_DIR / "prespecified_threshold_metrics_M3.csv", index=False)

    # Run the two-junior-reader reproducibility workflow before constructing Table S5.
    # Senior adjudication is not included as an independent reader.
    print("[READER AGREEMENT] Running corrected 622-case Reader 1 vs Reader 2 analysis ...", flush=True)
    run_reader_agreement_analysis()

    # Supplementary Table S5: corrected 622-case interreader agreement only.
    # Intrareader agreement is not calculated because no repeat-read data exist.
    s5_interobserver = interobserver_agreement_table()

    # Publication table map and statistical footnotes.
    table_notes = pd.DataFrame([
        {"Table": "Table 1", "Description": "Between-center comparison (Center 1 vs Center 2).",
         "Statistical_note": "Continuous variables are presented as mean ± SD and compared using Welch's t-test. Categorical variables are presented as n (%) and compared using Pearson chi-square or Fisher exact test, as appropriate; P values are omnibus. Postoperative nodal status (N) is descriptive only and is not included in LVI prediction models."},
        {"Table": "Table S1", "Description": "Within-center comparisons of LVI-negative and LVI-positive patients, shown separately for each center.",
         "Statistical_note": "Continuous variables are presented as mean ± SD and compared using Welch's t-test. Categorical variables are presented as n (%) and compared using Pearson chi-square or Fisher exact test, as appropriate. P values are unadjusted."},
        {"Table": "Table 2", "Description": "Performance of the four LVI prediction models in Center 1 and Center 2.",
         "Statistical_note": "Center 1 performance is estimated from nested-CV out-of-fold predictions. Center 2 is a locked external validation cohort and was not used for tuning, selection, threshold optimization, or recalibration."},
        {"Table": "Table 3", "Description": "Paired bootstrap differences in AUC between models.",
         "Statistical_note": "Paired patient-level bootstrap percentile 95% confidence intervals; no bootstrap P values."},
        {"Table": "Table 4", "Description": "Penalized logistic regression coefficients and exponentiated coefficients (ORs).",
         "Statistical_note": "Descriptive penalized-model coefficients; continuous variables were standardized. No Wald P values or conventional confidence intervals are provided."},
        {"Table": "Table S2A", "Description": "Fold-specific nested-CV results and selected hyperparameters across the five outer folds.",
         "Statistical_note": "Center 1 only; Center 2 was not used for tuning or selection."},
        {"Table": "Table S2B", "Description": "Final locked model specifications selected using Center 1 only.",
         "Statistical_note": "Center 2 is reserved for locked external validation."},
        {"Table": "Table S3", "Description": "M3 threshold grid for Center 1 nested-CV OOF and Center 2 locked external predictions.",
         "Statistical_note": "Thresholds 0.05-0.80 in 0.01 increments; descriptive only. No optimal or clinical cutoff is selected. Net benefit = TP/n - FP/n × threshold/(1-threshold)."},
        {"Table": "Table S4", "Description": "Paired patient-level bootstrap replicates for the AUC differences in Table 3.",
         "Statistical_note": "2,000 requested replicates per comparison; percentile 95% CI; no bootstrap P values."},
        {"Table": "Table S5", "Description": "Interreader agreement between Reader 1 (unsuffixed columns) and Reader 2 (suffix 1) across 622 cases.",
         "Statistical_note": "ICC(2,1) absolute agreement for continuous variables; unweighted Cohen's kappa for categorical variables; 95% CIs from 5,000 paired case-level bootstrap resamples. Senior adjudicator (suffix 2) excluded; no intrareader analysis."},
    ])

    # Write publication workbook with corrected numbering and clear table provenance.
    workbook = OUT_DIR / "Publication_Tables_Final.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        t1.to_excel(writer, sheet_name="Table1_Center_Comparison", index=False)
        s1_within.to_excel(writer, sheet_name="S1_Within_Center_LVI", index=False)
        s1a.to_excel(writer, sheet_name="S1_Center1", index=False)
        s1b.to_excel(writer, sheet_name="S1_Center2", index=False)
        t2.to_excel(writer, sheet_name="Table2_Performance", index=False)
        t3.to_excel(writer, sheet_name="Table3_AUC_Differences", index=False)
        t4.to_excel(writer, sheet_name="Table4_Coefficients_OR", index=False)
        s3_folds.to_excel(writer, sheet_name="S2A_CV_Folds", index=False)
        s3_specs.to_excel(writer, sheet_name="S2B_Locked_Specs", index=False)
        tgrid.to_excel(writer, sheet_name="S3_M3_Threshold_Grid", index=False)
        s4_bootstrap.to_excel(writer, sheet_name="S4_AUC_Bootstrap_Reps", index=False)
        s5_interobserver.to_excel(writer, sheet_name="S5_Interobserver", index=False)
        table_notes.to_excel(writer, sheet_name="Table_Notes", index=False)
        if not threshold_locked.empty:
            threshold_locked.to_excel(writer, sheet_name="S4_Locked_Threshold", index=False)

    # Individual CSV files are exported in manuscript-ready table layouts.
    for name, frame in {
        "Table1_Center_Comparison": t1,
        "TableS1_Within_Center_LVI": s1_within,
        "TableS1_Center1": s1a,
        "TableS1_Center2": s1b,
        "Table2_Performance": t2,
        "Table3_AUC_Differences": t3,
        "Table4_Coefficients_OR": t4,
        "TableS2A_CV_Folds": s3_folds,
        "TableS2B_Locked_Specs": s3_specs,
        "TableS3_M3_Threshold_Grid": tgrid,
        "TableS4_AUC_Bootstrap_Replicates": s4_bootstrap,
        "TableS5_Interobserver_Agreement": s5_interobserver,
        "Table_Notes": table_notes,
    }.items():
        frame.to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    # Exploratory DFS prognosis analysis: Center 1 uses OOF M3 predictions;
    # Center 2 uses locked M3 predictions from the Center-1-only model.
    print("[FIGURE 5-6] Running exploratory DFS analysis and exporting composite TIFFs ...", flush=True)
    run_dfs_analysis(a_raw, b_raw, internal_predictions[PRIMARY_MODEL],
                     external_predictions[PRIMARY_MODEL])


    config = {
        "development_cohort": CENTER1_FILE,
        "external_validation_cohort": CENTER2_FILE,
        "outcome": TARGET, "positive_class": 1,
        "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
        "seed": RANDOM_SEED, "model_features": MODEL_FEATURES,
        "memory_safe_plotting": {"GridSearchCV_n_jobs": N_JOBS, "individual_DFS_KM_exports": SAVE_INDIVIDUAL_DFS_KM, "matplotlib_backend": "Agg", "tiff_dpi": 600},
        "primary_model": PRIMARY_MODEL,
        "sensitivity_model": None,
        "missing_data_statement": "The study team confirmed that all variables included in the descriptive tables and LVI prediction models had no missing values; therefore, no missingness table is generated.",
        "Table1": "Between-center comparison; continuous variables mean ± SD with Welch t-test.",
        "TableS1": "Within-center LVI-negative vs LVI-positive comparisons.",
        "TableS5": "622-case interreader agreement: unsuffixed columns=Reader 1; suffix 1=Reader 2; suffix 2=senior adjudicator excluded. ICC(2,1) for continuous variables; unweighted Cohen kappa for categorical variables; 5,000 paired bootstrap 95% CIs; no intrareader analysis.",
         "TableS3": "M3 only; Center 1 nested-CV OOF probabilities and Center 2 locked external probabilities; thresholds 0.05-0.80 by 0.01; descriptive only; no optimized clinical cutoff.",
        "DFS_KM_y_axis": "Fixed display y-axis from 0.70 to 1.02 for Figures 5-6 to match the supplied reference layout; the axis starts above zero and this must be disclosed in the figure legend.",
        "DFS_KM_risk_table": "Risk-table axis uses exactly the same left/right plotting boundaries as the KM curve above, with a small shared blank gutter before month 0; x=0, 12, 24... coordinates align vertically and group labels are separated from the first risk-count column. Group labels are y tick labels, not a separate plotting column.",
        "publication_figures": {"Figure1": "Corrected two-center patient flowchart; Center 1 enrollment through December 2025.", "Figure2": "Paired ROC; A=Internal cohort, B=External cohort; no no-discrimination line; 600-dpi TIFF.", "Figure3": "Paired calibration; ideal-calibration line omitted; 600-dpi TIFF.", "Figure4": "Paired DCA; Treat none and Treat all retained; 600-dpi TIFF.", "Figure5": "Paired DFS KM: pathological LVI and M3-predicted group; 600-dpi TIFF.", "Figure6": "Paired DFS KM by N0 and N+ strata; 600-dpi TIFF.", "FigureS1": "M3 predicted-probability distributions by observed LVI status; 600-dpi TIFF."},
        "one_patient_one_lesion": "not verified by N; confirm source data; current CV is row-level",
        "fixed_threshold": FIXED_THRESHOLD,
        "DCA_xlim": DCA_XLIM, "DCA_ylim": DCA_YLIM,
        "DFS_time_col": DFS_TIME_COL,
        "DFS_event_col": DFS_EVENT_COL,
        "DFS_time_unit": DFS_TIME_UNIT,
        "postoperative_nodal_status_col": N_COL,
        "N_coding": {"0": "node-negative (N0)", "1": "node-positive (N+)"},
        "N_in_LVI_prediction_features": False,
        "N_DFS_analysis": "Exploratory pooled nodal-status KM/log-rank and pathological-LVI KM/log-rank within N0/N+ strata; no Cox adjustment or interaction due to 16 total events",
        "DFS_days_per_month": DAYS_PER_MONTH,
        "DFS_events_reported_by_user": {"Center_1": 14, "Center_2": 2},
        "DFS_method": "Exploratory pooled KM, log-rank, one-predictor Cox and descriptive C-index; A uses OOF M3 probabilities, B uses locked M3 probabilities",
        "DFS_multivariable_Cox": False,
        "DFS_B_standalone_Cox": False
    }
    with open(OUT_DIR / "analysis_configuration.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\nAnalysis completed.")
    print(f"Outputs: {OUT_DIR}")
    print(f"Publication tables workbook: {workbook}")
    print("Table 1 compares Center 1 vs Center 2; Table S1 contains within-center LVI comparisons.")
    print("Table 1, Table S1 and Table 2 are exported in manuscript-ready layouts without missing-value columns.")
    print("All variables in the descriptive tables and LVI prediction models were confirmed complete; report this in the manuscript text.")
    print("Table S3 is confirmed as M3 threshold analysis: A nested-CV OOF and B locked external predictions; 0.05-0.80 by 0.01.")
    print("Do not select a model or threshold based on Center 2. Only M1-M3 are fitted; Only the prespecified M1–M3 model sequence is used.")
    print("Table 4 ORs are penalized-model estimates; do not attach naive Wald P-values/CIs.")


if __name__ == "__main__":
    # Limit BLAS/OpenMP threads as well as GridSearchCV workers to reduce peak RAM.
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            main()
    except ImportError:
        main()
