"""Statistical analysis for RQ3 — signal value, confidence calibration, predictive model.

Public API
----------
calibration_table(df, n_bins)         -> DataFrame  (reliability curve data)
confidence_correlation(df)            -> dict        (r, p, 95% CI vs both outcomes)
build_feature_matrix(df)              -> (DataFrame, ndarray, list[str])
run_logistic_regression(X, y, names)  -> dict        (AUC, coefs, odds_ratios)
run_gradient_boosting(X, y, names)    -> dict        (AUC, importances)
compute_rq3(df)                       -> dict        (all RQ3 results)

Design notes
------------
- Features are tweet-time signals ONLY: sentiment, time_horizon, trade_type (dummies),
  confidence, and log1p-transformed engagement counts.  No account-level accuracy
  history is included (that would leak the answer from RQ2).
- LR uses StandardScaler so coefficients are standardised (comparability across scales).
- GB uses raw features; tree models are scale-invariant.
- Model evaluation: StratifiedKFold (k=5), metric = AUC-ROC.
- Calibration: equal-width 10-point bins from 0–100 on the confidence score;
  only non-empty bins are reported.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().eq("true")


def _pearson_ci(r: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """95% CI on Pearson r via Fisher z-transform."""
    if abs(r) >= 1.0 or n <= 3:
        return (float("nan"), float("nan"))
    z     = np.arctanh(r)
    se    = 1.0 / np.sqrt(n - 3)
    z_crit = scipy_stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    return (float(np.tanh(z - z_crit * se)), float(np.tanh(z + z_crit * se)))


# ---------------------------------------------------------------------------
# A) Confidence calibration
# ---------------------------------------------------------------------------

def calibration_table(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Reliability curve: per-bin confidence vs actual prediction_correct rate.

    Uses equal-width bins of width (100 / n_bins) over [0, 100].
    Only non-empty bins are returned.

    Columns: bin_lower, bin_upper, bin_mid, n_predictions, actual_accuracy,
             ci_low, ci_high  (95% Wilson CI on actual_accuracy).
    """
    from src.stats_rq1 import wilson_ci  # reuse Phase 4 Wilson CI

    val = df[_as_bool(df["validated"])].copy()
    val["correct"] = _as_bool(val["prediction_correct"])
    conf = pd.to_numeric(val["confidence"], errors="coerce")

    bin_edges = np.linspace(0.0, 100.0, n_bins + 1)
    rows = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (conf >= lo) & (conf < hi)
        # Include upper edge in last bin
        if hi == 100.0:
            mask = (conf >= lo) & (conf <= hi)
        sub = val[mask]
        n = len(sub)
        if n == 0:
            continue
        k = int(sub["correct"].sum())
        ci_lo, ci_hi = wilson_ci(k, n)
        rows.append({
            "bin_lower":        float(lo),
            "bin_upper":        float(hi),
            "bin_mid":          float((lo + hi) / 2.0),
            "n_predictions":    n,
            "actual_accuracy":  k / n,
            "ci_low":           ci_lo,
            "ci_high":          ci_hi,
        })
    return pd.DataFrame(rows)


def confidence_correlation(df: pd.DataFrame) -> dict:
    """Point-biserial correlation between confidence and both binary outcomes.

    Returns dict with keys:
        r_correct, p_correct, ci_correct_low, ci_correct_high,
        r_beats,   p_beats,   ci_beats_low,   ci_beats_high, n
    """
    val = df[_as_bool(df["validated"])].copy()
    conf    = pd.to_numeric(val["confidence"], errors="coerce")
    correct = _as_bool(val["prediction_correct"]).astype(float)
    beats   = _as_bool(val["beats_market"]).astype(float)

    # Drop any rows where confidence is NaN
    valid = conf.notna()
    conf    = conf[valid].values
    correct = correct[valid].values
    beats   = beats[valid].values
    n = len(conf)

    r_c, p_c = scipy_stats.pearsonr(conf, correct)
    r_b, p_b = scipy_stats.pearsonr(conf, beats)

    ci_c = _pearson_ci(r_c, n)
    ci_b = _pearson_ci(r_b, n)

    return {
        "r_correct":      float(r_c), "p_correct":      float(p_c),
        "ci_correct_low": ci_c[0],    "ci_correct_high": ci_c[1],
        "r_beats":        float(r_b), "p_beats":         float(p_b),
        "ci_beats_low":   ci_b[0],    "ci_beats_high":   ci_b[1],
        "n":              n,
    }


# ---------------------------------------------------------------------------
# B) Feature matrix (tweet-time signals only — NO account history)
# ---------------------------------------------------------------------------

# Reference categories (dropped dummies):
#   sentiment:    bearish (→ is_bullish = 1 means bullish)
#   time_horizon: short_term
#   trade_type:   analysis

CAT_DUMMIES = {
    # col_name: [(dummy_name, category_value), ...]
    "sentiment":    [("is_bullish",           "bullish")],
    "time_horizon": [("horizon_medium_term",  "medium_term"),
                     ("horizon_long_term",    "long_term"),
                     ("horizon_unknown",      "unknown")],
    "trade_type":   [("type_trade_suggestion","trade_suggestion"),
                     ("type_gen_discussion",  "general_discussion"),
                     ("type_news",            "news")],
}

ENG_COLS = ["likes", "retweets", "replies_count", "views", "author_followers"]


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Build feature matrix X and binary target y.

    Features (tweet-time only):
      - is_bullish              (1 = bullish, 0 = bearish)
      - horizon_medium_term, horizon_long_term, horizon_unknown
      - type_trade_suggestion, type_gen_discussion, type_news
      - confidence
      - log_likes, log_retweets, log_replies_count, log_views, log_author_followers

    Target y: prediction_correct (1 = correct).

    No account identifier or account-accuracy column is included.
    """
    val = df[_as_bool(df["validated"])].copy()

    feat: dict[str, pd.Series] = {}

    # Categorical dummies
    for col, pairs in CAT_DUMMIES.items():
        for dummy_name, cat_val in pairs:
            feat[dummy_name] = (val[col] == cat_val).astype(float)

    # Confidence (continuous)
    feat["confidence"] = pd.to_numeric(val["confidence"], errors="coerce").fillna(75.0)

    # Log-engagement (fill NA → 0 before log)
    for col in ENG_COLS:
        feat[f"log_{col}"] = np.log1p(
            pd.to_numeric(val[col], errors="coerce").fillna(0.0)
        )

    X = pd.DataFrame(feat, index=val.index)
    y = _as_bool(val["prediction_correct"]).astype(int).values
    feature_names = list(X.columns)

    return X, y, feature_names


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def _cv_auc(pipeline, X: pd.DataFrame, y: np.ndarray, cv_folds: int, seed: int) -> float:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    return float(scores.mean())


def run_logistic_regression(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    cv_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Fit LR (with StandardScaler) and return CV AUC, standardised coefs, odds ratios."""
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=seed),
    )
    auc = _cv_auc(pipe, X, y, cv_folds, seed)

    # Fit on full data to extract coefficients
    pipe.fit(X, y)
    coefs = pipe.named_steps["logisticregression"].coef_[0]

    coef_series = pd.Series(coefs,        index=feature_names).sort_values(key=abs, ascending=False)
    or_series   = pd.Series(np.exp(coefs), index=feature_names).sort_values(key=lambda s: abs(np.log(s)), ascending=False)

    return {
        "auc_cv":      auc,
        "coefs":       coef_series,
        "odds_ratios": or_series,
    }


def run_gradient_boosting(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    cv_folds: int = 5,
    seed: int = 42,
    n_estimators: int = 100,
) -> dict:
    """Fit GradientBoostingClassifier and return CV AUC + feature importances."""
    pipe = make_pipeline(
        GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=3,
            learning_rate=0.05, random_state=seed,
        )
    )
    auc = _cv_auc(pipe, X, y, cv_folds, seed)

    pipe.fit(X, y)
    importances = pipe.named_steps["gradientboostingclassifier"].feature_importances_
    imp_series  = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    return {
        "auc_cv":       auc,
        "importances":  imp_series,
    }


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def compute_rq3(
    df: pd.DataFrame,
    n_bins: int = 10,
    cv_folds: int = 5,
    seed: int = 42,
    n_estimators_gb: int = 100,
) -> dict:
    """Run the full RQ3 analysis.

    Returns dict with keys:
        calibration   (DataFrame),
        correlation   (dict),
        baseline_auc, baseline_accuracy,
        lr            (dict),
        gb            (dict),
        feature_names (list[str])
    """
    val = df[_as_bool(df["validated"])]
    y_all = _as_bool(val["prediction_correct"]).astype(int)
    baseline_acc = float(y_all.value_counts(normalize=True).max())

    calib = calibration_table(df, n_bins=n_bins)
    corr  = confidence_correlation(df)
    X, y, feature_names = build_feature_matrix(df)

    lr_res = run_logistic_regression(X, y, feature_names, cv_folds=cv_folds, seed=seed)
    gb_res = run_gradient_boosting(
        X, y, feature_names, cv_folds=cv_folds, seed=seed, n_estimators=n_estimators_gb
    )

    return {
        "calibration":        calib,
        "correlation":        corr,
        "baseline_auc":       0.5,
        "baseline_accuracy":  baseline_acc,
        "lr":                 lr_res,
        "gb":                 gb_res,
        "feature_names":      feature_names,
    }
