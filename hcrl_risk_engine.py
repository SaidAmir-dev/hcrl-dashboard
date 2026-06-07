"""Attrition risk estimation for HCRL.

The engine prefers observed outcomes and cross-validated out-of-fold probabilities.
If no outcome is supplied, it can only consume externally supplied probabilities; it does
not fabricate risk scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class RiskModelReport:
    method: str
    feature_columns: List[str]
    n_observations: int
    n_events: Optional[int]
    event_rate: Optional[float]
    auc_oof: Optional[float]
    brier_oof: Optional[float]
    log_loss_oof: Optional[float]
    warnings: List[str]


def _make_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])


def estimate_attrition_risk(
    df: pd.DataFrame,
    feature_columns: List[str],
    outcome_col: str = "separation_outcome",
    supplied_probability_col: str = "predicted_attrition_probability",
) -> Tuple[pd.DataFrame, RiskModelReport]:
    out = df.copy()
    warnings: List[str] = []

    if outcome_col not in out.columns:
        if supplied_probability_col not in out.columns:
            raise ValueError("No outcome or externally supplied probability column is available.")
        probs = pd.to_numeric(out[supplied_probability_col], errors="coerce")
        if probs.isna().any() or (probs < 0).any() or (probs > 1).any():
            raise ValueError("Externally supplied attrition probabilities must be complete and in [0, 1].")
        out["predicted_attrition_probability"] = probs
        return out, RiskModelReport(
            method="external_probability_input",
            feature_columns=[],
            n_observations=len(out),
            n_events=None,
            event_rate=None,
            auc_oof=None,
            brier_oof=None,
            log_loss_oof=None,
            warnings=["Risk probabilities were supplied by the user/company file; HCRL did not train a model."],
        )

    y = pd.to_numeric(out[outcome_col], errors="coerce")
    valid = y.isin([0, 1])
    if valid.sum() != len(out):
        warnings.append("Rows with missing/non-binary separation outcomes were excluded from model fitting.")
    model_df = out.loc[valid].copy()
    y = y.loc[valid].astype(int)

    if y.nunique() < 2:
        raise ValueError("Separation outcome must contain both events and non-events.")

    usable_features = [c for c in feature_columns if c in model_df.columns and model_df[c].nunique(dropna=True) > 1]
    if not usable_features:
        raise ValueError("No usable feature columns for attrition modeling.")

    X = model_df[usable_features]
    min_class_count = int(y.value_counts().min())
    n_splits = max(2, min(5, min_class_count))
    if min_class_count < 5:
        warnings.append(
            "Very few separation/non-separation cases exist. Cross-validated diagnostics may be unstable."
        )

    pipeline = _make_pipeline(X)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_probs = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]

    pipeline.fit(X, y)
    fitted_probs = pipeline.predict_proba(X)[:, 1]

    out["predicted_attrition_probability"] = np.nan
    out.loc[model_df.index, "predicted_attrition_probability"] = fitted_probs
    out["oof_attrition_probability"] = np.nan
    out.loc[model_df.index, "oof_attrition_probability"] = oof_probs

    try:
        auc = float(roc_auc_score(y, oof_probs))
    except Exception:
        auc = None

    report = RiskModelReport(
        method="cross_validated_logistic_regression",
        feature_columns=usable_features,
        n_observations=int(len(model_df)),
        n_events=int(y.sum()),
        event_rate=float(y.mean()),
        auc_oof=auc,
        brier_oof=float(brier_score_loss(y, oof_probs)),
        log_loss_oof=float(log_loss(y, oof_probs)),
        warnings=warnings,
    )
    return out, report
