"""HCRL attrition risk engine.

This module decides how attrition probabilities are produced.

Priority:
1. Train a company-specific model if historical separation outcomes exist.
2. Use precomputed probabilities only if supplied and valid.
3. Otherwise, mark risk as unavailable until the external labor-market baseline model is built.

HCRL must not fabricate attrition probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class RiskEngineReport:
    risk_source: str
    model_used: Optional[str]
    n_observations: int
    n_features: int
    warnings: List[str]
    errors: List[str]


def estimate_attrition_risk(
    df: pd.DataFrame,
    model_feature_columns: List[str],
) -> Tuple[pd.DataFrame, RiskEngineReport]:

    out = df.copy()
    warnings: List[str] = []
    errors: List[str] = []

    # Case 1: company has historical attrition outcomes
    if "separation_outcome" in out.columns:
        model_df = out.dropna(subset=["separation_outcome"]).copy()

        if len(model_df) < 50:
            errors.append(
                "Not enough labeled attrition observations to train a reliable company-specific model."
            )
            out["predicted_attrition_probability"] = pd.NA

            return out, RiskEngineReport(
                risk_source="unavailable",
                model_used=None,
                n_observations=len(model_df),
                n_features=0,
                warnings=warnings,
                errors=errors,
            )

        usable_features = [
            col for col in model_feature_columns
            if col in model_df.columns and model_df[col].nunique(dropna=True) > 1
        ]

        if len(usable_features) == 0:
            errors.append("No usable model features available for attrition modeling.")
            out["predicted_attrition_probability"] = pd.NA

            return out, RiskEngineReport(
                risk_source="unavailable",
                model_used=None,
                n_observations=len(model_df),
                n_features=0,
                warnings=warnings,
                errors=errors,
            )

        X = model_df[usable_features].copy()
        y = pd.to_numeric(model_df["separation_outcome"], errors="coerce")

        valid_idx = y.notna()
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        if y.nunique() < 2:
            errors.append(
                "Separation outcome must contain both 0 and 1 classes to train an attrition model."
            )
            out["predicted_attrition_probability"] = pd.NA

            return out, RiskEngineReport(
                risk_source="unavailable",
                model_used=None,
                n_observations=len(X),
                n_features=len(usable_features),
                warnings=warnings,
                errors=errors,
            )

        categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

        model.fit(X, y)

        prediction_X = out[usable_features].copy()
        out["predicted_attrition_probability"] = model.predict_proba(prediction_X)[:, 1]

        warnings.append(
            "Company-specific attrition model trained from uploaded historical separation outcomes. "
            "Enterprise use requires out-of-sample validation and calibration diagnostics."
        )

        return out, RiskEngineReport(
            risk_source="company_specific_model",
            model_used="logistic_regression",
            n_observations=len(X),
            n_features=len(usable_features),
            warnings=warnings,
            errors=errors,
        )

    # Case 2: company supplied existing probabilities
    if "predicted_attrition_probability" in out.columns:
        out["predicted_attrition_probability"] = pd.to_numeric(
            out["predicted_attrition_probability"],
            errors="coerce",
        )

        invalid = (
            out["predicted_attrition_probability"].dropna().lt(0).any()
            or out["predicted_attrition_probability"].dropna().gt(1).any()
        )

        if invalid:
            errors.append("Supplied attrition probabilities must be between 0 and 1.")
            out["predicted_attrition_probability"] = pd.NA

            return out, RiskEngineReport(
                risk_source="invalid_precomputed_probability",
                model_used=None,
                n_observations=len(out),
                n_features=0,
                warnings=warnings,
                errors=errors,
            )

        warnings.append(
            "Using supplied predicted attrition probabilities. HCRL did not generate these probabilities."
        )

        return out, RiskEngineReport(
            risk_source="precomputed_probability",
            model_used=None,
            n_observations=len(out),
            n_features=0,
            warnings=warnings,
            errors=errors,
        )

    # Case 3: no attrition outcome and no probability
    out["predicted_attrition_probability"] = pd.NA

    warnings.append(
        "No historical attrition outcome detected. HCRL cannot train a company-specific attrition model yet. "
        "Next architecture step: build external labor-market baseline risk model."
    )

    return out, RiskEngineReport(
        risk_source="external_baseline_required",
        model_used=None,
        n_observations=len(out),
        n_features=0,
        warnings=warnings,
        errors=errors,
    )
