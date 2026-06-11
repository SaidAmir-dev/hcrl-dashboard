"""HCRL AI readiness evidence engine.

No arbitrary thresholds.
No fake AI score.
No automated employment decisions.

This module attaches observable O*NET evidence dimensions:
- digital work
- analytical/cognitive work
- human interaction work
- physical/manual work

Percentiles are shown only as within-dataset rankings, not probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class AIReadinessReport:
    ai_reference_available: bool
    matched_rows: int
    unmatched_rows: int
    dimension_columns_used: Dict[str, List[str]]
    warnings: List[str]
    errors: List[str]


AI_DIMENSION_COLUMNS = {
    "digital_work": [
        "Working with Computers",
        "Processing Information",
        "Documenting/Recording Information",
    ],
    "analytical_cognitive_work": [
        "Analyzing Data or Information",
        "Critical Thinking",
        "Deductive Reasoning",
        "Inductive Reasoning",
        "Information Ordering",
        "Mathematical Reasoning",
        "Problem Sensitivity",
        "Making Decisions and Solving Problems",
        "Thinking Creatively",
    ],
    "human_interaction_work": [
        "Assisting and Caring for Others",
        "Communicating with People Outside the Organization",
        "Communicating with Supervisors, Peers, or Subordinates",
        "Establishing and Maintaining Interpersonal Relationships",
        "Performing for or Working Directly with the Public",
        "Speaking",
        "Active Listening",
        "Oral Comprehension",
        "Oral Expression",
    ],
    "physical_manual_work": [
        "Handling and Moving Objects",
        "Operating Vehicles, Mechanized Devices, or Equipment",
        "Performing General Physical Activities",
        "Repairing and Maintaining Mechanical Equipment",
        "Repairing and Maintaining Electronic Equipment",
        "Manual Dexterity",
        "Finger Dexterity",
        "Static Strength",
        "Stamina",
        "Control Precision",
    ],
}


def _standardize_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _find_workforce_code_col(df: pd.DataFrame) -> str | None:
    for col in ["matched_onet_code", "occupation_code", "O*NET-SOC Code", "soc_code"]:
        if col in df.columns:
            return col
    return None


def _available_columns(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [col for col in candidates if col in df.columns]


def _row_mean_numeric(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    if not cols:
        return pd.Series(pd.NA, index=df.index)

    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    return numeric.mean(axis=1)


def attach_ai_readiness(
    workforce_df: pd.DataFrame,
    onet_feature_path: str = "onet_occupation_feature_table.csv",
) -> Tuple[pd.DataFrame, AIReadinessReport]:

    out = workforce_df.copy()
    warnings: List[str] = []
    errors: List[str] = []

    code_col = _find_workforce_code_col(out)

    if code_col is None:
        errors.append("No O*NET code column found. AI readiness cannot be attached.")
        return out, AIReadinessReport(False, 0, len(out), {}, warnings, errors)

    try:
        onet = pd.read_csv(onet_feature_path)
    except Exception as e:
        errors.append(f"Could not load O*NET feature table: {e}")
        return out, AIReadinessReport(False, 0, len(out), {}, warnings, errors)

    if "O*NET-SOC Code" not in onet.columns:
        errors.append("O*NET feature table must contain 'O*NET-SOC Code'.")
        return out, AIReadinessReport(True, 0, len(out), {}, warnings, errors)

    out[code_col] = _standardize_code(out[code_col])
    onet["O*NET-SOC Code"] = _standardize_code(onet["O*NET-SOC Code"])

    feature_df = onet[["O*NET-SOC Code", "Title"]].copy()
    dimension_columns_used: Dict[str, List[str]] = {}

    for dimension, candidate_cols in AI_DIMENSION_COLUMNS.items():
        cols = _available_columns(onet, candidate_cols)
        dimension_columns_used[dimension] = cols

        raw_col = f"ai_{dimension}_evidence"
        pct_col = f"ai_{dimension}_percentile"

        if not cols:
            feature_df[raw_col] = pd.NA
            warnings.append(f"No usable O*NET columns found for {dimension}.")
            continue

        feature_df[raw_col] = _row_mean_numeric(onet, cols)
        feature_df[pct_col] = pd.to_numeric(
            feature_df[raw_col], errors="coerce"
        ).rank(pct=True)

    merged = out.merge(
        feature_df,
        left_on=code_col,
        right_on="O*NET-SOC Code",
        how="left",
        suffixes=("", "_ai_reference"),
    )

    matched = int(merged["O*NET-SOC Code"].notna().sum())
    unmatched = int(merged["O*NET-SOC Code"].isna().sum())

    missing_evidence_rows = int(
        merged[
            [
                c for c in merged.columns
                if c.endswith("_evidence")
            ]
        ].isna().all(axis=1).sum()
    )

    if missing_evidence_rows:
        warnings.append(
            f"{missing_evidence_rows} matched rows have no usable AI evidence values in the O*NET feature table."
        )

    warnings.append(
        "AI readiness fields are O*NET evidence indicators and workforce-relative percentiles. "
        "They are not probabilities, thresholds, or automated recommendations."
    )

    return merged, AIReadinessReport(
        ai_reference_available=True,
        matched_rows=matched,
        unmatched_rows=unmatched,
        dimension_columns_used=dimension_columns_used,
        warnings=warnings,
        errors=errors,
    )
