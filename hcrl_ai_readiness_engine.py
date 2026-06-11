"""HCRL AI readiness engine.

This module attaches O*NET-based AI readiness dimensions to workforce rows.

No fake thresholds.
No fake confidence scores.
No firing recommendations.

It creates transparent relative indices from observed O*NET variables:
- digital work
- analytical/cognitive work
- human-interaction work
- physical/manual work

The output is decision-support evidence, not a final automated decision.
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
        "Computers and Electronics",
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
        "Customer and Personal Service",
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
        return out, AIReadinessReport(
            ai_reference_available=False,
            matched_rows=0,
            unmatched_rows=len(out),
            dimension_columns_used={},
            warnings=warnings,
            errors=errors,
        )

    try:
        onet = pd.read_csv(onet_feature_path)
    except Exception as e:
        errors.append(f"Could not load O*NET feature table: {e}")
        return out, AIReadinessReport(
            ai_reference_available=False,
            matched_rows=0,
            unmatched_rows=len(out),
            dimension_columns_used={},
            warnings=warnings,
            errors=errors,
        )

    if "O*NET-SOC Code" not in onet.columns:
        errors.append("O*NET feature table must contain 'O*NET-SOC Code'.")
        return out, AIReadinessReport(
            ai_reference_available=True,
            matched_rows=0,
            unmatched_rows=len(out),
            dimension_columns_used={},
            warnings=warnings,
            errors=errors,
        )

    out[code_col] = _standardize_code(out[code_col])
    onet["O*NET-SOC Code"] = _standardize_code(onet["O*NET-SOC Code"])

    dimension_columns_used: Dict[str, List[str]] = {}

    feature_df = onet[["O*NET-SOC Code", "Title"]].copy()

    for dimension, candidate_cols in AI_DIMENSION_COLUMNS.items():
        cols = _available_columns(onet, candidate_cols)
        dimension_columns_used[dimension] = cols

        if not cols:
            warnings.append(f"No usable O*NET columns found for dimension: {dimension}")
            feature_df[f"ai_{dimension}_raw"] = pd.NA
        else:
            feature_df[f"ai_{dimension}_raw"] = _row_mean_numeric(onet, cols)

    merged = out.merge(
        feature_df,
        left_on=code_col,
        right_on="O*NET-SOC Code",
        how="left",
        suffixes=("", "_ai_reference"),
    )

    matched = int(merged["O*NET-SOC Code"].notna().sum())
    unmatched = int(merged["O*NET-SOC Code"].isna().sum())

    for dimension in AI_DIMENSION_COLUMNS:
        raw_col = f"ai_{dimension}_raw"
        rank_col = f"ai_{dimension}_percentile"

        if raw_col in merged.columns:
            merged[rank_col] = pd.to_numeric(
                merged[raw_col],
                errors="coerce",
            ).rank(pct=True)

    # This is not a probability and not a decision threshold.
    # It is a relative index based on workforce percentile ranks.
    positive_cols = [
        "ai_digital_work_percentile",
        "ai_analytical_cognitive_work_percentile",
    ]

    constraint_cols = [
        "ai_human_interaction_work_percentile",
        "ai_physical_manual_work_percentile",
    ]

    available_positive = [c for c in positive_cols if c in merged.columns]
    available_constraint = [c for c in constraint_cols if c in merged.columns]

    if available_positive and available_constraint:
        merged["ai_augmentation_readiness_index"] = (
            merged[available_positive].mean(axis=1)
            - merged[available_constraint].mean(axis=1)
        )
    else:
        merged["ai_augmentation_readiness_index"] = pd.NA
        warnings.append(
            "AI augmentation readiness index could not be computed because one or more dimension groups are missing."
        )

    warnings.append(
        "AI readiness dimensions are relative O*NET-based evidence indicators. "
        "They are not calibrated probabilities and should not be used as automated employment decisions."
    )

    return merged, AIReadinessReport(
        ai_reference_available=True,
        matched_rows=matched,
        unmatched_rows=unmatched,
        dimension_columns_used=dimension_columns_used,
        warnings=warnings,
        errors=errors,
    )
