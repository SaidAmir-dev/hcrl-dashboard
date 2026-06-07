"""HCRL decision intelligence engine.

This module translates workforce analytics into conservative decision-support options.

Important:
HCRL must not pretend to know the best intervention unless the data supports it.

Therefore, this engine separates:
1. observed risk signals
2. economic exposure signals
3. O*NET / AI exposure signals
4. decision-support options

It does NOT make firing recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class DecisionEngineReport:
    n_segments: int
    warnings: List[str]
    errors: List[str]


def _available_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    return [col for col in columns if col in df.columns]


def build_segment_decision_table(
    df: pd.DataFrame,
    segment_col: str = "job_title",
) -> Tuple[pd.DataFrame, DecisionEngineReport]:

    warnings: List[str] = []
    errors: List[str] = []

    if segment_col not in df.columns:
        errors.append(f"Segment column '{segment_col}' not found.")
        return pd.DataFrame(), DecisionEngineReport(
            n_segments=0,
            warnings=warnings,
            errors=errors,
        )

    aggregation_spec = {
        "n_workers": (segment_col, "count"),
    }

    if "predicted_attrition_probability" in df.columns:
        aggregation_spec["avg_attrition_probability"] = (
            "predicted_attrition_probability",
            "mean",
        )

    if "expected_attrition_cost" in df.columns:
        aggregation_spec["total_expected_attrition_cost"] = (
            "expected_attrition_cost",
            "sum",
        )
        aggregation_spec["avg_expected_attrition_cost"] = (
            "expected_attrition_cost",
            "mean",
        )

    if "ai_exposure_score" in df.columns:
        aggregation_spec["avg_ai_exposure_score"] = (
            "ai_exposure_score",
            "mean",
        )

    if "onet_match_status" in df.columns:
        aggregation_spec["onet_mapping_coverage"] = (
            "onet_match_status",
            lambda x: x.isin(["accepted", "review_required"]).mean(),
        )

    segment_table = (
        df.groupby(segment_col)
        .agg(**aggregation_spec)
        .reset_index()
    )

    if "total_expected_attrition_cost" in segment_table.columns:
        total_cost = segment_table["total_expected_attrition_cost"].sum()

        if total_cost > 0:
            segment_table["share_of_total_expected_attrition_cost"] = (
                segment_table["total_expected_attrition_cost"] / total_cost
            )
        else:
            segment_table["share_of_total_expected_attrition_cost"] = pd.NA

    segment_table["decision_support_status"] = "diagnostic_only"
    segment_table["decision_support_note"] = (
        "Insufficient intervention economics. HCRL can identify exposure, "
        "but intervention ROI has not yet been estimated."
    )

    # Conservative decision-support signals
    if (
        "avg_attrition_probability" in segment_table.columns
        and "total_expected_attrition_cost" in segment_table.columns
    ):
        segment_table["decision_support_status"] = "attrition_exposure_identified"
        segment_table["decision_support_note"] = (
            "This segment contributes to expected attrition exposure. "
            "Evaluate retention, compensation, management, workload, or career-path interventions."
        )

    if "avg_ai_exposure_score" in segment_table.columns:
        segment_table["ai_transformation_note"] = (
            "AI exposure detected at occupation level. This is not a replacement recommendation. "
            "Evaluate task redesign, AI augmentation, workflow redesign, and human oversight requirements."
        )
    else:
        segment_table["ai_transformation_note"] = (
            "AI exposure unavailable because O*NET/AI mapping is missing or incomplete."
        )

    sort_cols = _available_columns(
        segment_table,
        [
            "total_expected_attrition_cost",
            "avg_attrition_probability",
            "avg_ai_exposure_score",
            "n_workers",
        ],
    )

    if sort_cols:
        segment_table = segment_table.sort_values(
            sort_cols,
            ascending=False,
        )

    warnings.append(
        "Decision engine currently provides conservative decision-support signals only. "
        "Final intervention recommendations require an intervention economics module."
    )

    return segment_table, DecisionEngineReport(
        n_segments=len(segment_table),
        warnings=warnings,
        errors=errors,
    )
