"""
HCRL Workforce Segmentation Intelligence Engine

Purpose:
Identify where workforce risk and economic exposure are concentrated
inside the organization.

No arbitrary thresholds.
No hardcoded weights.
No automated employment decisions.

This engine ranks company segments by observed expected attrition cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class SegmentationReport:
    segment_column: str
    segments_analyzed: int
    total_expected_attrition_cost: float
    largest_cost_segment: str
    highest_risk_segment: str
    warnings: List[str]
    errors: List[str]


def build_segmentation_table(
    workforce_df: pd.DataFrame,
    segment_column: str,
) -> Tuple[pd.DataFrame, SegmentationReport]:

    warnings: List[str] = []
    errors: List[str] = []

    required_cols = [
        segment_column,
        "predicted_attrition_probability",
        "expected_attrition_cost",
    ]

    missing = [
        col for col in required_cols
        if col not in workforce_df.columns
    ]

    if missing:
        errors.append(f"Missing required columns: {missing}")

        return pd.DataFrame(), SegmentationReport(
            segment_column=segment_column,
            segments_analyzed=0,
            total_expected_attrition_cost=0.0,
            largest_cost_segment="",
            highest_risk_segment="",
            warnings=warnings,
            errors=errors,
        )

    df = workforce_df.copy()

    df = df[df[segment_column].notna()].copy()

    if df.empty:
        errors.append("No non-missing segment values available.")

        return pd.DataFrame(), SegmentationReport(
            segment_column=segment_column,
            segments_analyzed=0,
            total_expected_attrition_cost=0.0,
            largest_cost_segment="",
            highest_risk_segment="",
            warnings=warnings,
            errors=errors,
        )

    df["predicted_attrition_probability"] = pd.to_numeric(
        df["predicted_attrition_probability"],
        errors="coerce",
    )

    df["expected_attrition_cost"] = pd.to_numeric(
        df["expected_attrition_cost"],
        errors="coerce",
    ).fillna(0)

    agg_dict = {
        "employee_count": (
            segment_column,
            "count",
        ),
        "avg_attrition_probability": (
            "predicted_attrition_probability",
            "mean",
        ),
        "total_expected_attrition_cost": (
            "expected_attrition_cost",
            "sum",
        ),
    }

    optional_ai_columns = {
        "avg_digital_work":
            "ai_digital_work_percentile",
        "avg_analytical_work":
            "ai_analytical_cognitive_work_percentile",
        "avg_human_interaction":
            "ai_human_interaction_work_percentile",
        "avg_physical_work":
            "ai_physical_manual_work_percentile",
    }

    for output_col, source_col in optional_ai_columns.items():
        if source_col in df.columns:
            agg_dict[output_col] = (
                source_col,
                "mean",
            )

    segment_table = (
        df.groupby(segment_column)
        .agg(**agg_dict)
        .reset_index()
        .rename(columns={segment_column: "segment"})
    )

    total_cost = float(
        segment_table["total_expected_attrition_cost"].sum()
    )

    if total_cost > 0:
        segment_table["share_of_total_cost"] = (
            segment_table["total_expected_attrition_cost"]
            / total_cost
        )
    else:
        segment_table["share_of_total_cost"] = 0.0

    segment_table["share_of_total_cost_pct"] = (
        segment_table["share_of_total_cost"] * 100
    ).round(2)

    segment_table = segment_table.sort_values(
        "total_expected_attrition_cost",
        ascending=False,
    )

    largest_cost_segment = str(
        segment_table.iloc[0]["segment"]
    )

    highest_risk_segment = str(
        segment_table.sort_values(
            "avg_attrition_probability",
            ascending=False,
        ).iloc[0]["segment"]
    )

    final_columns = [
        "segment",
        "employee_count",
        "avg_attrition_probability",
        "total_expected_attrition_cost",
        "share_of_total_cost_pct",
    ]

    for col in [
        "avg_digital_work",
        "avg_analytical_work",
        "avg_human_interaction",
        "avg_physical_work",
    ]:
        if col in segment_table.columns:
            final_columns.append(col)

    segment_table = segment_table[final_columns]

    return segment_table, SegmentationReport(
        segment_column=segment_column,
        segments_analyzed=len(segment_table),
        total_expected_attrition_cost=total_cost,
        largest_cost_segment=largest_cost_segment,
        highest_risk_segment=highest_risk_segment,
        warnings=warnings,
        errors=errors,
    )
