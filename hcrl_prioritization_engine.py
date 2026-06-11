"""
HCRL Workforce Prioritization Engine

Purpose:
Identify which occupations create the largest workforce risk and
economic exposure.

No arbitrary thresholds.
No hardcoded weights.
No automated employment decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class PrioritizationReport:
    occupations_analyzed: int
    warnings: List[str]
    errors: List[str]


def build_prioritization_table(
    workforce_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, PrioritizationReport]:

    warnings: List[str] = []
    errors: List[str] = []

    required_cols = [
        "matched_onet_title",
        "predicted_attrition_probability",
        "expected_attrition_cost",
    ]

    missing = [
        c for c in required_cols
        if c not in workforce_df.columns
    ]

    if missing:
        errors.append(
            f"Missing required columns: {missing}"
        )

        return pd.DataFrame(), PrioritizationReport(
            occupations_analyzed=0,
            warnings=warnings,
            errors=errors,
        )

    df = workforce_df.copy()

    df = df[
        df["matched_onet_title"].notna()
    ].copy()

    if df.empty:
        errors.append(
            "No mapped occupations available."
        )

        return pd.DataFrame(), PrioritizationReport(
            occupations_analyzed=0,
            warnings=warnings,
            errors=errors,
        )

    agg_dict = {
        "n_workers": (
            "matched_onet_title",
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

    summary = (
        df.groupby("matched_onet_title")
        .agg(**agg_dict)
        .reset_index()
    )

    total_cost = summary[
        "total_expected_attrition_cost"
    ].sum()

    if total_cost > 0:
        summary["share_of_total_cost"] = (
            summary["total_expected_attrition_cost"]
            / total_cost
        )
    else:
        summary["share_of_total_cost"] = 0.0

    summary["share_of_total_cost_pct"] = (
        summary["share_of_total_cost"] * 100
    ).round(2)

    dimension_cols = [
        c for c in [
            "avg_digital_work",
            "avg_analytical_work",
            "avg_human_interaction",
            "avg_physical_work",
        ]
        if c in summary.columns
    ]

    dimension_labels = {
        "avg_digital_work": "Digital",
        "avg_analytical_work": "Analytical",
        "avg_human_interaction": "Human Interaction",
        "avg_physical_work": "Physical / Manual",
    }

    def get_primary_dimension(row):

        values = row[dimension_cols].dropna()

        if values.empty:
            return "Unknown"

        return dimension_labels[
            values.idxmax()
        ]

    def get_secondary_dimension(row):

        values = row[dimension_cols].dropna()

        if len(values) < 2:
            return None

        ordered = values.sort_values(
            ascending=False
        )

        return dimension_labels[
            ordered.index[1]
        ]

    if dimension_cols:

        summary["primary_work_type"] = (
            summary.apply(
                get_primary_dimension,
                axis=1,
            )
        )

        summary["secondary_work_type"] = (
            summary.apply(
                get_secondary_dimension,
                axis=1,
            )
        )

    else:

        summary["primary_work_type"] = (
            "Unknown"
        )

        summary["secondary_work_type"] = (
            None
        )

        warnings.append(
            "No AI readiness columns available."
        )

    summary = summary.sort_values(
        "total_expected_attrition_cost",
        ascending=False,
    )

    summary["priority_rank"] = range(
        1,
        len(summary) + 1,
    )

    summary["executive_summary"] = (
        "Cost Share="
        + summary[
            "share_of_total_cost_pct"
        ].astype(str)
        + "% | Primary Work="
        + summary[
            "primary_work_type"
        ].fillna("Unknown")
    )

    final_columns = [
        "priority_rank",
        "matched_onet_title",
        "n_workers",
        "avg_attrition_probability",
        "total_expected_attrition_cost",
        "share_of_total_cost",
        "share_of_total_cost_pct",
        "primary_work_type",
        "secondary_work_type",
        "executive_summary",
    ]

    for col in [
        "avg_digital_work",
        "avg_analytical_work",
        "avg_human_interaction",
        "avg_physical_work",
    ]:
        if col in summary.columns:
            final_columns.append(col)

    summary = summary[final_columns]

    return summary, PrioritizationReport(
        occupations_analyzed=len(summary),
        warnings=warnings,
        errors=errors,
    )
