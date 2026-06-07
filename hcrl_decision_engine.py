"""Decision-support summaries for HCRL.

This module avoids prescriptive firing logic. It ranks exposure concentration and
creates evidence labels based on observed data availability, not arbitrary cutoffs.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def build_segment_exposure(df: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    if segment_col not in df.columns:
        raise ValueError(f"Segment column not found: {segment_col}")

    aggregations = {
        "n_workers": (segment_col, "count"),
        "avg_attrition_probability": ("predicted_attrition_probability", "mean"),
    }
    if "expected_attrition_cost" in df.columns:
        aggregations["total_expected_attrition_cost"] = ("expected_attrition_cost", "sum")
        aggregations["avg_expected_attrition_cost"] = ("expected_attrition_cost", "mean")
    if "annual_wage" in df.columns:
        aggregations["avg_annual_wage"] = ("annual_wage", "mean")

    summary = df.groupby(segment_col, dropna=False).agg(**aggregations).reset_index()

    if "total_expected_attrition_cost" in summary.columns:
        denom = summary["total_expected_attrition_cost"].sum()
        if pd.notna(denom) and denom != 0:
            summary["risk_concentration_share"] = summary["total_expected_attrition_cost"] / denom
        summary = summary.sort_values("total_expected_attrition_cost", ascending=False)
    else:
        summary = summary.sort_values("avg_attrition_probability", ascending=False)

    summary["exposure_rank"] = range(1, len(summary) + 1)
    return summary


def build_decision_support_table(df: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    summary = build_segment_exposure(df, segment_col)

    labels = []
    for _, row in summary.iterrows():
        parts = ["Review workforce stability drivers for this segment."]
        if "risk_concentration_share" in summary.columns:
            parts.append("Prioritization is based on share of total expected attrition cost, not a hard risk cutoff.")
        if "matched_onet_title" in df.columns:
            parts.append("Use O*NET mapping as occupational context; fuzzy matches require review before enterprise use.")
        labels.append(" ".join(parts))
    summary["decision_support_interpretation"] = labels
    return summary
