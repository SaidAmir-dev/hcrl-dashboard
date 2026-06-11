"""HCRL Workforce Prioritization Engine.

Purpose:
Identify which occupations create the largest workforce risk and
economic exposure.

No arbitrary thresholds.
No hardcoded weights.
No automated employment decisions.

Ranking is based on observed economic exposure:

* expected attrition cost
  """

from **future** import annotations

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

```
warnings = []
errors = []

required_cols = [
    "matched_onet_title",
    "predicted_attrition_probability",
    "expected_attrition_cost",
]

missing = [c for c in required_cols if c not in workforce_df.columns]

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

if len(df) == 0:
    errors.append("No mapped occupations available.")

    return pd.DataFrame(), PrioritizationReport(
        occupations_analyzed=0,
        warnings=warnings,
        errors=errors,
    )

summary = (
    df.groupby("matched_onet_title")
    .agg(
        n_workers=("matched_onet_title", "count"),

        avg_attrition_probability=(
            "predicted_attrition_probability",
            "mean",
        ),

        total_expected_attrition_cost=(
            "expected_attrition_cost",
            "sum",
        ),

        avg_digital_work=(
            "ai_digital_work_percentile",
            "mean",
        ),

        avg_analytical_work=(
            "ai_analytical_cognitive_work_percentile",
            "mean",
        ),

        avg_human_interaction=(
            "ai_human_interaction_work_percentile",
            "mean",
        ),

        avg_physical_work=(
            "ai_physical_manual_work_percentile",
            "mean",
        ),
    )
    .reset_index()
)

# --------------------------------------------
# COST SHARE
# --------------------------------------------

total_cost = summary[
    "total_expected_attrition_cost"
].sum()

summary["share_of_total_cost"] = (
    summary["total_expected_attrition_cost"]
    / total_cost
)

summary["share_of_total_cost_pct"] = (
    summary["share_of_total_cost"] * 100
).round(2)

# --------------------------------------------
# PRIMARY WORK TYPE
# --------------------------------------------

dimension_cols = [
    "avg_digital_work",
    "avg_analytical_work",
    "avg_human_interaction",
    "avg_physical_work",
]

dimension_labels = {
    "avg_digital_work": "Digital",
    "avg_analytical_work": "Analytical",
    "avg_human_interaction": "Human Interaction",
    "avg_physical_work": "Physical / Manual",
}

summary["primary_work_type"] = (
    summary[dimension_cols]
    .idxmax(axis=1)
    .map(dimension_labels)
)

# --------------------------------------------
# SECONDARY WORK TYPE
# --------------------------------------------

def second_highest_dimension(row):

    values = row[dimension_cols].dropna()

    if len(values) < 2:
        return None

    return (
        values.sort_values(ascending=False)
        .index[1]
    )

summary["secondary_work_type"] = (
    summary.apply(
        second_highest_dimension,
        axis=1,
    )
    .map(dimension_labels)
)

# --------------------------------------------
# PRIORITY RANKS
# --------------------------------------------

summary = summary.sort_values(
    "total_expected_attrition_cost",
    ascending=False,
)

summary["priority_rank"] = range(
    1,
    len(summary) + 1,
)

# --------------------------------------------
# EXECUTIVE SUMMARY
# --------------------------------------------

summary["executive_summary"] = (
    "Cost Share="
    + summary["share_of_total_cost_pct"]
    .astype(str)
    + "% | Primary Work="
    + summary["primary_work_type"]
    .fillna("Unknown")
)

# --------------------------------------------
# FINAL OUTPUT
# --------------------------------------------

summary = summary[
    [
        "priority_rank",
        "matched_onet_title",
        "n_workers",
        "avg_attrition_probability",
        "total_expected_attrition_cost",
        "share_of_total_cost_pct",
        "primary_work_type",
        "secondary_work_type",
        "executive_summary",
        "avg_digital_work",
        "avg_analytical_work",
        "avg_human_interaction",
        "avg_physical_work",
    ]
]

return summary, PrioritizationReport(
    occupations_analyzed=len(summary),
    warnings=warnings,
    errors=errors,
)
```
