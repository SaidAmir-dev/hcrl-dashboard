"""
HCRL Attrition Driver Intelligence Engine

Purpose:
Identify variables statistically associated with predicted attrition risk.

No causal claims.
No arbitrary thresholds.
No fake recommendations.
No automated employment decisions.

This engine answers:
"Which observed variables are most associated with attrition risk in this dataset?"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class AttritionDriverReport:
    drivers_analyzed: int
    warnings: List[str]
    errors: List[str]


EXCLUDED_COLUMNS = {
    "employee_id",

    "predicted_attrition_probability",

    "expected_attrition_cost",
    "expected_attrition_cost_low",
    "expected_attrition_cost_high",

    "replacement_cost",
    "replacement_cost_low",
    "replacement_cost_high",

    "replacement_cost_multiplier_base",
    "replacement_cost_multiplier_low",
    "replacement_cost_multiplier_high",

    "annual_wage",

    "Attrition",
    "separation_outcome",
}


def build_attrition_driver_table(
    workforce_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, AttritionDriverReport]:

    warnings: List[str] = []
    errors: List[str] = []

    target_col = "predicted_attrition_probability"

    if target_col not in workforce_df.columns:
        errors.append("predicted_attrition_probability is required.")
        return pd.DataFrame(), AttritionDriverReport(0, warnings, errors)

    df = workforce_df.copy()

    df[target_col] = pd.to_numeric(
        df[target_col],
        errors="coerce",
    )

    df = df[df[target_col].notna()].copy()

    if df.empty:
        errors.append("No valid attrition probability values available.")
        return pd.DataFrame(), AttritionDriverReport(0, warnings, errors)

    global_risk = df[target_col].mean()

    rows = []

    candidate_cols = [
        col for col in df.columns
        if col not in EXCLUDED_COLUMNS
        and not col.startswith("ai_")
        and not col.startswith("task_")
        and df[col].nunique(dropna=True) > 1
    ]

    for col in candidate_cols:

        series = df[col]

        numeric_series = pd.to_numeric(series, errors="coerce")

        if numeric_series.notna().mean() >= 0.8:

            temp = pd.DataFrame(
                {
                    "x": numeric_series,
                    "risk": df[target_col],
                }
            ).dropna()

            if len(temp) < 2:
                continue

            association = temp["x"].corr(
                temp["risk"],
                method="spearman",
            )

            rows.append(
                {
                    "driver_variable": col,
                    "driver_type": "numeric",
                    "association_metric": "spearman_correlation_with_predicted_attrition_probability",
                    "association_value": association,
                    "direction": (
                        "higher_values_associated_with_higher_risk"
                        if association > 0
                        else "higher_values_associated_with_lower_risk"
                    ),
                    "evidence_summary": (
                        f"{col} has a Spearman association of "
                        f"{association:.3f} with predicted attrition probability."
                    ),
                }
            )

        else:

            temp = pd.DataFrame(
                {
                    "category": series.astype(str),
                    "risk": df[target_col],
                }
            ).dropna()

            group_stats = (
                temp.groupby("category")
                .agg(
                    n_workers=("risk", "count"),
                    avg_attrition_probability=("risk", "mean"),
                )
                .reset_index()
            )

            if group_stats.empty:
                continue

            group_stats["risk_difference_from_company_average"] = (
                group_stats["avg_attrition_probability"] - global_risk
            )

            strongest = group_stats.reindex(
                group_stats["risk_difference_from_company_average"]
                .abs()
                .sort_values(ascending=False)
                .index
            ).iloc[0]

            rows.append(
                {
                    "driver_variable": col,
                    "driver_type": "categorical",
                    "association_metric": "largest_category_difference_from_company_average",
                    "association_value": strongest["risk_difference_from_company_average"],
                    "direction": (
                        "category_associated_with_higher_risk"
                        if strongest["risk_difference_from_company_average"] > 0
                        else "category_associated_with_lower_risk"
                    ),
                    "evidence_summary": (
                        f"For {col}, category '{strongest['category']}' differs from "
                        f"the company average predicted attrition probability by "
                        f"{strongest['risk_difference_from_company_average']:.3f}."
                    ),
                }
            )

    driver_table = pd.DataFrame(rows)

    if driver_table.empty:
        warnings.append("No usable driver associations were found.")
        return driver_table, AttritionDriverReport(0, warnings, errors)

    driver_table["absolute_association_value"] = (
        driver_table["association_value"].abs()
    )

    driver_table = driver_table.sort_values(
        "absolute_association_value",
        ascending=False,
    ).drop(columns=["absolute_association_value"])
    driver_table["driver_rank"] = range(
    1,
    len(driver_table) + 1,
    )

    warnings.append(
        "Driver analysis identifies statistical associations with predicted attrition risk. "
        "It does not prove causality or prescribe interventions."
    )

    return driver_table, AttritionDriverReport(
        drivers_analyzed=len(driver_table),
        warnings=warnings,
        errors=errors,
    )
