"""
HCRL Attrition Driver Intelligence Engine

Purpose:
Identify business-level workforce variables statistically associated with
predicted attrition risk.

No causal claims.
No arbitrary thresholds.
No fake recommendations.
No automated employment decisions.
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
    "EmployeeNumber",
    "EmployeeCount",
    "StandardHours",
    "Over18",

    "DailyRate",
    "HourlyRate",
    "MonthlyRate",

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
    "replacement_cost_tier",
    "annual_wage",

    "Attrition",
    "separation_outcome",

    "matched_onet_title",
    "matched_onet_code",
    "normalized_title",
    "candidate_titles",
    "O*NET-SOC Code",
    "O*NET-SOC Code_ai_reference",
    "Title",

    "onet_match_score",
    "onet_match_method",
    "onet_match_status",
    "title_function",
    "title_level",
    "title_normalization_method",

    "primary_work_type",
    "secondary_work_type",
    "priority_rank",
    "share_of_total_cost",
    "share_of_total_cost_pct",
}


NON_ACTIONABLE_DEMOGRAPHICS = {
    "Age",
    "age",
    "Gender",
    "gender",
    "MaritalStatus",
    "marital_status",
}


DRIVER_GROUPS = {
    "MonthlyIncome": "Compensation",
    "monthly_income": "Compensation",
    "PercentSalaryHike": "Compensation",
    "StockOptionLevel": "Compensation",

    "JobLevel": "Career Progression",
    "YearsInCurrentRole": "Career Progression",
    "YearsSinceLastPromotion": "Career Progression",
    "YearsAtCompany": "Career Progression",
    "tenure_years": "Career Progression",

    "TotalWorkingYears": "Employee Experience",
    "NumCompaniesWorked": "Employee Experience",

    "YearsWithCurrManager": "Manager Stability",

    "OverTime": "Workload",

    "EnvironmentSatisfaction": "Work Environment",
    "JobSatisfaction": "Work Environment",
    "RelationshipSatisfaction": "Work Environment",
    "JobInvolvement": "Work Environment",
    "WorkLifeBalance": "Work Environment",

    "BusinessTravel": "Travel / Commute Burden",
    "DistanceFromHome": "Travel / Commute Burden",

    "TrainingTimesLastYear": "Training and Development",

    "Department": "Department",
    "department": "Department",

    "JobRole": "Occupation",
    "job_title": "Occupation",

    "Education": "Education",
    "EducationField": "Education",
    "PerformanceRating": "Performance",
}


ACTIONABLE_GROUPS = {
    "Compensation",
    "Career Progression",
    "Manager Stability",
    "Workload",
    "Work Environment",
    "Travel / Commute Burden",
    "Training and Development",
    "Department",
    "Performance",
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
        if col in DRIVER_GROUPS
        and col not in EXCLUDED_COLUMNS
        and col not in NON_ACTIONABLE_DEMOGRAPHICS
        and df[col].nunique(dropna=True) > 1
    ]

    for col in candidate_cols:

        driver_group = DRIVER_GROUPS.get(col)

        if driver_group is None:
            continue

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

            if pd.isna(association):
                continue

            rows.append(
                {
                    "driver_variable": col,
                    "driver_group": driver_group,
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

            association = strongest[
                "risk_difference_from_company_average"
            ]

            if pd.isna(association):
                continue

            rows.append(
                {
                    "driver_variable": col,
                    "driver_group": driver_group,
                    "driver_type": "categorical",
                    "association_metric": "largest_category_difference_from_company_average",
                    "association_value": association,
                    "direction": (
                        "category_associated_with_higher_risk"
                        if association > 0
                        else "category_associated_with_lower_risk"
                    ),
                    "evidence_summary": (
                        f"For {col}, category '{strongest['category']}' differs from "
                        f"the company average predicted attrition probability by "
                        f"{association:.3f}."
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

    driver_table = (
        driver_table
        .sort_values(
            "absolute_association_value",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    driver_table["actionability"] = (
        driver_table["driver_group"]
        .isin(ACTIONABLE_GROUPS)
        .map(
            {
                True: "Actionable",
                False: "Descriptive",
            }
        )
    )

    driver_table["driver_rank"] = range(
        1,
        len(driver_table) + 1,
    )

    driver_table = driver_table[
        [
            "driver_rank",
            "driver_group",
            "driver_variable",
            "driver_type",
            "association_metric",
            "association_value",
            "direction",
            "actionability",
            "evidence_summary",
        ]
    ]

    warnings.append(
        "Driver analysis identifies statistical associations with predicted attrition risk. "
        "It does not prove causality or prescribe interventions."
    )

    return driver_table, AttritionDriverReport(
        drivers_analyzed=len(driver_table),
        warnings=warnings,
        errors=errors,
    )
