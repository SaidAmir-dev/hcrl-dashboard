"""
HCRL Driver Recommendation Engine

Purpose:
Convert statistical attrition drivers into evidence-based management
hypotheses.

No causal claims.
No automated employment decisions.
No firing recommendations.
No unsupported prescriptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class DriverRecommendationReport:
    recommendations_generated: int
    warnings: List[str]
    errors: List[str]


HYPOTHESIS_LIBRARY = {
    "Compensation": {
        "review_area": "Compensation",
        "management_hypothesis": (
            "Compensation differences may be associated with retention outcomes. "
            "Review pay competitiveness, internal pay equity, and compensation "
            "patterns in high-risk groups."
        ),
    },
    "Compensation Growth": {
        "review_area": "Compensation Growth",
        "management_hypothesis": (
            "Salary growth patterns may be associated with retention outcomes. "
            "Review whether compensation progression is competitive across roles, "
            "levels, and departments."
        ),
    },
    "Career Progression": {
        "review_area": "Career Progression",
        "management_hypothesis": (
            "Career advancement indicators may be associated with retention outcomes. "
            "Review promotion pathways, internal mobility, and role progression patterns."
        ),
    },
    "Manager Stability": {
        "review_area": "Management Quality",
        "management_hypothesis": (
            "Manager continuity may be associated with retention outcomes. "
            "Review leadership stability, team structure, and manager-level retention patterns."
        ),
    },
    "Workload": {
        "review_area": "Workload",
        "management_hypothesis": (
            "Workload indicators may be associated with retention outcomes. "
            "Review overtime concentration, staffing levels, and workload distribution."
        ),
    },
    "Work Environment": {
        "review_area": "Work Environment",
        "management_hypothesis": (
            "Work environment indicators may be associated with retention outcomes. "
            "Review employee feedback, workplace conditions, and department-level experience gaps."
        ),
    },
    "Job Satisfaction": {
        "review_area": "Job Satisfaction",
        "management_hypothesis": (
            "Job satisfaction indicators may be associated with retention outcomes. "
            "Review employee engagement, role fit, and satisfaction differences across teams."
        ),
    },
    "Relationship Satisfaction": {
        "review_area": "Relationship Quality",
        "management_hypothesis": (
            "Relationship satisfaction indicators may be associated with retention outcomes. "
            "Review team climate, manager relationships, and collaboration patterns."
        ),
    },
    "Work-Life Balance": {
        "review_area": "Work-Life Balance",
        "management_hypothesis": (
            "Work-life balance indicators may be associated with retention outcomes. "
            "Review workload patterns, scheduling pressure, and employee experience metrics."
        ),
    },
    "Long-Term Incentives": {
        "review_area": "Incentive Structure",
        "management_hypothesis": (
            "Long-term incentive structures may be associated with retention outcomes. "
            "Review equity, retention incentives, and incentive eligibility across high-risk roles."
        ),
    },
    "Travel Burden": {
        "review_area": "Travel Burden",
        "management_hypothesis": (
            "Business travel patterns may be associated with retention outcomes. "
            "Review travel frequency, role expectations, and travel concentration in high-risk groups."
        ),
    },
    "Commute Burden": {
        "review_area": "Commute Burden",
        "management_hypothesis": (
            "Commute burden may be associated with retention outcomes. "
            "Review location strategy, flexibility options, and commute exposure for high-risk groups."
        ),
    },
    "Training": {
        "review_area": "Training and Development",
        "management_hypothesis": (
            "Training participation may be associated with retention outcomes. "
            "Review learning access, onboarding quality, and development investment by segment."
        ),
    },
    "Role Stability": {
        "review_area": "Role Stability",
        "management_hypothesis": (
            "Time in current role may be associated with retention outcomes. "
            "Review role transitions, internal mobility timing, and role stagnation patterns."
        ),
    },
    "Job Engagement": {
        "review_area": "Job Engagement",
        "management_hypothesis": (
            "Job involvement indicators may be associated with retention outcomes. "
            "Review engagement patterns, role ownership, and participation across teams."
        ),
    },
    "Department": {
        "review_area": "Department-Level Risk",
        "management_hypothesis": (
            "Department differences may be associated with retention outcomes. "
            "Review whether risk is concentrated in specific functions or operating units."
        ),
    },
    "Occupation": {
        "review_area": "Occupational Structure",
        "management_hypothesis": (
            "Occupation differences may be associated with retention outcomes. "
            "Review role-specific workforce dynamics, labor-market exposure, and occupation-level risk."
        ),
    },
    "Career Mobility": {
        "review_area": "Career Mobility",
        "management_hypothesis": (
            "Prior career mobility may be associated with retention outcomes. "
            "Review whether external mobility history differs across high-risk employee groups."
        ),
    },
    "Performance": {
        "review_area": "Performance",
        "management_hypothesis": (
            "Performance indicators may be associated with retention outcomes. "
            "Review whether performance ratings align with retention, rewards, and progression patterns."
        ),
    },
    "Education": {
        "review_area": "Education Profile",
        "management_hypothesis": (
            "Education profile may be associated with retention outcomes. "
            "Review whether education-related differences reflect role mix, career path, or labor-market dynamics."
        ),
    },
    "Education Field": {
        "review_area": "Education Field",
        "management_hypothesis": (
            "Education field may be associated with retention outcomes. "
            "Review whether field-of-study differences reflect role specialization or labor-market alternatives."
        ),
    },
        "Employee Experience": {
        "review_area": "Employee Experience",
        "management_hypothesis": (
            "Workforce experience may be associated with retention outcomes. "
            "Review whether less-experienced employee groups exhibit elevated attrition risk, "
            "and evaluate onboarding, mentoring, development, and career support programs."
        ),
    },
    "Employee Tenure": {
        "review_area": "Employee Tenure",
        "management_hypothesis": (
            "Employee tenure may be associated with retention outcomes. "
            "Review whether attrition is concentrated among newer employees and evaluate "
            "onboarding quality, early-tenure support, and retention during critical tenure periods."
        ),
    },
    "Role Stability": {
        "review_area": "Role Stability",
        "management_hypothesis": (
            "Time in current role may be associated with retention outcomes. "
            "Review role stagnation, internal mobility timing, promotion timing, and role transition patterns."
        ),
    },
}


def build_driver_recommendations(
    driver_table: pd.DataFrame,
) -> Tuple[pd.DataFrame, DriverRecommendationReport]:

    warnings: List[str] = []
    errors: List[str] = []

    required_cols = [
        "driver_group",
        "driver_variable",
        "association_value",
        "direction",
        "actionability",
    ]

    missing = [
        col for col in required_cols
        if col not in driver_table.columns
    ]

    if missing:
        errors.append(f"Missing required columns: {missing}")

        return pd.DataFrame(), DriverRecommendationReport(
            recommendations_generated=0,
            warnings=warnings,
            errors=errors,
        )

    df = driver_table.copy()

    df["association_value"] = pd.to_numeric(
        df["association_value"],
        errors="coerce",
    )

    df = df[df["association_value"].notna()].copy()

    if df.empty:
        errors.append("No valid driver associations available.")

        return pd.DataFrame(), DriverRecommendationReport(
            recommendations_generated=0,
            warnings=warnings,
            errors=errors,
        )

    rows = []

    for _, row in df.iterrows():

        driver_group = str(row["driver_group"])
        driver_variable = str(row["driver_variable"])
        actionability = str(row["actionability"])
        association_value = float(row["association_value"])
        direction = str(row["direction"])

        hypothesis_entry = HYPOTHESIS_LIBRARY.get(
            driver_group,
            {
                "review_area": "Further Investigation",
                "management_hypothesis": (
                    "This factor shows a statistical association with attrition risk. "
                    "Additional organizational analysis may be required to determine "
                    "whether intervention opportunities exist."
                ),
            },
        )

        if association_value > 0:
            observed_pattern = (
                "Higher values or the highlighted category are associated with "
                "higher predicted attrition risk."
            )
        else:
            observed_pattern = (
                "Higher values or the highlighted category are associated with "
                "lower predicted attrition risk."
            )

        rows.append(
            {
                "driver_group": driver_group,
                "supporting_variable": driver_variable,
                "association_value": association_value,
                "direction": direction,
                "actionability": actionability,
                "review_area": hypothesis_entry["review_area"],
                "observed_pattern": observed_pattern,
                "management_hypothesis": hypothesis_entry["management_hypothesis"],
            }
        )

    output = pd.DataFrame(rows)

    output["absolute_association_value"] = (
        output["association_value"].abs()
    )

    output = (
        output
        .sort_values(
            "absolute_association_value",
            ascending=False,
        )
        .drop(columns=["absolute_association_value"])
        .reset_index(drop=True)
    )

    output["hypothesis_rank"] = range(
        1,
        len(output) + 1,
    )

    output = output[
        [
            "hypothesis_rank",
            "driver_group",
            "supporting_variable",
            "association_value",
            "direction",
            "actionability",
            "review_area",
            "observed_pattern",
            "management_hypothesis",
        ]
    ]

    warnings.append(
        "Management hypotheses are evidence-based investigation areas only. "
        "They do not establish causality and should not be interpreted as automatic personnel decisions."
    )

    return output, DriverRecommendationReport(
        recommendations_generated=len(output),
        warnings=warnings,
        errors=errors,
    )
