"""
HCRL Driver Recommendation Engine

Purpose:
Convert statistical attrition driver groups into evidence-based management
hypotheses.

No causal claims.
No automated employment decisions.
No firing recommendations.
No unsupported prescriptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import math
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
            "Compensation signals may be associated with retention outcomes. "
            "Review pay competitiveness, internal pay equity, salary progression, "
            "and long-term incentive structures in high-risk workforce groups."
        ),
    },
    "Career Progression": {
        "review_area": "Career Progression",
        "management_hypothesis": (
            "Career progression signals may be associated with retention outcomes. "
            "Review promotion pathways, internal mobility, role progression, and "
            "time-in-role patterns."
        ),
    },
    "Employee Experience": {
        "review_area": "Employee Experience",
        "management_hypothesis": (
            "Employee experience signals may be associated with retention outcomes. "
            "Review onboarding quality, early-career support, prior mobility, and "
            "experience differences across workforce segments."
        ),
    },
    "Manager Stability": {
        "review_area": "Management Quality",
        "management_hypothesis": (
            "Manager continuity may be associated with retention outcomes. "
            "Review leadership stability, manager transitions, team structure, "
            "and manager-level retention patterns."
        ),
    },
    "Workload": {
        "review_area": "Workload",
        "management_hypothesis": (
            "Workload signals may be associated with retention outcomes. "
            "Review overtime concentration, staffing levels, scheduling pressure, "
            "and workload distribution."
        ),
    },
    "Work Environment": {
        "review_area": "Work Environment",
        "management_hypothesis": (
            "Work environment signals may be associated with retention outcomes. "
            "Review job satisfaction, employee involvement, relationship quality, "
            "work-life balance, and department-level experience gaps."
        ),
    },
    "Travel / Commute Burden": {
        "review_area": "Travel and Commute Burden",
        "management_hypothesis": (
            "Travel or commute burden may be associated with retention outcomes. "
            "Review travel frequency, commute exposure, location strategy, and "
            "flexibility options for affected groups."
        ),
    },
    "Training and Development": {
        "review_area": "Training and Development",
        "management_hypothesis": (
            "Training participation may be associated with retention outcomes. "
            "Review learning access, onboarding support, skill development, and "
            "development investment by segment."
        ),
    },
    "Department": {
        "review_area": "Department-Level Risk",
        "management_hypothesis": (
            "Department differences may be associated with retention outcomes. "
            "Review whether risk is concentrated in specific functions, operating "
            "units, or management environments."
        ),
    },
    "Occupation": {
        "review_area": "Occupational Structure",
        "management_hypothesis": (
            "Occupation-level differences may be associated with retention outcomes. "
            "Review role-specific workforce dynamics, labor-market exposure, hiring "
            "pipelines, and occupation-level retention risk."
        ),
    },
    "Education": {
        "review_area": "Education Profile",
        "management_hypothesis": (
            "Education profile may be associated with retention outcomes. "
            "Review whether education-related differences reflect role mix, career "
            "path, specialization, or external labor-market alternatives."
        ),
    },
    "Performance": {
        "review_area": "Performance",
        "management_hypothesis": (
            "Performance indicators may be associated with retention outcomes. "
            "Review whether performance ratings align with rewards, progression, "
            "manager feedback, and retention patterns."
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

    df["absolute_association"] = df["association_value"].abs()

    grouped = (
        df.groupby("driver_group")
        .agg(
            evidence_drivers=("driver_variable", "nunique"),
            supporting_variables=(
                "driver_variable",
                lambda x: " | ".join(sorted(set(map(str, x)))),
            ),
            strongest_association=("absolute_association", "max"),
            average_association=("absolute_association", "mean"),
            actionability=("actionability", "first"),
        )
        .reset_index()
    )

    grouped["hypothesis_score"] = (
        grouped["average_association"]
        * grouped["evidence_drivers"].apply(math.sqrt)
    )

    grouped = grouped.sort_values(
        ["hypothesis_score", "strongest_association"],
        ascending=[False, False],
    ).reset_index(drop=True)

    rows = []

    for _, row in grouped.iterrows():

        driver_group = str(row["driver_group"])

        hypothesis_entry = HYPOTHESIS_LIBRARY.get(
            driver_group,
            {
                "review_area": "Further Investigation",
                "management_hypothesis": (
                    "This workforce domain shows a statistical association with attrition risk. "
                    "Additional organizational analysis may be required to determine whether "
                    "intervention opportunities exist."
                ),
            },
        )

        rows.append(
            {
                "driver_group": driver_group,
                "evidence_drivers": int(row["evidence_drivers"]),
                "supporting_variables": row["supporting_variables"],
                "strongest_association": float(row["strongest_association"]),
                "average_association": float(row["average_association"]),
                "hypothesis_score": float(row["hypothesis_score"]),
                "actionability": row["actionability"],
                "review_area": hypothesis_entry["review_area"],
                "management_hypothesis": hypothesis_entry["management_hypothesis"],
            }
        )

    output = pd.DataFrame(rows)

    output["hypothesis_rank"] = range(
        1,
        len(output) + 1,
    )

    output = output[
        [
            "hypothesis_rank",
            "driver_group",
            "evidence_drivers",
            "supporting_variables",
            "strongest_association",
            "average_association",
            "hypothesis_score",
            "actionability",
            "review_area",
            "management_hypothesis",
        ]
    ]

    warnings.append(
        "Management hypotheses are grouped by workforce domain and based on statistical "
        "associations only. They do not establish causality and should not be interpreted "
        "as automatic personnel decisions."
    )

    return output, DriverRecommendationReport(
        recommendations_generated=len(output),
        warnings=warnings,
        errors=errors,
    )
