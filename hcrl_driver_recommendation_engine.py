"""
HCRL Driver Recommendation Engine

Purpose
-------
Convert statistical attrition drivers into
evidence-based management hypotheses.

This module DOES NOT:

- prove causality
- prescribe actions
- automate decisions
- recommend layoffs

It only translates observed statistical
associations into areas for management review.
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


def build_driver_recommendations(
    driver_table: pd.DataFrame,
) -> Tuple[pd.DataFrame, DriverRecommendationReport]:

    warnings: List[str] = []
    errors: List[str] = []

    required_cols = [
        "driver_variable",
        "association_value",
        "direction",
    ]

    missing = [
        col
        for col in required_cols
        if col not in driver_table.columns
    ]

    if missing:
        errors.append(
            f"Missing required columns: {missing}"
        )

        return (
            pd.DataFrame(),
            DriverRecommendationReport(
                recommendations_generated=0,
                warnings=warnings,
                errors=errors,
            ),
        )

    df = driver_table.copy()

    recommendations = []

    for _, row in df.iterrows():

        driver = str(row["driver_variable"])

        evidence = float(
            row["association_value"]
        )

        direction = str(
            row["direction"]
        )

        hypothesis = (
            "Additional review may be warranted."
        )

        review_area = (
            "Management review"
        )

        driver_lower = driver.lower()

        # -------------------------------------
        # Compensation
        # -------------------------------------

        if (
            "income" in driver_lower
            or "salary" in driver_lower
            or "pay" in driver_lower
        ):

            review_area = "Compensation"

            hypothesis = (
                "Compensation differences may be associated "
                "with workforce retention outcomes. "
                "Review market competitiveness and pay structure."
            )

        # -------------------------------------
        # Tenure
        # -------------------------------------

        elif (
            "yearsatcompany" in driver_lower
            or "totalworkingyears" in driver_lower
            or "tenure" in driver_lower
        ):

            review_area = "Employee Tenure"

            hypothesis = (
                "Workforce tenure may be associated with "
                "retention outcomes. Review onboarding, "
                "early-career support, and retention patterns."
            )

        # -------------------------------------
        # Manager relationship
        # -------------------------------------

        elif (
            "manager" in driver_lower
            or "currmanager" in driver_lower
        ):

            review_area = "Management Quality"

            hypothesis = (
                "Manager continuity or management structure "
                "may be associated with retention outcomes. "
                "Review leadership effectiveness and team stability."
            )

        # -------------------------------------
        # Career progression
        # -------------------------------------

        elif (
            "joblevel" in driver_lower
            or "promotion" in driver_lower
            or "role" in driver_lower
        ):

            review_area = "Career Progression"

            hypothesis = (
                "Career advancement opportunities may be associated "
                "with retention outcomes. Review internal mobility "
                "and promotion pathways."
            )

        # -------------------------------------
        # Work-life balance
        # -------------------------------------

        elif (
            "worklifebalance" in driver_lower
        ):

            review_area = "Work-Life Balance"

            hypothesis = (
                "Work-life balance indicators may be associated "
                "with retention outcomes. Review workload patterns "
                "and employee experience metrics."
            )

        # -------------------------------------
        # Stock compensation
        # -------------------------------------

        elif (
            "stockoption" in driver_lower
        ):

            review_area = "Incentive Structure"

            hypothesis = (
                "Long-term incentive structures may be associated "
                "with retention outcomes. Review compensation design "
                "and retention incentives."
            )

        # -------------------------------------
        # Occupation
        # -------------------------------------

        elif (
            "job_title" in driver_lower
            or "jobrole" in driver_lower
            or "onet" in driver_lower
            or "occupation" in driver_lower
        ):

            review_area = "Occupational Structure"

            hypothesis = (
                "Certain occupations may exhibit different retention "
                "patterns. Review occupation-specific workforce dynamics."
            )

        recommendations.append(
            {
                "driver_variable": driver,
                "association_value": evidence,
                "direction": direction,
                "review_area": review_area,
                "management_hypothesis": hypothesis,
            }
        )

    output = pd.DataFrame(
        recommendations
    )

    output = output.sort_values(
        "association_value",
        key=abs,
        ascending=False,
    )

    warnings.append(
        "Recommendations are evidence-based hypotheses only. "
        "They do not establish causality and should not be "
        "interpreted as automatic personnel decisions."
    )

    return (
        output,
        DriverRecommendationReport(
            recommendations_generated=len(output),
            warnings=warnings,
            errors=errors,
        ),
    )
