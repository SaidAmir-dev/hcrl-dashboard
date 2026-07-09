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
    "Compensation": {"review_area": "Compensation", "management_hypothesis": "Compensation signals may be associated with retention outcomes. Review pay competitiveness, internal pay equity, salary progression, and long-term incentive structures in high-risk workforce groups."},
    "Career Progression": {"review_area": "Career Progression", "management_hypothesis": "Career progression signals may be associated with retention outcomes. Review promotion pathways, internal mobility, role progression, and time-in-role patterns."},
    "Employee Experience": {"review_area": "Employee Experience", "management_hypothesis": "Employee experience signals may be associated with retention outcomes. Review onboarding quality, early-career support, prior mobility, and experience differences."},
    "Manager Stability": {"review_area": "Management Quality", "management_hypothesis": "Manager continuity may be associated with retention outcomes. Review leadership stability, manager transitions, team structure, and manager-level retention patterns."},
    "Workload": {"review_area": "Workload", "management_hypothesis": "Workload signals may be associated with retention outcomes. Review overtime concentration, staffing levels, scheduling pressure, and workload distribution."},
    "Work Environment": {"review_area": "Work Environment", "management_hypothesis": "Work environment signals may be associated with retention outcomes. Review job satisfaction, employee involvement, relationship quality, work-life balance, and department gaps."},
    "Travel / Commute Burden": {"review_area": "Travel and Commute Burden", "management_hypothesis": "Travel or commute burden may be associated with retention outcomes. Review travel frequency, commute exposure, location strategy, and flexibility options."},
    "Training and Development": {"review_area": "Training and Development", "management_hypothesis": "Training participation may be associated with retention outcomes. Review learning access, onboarding support, skill development, and development investment."},
    "Department": {"review_area": "Department-Level Risk", "management_hypothesis": "Department differences may be associated with retention outcomes. Review whether risk is concentrated in specific functions, operating units, or management environments."},
    "Occupation": {"review_area": "Occupational Structure", "management_hypothesis": "Occupation-level differences may be associated with retention outcomes. Review role-specific workforce dynamics, labor-market exposure, hiring pipelines, and occupation-level risk."},
    "Education": {"review_area": "Education Profile", "management_hypothesis": "Education profile may be associated with retention outcomes. Review whether education-related differences reflect role mix, career path, specialization, or labor-market alternatives."},
    "Performance": {"review_area": "Performance", "management_hypothesis": "Performance indicators may be associated with retention outcomes. Review whether performance ratings align with rewards, progression, manager feedback, and retention patterns."},
}

def build_driver_recommendations(driver_table: pd.DataFrame) -> Tuple[pd.DataFrame, DriverRecommendationReport]:
    warnings: List[str] = []
    errors: List[str] = []
    required_cols = ["driver_group", "driver_variable", "association_value", "direction", "actionability"]
    missing = [col for col in required_cols if col not in driver_table.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return pd.DataFrame(), DriverRecommendationReport(0, warnings, errors)
    df = driver_table.copy()
    df["association_value"] = pd.to_numeric(df["association_value"], errors="coerce")
    df = df[df["association_value"].notna()].copy()
    if df.empty:
        errors.append("No valid driver associations available.")
        return pd.DataFrame(), DriverRecommendationReport(0, warnings, errors)
    df["absolute_association"] = df["association_value"].abs()
    grouped = df.groupby("driver_group").agg(evidence_drivers=("driver_variable", "nunique"), supporting_variables=("driver_variable", lambda x: " | ".join(sorted(set(map(str, x))))), strongest_association=("absolute_association", "max"), average_association=("absolute_association", "mean"), actionability=("actionability", "first")).reset_index()
    grouped["actionability_priority"] = grouped["actionability"].map({"Actionable": 1, "Descriptive": 0}).fillna(0)
    grouped = grouped.sort_values(["actionability_priority", "evidence_drivers", "strongest_association", "average_association"], ascending=[False, False, False, False]).reset_index(drop=True)
    rows = []
    for _, row in grouped.iterrows():
        driver_group = str(row["driver_group"])
        hypothesis_entry = HYPOTHESIS_LIBRARY.get(driver_group, {"review_area": "Further Investigation", "management_hypothesis": "This workforce domain shows a statistical association with attrition risk. Additional organizational analysis may be required to determine whether decision opportunities exist."})
        rows.append({"driver_group": driver_group, "evidence_drivers": int(row["evidence_drivers"]), "supporting_variables": row["supporting_variables"], "strongest_association": float(row["strongest_association"]), "average_association": float(row["average_association"]), "actionability": row["actionability"], "review_area": hypothesis_entry["review_area"], "management_hypothesis": hypothesis_entry["management_hypothesis"], "ranking_basis": "Ranked by actionability, number of evidence drivers, strongest observed association, and average observed association. No weighted hypothesis score is used."})
    output = pd.DataFrame(rows)
    output["hypothesis_rank"] = range(1, len(output) + 1)
    output = output[["hypothesis_rank", "driver_group", "evidence_drivers", "supporting_variables", "strongest_association", "average_association", "actionability", "review_area", "management_hypothesis", "ranking_basis"]]
    warnings.append("Management hypotheses are based on statistical associations only. They do not establish causality or prescribe automatic personnel decisions.")
    return output, DriverRecommendationReport(len(output), warnings, errors)
