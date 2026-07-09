from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

@dataclass
class ActionIntelligenceReport:
    occupations_analyzed: int
    warnings: List[str]
    errors: List[str]

def build_action_intelligence_table(prioritization_df: pd.DataFrame) -> Tuple[pd.DataFrame, ActionIntelligenceReport]:
    warnings: List[str] = []
    errors: List[str] = []
    required_cols = ["priority_rank", "matched_onet_title", "share_of_total_cost_pct", "avg_attrition_probability", "primary_work_type"]
    missing = [c for c in required_cols if c not in prioritization_df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return pd.DataFrame(), ActionIntelligenceReport(0, warnings, errors)
    df = prioritization_df.copy()
    df["share_of_total_cost_pct"] = pd.to_numeric(df["share_of_total_cost_pct"], errors="coerce")
    df["avg_attrition_probability"] = pd.to_numeric(df["avg_attrition_probability"], errors="coerce")
    df = df.sort_values("priority_rank", ascending=True).reset_index(drop=True)
    def build_decision_focus(row) -> str:
        work_type = str(row["primary_work_type"])
        if work_type == "Digital": return "Role-level workforce planning and AI-readiness review"
        if work_type == "Analytical": return "Analytical work redesign and AI-augmentation assessment"
        if work_type == "Human Interaction": return "Human-centered workforce stability review"
        if work_type == "Physical / Manual": return "Process, staffing, and operational capacity review"
        return "Further executive review"
    def build_action_rationale(row) -> str:
        return (f"Occupation ranked #{int(row['priority_rank'])} by modeled economic exposure. Cost share={row['share_of_total_cost_pct']:.1f}% | Average modeled attrition probability={row['avg_attrition_probability']:.1%} | Primary work profile={row['primary_work_type']}. This identifies where executives should review decision options; it does not prescribe an intervention.")
    df["decision_focus"] = df.apply(build_decision_focus, axis=1)
    df["recommended_action"] = df["decision_focus"]
    df["action_rationale"] = df.apply(build_action_rationale, axis=1)
    final = df[["priority_rank", "matched_onet_title", "share_of_total_cost_pct", "avg_attrition_probability", "primary_work_type", "recommended_action", "action_rationale"]]
    warnings.append("Action Intelligence now identifies executive decision focus areas only. It does not use percentile thresholds or unsupported automation recommendations.")
    return final, ActionIntelligenceReport(len(final), warnings, errors)
