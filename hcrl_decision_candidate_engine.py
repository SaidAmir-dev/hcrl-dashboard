from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from hcrl_objective_engine import get_objective

@dataclass
class DecisionCandidateReport:
    candidates_generated: int
    warnings: List[str]
    errors: List[str]

DECISION_LIBRARY: Dict[str, List[Dict[str, object]]] = {
    "Career Progression": [
        {"decision_id":"D_CP_001","decision_name":"Promotion Process Review","decision_family":"Promotion governance","decision_question":"Should leadership prioritize reviewing promotion timing, criteria, and approval pathways?","required_evidence_variables":["YearsSinceLastPromotion","YearsInCurrentRole","JobLevel","YearsAtCompany"],"executive_action_language":"Begin a structured review of promotion timing, promotion criteria, and approval pathways in the most exposed workforce segments."},
        {"decision_id":"D_CP_002","decision_name":"Internal Mobility Review","decision_family":"Internal mobility","decision_question":"Should leadership review whether internal movement is available before external hiring?","required_evidence_variables":["YearsAtCompany","YearsInCurrentRole","JobRole","Department"],"executive_action_language":"Review whether internal mobility paths are visible, accessible, and used before external hiring in exposed groups."},
        {"decision_id":"D_CP_003","decision_name":"Career Ladder Review","decision_family":"Career ladder design","decision_question":"Should leadership review whether role levels and progression paths are clear?","required_evidence_variables":["JobLevel","YearsInCurrentRole","YearsSinceLastPromotion"],"executive_action_language":"Review whether career ladders, role levels, and advancement criteria are clear for exposed workforce segments."},
    ],
    "Compensation": [
        {"decision_id":"D_COMP_001","decision_name":"Compensation Competitiveness Review","decision_family":"Compensation benchmarking","decision_question":"Should leadership review compensation competitiveness for exposed roles?","required_evidence_variables":["MonthlyIncome","PercentSalaryHike","StockOptionLevel","JobLevel"],"executive_action_language":"Review compensation competitiveness and pay progression for exposed roles."},
        {"decision_id":"D_COMP_002","decision_name":"Pay Compression Review","decision_family":"Pay compression review","decision_question":"Should leadership review pay compression across levels and roles?","required_evidence_variables":["MonthlyIncome","JobLevel","PercentSalaryHike"],"executive_action_language":"Review whether pay progression and job-level differences indicate possible pay compression in exposed workforce groups."},
    ],
    "Manager Stability": [
        {"decision_id":"D_MGR_001","decision_name":"Manager Continuity Review","decision_family":"Manager continuity review","decision_question":"Should leadership review manager continuity in exposed teams?","required_evidence_variables":["YearsWithCurrManager","Manager","Department"],"executive_action_language":"Review manager continuity, team stability, and leadership transitions in exposed workforce areas."},
    ],
    "Work Environment": [
        {"decision_id":"D_ENV_001","decision_name":"Employee Experience Review","decision_family":"Team climate review","decision_question":"Should leadership review employee experience signals in exposed segments?","required_evidence_variables":["EnvironmentSatisfaction","JobSatisfaction","RelationshipSatisfaction","JobInvolvement","WorkLifeBalance"],"executive_action_language":"Review employee experience, satisfaction, involvement, and work-life balance patterns in exposed workforce groups."},
    ],
    "Workload": [
        {"decision_id":"D_WORK_001","decision_name":"Workload and Overtime Review","decision_family":"Workload allocation review","decision_question":"Should leadership review workload and overtime concentration?","required_evidence_variables":["OverTime","WorkLifeBalance","DistanceFromHome"],"executive_action_language":"Review workload, overtime concentration, staffing pressure, and scheduling patterns in exposed workforce segments."},
    ],
    "Training and Development": [
        {"decision_id":"D_LD_001","decision_name":"Training Access Review","decision_family":"Training prioritization","decision_question":"Should leadership review whether development access is aligned with exposed groups?","required_evidence_variables":["TrainingTimesLastYear","Education","EducationField","YearsAtCompany"],"executive_action_language":"Review training access, development participation, and skills pathways for exposed groups."},
    ],
    "Travel / Commute Burden": [
        {"decision_id":"D_TRAVEL_001","decision_name":"Travel and Flexibility Review","decision_family":"Location and flexibility","decision_question":"Should leadership review travel, commute, and flexibility requirements?","required_evidence_variables":["BusinessTravel","DistanceFromHome"],"executive_action_language":"Review travel burden, commute exposure, location strategy, and flexibility options for exposed groups."},
    ],
    "Occupation": [
        {"decision_id":"D_OCC_001","decision_name":"Role-Level Workforce Planning Review","decision_family":"Critical role protection","decision_question":"Should leadership review role-level exposure and workforce supply?","required_evidence_variables":["JobRole","job_title","matched_onet_title","expected_attrition_cost"],"executive_action_language":"Review role-level workforce exposure, hiring pipeline strength, and critical occupation protection."},
        {"decision_id":"D_OCC_002","decision_name":"AI Augmentation Assessment","decision_family":"AI augmentation planning","decision_question":"Should leadership evaluate AI augmentation or task redesign for exposed roles?","required_evidence_variables":["primary_work_type","ai_exposure_score","avg_digital_work","avg_analytical_work","matched_onet_title"],"executive_action_language":"Evaluate whether exposed roles require AI augmentation, task redesign, reskilling, or human oversight planning."},
    ],
    "Department": [
        {"decision_id":"D_DEPT_001","decision_name":"Department-Level Operating Review","decision_family":"Department-level review","decision_question":"Should leadership review local workforce conditions in exposed departments?","required_evidence_variables":["Department","department","expected_attrition_cost","predicted_attrition_probability"],"executive_action_language":"Review local workforce conditions, leadership practices, and operating context in the highest-exposure departments."},
    ],
}

def _split_supporting_variables(value) -> List[str]:
    if pd.isna(value): return []
    parts = []
    for raw in str(value).replace(",", "|").split("|"):
        item = raw.strip()
        if item: parts.append(item)
    return sorted(set(parts))

def build_decision_candidates(driver_evidence_df: pd.DataFrame, objective_key: str = "reduce_attrition") -> Tuple[pd.DataFrame, DecisionCandidateReport]:
    warnings: List[str] = []
    errors: List[str] = []
    objective, objective_warnings = get_objective(objective_key)
    warnings.extend(objective_warnings)
    required_cols = ["driver_group", "supporting_variables"]
    missing = [col for col in required_cols if col not in driver_evidence_df.columns]
    if missing:
        errors.append(f"Missing required driver evidence columns: {missing}")
        return pd.DataFrame(), DecisionCandidateReport(0, warnings, errors)
    df = driver_evidence_df.copy()
    if "linked_modeled_exposure" not in df.columns:
        if "exposure_linked_to_intervention_area" in df.columns:
            df["linked_modeled_exposure"] = df["exposure_linked_to_intervention_area"]
        elif "total_expected_attrition_cost" in df.columns:
            df["linked_modeled_exposure"] = df["total_expected_attrition_cost"]
        else:
            df["linked_modeled_exposure"] = 0.0
    rows = []
    for _, evidence in df.iterrows():
        driver_group = str(evidence["driver_group"])
        supporting_variables = _split_supporting_variables(evidence["supporting_variables"])
        if driver_group not in DECISION_LIBRARY: continue
        for candidate in DECISION_LIBRARY[driver_group]:
            required_vars = list(candidate["required_evidence_variables"])
            matched_vars = sorted(set(supporting_variables).intersection(required_vars))
            exposure_value = pd.to_numeric(evidence.get("linked_modeled_exposure", 0.0), errors="coerce")
            if pd.isna(exposure_value): exposure_value = 0.0
            rows.append({
                "objective_key": objective.key,
                "objective_name": objective.name,
                "objective_domain_aligned": driver_group in objective.primary_domains,
                "driver_group": driver_group,
                "decision_id": candidate["decision_id"],
                "decision_name": candidate["decision_name"],
                "decision_family": candidate["decision_family"],
                "decision_question": candidate["decision_question"],
                "executive_action_language": candidate["executive_action_language"],
                "required_evidence_variables": " | ".join(required_vars),
                "matched_company_evidence_variables": " | ".join(matched_vars),
                "company_evidence_link_count": len(matched_vars),
                "supporting_variables_from_model": " | ".join(supporting_variables),
                "linked_modeled_exposure": float(exposure_value),
                "evidence_drivers": evidence.get("evidence_drivers", pd.NA),
                "actionability": evidence.get("actionability", pd.NA),
            })
    output = pd.DataFrame(rows)
    if output.empty:
        warnings.append("No decision candidates could be generated from the available driver evidence.")
        return output, DecisionCandidateReport(0, warnings, errors)
    warnings.append("Decision candidates are generated from observed company evidence and domain mappings. They are not ranked here and do not claim causal effects.")
    return output, DecisionCandidateReport(len(output), warnings, errors)
