from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

@dataclass
class EvidenceEvaluationReport:
    decisions_evaluated: int
    warnings: List[str]
    errors: List[str]

def evaluate_decision_evidence(decision_candidates_df: pd.DataFrame, references_df: pd.DataFrame | None = None) -> Tuple[pd.DataFrame, EvidenceEvaluationReport]:
    warnings: List[str] = []
    errors: List[str] = []
    required_cols = ["decision_id", "decision_name", "driver_group", "objective_domain_aligned", "company_evidence_link_count", "matched_company_evidence_variables", "linked_modeled_exposure"]
    missing = [col for col in required_cols if col not in decision_candidates_df.columns]
    if missing:
        errors.append(f"Missing required decision candidate columns: {missing}")
        return pd.DataFrame(), EvidenceEvaluationReport(0, warnings, errors)
    df = decision_candidates_df.copy()
    df["company_evidence_link_count"] = pd.to_numeric(df["company_evidence_link_count"], errors="coerce").fillna(0).astype(int)
    df["linked_modeled_exposure"] = pd.to_numeric(df["linked_modeled_exposure"], errors="coerce").fillna(0.0)
    df["company_evidence_status"] = df["company_evidence_link_count"].apply(lambda x: "Company evidence present" if x > 0 else "No direct company variable match")
    df["objective_alignment_status"] = df["objective_domain_aligned"].apply(lambda x: "Aligned with selected objective" if bool(x) else "Not primary for selected objective")
    df["company_evidence_basis"] = df.apply(lambda row: f"Matched company evidence variables: {row['matched_company_evidence_variables']}" if int(row["company_evidence_link_count"]) > 0 else "No direct company evidence variables matched this decision.", axis=1)
    df["economic_exposure_basis"] = df["linked_modeled_exposure"].apply(lambda x: f"Linked modeled exposure: ${float(x):,.0f}" if float(x) > 0 else "No linked modeled exposure available.")
    df["external_evidence_status"] = "External evidence library available" if references_df is not None and not references_df.empty else "External evidence not yet attached"
    df["decision_readiness"] = df.apply(lambda row: "Ready for executive review" if row["objective_domain_aligned"] and row["company_evidence_link_count"] > 0 else "Needs more evidence or different objective", axis=1)
    df["evidence_explanation"] = df.apply(lambda row: f"{row['decision_name']} is supported by {int(row['company_evidence_link_count'])} matched company evidence variable(s) and ${float(row['linked_modeled_exposure']):,.0f} of linked modeled exposure. This is evidence for executive review, not proof of causal effect." if int(row["company_evidence_link_count"]) > 0 else f"{row['decision_name']} is available as a decision option, but the current company evidence does not directly match its required variables.", axis=1)
    warnings.append("Evidence evaluation reports evidence availability only. It does not estimate intervention effects.")
    return df, EvidenceEvaluationReport(len(df), warnings, errors)
