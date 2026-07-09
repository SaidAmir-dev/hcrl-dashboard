from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

@dataclass
class DecisionRankingReport:
    decisions_ranked: int
    warnings: List[str]
    errors: List[str]

def rank_decisions(evaluated_decisions_df: pd.DataFrame) -> Tuple[pd.DataFrame, DecisionRankingReport]:
    warnings: List[str] = []
    errors: List[str] = []
    required_cols = ["decision_id", "decision_name", "objective_domain_aligned", "company_evidence_link_count", "linked_modeled_exposure"]
    missing = [col for col in required_cols if col not in evaluated_decisions_df.columns]
    if missing:
        errors.append(f"Missing required evaluated decision columns: {missing}")
        return pd.DataFrame(), DecisionRankingReport(0, warnings, errors)
    df = evaluated_decisions_df.copy()
    df["objective_domain_aligned"] = df["objective_domain_aligned"].astype(bool)
    df["company_evidence_link_count"] = pd.to_numeric(df["company_evidence_link_count"], errors="coerce").fillna(0).astype(int)
    df["linked_modeled_exposure"] = pd.to_numeric(df["linked_modeled_exposure"], errors="coerce").fillna(0.0)
    df["evidence_drivers_sort"] = pd.to_numeric(df["evidence_drivers"], errors="coerce").fillna(0) if "evidence_drivers" in df.columns else 0
    df = df.sort_values(["objective_domain_aligned", "company_evidence_link_count", "linked_modeled_exposure", "evidence_drivers_sort", "decision_name"], ascending=[False, False, False, False, True]).reset_index(drop=True)
    df["decision_rank"] = range(1, len(df) + 1)
    df["ranking_basis"] = "Ranked lexicographically by objective alignment, matched company evidence count, linked modeled exposure, and evidence-driver count. No weighted score is used."
    df = df.drop(columns=["evidence_drivers_sort"])
    warnings.append("Decision ranking uses lexicographic evidence ordering rather than arbitrary weights.")
    return df, DecisionRankingReport(len(df), warnings, errors)
