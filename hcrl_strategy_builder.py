from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

@dataclass
class StrategyBuilderReport:
    strategies_built: int
    warnings: List[str]
    errors: List[str]

def build_strategy_portfolio(ranked_decisions_df: pd.DataFrame) -> Tuple[pd.DataFrame, StrategyBuilderReport]:
    warnings: List[str] = []
    errors: List[str] = []
    required_cols = ["decision_rank", "objective_name", "decision_name", "decision_family", "driver_group", "executive_action_language", "company_evidence_link_count", "linked_modeled_exposure", "evidence_explanation"]
    missing = [col for col in required_cols if col not in ranked_decisions_df.columns]
    if missing:
        errors.append(f"Missing required ranked decision columns: {missing}")
        return pd.DataFrame(), StrategyBuilderReport(0, warnings, errors)
    df = ranked_decisions_df.copy()
    df["linked_modeled_exposure"] = pd.to_numeric(df["linked_modeled_exposure"], errors="coerce").fillna(0.0)
    top_decisions = df.head(5).copy()
    exposure_decisions = df.sort_values("linked_modeled_exposure", ascending=False).head(5).copy()
    broad_rows, used_groups = [], set()
    for _, row in df.iterrows():
        group = str(row["driver_group"])
        if group not in used_groups:
            broad_rows.append(row)
            used_groups.add(group)
        if len(broad_rows) >= 5: break
    broad_decisions = pd.DataFrame(broad_rows)
    specs = [("S001", "Best-Supported Decision Portfolio", "Uses the highest-ranked decisions under the selected objective.", top_decisions), ("S002", "Economic Exposure Portfolio", "Uses decisions linked to the largest modeled workforce exposure.", exposure_decisions), ("S003", "Broad Evidence Coverage Portfolio", "Uses decisions spanning multiple workforce domains to avoid focusing on one signal only.", broad_decisions)]
    rows = []
    for strategy_id, strategy_name, description, subset in specs:
        if subset is None or subset.empty: continue
        rows.append({"strategy_id": strategy_id, "strategy_name": strategy_name, "objective_name": subset["objective_name"].iloc[0], "strategy_description": description, "included_decisions": " | ".join(subset["decision_name"].astype(str).tolist()), "included_domains": " | ".join(sorted(set(subset["driver_group"].astype(str)))), "decision_families": " | ".join(sorted(set(subset["decision_family"].astype(str)))), "linked_modeled_exposure": float(subset["linked_modeled_exposure"].sum()), "company_evidence_links": int(subset["company_evidence_link_count"].sum()), "executive_actions": " | ".join(subset["executive_action_language"].astype(str).tolist()), "evidence_summary": " | ".join(subset["evidence_explanation"].astype(str).tolist()), "limitations": "This strategy portfolio is based on available company evidence and modeled exposure. It does not estimate causal impact or guaranteed financial savings."})
    out = pd.DataFrame(rows)
    warnings.append("Strategy portfolios group evidence-supported decisions. They do not claim guaranteed outcomes.")
    return out, StrategyBuilderReport(len(out), warnings, errors)
