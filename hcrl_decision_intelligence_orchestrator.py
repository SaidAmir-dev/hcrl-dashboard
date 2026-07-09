from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from hcrl_decision_candidate_engine import build_decision_candidates
from hcrl_evidence_evaluation_engine import evaluate_decision_evidence
from hcrl_decision_ranking_engine import rank_decisions
from hcrl_strategy_builder import build_strategy_portfolio

@dataclass
class DecisionIntelligenceReport:
    warnings: List[str]
    errors: List[str]

def build_decision_intelligence_outputs(driver_evidence_df: pd.DataFrame, objective_key: str = "reduce_attrition", references_df: pd.DataFrame | None = None) -> Tuple[Dict[str, pd.DataFrame], DecisionIntelligenceReport]:
    warnings: List[str] = []
    errors: List[str] = []
    candidates, candidate_report = build_decision_candidates(driver_evidence_df=driver_evidence_df, objective_key=objective_key)
    warnings.extend(candidate_report.warnings); errors.extend(candidate_report.errors)
    if errors: return {}, DecisionIntelligenceReport(warnings, errors)
    evaluated, evaluation_report = evaluate_decision_evidence(decision_candidates_df=candidates, references_df=references_df)
    warnings.extend(evaluation_report.warnings); errors.extend(evaluation_report.errors)
    if errors: return {"decision_candidates": candidates}, DecisionIntelligenceReport(warnings, errors)
    ranked, ranking_report = rank_decisions(evaluated)
    warnings.extend(ranking_report.warnings); errors.extend(ranking_report.errors)
    if errors: return {"decision_candidates": candidates, "evaluated_decisions": evaluated}, DecisionIntelligenceReport(warnings, errors)
    strategies, strategy_report = build_strategy_portfolio(ranked)
    warnings.extend(strategy_report.warnings); errors.extend(strategy_report.errors)
    return {"decision_candidates": candidates, "evaluated_decisions": evaluated, "ranked_decisions": ranked, "strategy_portfolio": strategies}, DecisionIntelligenceReport(warnings, errors)
