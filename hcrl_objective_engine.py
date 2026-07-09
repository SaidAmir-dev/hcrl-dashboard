from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class ExecutiveObjective:
    key: str
    id: str
    name: str
    description: str
    primary_domains: List[str]

EXECUTIVE_OBJECTIVES: Dict[str, ExecutiveObjective] = {
    "reduce_attrition": ExecutiveObjective(
        key="reduce_attrition", id="EO001", name="Reduce Voluntary Attrition",
        description="Prioritize executive decisions supported by company evidence related to avoidable voluntary workforce loss.",
        primary_domains=["Career Progression", "Compensation", "Manager Stability", "Work Environment", "Workload", "Travel / Commute Burden", "Training and Development", "Department", "Occupation", "Performance"],
    ),
    "protect_critical_talent": ExecutiveObjective(
        key="protect_critical_talent", id="EO002", name="Protect Critical Talent",
        description="Prioritize decisions that protect workforce segments with high modeled economic exposure and strategically important roles.",
        primary_domains=["Career Progression", "Compensation", "Manager Stability", "Work Environment", "Occupation", "Performance"],
    ),
    "increase_internal_mobility": ExecutiveObjective(
        key="increase_internal_mobility", id="EO003", name="Increase Internal Mobility",
        description="Prioritize decisions supported by evidence around advancement, career movement, role progression, and workforce supply.",
        primary_domains=["Career Progression", "Training and Development", "Department", "Occupation"],
    ),
    "prepare_workforce_for_ai": ExecutiveObjective(
        key="prepare_workforce_for_ai", id="EO004", name="Prepare Workforce for AI",
        description="Prioritize decisions related to AI augmentation, task redesign, reskilling, and workforce transformation.",
        primary_domains=["Occupation", "Training and Development", "Department", "Workload"],
    ),
    "reduce_workforce_cost": ExecutiveObjective(
        key="reduce_workforce_cost", id="EO005", name="Reduce Workforce Cost",
        description="Prioritize decisions that reduce expected workforce economic exposure while preserving operational capability.",
        primary_domains=["Occupation", "Department", "Workload", "Career Progression", "Compensation"],
    ),
}

def list_objectives() -> List[ExecutiveObjective]:
    return list(EXECUTIVE_OBJECTIVES.values())

def get_objective(objective_key: str) -> Tuple[ExecutiveObjective, List[str]]:
    warnings: List[str] = []
    if objective_key in EXECUTIVE_OBJECTIVES:
        return EXECUTIVE_OBJECTIVES[objective_key], warnings
    warnings.append(f"Unknown objective '{objective_key}'. Falling back to Reduce Voluntary Attrition.")
    return EXECUTIVE_OBJECTIVES["reduce_attrition"], warnings
