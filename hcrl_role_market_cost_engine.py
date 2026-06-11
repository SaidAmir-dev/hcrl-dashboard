"""HCRL role-market replacement cost engine.

Purpose:
Estimate replacement-cost exposure when a company does not provide its own
replacement cost inputs.

This engine avoids fake precision.

It uses role complexity tiers derived from observable role signals:
- wage level
- seniority/title
- management status
- occupation family

Outputs are low/base/high replacement-cost estimates.

Important:
These are not final audited accounting numbers.
They are economic exposure estimates for decision support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class RoleMarketCostReport:
    cost_source: str
    n_observations: int
    warnings: List[str]
    errors: List[str]


def _clean(x) -> str:
    return str(x).lower().strip()


def _detect_title_complexity(title: str) -> str:
    title = _clean(title)

    if any(x in title for x in ["chief", "vp", "vice president", "director", "head"]):
        return "executive_or_director"

    if any(x in title for x in ["manager", "lead", "principal", "senior"]):
        return "manager_or_senior"

    if any(x in title for x in ["scientist", "engineer", "developer", "analyst", "specialist"]):
        return "specialized_professional"

    if any(x in title for x in ["technician", "representative", "associate", "coordinator"]):
        return "operational_role"

    return "general_role"


def _detect_cost_tier(row: pd.Series) -> str:
    title = _clean(row.get("job_title", ""))
    family = _clean(row.get("title_function", ""))
    wage = pd.to_numeric(row.get("annual_wage", None), errors="coerce")

    title_complexity = _detect_title_complexity(title)

    if title_complexity == "executive_or_director":
        return "high_complexity"

    if title_complexity == "manager_or_senior":
        return "upper_mid_complexity"

    if family in ["software_it", "research_science", "finance"]:
        return "upper_mid_complexity"

    if title_complexity == "specialized_professional":
        return "mid_complexity"

    if title_complexity == "operational_role":
        return "lower_mid_complexity"

    if pd.notna(wage):
        if wage >= 120000:
            return "high_complexity"
        if wage >= 80000:
            return "upper_mid_complexity"
        if wage >= 45000:
            return "mid_complexity"

    return "lower_mid_complexity"


REPLACEMENT_COST_MULTIPLIERS = {
    # Low, base, high.
    # These are deliberately ranges, not fake precise values.
    "lower_mid_complexity": (0.20, 0.35, 0.50),
    "mid_complexity": (0.35, 0.60, 0.90),
    "upper_mid_complexity": (0.60, 1.00, 1.50),
    "high_complexity": (1.00, 1.50, 2.00),
}


def estimate_role_market_replacement_cost(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, RoleMarketCostReport]:

    out = df.copy()
    warnings: List[str] = []
    errors: List[str] = []

    if "annual_wage" not in out.columns:
        errors.append(
            "annual_wage is required for market replacement-cost estimation."
        )
        return out, RoleMarketCostReport(
            cost_source="replacement_cost_unavailable",
            n_observations=len(out),
            warnings=warnings,
            errors=errors,
        )

    out["annual_wage"] = pd.to_numeric(out["annual_wage"], errors="coerce")

    if out["annual_wage"].isna().all():
        errors.append(
            "annual_wage exists but contains no usable numeric values."
        )
        return out, RoleMarketCostReport(
            cost_source="replacement_cost_unavailable",
            n_observations=len(out),
            warnings=warnings,
            errors=errors,
        )

    out["replacement_cost_tier"] = out.apply(_detect_cost_tier, axis=1)

    out["replacement_cost_multiplier_low"] = out["replacement_cost_tier"].map(
        lambda tier: REPLACEMENT_COST_MULTIPLIERS[tier][0]
    )

    out["replacement_cost_multiplier_base"] = out["replacement_cost_tier"].map(
        lambda tier: REPLACEMENT_COST_MULTIPLIERS[tier][1]
    )

    out["replacement_cost_multiplier_high"] = out["replacement_cost_tier"].map(
        lambda tier: REPLACEMENT_COST_MULTIPLIERS[tier][2]
    )

    out["replacement_cost_low"] = (
        out["annual_wage"] * out["replacement_cost_multiplier_low"]
    )

    out["replacement_cost"] = (
        out["annual_wage"] * out["replacement_cost_multiplier_base"]
    )

    out["replacement_cost_high"] = (
        out["annual_wage"] * out["replacement_cost_multiplier_high"]
    )

    warnings.append(
        "Replacement costs are estimated using role-market complexity tiers and wage-based ranges. "
        "For enterprise deployment, these ranges should be calibrated with company-specific HR finance data."
    )

    return out, RoleMarketCostReport(
        cost_source="role_market_replacement_cost_estimate",
        n_observations=len(out),
        warnings=warnings,
        errors=errors,
    )
