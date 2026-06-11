"""HCRL economic exposure engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from hcrl_role_market_cost_engine import estimate_role_market_replacement_cost


@dataclass
class CostReport:
    cost_source: str
    n_observations: int
    warnings: List[str]
    errors: List[str]


def estimate_expected_attrition_cost(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, CostReport]:

    out = df.copy()
    warnings: List[str] = []
    errors: List[str] = []

    if "predicted_attrition_probability" not in out.columns:
        errors.append(
            "predicted_attrition_probability is required before economic exposure can be estimated."
        )
        return out, CostReport(
            cost_source="cost_unavailable_no_risk",
            n_observations=len(out),
            warnings=warnings,
            errors=errors,
        )

    out["predicted_attrition_probability"] = pd.to_numeric(
        out["predicted_attrition_probability"],
        errors="coerce",
    )

    if "replacement_cost" not in out.columns:
        out, market_report = estimate_role_market_replacement_cost(out)

        warnings.extend(market_report.warnings)
        errors.extend(market_report.errors)

        if market_report.errors:
            return out, CostReport(
                cost_source=market_report.cost_source,
                n_observations=len(out),
                warnings=warnings,
                errors=errors,
            )

        cost_source = market_report.cost_source
    else:
        out["replacement_cost"] = pd.to_numeric(
            out["replacement_cost"],
            errors="coerce",
        )
        cost_source = "company_supplied_replacement_cost"

    out["expected_attrition_cost"] = (
        out["predicted_attrition_probability"] * out["replacement_cost"]
    )

    if "replacement_cost_low" in out.columns:
        out["expected_attrition_cost_low"] = (
            out["predicted_attrition_probability"] * out["replacement_cost_low"]
        )

    if "replacement_cost_high" in out.columns:
        out["expected_attrition_cost_high"] = (
            out["predicted_attrition_probability"] * out["replacement_cost_high"]
        )

    warnings.append(
        "Expected attrition cost is a decision-support estimate, not an audited accounting loss."
    )

    return out, CostReport(
        cost_source=cost_source,
        n_observations=len(out),
        warnings=warnings,
        errors=errors,
    )
