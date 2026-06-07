"""Economic exposure engine for HCRL.

No default replacement-cost multiplier is imposed. The company must supply either a
replacement_cost field or a replacement_cost_multiplier field. Otherwise cost exposure
is intentionally unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class CostEngineReport:
    method: str
    warnings: List[str]


def estimate_expected_cost(
    df: pd.DataFrame,
    probability_col: str = "predicted_attrition_probability",
    annual_wage_col: str = "annual_wage",
    multiplier_col: str = "replacement_cost_multiplier",
    replacement_cost_col: str = "replacement_cost",
) -> Tuple[pd.DataFrame, CostEngineReport]:
    out = df.copy()
    warnings: List[str] = []

    if probability_col not in out.columns:
        raise ValueError("Attrition probability is required before estimating expected cost.")
    if annual_wage_col not in out.columns:
        raise ValueError("Annual wage is required for wage-based exposure estimates.")

    out[probability_col] = pd.to_numeric(out[probability_col], errors="coerce")
    out[annual_wage_col] = pd.to_numeric(out[annual_wage_col], errors="coerce")

    if replacement_cost_col in out.columns:
        out[replacement_cost_col] = pd.to_numeric(out[replacement_cost_col], errors="coerce")
        out["expected_attrition_cost"] = out[probability_col] * out[replacement_cost_col]
        return out, CostEngineReport(
            method="direct_replacement_cost_input",
            warnings=warnings,
        )

    if multiplier_col in out.columns:
        out[multiplier_col] = pd.to_numeric(out[multiplier_col], errors="coerce")
        out["replacement_cost"] = out[multiplier_col] * out[annual_wage_col]
        out["expected_attrition_cost"] = out[probability_col] * out["replacement_cost"]
        return out, CostEngineReport(
            method="company_supplied_replacement_cost_multiplier",
            warnings=warnings,
        )

    out["replacement_cost"] = pd.NA
    out["expected_attrition_cost"] = pd.NA
    warnings.append(
        "Cost exposure not estimated because no replacement_cost or replacement_cost_multiplier field was supplied."
    )
    return out, CostEngineReport(method="cost_unavailable_missing_company_input", warnings=warnings)
