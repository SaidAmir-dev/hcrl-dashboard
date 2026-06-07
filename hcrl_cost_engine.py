"""HCRL economic exposure engine.

This module converts attrition probabilities into expected workforce cost exposure.

Core formula:
    Expected Attrition Cost = P(separation) * Replacement Cost

Replacement cost can come from:
1. company-supplied replacement_cost
2. company-supplied replacement_cost_multiplier * annual_wage
3. later: externally calibrated role-market replacement cost model

HCRL must not silently invent replacement cost multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd


@dataclass
class CostEngineReport:
    cost_source: str
    n_observations: int
    warnings: List[str]
    errors: List[str]


def estimate_expected_attrition_cost(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, CostEngineReport]:

    out = df.copy()
    warnings: List[str] = []
    errors: List[str] = []

    if "predicted_attrition_probability" not in out.columns:
        errors.append(
            "Predicted attrition probability is missing. Expected cost cannot be estimated."
        )
        out["expected_attrition_cost"] = pd.NA

        return out, CostEngineReport(
            cost_source="unavailable_no_risk_probability",
            n_observations=len(out),
            warnings=warnings,
            errors=errors,
        )

    out["predicted_attrition_probability"] = pd.to_numeric(
        out["predicted_attrition_probability"],
        errors="coerce",
    )

    if out["predicted_attrition_probability"].dropna().empty:
        errors.append(
            "Predicted attrition probability exists but contains no usable numeric values."
        )
        out["expected_attrition_cost"] = pd.NA

        return out, CostEngineReport(
            cost_source="unavailable_invalid_risk_probability",
            n_observations=len(out),
            warnings=warnings,
            errors=errors,
        )

    # Case 1: company gives direct replacement cost estimate
    if "replacement_cost" in out.columns:
        out["replacement_cost"] = pd.to_numeric(
            out["replacement_cost"],
            errors="coerce",
        )

        if out["replacement_cost"].dropna().empty:
            errors.append(
                "Replacement cost field exists but contains no usable numeric values."
            )
            out["expected_attrition_cost"] = pd.NA

            return out, CostEngineReport(
                cost_source="unavailable_invalid_replacement_cost",
                n_observations=len(out),
                warnings=warnings,
                errors=errors,
            )

        if out["replacement_cost"].dropna().lt(0).any():
            errors.append("Replacement cost cannot contain negative values.")
            out["expected_attrition_cost"] = pd.NA

            return out, CostEngineReport(
                cost_source="invalid_replacement_cost",
                n_observations=len(out),
                warnings=warnings,
                errors=errors,
            )

        out["expected_attrition_cost"] = (
            out["predicted_attrition_probability"] * out["replacement_cost"]
        )

        warnings.append(
            "Expected attrition cost estimated using company-supplied replacement cost."
        )

        return out, CostEngineReport(
            cost_source="company_supplied_replacement_cost",
            n_observations=len(out),
            warnings=warnings,
            errors=errors,
        )

    # Case 2: company gives replacement multiplier and wage
    if "replacement_cost_multiplier" in out.columns and "annual_wage" in out.columns:
        out["replacement_cost_multiplier"] = pd.to_numeric(
            out["replacement_cost_multiplier"],
            errors="coerce",
        )
        out["annual_wage"] = pd.to_numeric(
            out["annual_wage"],
            errors="coerce",
        )

        if out["replacement_cost_multiplier"].dropna().empty:
            errors.append(
                "Replacement cost multiplier exists but contains no usable numeric values."
            )
            out["expected_attrition_cost"] = pd.NA

            return out, CostEngineReport(
                cost_source="unavailable_invalid_multiplier",
                n_observations=len(out),
                warnings=warnings,
                errors=errors,
            )

        if out["annual_wage"].dropna().empty:
            errors.append(
                "Annual wage exists but contains no usable numeric values."
            )
            out["expected_attrition_cost"] = pd.NA

            return out, CostEngineReport(
                cost_source="unavailable_invalid_wage",
                n_observations=len(out),
                warnings=warnings,
                errors=errors,
            )

        if out["replacement_cost_multiplier"].dropna().lt(0).any():
            errors.append("Replacement cost multiplier cannot contain negative values.")
            out["expected_attrition_cost"] = pd.NA

            return out, CostEngineReport(
                cost_source="invalid_multiplier",
                n_observations=len(out),
                warnings=warnings,
                errors=errors,
            )

        if out["annual_wage"].dropna().lt(0).any():
            errors.append("Annual wage cannot contain negative values.")
            out["expected_attrition_cost"] = pd.NA

            return out, CostEngineReport(
                cost_source="invalid_wage",
                n_observations=len(out),
                warnings=warnings,
                errors=errors,
            )

        out["replacement_cost"] = (
            out["replacement_cost_multiplier"] * out["annual_wage"]
        )

        out["expected_attrition_cost"] = (
            out["predicted_attrition_probability"] * out["replacement_cost"]
        )

        warnings.append(
            "Expected attrition cost estimated using company-supplied replacement cost multiplier."
        )

        return out, CostEngineReport(
            cost_source="company_supplied_replacement_multiplier",
            n_observations=len(out),
            warnings=warnings,
            errors=errors,
        )

    # Case 3: no defensible replacement-cost source yet
    out["replacement_cost"] = pd.NA
    out["expected_attrition_cost"] = pd.NA

    warnings.append(
        "No company-supplied replacement cost or replacement cost multiplier detected. "
        "HCRL will not fabricate replacement costs. Next architecture step: build "
        "externally calibrated role-market replacement cost model."
    )

    return out, CostEngineReport(
        cost_source="external_replacement_cost_model_required",
        n_observations=len(out),
        warnings=warnings,
        errors=errors,
    )
