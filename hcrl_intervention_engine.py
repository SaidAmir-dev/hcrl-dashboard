"""HCRL intervention economics engine.

Compares possible workforce interventions only when cost/impact assumptions
are supplied by the company or a validated external model.

No fake ROI.
No arbitrary savings.
No automatic firing recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class InterventionEngineReport:
    n_options: int
    warnings: List[str]
    errors: List[str]


REQUIRED_COLUMNS = [
    "segment",
    "intervention_type",
    "implementation_cost",
    "expected_gross_benefit",
]


VALID_INTERVENTIONS = {
    "retain",
    "retrain",
    "augment",
    "redesign",
    "automate",
}


def evaluate_interventions(
    intervention_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, InterventionEngineReport]:

    warnings: List[str] = []
    errors: List[str] = []

    missing = [col for col in REQUIRED_COLUMNS if col not in intervention_df.columns]

    if missing:
        errors.append(
            f"Missing required intervention assumption columns: {missing}"
        )
        return pd.DataFrame(), InterventionEngineReport(
            n_options=0,
            warnings=warnings,
            errors=errors,
        )

    out = intervention_df.copy()

    out["intervention_type"] = (
        out["intervention_type"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    invalid_interventions = sorted(
        set(out["intervention_type"]) - VALID_INTERVENTIONS
    )

    if invalid_interventions:
        errors.append(
            f"Invalid intervention types detected: {invalid_interventions}. "
            f"Valid types are: {sorted(VALID_INTERVENTIONS)}"
        )
        return pd.DataFrame(), InterventionEngineReport(
            n_options=0,
            warnings=warnings,
            errors=errors,
        )

    out["implementation_cost"] = pd.to_numeric(
        out["implementation_cost"],
        errors="coerce",
    )

    out["expected_gross_benefit"] = pd.to_numeric(
        out["expected_gross_benefit"],
        errors="coerce",
    )

    if out["implementation_cost"].isna().any():
        errors.append("Implementation cost contains non-numeric or missing values.")

    if out["expected_gross_benefit"].isna().any():
        errors.append("Expected gross benefit contains non-numeric or missing values.")

    if out["implementation_cost"].dropna().lt(0).any():
        errors.append("Implementation cost cannot be negative.")

    if out["expected_gross_benefit"].dropna().lt(0).any():
        errors.append("Expected gross benefit cannot be negative.")

    if errors:
        return pd.DataFrame(), InterventionEngineReport(
            n_options=0,
            warnings=warnings,
            errors=errors,
        )

    out["expected_net_value"] = (
        out["expected_gross_benefit"] - out["implementation_cost"]
    )

    out["roi"] = out["expected_net_value"] / out["implementation_cost"]

    out.loc[out["implementation_cost"] == 0, "roi"] = pd.NA

    out["decision_status"] = "economically_evaluated"

    out["decision_note"] = (
        "This intervention is evaluated using supplied or externally validated "
        "cost and benefit assumptions. It is not an automatic personnel decision."
    )

    out = out.sort_values(
        ["segment", "expected_net_value"],
        ascending=[True, False],
    )

    warnings.append(
        "Intervention ROI depends entirely on supplied or externally validated assumptions. "
        "HCRL does not fabricate intervention effects."
    )

    return out, InterventionEngineReport(
        n_options=len(out),
        warnings=warnings,
        errors=errors,
    )
