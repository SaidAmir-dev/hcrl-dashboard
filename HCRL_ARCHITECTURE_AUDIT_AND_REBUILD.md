# HCRL Architecture Audit and Rebuild Plan

## Current weakness fixed in this rebuild

The previous Streamlit app mixed IBM-specific data detection, model fitting, economic assumptions, O*NET matching, decision text, and UI in one file. That made the product look complete, but it was not yet a general enterprise architecture.

This rebuild separates HCRL into modules:

1. `hcrl_schema.py` — validates uploaded company data and maps it into a universal HCRL schema.
2. `hcrl_risk_engine.py` — estimates attrition probability only from observed outcomes or externally supplied probabilities.
3. `hcrl_cost_engine.py` — estimates expected cost only when replacement-cost inputs are supplied; it does not impose a default multiplier.
4. `hcrl_onet_mapper.py` — maps roles to O*NET titles/codes and flags fuzzy matches for review.
5. `hcrl_decision_engine.py` — produces exposure concentration and decision-support interpretation without hard risk cutoffs.
6. `upload_app_enterprise.py` — Streamlit UI that calls the modules instead of embedding business logic directly.

## Important methodological decisions

### No arbitrary replacement-cost multiplier

The new cost engine does not use a default `0.5` multiplier. A company must provide either:

- `replacement_cost`, or
- `replacement_cost_multiplier`.

If neither exists, expected cost remains unavailable rather than fabricated.

### No arbitrary stress multiplier

The old `predicted_risk * stress` logic was removed. A future stress engine should be tied to macroeconomic or labor-market covariates, not a free slider.

### No IBM dependency in core architecture

IBM is now treated as one possible source adapter. The target architecture is:

Company Data → Validation Layer → HCRL Schema → Risk Engine → Cost Engine → O*NET Layer → Decision-Support Engine → Dashboard

### No prescriptive firing recommendations

The decision engine ranks workforce exposure and provides decision-support interpretation. It does not recommend terminating employees.

## Recommended next step

Replace the existing GitHub `upload_app.py` with `upload_app_enterprise.py` after uploading the new module files to the same repository.

Then test with three CSV types:

1. IBM HR attrition dataset.
2. A company-like file with `predicted_attrition_probability`, `annual_wage`, and `replacement_cost_multiplier`.
3. A company-like file with `separation_outcome`, employee features, `annual_wage`, and `replacement_cost`.
