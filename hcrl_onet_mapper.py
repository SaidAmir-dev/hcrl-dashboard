"""O*NET occupation intelligence mapping for HCRL."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional, Tuple

import pandas as pd


@dataclass
class OnetMappingReport:
    role_column: Optional[str]
    coverage: float
    exact_matches: int
    fuzzy_matches: int
    unmatched: int
    note: str


def _clean(x) -> str:
    return str(x).lower().strip()


def _best_title_match(role: str, ref: pd.DataFrame) -> Tuple[pd.Series, float, str]:
    role_clean = _clean(role)
    if role_clean in ref.index:
        row = ref.loc[role_clean]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row, 1.0, "exact_title"

    scores = ref.index.to_series().apply(lambda t: SequenceMatcher(None, role_clean, t).ratio())
    best_key = scores.idxmax()
    best_score = float(scores.loc[best_key])
    return ref.loc[best_key], best_score, "best_fuzzy_title_review_required"


def map_to_onet(df: pd.DataFrame, onet_reference: pd.DataFrame) -> Tuple[pd.DataFrame, OnetMappingReport]:
    out = df.copy()
    role_col = None
    for c in ["occupation_code", "job_title", "JobRole", "job_role", "Title", "occupation", "Occupation"]:
        if c in out.columns:
            role_col = c
            break

    if role_col is None:
        out["matched_onet_title"] = pd.NA
        out["onet_match_score"] = pd.NA
        out["onet_match_method"] = "no_role_column"
        return out, OnetMappingReport(None, 0.0, 0, 0, len(out), "No role/title/code column available.")

    ref = onet_reference.copy()
    title_col = "Title" if "Title" in ref.columns else None
    code_col = "O*NET-SOC Code" if "O*NET-SOC Code" in ref.columns else None
    if title_col is None:
        raise ValueError("O*NET reference must contain a Title column.")

    ref["_clean_title"] = ref[title_col].apply(_clean)
    ref_by_title = ref.set_index("_clean_title", drop=False)
    ref_by_code = ref.set_index(code_col, drop=False) if code_col else None

    rows = []
    for value in out[role_col]:
        if pd.isna(value) or str(value).strip() == "":
            rows.append({"matched_onet_title": pd.NA, "matched_onet_code": pd.NA, "onet_match_score": pd.NA, "onet_match_method": "missing_role"})
            continue

        if role_col == "occupation_code" and ref_by_code is not None and str(value) in ref_by_code.index:
            r = ref_by_code.loc[str(value)]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            rows.append({"matched_onet_title": r[title_col], "matched_onet_code": r.get(code_col, pd.NA), "onet_match_score": 1.0, "onet_match_method": "exact_soc_code"})
            continue

        r, score, method = _best_title_match(str(value), ref_by_title)
        rows.append({"matched_onet_title": r[title_col], "matched_onet_code": r.get(code_col, pd.NA), "onet_match_score": score, "onet_match_method": method})

    match_df = pd.DataFrame(rows, index=out.index)
    out = pd.concat([out, match_df], axis=1)

    exact = int(out["onet_match_method"].isin(["exact_title", "exact_soc_code"]).sum())
    fuzzy = int((out["onet_match_method"] == "best_fuzzy_title_review_required").sum())
    unmatched = int(out["onet_match_method"].isin(["missing_role", "no_role_column"]).sum())
    coverage = float(out["matched_onet_title"].notna().mean()) if len(out) else 0.0

    return out, OnetMappingReport(
        role_column=role_col,
        coverage=coverage,
        exact_matches=exact,
        fuzzy_matches=fuzzy,
        unmatched=unmatched,
        note="Fuzzy matches are not final enterprise mappings; they require human review or SOC-code confirmation.",
    )
