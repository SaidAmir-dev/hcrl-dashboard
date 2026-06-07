"""O*NET occupation intelligence mapping for HCRL.

This module maps company job titles or occupation codes to O*NET occupations.

Enterprise principle:
Never silently force a weak occupation match.

A weak job-title match can corrupt every downstream HCRL module:
- AI exposure
- strategic human capital importance
- augmentation potential
- automation feasibility
- intervention logic

Therefore, this mapper assigns a mapping status:
    accepted
    review_required
    unmatched
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional, Tuple, Dict, List

import pandas as pd


AUTO_ACCEPT_THRESHOLD = 0.90
REVIEW_THRESHOLD = 0.70


@dataclass
class OnetMappingReport:
    role_column: Optional[str]
    coverage: float
    accepted_matches: int
    review_required_matches: int
    unmatched: int
    exact_matches: int
    fuzzy_matches: int
    note: str


def _clean(x) -> str:
    return (
        str(x)
        .lower()
        .strip()
        .replace("&", "and")
        .replace("/", " ")
        .replace("-", " ")
        .replace("_", " ")
    )


def _compact(x) -> str:
    return (
        _clean(x)
        .replace(" ", "")
        .replace(",", "")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
    )


def _token_set(x: str) -> set:
    return set(_clean(x).split())


def _token_overlap_score(a: str, b: str) -> float:
    a_tokens = _token_set(a)
    b_tokens = _token_set(b)

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens.intersection(b_tokens)) / len(a_tokens.union(b_tokens))


def _combined_similarity(a: str, b: str) -> float:
    """Combine sequence similarity and token-overlap similarity.

    This is not a causal or economic score.
    It is only a mapping-confidence heuristic.
    """

    seq_score = SequenceMatcher(None, _clean(a), _clean(b)).ratio()
    compact_score = SequenceMatcher(None, _compact(a), _compact(b)).ratio()
    token_score = _token_overlap_score(a, b)

    return max(seq_score, compact_score, token_score)


def _detect_role_column(df: pd.DataFrame) -> Optional[str]:
    for col in [
        "occupation_code",
        "matched_onet_code",
        "soc_code",
        "job_title",
        "JobRole",
        "job_role",
        "Title",
        "occupation",
        "Occupation",
        "position",
        "role",
    ]:
        if col in df.columns:
            return col

    return None


def _prepare_reference(onet_reference: pd.DataFrame) -> Tuple[pd.DataFrame, str, Optional[str]]:
    ref = onet_reference.copy()

    title_col = "Title" if "Title" in ref.columns else None
    code_col = "O*NET-SOC Code" if "O*NET-SOC Code" in ref.columns else None

    if title_col is None:
        raise ValueError("O*NET reference must contain a Title column.")

    ref["_clean_title"] = ref[title_col].apply(_clean)
    ref["_compact_title"] = ref[title_col].apply(_compact)

    if code_col is not None:
        ref[code_col] = ref[code_col].astype(str)

    return ref, title_col, code_col


def _exact_code_match(value: str, ref: pd.DataFrame, code_col: Optional[str]) -> Optional[pd.Series]:
    if code_col is None:
        return None

    value_str = str(value).strip()

    matches = ref[ref[code_col].astype(str) == value_str]

    if len(matches) == 0:
        return None

    return matches.iloc[0]


def _exact_title_match(value: str, ref: pd.DataFrame, title_col: str) -> Optional[pd.Series]:
    value_clean = _clean(value)
    value_compact = _compact(value)

    matches = ref[
        (ref["_clean_title"] == value_clean)
        | (ref["_compact_title"] == value_compact)
    ]

    if len(matches) == 0:
        return None

    return matches.iloc[0]


def _best_fuzzy_title_match(value: str, ref: pd.DataFrame, title_col: str) -> Tuple[pd.Series, float]:
    scores = ref[title_col].apply(lambda title: _combined_similarity(value, title))

    best_idx = scores.idxmax()
    best_score = float(scores.loc[best_idx])

    return ref.loc[best_idx], best_score


def _classify_match(score: float, method: str) -> str:
    if method in {"exact_soc_code", "exact_title"}:
        return "accepted"

    if score >= AUTO_ACCEPT_THRESHOLD:
        return "accepted"

    if score >= REVIEW_THRESHOLD:
        return "review_required"

    return "unmatched"


def _map_single_role(
    value,
    role_col: str,
    ref: pd.DataFrame,
    title_col: str,
    code_col: Optional[str],
) -> Dict[str, object]:

    if pd.isna(value) or str(value).strip() == "":
        return {
            "matched_onet_title": pd.NA,
            "matched_onet_code": pd.NA,
            "onet_match_score": pd.NA,
            "onet_match_method": "missing_role",
            "onet_match_status": "unmatched",
        }

    # If role column is occupation-code-like, try exact code first
    if role_col in {"occupation_code", "matched_onet_code", "soc_code"}:
        code_match = _exact_code_match(str(value), ref, code_col)

        if code_match is not None:
            return {
                "matched_onet_title": code_match[title_col],
                "matched_onet_code": code_match.get(code_col, pd.NA),
                "onet_match_score": 1.0,
                "onet_match_method": "exact_soc_code",
                "onet_match_status": "accepted",
            }

    # Try exact title match
    title_match = _exact_title_match(str(value), ref, title_col)

    if title_match is not None:
        return {
            "matched_onet_title": title_match[title_col],
            "matched_onet_code": title_match.get(code_col, pd.NA),
            "onet_match_score": 1.0,
            "onet_match_method": "exact_title",
            "onet_match_status": "accepted",
        }

    # Try fuzzy title match
    best_match, best_score = _best_fuzzy_title_match(str(value), ref, title_col)
    status = _classify_match(best_score, "fuzzy_title")

    if status == "unmatched":
        return {
            "matched_onet_title": pd.NA,
            "matched_onet_code": pd.NA,
            "onet_match_score": best_score,
            "onet_match_method": "fuzzy_title_below_review_threshold",
            "onet_match_status": "unmatched",
        }

    return {
        "matched_onet_title": best_match[title_col],
        "matched_onet_code": best_match.get(code_col, pd.NA),
        "onet_match_score": best_score,
        "onet_match_method": "fuzzy_title",
        "onet_match_status": status,
    }


def map_to_onet(
    df: pd.DataFrame,
    onet_reference: pd.DataFrame,
) -> Tuple[pd.DataFrame, OnetMappingReport]:

    out = df.copy()
    role_col = _detect_role_column(out)

    if role_col is None:
        out["matched_onet_title"] = pd.NA
        out["matched_onet_code"] = pd.NA
        out["onet_match_score"] = pd.NA
        out["onet_match_method"] = "no_role_column"
        out["onet_match_status"] = "unmatched"

        return out, OnetMappingReport(
            role_column=None,
            coverage=0.0,
            accepted_matches=0,
            review_required_matches=0,
            unmatched=len(out),
            exact_matches=0,
            fuzzy_matches=0,
            note="No role/title/code column available. O*NET mapping cannot be performed.",
        )

    ref, title_col, code_col = _prepare_reference(onet_reference)

    mapped_rows: List[Dict[str, object]] = []

    for value in out[role_col]:
        mapped_rows.append(
            _map_single_role(
                value=value,
                role_col=role_col,
                ref=ref,
                title_col=title_col,
                code_col=code_col,
            )
        )

    mapped_df = pd.DataFrame(mapped_rows, index=out.index)
    out = pd.concat([out, mapped_df], axis=1)

    accepted = int((out["onet_match_status"] == "accepted").sum())
    review_required = int((out["onet_match_status"] == "review_required").sum())
    unmatched = int((out["onet_match_status"] == "unmatched").sum())

    exact = int(out["onet_match_method"].isin(["exact_title", "exact_soc_code"]).sum())
    fuzzy = int(out["onet_match_method"].isin(["fuzzy_title"]).sum())

    coverage = float(
        out["onet_match_status"].isin(["accepted", "review_required"]).mean()
    ) if len(out) else 0.0

    return out, OnetMappingReport(
        role_column=role_col,
        coverage=coverage,
        accepted_matches=accepted,
        review_required_matches=review_required,
        unmatched=unmatched,
        exact_matches=exact,
        fuzzy_matches=fuzzy,
        note=(
            "Accepted matches can be used automatically. Review-required matches "
            "should be inspected before enterprise reporting. Unmatched roles are "
            "excluded from O*NET-based AI exposure and workforce transformation modules."
        ),
    )
