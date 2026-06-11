"""HCRL O*NET occupation resolver.

Generalized occupation mapping for any company.

Principle:
Do NOT rely on raw fuzzy title matching alone.

Use:
1. exact O*NET-SOC code match
2. exact title match
3. title normalization
4. occupation candidate generation
5. candidate-constrained fuzzy matching
6. confidence/status audit layer
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import pandas as pd

from hcrl_title_normalizer import normalize_title
from hcrl_occupation_candidate_engine import (
    generate_candidate_titles,
    filter_onet_reference_to_candidates,
)


AUTO_ACCEPT_THRESHOLD = 0.88
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
        .replace(",", " ")
        .replace(".", " ")
    )


def _compact(x) -> str:
    return "".join(_clean(x).split())


def _token_set(x: str) -> set:
    return set(_clean(x).split())


def _token_overlap_score(a: str, b: str) -> float:
    a_tokens = _token_set(a)
    b_tokens = _token_set(b)

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _combined_similarity(a: str, b: str) -> float:
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


def _prepare_reference(
    onet_reference: pd.DataFrame,
) -> Tuple[pd.DataFrame, str, Optional[str]]:

    ref = onet_reference.copy()

    title_col = "Title" if "Title" in ref.columns else None
    code_col = "O*NET-SOC Code" if "O*NET-SOC Code" in ref.columns else None

    if title_col is None:
        raise ValueError("O*NET reference must contain a Title column.")

    ref["_clean_title"] = ref[title_col].apply(_clean)
    ref["_compact_title"] = ref[title_col].apply(_compact)

    if code_col is not None:
        ref[code_col] = ref[code_col].astype(str).str.strip()

    return ref, title_col, code_col


def _exact_code_match(
    value: str,
    ref: pd.DataFrame,
    code_col: Optional[str],
) -> Optional[pd.Series]:

    if code_col is None:
        return None

    value_str = str(value).strip()
    matches = ref[ref[code_col].astype(str).str.strip() == value_str]

    if matches.empty:
        return None

    return matches.iloc[0]


def _exact_title_match(
    value: str,
    ref: pd.DataFrame,
    title_col: str,
) -> Optional[pd.Series]:

    value_clean = _clean(value)
    value_compact = _compact(value)

    matches = ref[
        (ref["_clean_title"] == value_clean)
        | (ref["_compact_title"] == value_compact)
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def _best_fuzzy_title_match(
    value: str,
    ref: pd.DataFrame,
    title_col: str,
) -> Tuple[pd.Series, float]:

    scores = ref[title_col].apply(lambda title: _combined_similarity(value, title))
    best_idx = scores.idxmax()

    return ref.loc[best_idx], float(scores.loc[best_idx])


def _classify_match(score: float, method: str) -> str:
    if method in {
        "exact_soc_code",
        "exact_title",
        "normalized_exact_title",
        "candidate_exact_title",
    }:
        return "accepted"

    if score >= AUTO_ACCEPT_THRESHOLD:
        return "accepted"

    if score >= REVIEW_THRESHOLD:
        return "review_required"

    return "unmatched"


def _base_unmatched_return(
    method: str,
    score,
    normalized,
    candidate_titles: List[str],
) -> Dict[str, object]:

    return {
        "matched_onet_title": pd.NA,
        "matched_onet_code": pd.NA,
        "onet_match_score": score,
        "onet_match_method": method,
        "onet_match_status": "unmatched",
        "normalized_title": normalized.canonical_title,
        "title_function": normalized.detected_function,
        "title_level": normalized.detected_level,
        "title_normalization_method": normalized.normalization_method,
        "candidate_titles": " | ".join(candidate_titles) if candidate_titles else pd.NA,
    }


def _map_single_role(
    value,
    role_col: str,
    ref: pd.DataFrame,
    title_col: str,
    code_col: Optional[str],
    department_value=None,
) -> Dict[str, object]:

    normalized = normalize_title(value, department_value)

    if pd.isna(value) or str(value).strip() == "":
        return _base_unmatched_return(
            method="missing_role",
            score=pd.NA,
            normalized=normalized,
            candidate_titles=[],
        )

    if role_col in {"occupation_code", "matched_onet_code", "soc_code"}:
        code_match = _exact_code_match(str(value), ref, code_col)

        if code_match is not None:
            return {
                "matched_onet_title": code_match[title_col],
                "matched_onet_code": code_match.get(code_col, pd.NA),
                "onet_match_score": 1.0,
                "onet_match_method": "exact_soc_code",
                "onet_match_status": "accepted",
                "normalized_title": normalized.canonical_title,
                "title_function": normalized.detected_function,
                "title_level": normalized.detected_level,
                "title_normalization_method": normalized.normalization_method,
                "candidate_titles": pd.NA,
            }

    title_match = _exact_title_match(str(value), ref, title_col)

    if title_match is not None:
        return {
            "matched_onet_title": title_match[title_col],
            "matched_onet_code": title_match.get(code_col, pd.NA),
            "onet_match_score": 1.0,
            "onet_match_method": "exact_title",
            "onet_match_status": "accepted",
            "normalized_title": normalized.canonical_title,
            "title_function": normalized.detected_function,
            "title_level": normalized.detected_level,
            "title_normalization_method": normalized.normalization_method,
            "candidate_titles": pd.NA,
        }

    role_for_matching = (
        normalized.canonical_title
        if normalized.canonical_title is not None
        else str(value)
    )

    normalized_title_match = _exact_title_match(role_for_matching, ref, title_col)

    if normalized_title_match is not None and normalized.canonical_title is not None:
        return {
            "matched_onet_title": normalized_title_match[title_col],
            "matched_onet_code": normalized_title_match.get(code_col, pd.NA),
            "onet_match_score": 1.0,
            "onet_match_method": "normalized_exact_title",
            "onet_match_status": "accepted",
            "normalized_title": normalized.canonical_title,
            "title_function": normalized.detected_function,
            "title_level": normalized.detected_level,
            "title_normalization_method": normalized.normalization_method,
            "candidate_titles": normalized.canonical_title,
        }

    candidate_titles = generate_candidate_titles(
        job_title=value,
        detected_function=normalized.detected_function,
        department=department_value,
    )

    candidate_ref = filter_onet_reference_to_candidates(
        onet_reference=ref,
        candidate_titles=candidate_titles,
        title_col=title_col,
    )

    best_match, best_score = _best_fuzzy_title_match(
        role_for_matching,
        candidate_ref,
        title_col,
    )

    method = (
        "candidate_constrained_fuzzy_title"
        if candidate_titles
        else "fallback_full_reference_fuzzy_title"
    )

    status = _classify_match(best_score, method)

    if status == "unmatched":
        return _base_unmatched_return(
            method=f"{method}_below_review_threshold",
            score=best_score,
            normalized=normalized,
            candidate_titles=candidate_titles,
        )

    return {
        "matched_onet_title": best_match[title_col],
        "matched_onet_code": best_match.get(code_col, pd.NA),
        "onet_match_score": best_score,
        "onet_match_method": method,
        "onet_match_status": status,
        "normalized_title": normalized.canonical_title,
        "title_function": normalized.detected_function,
        "title_level": normalized.detected_level,
        "title_normalization_method": normalized.normalization_method,
        "candidate_titles": " | ".join(candidate_titles) if candidate_titles else pd.NA,
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
        out["normalized_title"] = pd.NA
        out["title_function"] = pd.NA
        out["title_level"] = pd.NA
        out["title_normalization_method"] = pd.NA
        out["candidate_titles"] = pd.NA

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

    department_col = "department" if "department" in out.columns else None

    mapped_rows: List[Dict[str, object]] = []

    for idx, value in out[role_col].items():
        department_value = out.loc[idx, department_col] if department_col else None

        mapped_rows.append(
            _map_single_role(
                value=value,
                role_col=role_col,
                ref=ref,
                title_col=title_col,
                code_col=code_col,
                department_value=department_value,
            )
        )

    mapped_df = pd.DataFrame(mapped_rows, index=out.index)
    out = pd.concat([out, mapped_df], axis=1)

    accepted = int((out["onet_match_status"] == "accepted").sum())
    review_required = int((out["onet_match_status"] == "review_required").sum())
    unmatched = int((out["onet_match_status"] == "unmatched").sum())

    exact = int(
        out["onet_match_method"]
        .isin([
            "exact_title",
            "exact_soc_code",
            "normalized_exact_title",
            "candidate_exact_title",
        ])
        .sum()
    )

    fuzzy = int(
        out["onet_match_method"]
        .astype(str)
        .str.contains("fuzzy", na=False)
        .sum()
    )

    coverage = (
        float(out["onet_match_status"].isin(["accepted", "review_required"]).mean())
        if len(out)
        else 0.0
    )

    return out, OnetMappingReport(
        role_column=role_col,
        coverage=coverage,
        accepted_matches=accepted,
        review_required_matches=review_required,
        unmatched=unmatched,
        exact_matches=exact,
        fuzzy_matches=fuzzy,
        note=(
            "Occupation mapping uses exact matching, title normalization, "
            "candidate occupation generation, candidate-constrained fuzzy scoring, "
            "and audit status labels. Accepted matches can be used automatically. "
            "Review-required matches should be inspected before enterprise reporting."
        ),
    )
