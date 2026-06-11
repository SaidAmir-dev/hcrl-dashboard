"""HCRL O*NET occupation resolver."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import pandas as pd

from hcrl_occupation_aliases import OCCUPATION_ALIASES
from hcrl_title_normalizer import normalize_title
from hcrl_occupation_candidate_engine import generate_candidate_titles


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
        str(x).lower().strip()
        .replace("&", "and").replace("/", " ")
        .replace("-", " ").replace("_", " ")
        .replace(",", " ").replace(".", " ")
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
        "occupation_code", "matched_onet_code", "soc_code",
        "job_title", "JobRole", "job_role", "Title",
        "occupation", "Occupation", "position", "role",
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


def _exact_code_match(value: str, ref: pd.DataFrame, code_col: Optional[str]):
    if code_col is None:
        return None

    matches = ref[ref[code_col].astype(str).str.strip() == str(value).strip()]
    return None if matches.empty else matches.iloc[0]


def _exact_title_match(value: str, ref: pd.DataFrame, title_col: str):
    value_clean = _clean(value)
    value_compact = _compact(value)

    matches = ref[
        (ref["_clean_title"] == value_clean)
        | (ref["_compact_title"] == value_compact)
    ]

    return None if matches.empty else matches.iloc[0]


def _filter_ref_to_candidate_titles(
    ref: pd.DataFrame,
    candidate_titles: List[str],
    title_col: str,
) -> pd.DataFrame:
    if not candidate_titles:
        return ref.iloc[0:0].copy()

    candidate_clean = {_clean(t) for t in candidate_titles}

    return ref[
        ref[title_col].astype(str).apply(lambda t: _clean(t) in candidate_clean)
    ].copy()


def _best_fuzzy_title_match(value: str, ref: pd.DataFrame, title_col: str):
    if ref.empty:
        return None, pd.NA

    scores = ref[title_col].apply(lambda title: _combined_similarity(value, title))
    best_idx = scores.idxmax()

    return ref.loc[best_idx], float(scores.loc[best_idx])


def _return_row(
    matched_title,
    matched_code,
    score,
    method: str,
    status: str,
    normalized,
    candidate_titles=None,
) -> Dict[str, object]:
    return {
        "matched_onet_title": matched_title,
        "matched_onet_code": matched_code,
        "onet_match_score": score,
        "onet_match_method": method,
        "onet_match_status": status,
        "normalized_title": normalized.canonical_title,
        "title_function": normalized.detected_function,
        "title_level": normalized.detected_level,
        "title_normalization_method": normalized.normalization_method,
        "candidate_titles": (
            " | ".join(candidate_titles)
            if isinstance(candidate_titles, list) and candidate_titles
            else pd.NA
        ),
    }


def _matched_return(
    match_row,
    title_col: str,
    code_col: Optional[str],
    score,
    method: str,
    status: str,
    normalized,
    candidate_titles=None,
):
    return _return_row(
        matched_title=match_row[title_col],
        matched_code=match_row.get(code_col, pd.NA) if code_col else pd.NA,
        score=score,
        method=method,
        status=status,
        normalized=normalized,
        candidate_titles=candidate_titles,
    )


def _unmatched_return(method: str, normalized, candidate_titles=None, score=pd.NA):
    return _return_row(
        matched_title=pd.NA,
        matched_code=pd.NA,
        score=score,
        method=method,
        status="unmatched",
        normalized=normalized,
        candidate_titles=candidate_titles,
    )


def _alias_match(raw_title, ref: pd.DataFrame, title_col: str):
    alias_title = OCCUPATION_ALIASES.get(_clean(raw_title))
    if alias_title is None:
        return None
    return _exact_title_match(alias_title, ref, title_col)


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
        return _unmatched_return("missing_role", normalized)

    if role_col in {"occupation_code", "matched_onet_code", "soc_code"}:
        code_match = _exact_code_match(str(value), ref, code_col)

        if code_match is not None:
            return _matched_return(
                code_match,
                title_col,
                code_col,
                score=1.0,
                method="exact_soc_code",
                status="accepted",
                normalized=normalized,
            )

    title_match = _exact_title_match(str(value), ref, title_col)

    if title_match is not None:
        return _matched_return(
            title_match,
            title_col,
            code_col,
            score=1.0,
            method="exact_title",
            status="accepted",
            normalized=normalized,
        )


    role_for_matching = (
        normalized.canonical_title
        if normalized.canonical_title is not None
        else str(value)
    )

    normalized_exact = _exact_title_match(role_for_matching, ref, title_col)

    if normalized_exact is not None and normalized.canonical_title is not None:
        return _matched_return(
            normalized_exact,
            title_col,
            code_col,
            score=1.0,
            method="normalized_exact_title",
            status="accepted",
            normalized=normalized,
            candidate_titles=[normalized.canonical_title],
        )

    candidate_titles = generate_candidate_titles(
        job_title=value,
        detected_function=normalized.detected_function,
        department=department_value,
    )

    candidate_ref = _filter_ref_to_candidate_titles(
        ref=ref,
        candidate_titles=candidate_titles,
        title_col=title_col,
    )

    if candidate_ref.empty:
        return _unmatched_return(
            method="candidate_titles_not_found_in_onet_reference",
            normalized=normalized,
            candidate_titles=candidate_titles,
        )

    best_match, best_score = _best_fuzzy_title_match(
        role_for_matching,
        candidate_ref,
        title_col,
    )

    if best_match is None:
        return _unmatched_return(
            method="candidate_fuzzy_no_match",
            normalized=normalized,
            candidate_titles=candidate_titles,
        )

    return _matched_return(
        best_match,
        title_col,
        code_col,
        score=best_score,
        method="candidate_constrained_fuzzy_review_required",
        status="review_required",
        normalized=normalized,
        candidate_titles=candidate_titles,
    )


def map_to_onet(
    df: pd.DataFrame,
    onet_reference: pd.DataFrame,
) -> Tuple[pd.DataFrame, OnetMappingReport]:

    out = df.copy()
    role_col = _detect_role_column(out)

    if role_col is None:
        for col in [
            "matched_onet_title",
            "matched_onet_code",
            "onet_match_score",
            "onet_match_method",
            "onet_match_status",
            "normalized_title",
            "title_function",
            "title_level",
            "title_normalization_method",
            "candidate_titles",
        ]:
            out[col] = pd.NA

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

    department_col = "department" if "department" in out.columns else None
    mapped_rows = []

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
        .isin(["exact_title", "exact_soc_code", "normalized_exact_title"])
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
            "Occupation mapping uses exact matches, title normalization, occupation aliases, "
            "and candidate-constrained fuzzy review. Alias and fuzzy matches are review-required, "
            "not automatically accepted."
        ),
    )
