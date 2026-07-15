"""Intent-based reference scoring.

The whole point of this MCP (per the spec) is *not* to find visually similar
dashboards but the ones that best fit the user's decision structure. So visual
similarity is deliberately weighted at only 5%; business goal, user role and
information hierarchy dominate.

    business goal fit        30%
    user role fit            20%
    information hierarchy    20%
    data structure fit       15%
    implementation feasibility 10%
    visual similarity         5%

Scoring is deterministic keyword/tag overlap so results are explainable and
unit-testable with no external embedding service. The function signatures leave
room to swap in vector similarity later without changing callers.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

WEIGHTS = {
    "business_goal": 0.30,
    "user_role": 0.20,
    "information_hierarchy": 0.20,
    "data_structure": 0.15,
    "implementation": 0.10,
    "visual_similarity": 0.05,
}

_DIFFICULTY_FEASIBILITY = {"low": 1.0, "medium": 0.6, "high": 0.3}
_DENSITY_ORDER = {"low": 0, "medium": 1, "high": 2}

# Role synonyms so "CEO", "executive", "경영자" all resonate with the same refs.
_ROLE_SYNONYMS = {
    "ceo": {"ceo", "executive", "cfo", "founder", "leadership", "경영자", "경영진", "대표"},
    "executive": {"ceo", "executive", "cfo", "founder", "leadership", "경영자", "경영진"},
    "cfo": {"cfo", "executive", "finance", "재무"},
    "operations manager": {"operations", "manager", "ops", "운영", "관리자"},
    "operations": {"operations", "manager", "ops", "운영", "관리자"},
    "manager": {"manager", "operations", "관리자"},
    "marketing manager": {"marketing", "manager", "growth", "마케팅"},
    "member": {"member", "individual", "조합원", "개인"},
    "regional chapter lead": {"regional", "chapter", "lead", "지역", "조합장"},
    "cooperative executive": {"cooperative", "executive", "조합", "경영자", "경영진"},
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(*values: Any) -> set:
    out: set = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            for item in value:
                out |= _tokens(item)
        else:
            out |= set(_TOKEN_RE.findall(str(value).lower()))
    return out


def _overlap(query: set, target: set) -> float:
    """Fraction of query tokens present in the target (0..1)."""
    if not query:
        return 0.0
    hits = sum(1 for t in query if t in target)
    return hits / len(query)


def _role_tokens(role: Optional[str]) -> set:
    base = _tokens(role)
    for key, syns in _ROLE_SYNONYMS.items():
        if key in (role or "").lower():
            base |= _tokens(list(syns))
    return base


def score_reference(query: Dict[str, Any], ref: Dict[str, Any]) -> Dict[str, Any]:
    """Score one reference against a query, returning score + component breakdown."""
    va = ref.get("visual_analysis") or {}

    ref_goal_tokens = _tokens(
        ref.get("industry"),
        ref.get("business_goal"),
        ref.get("dashboard_type"),
        ref.get("tags"),
        ref.get("decision_purpose"),
    )
    ref_role_tokens = _role_tokens(ref.get("primary_user"))
    ref_hierarchy_tokens = _tokens(
        va.get("layout_pattern"),
        va.get("information_hierarchy"),
        ref.get("decision_purpose"),
        ref.get("dashboard_type"),
    )
    ref_metric_tokens = _tokens(ref.get("supported_metrics"), ref.get("tags"))

    # --- component scores (0..1) ---------------------------------------
    goal_query = _tokens(
        query.get("business_domain"),
        query.get("business_goal"),
        query.get("dashboard_type"),
        query.get("keywords"),
    )
    business_goal = _overlap(goal_query, ref_goal_tokens)

    role_query = _role_tokens(query.get("primary_user"))
    user_role = _overlap(role_query, ref_role_tokens)

    hierarchy_query = _tokens(
        query.get("dashboard_type"),
        query.get("decision_purpose"),
        query.get("business_goal"),
    )
    hierarchy = _overlap(hierarchy_query, ref_hierarchy_tokens)

    metric_query = _tokens(query.get("main_metrics"))
    data_structure = _overlap(metric_query, ref_metric_tokens)

    difficulty = (va.get("implementation_difficulty") or "medium").lower()
    implementation = _DIFFICULTY_FEASIBILITY.get(difficulty, 0.6)

    visual = _visual_similarity(query, ref, va)

    components = {
        "business_goal": business_goal,
        "user_role": user_role,
        "information_hierarchy": hierarchy,
        "data_structure": data_structure,
        "implementation": implementation,
        "visual_similarity": visual,
    }
    total = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)

    return {
        "reference_id": ref.get("id"),
        "title": ref.get("title"),
        "industry": ref.get("industry"),
        "primary_user": ref.get("primary_user"),
        "dashboard_type": ref.get("dashboard_type"),
        "score": round(total * 100, 1),
        "components": {k: round(v * 100, 1) for k, v in components.items()},
        "why": _explain(components, ref),
    }


def _visual_similarity(query: Dict[str, Any], ref: Dict[str, Any], va: Dict[str, Any]) -> float:
    score = 0.0
    parts = 0
    q_density = (query.get("information_density") or "").lower()
    if q_density:
        parts += 1
        r_density = (va.get("visual_density") or "").lower()
        if q_density and r_density:
            dist = abs(_DENSITY_ORDER.get(q_density, 1) - _DENSITY_ORDER.get(r_density, 1))
            score += 1.0 - (dist / 2.0)
    q_layout = (query.get("layout_style") or "").lower()
    if q_layout:
        parts += 1
        haystack = _tokens(va.get("layout_pattern"), ref.get("tags"), va.get("color_strategy"))
        score += 1.0 if any(t in haystack for t in _tokens(q_layout)) else 0.0
    if parts == 0:
        return 0.5  # neutral when the query says nothing about visuals
    return score / parts


def _device_ok(query: Dict[str, Any], va: Dict[str, Any]) -> bool:
    device = (query.get("device") or "").lower()
    if device in ("mobile", "tablet"):
        return (va.get("mobile_suitability") or "medium").lower() != "low"
    return True


def _explain(components: Dict[str, float], ref: Dict[str, Any]) -> str:
    ranked = sorted(components.items(), key=lambda kv: kv[1], reverse=True)
    strong = [k.replace("_", " ") for k, v in ranked if v >= 0.5][:3]
    if strong:
        return f"Strong on {', '.join(strong)} for {ref.get('primary_user')} / {ref.get('industry')}."
    return f"Partial fit for {ref.get('primary_user')} / {ref.get('industry')}; review manually."


def search_references(
    store,
    query: Dict[str, Any],
    result_count: int = 6,
) -> List[Dict[str, Any]]:
    """Rank all references against the query and return the top ``result_count``.

    A device constraint (mobile/tablet) filters out references whose
    ``mobile_suitability`` is ``low`` before ranking.
    """
    candidates = []
    for ref in store.all():
        va = ref.get("visual_analysis") or {}
        if not _device_ok(query, va):
            continue
        candidates.append(score_reference(query, ref))
    candidates.sort(key=lambda r: r["score"], reverse=True)
    n = max(1, int(result_count or 6))
    return candidates[:n]
