"""Reference structure analysis and comparison.

``analyze_reference`` deconstructs a stored reference (or an ad-hoc description)
into the design vocabulary the spec asks for: primary story, hierarchy, grid,
chart types, removable decoration, mobile-conversion and accessibility risk.

The analysis is heuristic and deterministic. When an image is supplied and an
OpenAI key is configured, ``analysis.vision`` (optional) can enrich it, but the
core never *requires* a network call, so it stays testable offline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_ROLE_FIT = {
    "executive": {"CEO dashboard", "sales management", "regional performance"},
    "operations": {"operations control", "exception handling", "execution management"},
    "member": {"personal performance", "self-service"},
    "analytical": {"deep analysis", "channel comparison"},
}

_DENSITY_MOBILE = {"low": "high", "medium": "medium", "high": "low"}


def analyze_reference(ref: Dict[str, Any]) -> Dict[str, Any]:
    """Return a structured design analysis for a reference record."""
    va = ref.get("visual_analysis") or {}
    hierarchy: List[str] = list(va.get("information_hierarchy") or [])
    primary_story = hierarchy[0] if hierarchy else ref.get("business_goal") or "(unknown primary story)"

    kpi_count = va.get("primary_kpi_count")
    layout = _layout_zones(hierarchy, va)
    density = (va.get("visual_density") or "medium").lower()

    removable = _removable_decoration(va)
    accessibility_risk = _accessibility_risk(va)
    mobile = va.get("mobile_suitability") or _DENSITY_MOBILE.get(density, "medium")

    return {
        "reference_id": ref.get("id"),
        "title": ref.get("title"),
        "primary_story": primary_story,
        "primary_user": ref.get("primary_user"),
        "hierarchy": hierarchy,
        "layout": layout,
        "grid_hint": _grid_hint(va),
        "primary_kpi_count": kpi_count,
        "chart_types": va.get("chart_types") or [],
        "component_types": va.get("component_types") or [],
        "filter_placement": va.get("navigation_pattern") or "top_filter_bar",
        "color_strategy": va.get("color_strategy"),
        "design_principle": va.get("layout_pattern") or "KPI -> Trend -> Detail",
        "visual_density": density,
        "removable_decoration": removable,
        "mobile_convertibility": mobile,
        "accessibility_risk": accessibility_risk,
        "recommended_for": sorted(_ROLE_FIT.get((ref.get("dashboard_type") or "").lower(), set())),
    }


def _layout_zones(hierarchy: List[str], va: Dict[str, Any]) -> Dict[str, str]:
    kpi = va.get("primary_kpi_count") or 4
    top = f"{kpi} KPI cards"
    charts = va.get("chart_types") or []
    middle = "main trend and comparison charts"
    if charts:
        middle = " and ".join(c.replace("_", " ") for c in charts[:2])
    bottom = "detail table / lists"
    if len(hierarchy) >= 4:
        bottom = hierarchy[-1]
    return {"top": top, "middle": middle, "bottom": bottom}


def _grid_hint(va: Dict[str, Any]) -> str:
    density = (va.get("visual_density") or "medium").lower()
    return {
        "low": "12-col grid, generous whitespace, <= 6 blocks per view",
        "medium": "12-col grid, ~8-10 blocks per view",
        "high": "12-col dense grid, small multiples, >= 12 blocks",
    }.get(density, "12-col grid")


def _removable_decoration(va: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    color = (va.get("color_strategy") or "").lower()
    if "gradient" in color or "decorative" in color:
        out.append("decorative gradients")
    if (va.get("visual_density") or "") == "high":
        out.append("redundant icons and heavy shadows")
    if not out:
        out.append("no obvious decorative debt; keep it lean")
    return out


def _accessibility_risk(va: Dict[str, Any]) -> Dict[str, Any]:
    score = va.get("accessibility_score")
    color = (va.get("color_strategy") or "").lower()
    risks: List[str] = []
    if "status color" in color or "green/amber/red" in color:
        risks.append("status conveyed by color alone — add labels/icons")
    if score is not None and score < 75:
        risks.append("contrast/legibility below target — verify WCAG AA")
    level = "low"
    if score is not None:
        level = "high" if score < 70 else ("medium" if score < 80 else "low")
    return {"level": level, "score": score, "notes": risks or ["no major risks flagged"]}


def compare_references(refs: List[Dict[str, Any]], query: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compare references on decision-fit criteria, not aesthetics.

    Returns a per-criterion matrix plus a recommended winner. When ``query`` is
    provided, the fit rows are computed against it; otherwise generic scores are
    used.
    """
    from .scoring import score_reference  # local import to avoid cycle

    rows = ["decision_fit", "information_hierarchy", "data_density", "implementation_ease", "mobile", "decoration_dependence"]
    matrix: Dict[str, Dict[str, Any]] = {r: {} for r in rows}
    winner_scores: Dict[str, float] = {}

    for ref in refs:
        va = ref.get("visual_analysis") or {}
        rid = ref.get("id")
        if query:
            scored = score_reference(query, ref)
            fit = scored["score"]
            hierarchy_fit = scored["components"]["information_hierarchy"]
        else:
            fit = 100.0 - 10.0 * {"high": 2, "medium": 1, "low": 0}.get(va.get("visual_density", "medium"), 1)
            hierarchy_fit = 100.0 if va.get("information_hierarchy") else 50.0
        winner_scores[rid] = fit

        matrix["decision_fit"][rid] = round(fit, 1)
        matrix["information_hierarchy"][rid] = round(hierarchy_fit, 1)
        matrix["data_density"][rid] = va.get("visual_density", "medium")
        matrix["implementation_ease"][rid] = va.get("implementation_difficulty", "medium")
        matrix["mobile"][rid] = va.get("mobile_suitability", "medium")
        matrix["decoration_dependence"][rid] = "high" if (va.get("visual_density") == "high") else "low"

    winner = max(winner_scores, key=winner_scores.get) if winner_scores else None
    return {
        "references": [r.get("id") for r in refs],
        "criteria": rows,
        "matrix": matrix,
        "recommended": winner,
        "rationale": (
            f"{winner} scores highest on decision-fit; it is selected for purpose-fit, "
            "not visual appeal."
            if winner
            else "no references supplied"
        ),
    }
