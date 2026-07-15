"""Data-mapped blueprint and build-prompt generation.

``generate_blueprint`` applies the *design grammar* of reference dashboards
(their information hierarchy) to the user's actual data fields, producing the
four-tier structure the spec describes:

    Tier 1  Conclusion  — the 3-5 primary KPIs
    Tier 2  Cause       — trends and comparisons that explain the KPIs
    Tier 3  Risk        — anomalies and things needing action
    Tier 4  Detail      — drill-down lists / tables

``generate_build_prompt`` renders that blueprint as an instruction for AI coding
tools (Lovable, Bolt, Claude Code, Cursor, Codex) that explicitly forbids
cloning the reference's colors/brand and only reuses its layout grammar.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Heuristic classification of a metric name into a tier + widget.
_PRIMARY_HINTS = ("total", "revenue", "sales", "profit", "margin", "balance", "gmv", "member")
_TREND_HINTS = ("trend", "monthly", "growth", "over_time", "series")
_COMPARISON_HINTS = ("regional", "region", "by_", "channel", "category", "product", "segment")
_RISK_HINTS = ("turnover", "inactive", "declining", "delay", "exception", "risk", "churn", "low")
_DETAIL_HINTS = ("list", "detail", "per_", "individual", "settlement", "record")


def _pretty(field: str) -> str:
    return field.replace("_", " ").strip()


def _classify(field: str) -> str:
    f = field.lower()
    if any(h in f for h in _RISK_HINTS):
        return "risk"
    if any(h in f for h in _TREND_HINTS):
        return "cause"
    if any(h in f for h in _COMPARISON_HINTS):
        return "cause"
    if any(h in f for h in _DETAIL_HINTS):
        return "detail"
    if any(h in f for h in _PRIMARY_HINTS):
        return "conclusion"
    return "detail"


def generate_blueprint(
    available_data: List[str],
    user_role: str,
    business_goal: str,
    references: Optional[List[Dict[str, Any]]] = None,
    max_primary_kpis: int = 4,
) -> Dict[str, Any]:
    """Map available data fields onto a four-tier dashboard blueprint."""
    references = references or []
    tiers: Dict[str, List[str]] = {"conclusion": [], "cause": [], "risk": [], "detail": []}
    for field in available_data:
        tiers[_classify(field)].append(field)

    # A dashboard with no headline number is unusable. If nothing classified as
    # a conclusion (e.g. all metrics are breakdowns like `regional_sales`),
    # promote the strongest headline-looking fields out of the cause tier.
    if not tiers["conclusion"]:
        promotable = [f for f in tiers["cause"] if any(h in f.lower() for h in _PRIMARY_HINTS)]
        for field in promotable[:max_primary_kpis]:
            tiers["cause"].remove(field)
            tiers["conclusion"].append(field)

    # Enforce a small primary set — demote overflow KPIs to secondary emphasis.
    primary = tiers["conclusion"][:max_primary_kpis]
    secondary = tiers["conclusion"][max_primary_kpis:]
    tiers["detail"] = secondary + tiers["detail"]

    ref_pattern = None
    if references:
        va = references[0].get("visual_analysis") or {}
        ref_pattern = va.get("layout_pattern")

    blueprint = {
        "user_role": user_role,
        "business_goal": business_goal,
        "based_on_references": [r.get("id") for r in references],
        "layout_grammar": ref_pattern or "Conclusion -> Cause -> Risk -> Detail",
        "tiers": [
            {
                "tier": 1,
                "name": "Conclusion (핵심 결론)",
                "emphasis": "primary",
                "widgets": [
                    {"field": f, "widget": "kpi_card", "label": _pretty(f)} for f in primary
                ],
            },
            {
                "tier": 2,
                "name": "Cause (변화와 원인)",
                "emphasis": "supporting",
                "widgets": [
                    {"field": f, "widget": _trend_or_comparison(f), "label": _pretty(f)}
                    for f in tiers["cause"]
                ],
            },
            {
                "tier": 3,
                "name": "Risk (위험과 실행)",
                "emphasis": "attention",
                "widgets": [
                    {"field": f, "widget": "risk_list", "label": _pretty(f)} for f in tiers["risk"]
                ],
            },
            {
                "tier": 4,
                "name": "Detail (상세 정보)",
                "emphasis": "on_demand",
                "widgets": [
                    {"field": f, "widget": "data_table", "label": _pretty(f)} for f in tiers["detail"]
                ],
            },
        ],
        "notes": _blueprint_notes(primary, secondary),
    }
    return blueprint


def _trend_or_comparison(field: str) -> str:
    f = field.lower()
    if any(h in f for h in _COMPARISON_HINTS):
        return "comparison_chart"
    return "trend_chart"


def _blueprint_notes(primary: List[str], secondary: List[str]) -> List[str]:
    notes = [
        "Do not give every KPI equal visual weight — Tier 1 is primary, others recede.",
    ]
    if not primary:
        notes.append("No obvious primary KPI in the data; confirm the single most important number with the user.")
    if secondary:
        notes.append(
            f"Demoted {len(secondary)} extra headline metric(s) to detail to keep Tier 1 scannable."
        )
    return notes


def generate_build_prompt(
    blueprint: Dict[str, Any],
    target_tool: str = "generic",
    framework: str = "react",
    language: str = "en",
) -> str:
    """Render a copy-paste build prompt from a blueprint."""
    tiers = blueprint.get("tiers", [])
    role = blueprint.get("user_role", "the primary user")
    goal = blueprint.get("business_goal", "")

    def kpi_lines(tier: Dict[str, Any]) -> str:
        return "\n".join(
            f"    {i+1}. {w['label']}" for i, w in enumerate(tier.get("widgets", []))
        ) or "    (none)"

    tier_map = {t["tier"]: t for t in tiers}
    header_ko = (
        "첨부한 레퍼런스의 색상이나 브랜드는 복제하지 말고, 정보 위계와 레이아웃 문법만 참고하라."
    )
    header_en = (
        "Do NOT clone the reference's colors or brand. Reuse only its information "
        "hierarchy and layout grammar."
    )
    header = header_ko if language == "ko" else header_en

    parts = [
        f"# Dashboard build prompt ({framework}, target: {target_tool})",
        "",
        header,
        "",
        f"User: {role}",
        f"Goal: {goal}",
        f"Layout grammar: {blueprint.get('layout_grammar')}",
        "",
        "Tier 1 — top of the screen, place ONLY these KPIs (equal weight is wrong; "
        "the first two are primary, the rest are secondary):",
        kpi_lines(tier_map.get(1, {})),
        "",
        "Tier 2 — below the KPIs, trends and comparisons that explain them "
        "(main trend ~65% width, comparison ~35%):",
        kpi_lines(tier_map.get(2, {})),
        "",
        "Tier 3 — show ONLY items with a risk signal (do not list everything):",
        kpi_lines(tier_map.get(3, {})),
        "",
        "Tier 4 — drill-down tables, collapsed by default:",
        kpi_lines(tier_map.get(4, {})),
        "",
        "Constraints:",
        "- No decorative icons, gradients, or heavy shadows.",
        "- Status must not rely on color alone; add a label or icon.",
        "- The screen must be readable at a glance in ~5 seconds.",
        f"- Build with {framework}; keep components small and data-driven.",
    ]
    return "\n".join(parts)
