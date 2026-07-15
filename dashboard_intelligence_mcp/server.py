"""Dashboard Intelligence MCP server.

Thin MCP wrapper over the pure-Python ``core``. Exposes:

  Tools
    - register_dashboard_reference
    - search_dashboard_references
    - analyze_dashboard_reference
    - compare_dashboard_references
    - generate_dashboard_blueprint
    - generate_dashboard_build_prompt
    - audit_existing_dashboard

  Resources
    - dashboard://reference/{id}
    - dashboard://industry/{industry}
    - dashboard://pattern/{slug}
    - dashboard://guideline/information-hierarchy

  Prompts
    - executive-dashboard / operations-dashboard / member-analysis-dashboard
    - reference-to-react / dashboard-ux-audit

Run with:  python -m dashboard_intelligence_mcp.server   (stdio transport)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .core import (
    ReferenceStore,
    analyze_reference,
    audit_dashboard,
    compare_references,
    generate_blueprint,
    generate_build_prompt,
    search_references,
)

mcp = FastMCP("dashboard-intelligence")
_store = ReferenceStore()


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
@mcp.tool()
def search_dashboard_references(
    business_domain: str = "",
    dashboard_type: str = "",
    primary_user: str = "",
    business_goal: str = "",
    main_metrics: Optional[List[str]] = None,
    decision_purpose: Optional[List[str]] = None,
    layout_style: str = "",
    information_density: str = "",
    device: str = "",
    result_count: int = 6,
) -> str:
    """Find the dashboard references that best fit a business/decision intent.

    Ranks by decision-fit (business goal 30%, user role 20%, information
    hierarchy 20%, data structure 15%, feasibility 10%, visual 5%), not by
    visual similarity. Returns a scored, explained list.
    """
    query = {
        "business_domain": business_domain,
        "dashboard_type": dashboard_type,
        "primary_user": primary_user,
        "business_goal": business_goal,
        "main_metrics": main_metrics or [],
        "decision_purpose": decision_purpose or [],
        "layout_style": layout_style,
        "information_density": information_density,
        "device": device,
    }
    results = search_references(_store, query, result_count=result_count)
    return _dumps({"query": query, "result_count": len(results), "results": results})


@mcp.tool()
def analyze_dashboard_reference(reference_id: str) -> str:
    """Deconstruct a reference into primary story, hierarchy, layout zones,
    chart types, removable decoration, mobile-convertibility and a11y risk."""
    ref = _store.get(reference_id)
    if not ref:
        return _dumps({"error": f"reference '{reference_id}' not found"})
    return _dumps(analyze_reference(ref))


@mcp.tool()
def compare_dashboard_references(
    reference_ids: List[str],
    business_goal: str = "",
    primary_user: str = "",
    dashboard_type: str = "",
    main_metrics: Optional[List[str]] = None,
) -> str:
    """Compare references on decision-fit criteria and recommend the best fit
    for the *purpose*, not the prettiest one."""
    refs = [r for r in (_store.get(rid) for rid in reference_ids) if r]
    if not refs:
        return _dumps({"error": "no valid references found for the given ids"})
    query = None
    if business_goal or primary_user or dashboard_type or main_metrics:
        query = {
            "business_goal": business_goal,
            "primary_user": primary_user,
            "dashboard_type": dashboard_type,
            "main_metrics": main_metrics or [],
        }
    return _dumps(compare_references(refs, query))


@mcp.tool()
def generate_dashboard_blueprint(
    available_data: List[str],
    user_role: str,
    business_goal: str,
    reference_ids: Optional[List[str]] = None,
    max_primary_kpis: int = 4,
) -> str:
    """Apply reference layout grammar to the user's real data fields, producing a
    four-tier blueprint (Conclusion -> Cause -> Risk -> Detail)."""
    refs = [r for r in (_store.get(rid) for rid in (reference_ids or [])) if r]
    blueprint = generate_blueprint(
        available_data=available_data,
        user_role=user_role,
        business_goal=business_goal,
        references=refs,
        max_primary_kpis=max_primary_kpis,
    )
    return _dumps(blueprint)


@mcp.tool()
def generate_dashboard_build_prompt(
    available_data: List[str],
    user_role: str,
    business_goal: str,
    reference_ids: Optional[List[str]] = None,
    target_tool: str = "generic",
    framework: str = "react",
    language: str = "en",
    max_primary_kpis: int = 4,
) -> str:
    """Produce a copy-paste build prompt for AI coding tools (Lovable, Bolt,
    Claude Code, Cursor, Codex) from a data-mapped blueprint. The prompt forbids
    cloning reference colors/brand and reuses only layout grammar."""
    refs = [r for r in (_store.get(rid) for rid in (reference_ids or [])) if r]
    blueprint = generate_blueprint(
        available_data=available_data,
        user_role=user_role,
        business_goal=business_goal,
        references=refs,
        max_primary_kpis=max_primary_kpis,
    )
    prompt = generate_build_prompt(
        blueprint, target_tool=target_tool, framework=framework, language=language
    )
    return prompt


@mcp.tool()
def audit_existing_dashboard(dashboard: Dict[str, Any]) -> str:
    """Audit a structured description of an existing dashboard against the design
    checklist and return prioritized findings (Critical/High/Medium)."""
    return _dumps(audit_dashboard(dashboard))


@mcp.tool()
def register_dashboard_reference(
    title: str,
    source_url: str = "",
    industry: str = "",
    dashboard_type: str = "",
    primary_user: str = "",
    business_goal: str = "",
    source_platform: str = "",
    creator: str = "",
    original_image_url: str = "",
    tags: Optional[List[str]] = None,
    supported_metrics: Optional[List[str]] = None,
    decision_purpose: Optional[List[str]] = None,
    visual_analysis: Optional[Dict[str, Any]] = None,
    rights_status: str = "reference_only",
) -> str:
    """Register a new reference (image URL / source URL + metadata) into the
    archive. Records source and rights; stores layout grammar, not pixels."""
    record = {
        "title": title,
        "source_url": source_url or None,
        "original_image_url": original_image_url or None,
        "industry": industry,
        "dashboard_type": dashboard_type,
        "primary_user": primary_user,
        "business_goal": business_goal,
        "source_platform": source_platform,
        "creator": creator,
        "tags": tags or [],
        "supported_metrics": supported_metrics or [],
        "decision_purpose": decision_purpose or [],
        "visual_analysis": visual_analysis or {},
        "rights_status": rights_status,
    }
    try:
        stored = _store.register(record)
    except ValueError as exc:
        return _dumps({"error": str(exc)})
    return _dumps({"registered": stored["id"], "reference": stored})


# --------------------------------------------------------------------------
# Resources
# --------------------------------------------------------------------------
@mcp.resource("dashboard://reference/{reference_id}")
def resource_reference(reference_id: str) -> str:
    ref = _store.get(reference_id)
    return _dumps(ref or {"error": f"reference '{reference_id}' not found"})


@mcp.resource("dashboard://industry/{industry}")
def resource_industry(industry: str) -> str:
    refs = _store.by_industry(industry)
    return _dumps({"industry": industry, "count": len(refs), "references": refs})


@mcp.resource("dashboard://pattern/{slug}")
def resource_pattern(slug: str) -> str:
    refs = _store.by_pattern_slug(slug)
    return _dumps({"pattern": slug, "count": len(refs), "references": refs})


@mcp.resource("dashboard://guideline/information-hierarchy")
def resource_guideline() -> str:
    return _dumps(
        {
            "title": "Information hierarchy guideline",
            "principle": "Conclusion -> Cause -> Risk -> Detail",
            "rules": [
                "Lead with 3-5 primary KPIs; never give every KPI equal weight.",
                "Explain KPIs with trend + comparison before showing detail.",
                "Surface only risk-flagged items in the action tier.",
                "Push tables/lists to the bottom, collapsed by default.",
                "Reserve accent color for the single primary signal.",
                "Do not encode status by color alone.",
                "Preserve tier order on mobile (Tier 1 first).",
            ],
        }
    )


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
@mcp.prompt()
def executive_dashboard(business_goal: str = "", industry: str = "", available_data: str = "") -> str:
    """Guided flow to design an executive (CEO) dashboard end to end."""
    return (
        "Design an executive dashboard. Follow this order and use the "
        "dashboard-intelligence tools at each step:\n"
        f"1. Confirm the decision purpose for goal: '{business_goal}' (industry: '{industry}').\n"
        "2. Pick the 3-5 primary KPIs.\n"
        "3. Call search_dashboard_references (primary_user='CEO', dashboard_type='executive').\n"
        "4. Compare the top candidates with compare_dashboard_references.\n"
        "5. Choose the best-fit layout grammar.\n"
        f"6. Call generate_dashboard_blueprint with available_data: {available_data or '[fill in]'}.\n"
        "7. Call generate_dashboard_build_prompt for the target coding tool.\n"
        "8. Verify no required metric was dropped."
    )


@mcp.prompt()
def operations_dashboard(business_goal: str = "", available_data: str = "") -> str:
    """Guided flow to design an operations/monitoring dashboard."""
    return (
        "Design an operations dashboard focused on anomaly detection and "
        "execution.\n"
        f"Goal: {business_goal}\n"
        "Use search_dashboard_references (dashboard_type='operations'), compare, "
        "then blueprint + build prompt.\n"
        f"Available data: {available_data or '[fill in]'}\n"
        "Prioritize status, exceptions, and queues; keep detail at the bottom."
    )


@mcp.prompt()
def member_analysis_dashboard(available_data: str = "") -> str:
    """Guided flow for an individual member / self-service dashboard."""
    return (
        "Design a personal member dashboard (self-service). Keep density low and "
        "mobile-first.\n"
        "Search with primary_user='member', dashboard_type='member', then "
        "blueprint the member's own metrics (my sales, earned profit, expected "
        "settlement).\n"
        f"Available data: {available_data or '[fill in]'}"
    )


@mcp.prompt()
def reference_to_react(reference_id: str = "", available_data: str = "") -> str:
    """Turn a chosen reference into a React build prompt mapped to real data."""
    return (
        f"Analyze reference {reference_id or '[id]'} with "
        "analyze_dashboard_reference, then call generate_dashboard_build_prompt "
        f"(framework='react') for available_data: {available_data or '[fill in]'}. "
        "Reuse only the layout grammar — never the reference's colors or brand."
    )


@mcp.prompt()
def dashboard_ux_audit(dashboard_description: str = "") -> str:
    """Audit an existing dashboard for hierarchy and UX problems."""
    return (
        "Audit an existing dashboard. Build a structured description "
        "(cards, kpi_count, top_widgets, charts, filters, audience, "
        "list_position, mobile_hierarchy_preserved) and call "
        "audit_existing_dashboard. Report Critical/High/Medium findings.\n"
        f"Dashboard: {dashboard_description or '[describe or attach]'}"
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
