"""Heuristic audit of an existing dashboard description.

Since the core runs offline, the audit works on a *structured description* of a
dashboard (the caller — a vision-capable client — supplies it) rather than
pixels. It applies the spec's checklist and returns prioritized findings
(Critical / High / Medium).
"""

from __future__ import annotations

from typing import Any, Dict, List

CRITICAL = "Critical"
HIGH = "High"
MEDIUM = "Medium"


def audit_dashboard(dashboard: Dict[str, Any]) -> Dict[str, Any]:
    """Audit a dashboard description and return prioritized findings.

    Expected (all optional) keys on ``dashboard``:
        cards: list of {name, size, emphasis}
        kpi_count: int
        top_widgets: int  (widgets above the fold)
        charts: list of {name, tied_to_decision: bool}
        filters: list of str  (may repeat -> duplication)
        colors_meaningful: bool
        list_position: "top"|"middle"|"bottom"
        audience: "executive"|"operations"|"mixed"
        mobile_hierarchy_preserved: bool
        duplicated_metrics: list of str
    """
    findings: List[Dict[str, str]] = []

    def add(severity: str, issue: str, fix: str) -> None:
        findings.append({"severity": severity, "issue": issue, "recommendation": fix})

    cards = dashboard.get("cards") or []
    emphases = {c.get("emphasis") for c in cards if isinstance(c, dict)}
    sizes = {c.get("size") for c in cards if isinstance(c, dict)}
    if cards and len(emphases) <= 1 and len(sizes) <= 1:
        add(
            CRITICAL,
            "Primary and secondary KPIs have no visual difference — every card looks equally important.",
            "Enlarge/accent the 1-2 primary KPIs; mute the rest.",
        )

    top = dashboard.get("top_widgets")
    if isinstance(top, int) and top >= 6:
        add(
            CRITICAL,
            f"{top} widgets above the fold, several unrelated to action.",
            "Keep 3-5 decision-critical widgets at the top; move the rest down.",
        )

    kpi_count = dashboard.get("kpi_count")
    if isinstance(kpi_count, int) and kpi_count > 6:
        add(HIGH, f"{kpi_count} KPIs compete for attention.", "Reduce to 3-5 primary KPIs.")

    dupes = dashboard.get("duplicated_metrics") or []
    if dupes:
        add(HIGH, f"Duplicated information: {', '.join(dupes)}.", "Show each metric once; remove redundancy.")

    filters = dashboard.get("filters") or []
    if len(filters) != len(set(filters)):
        repeated = sorted({f for f in filters if filters.count(f) > 1})
        add(HIGH, f"Filter(s) repeated across the screen: {', '.join(repeated)}.", "Consolidate into one global filter bar.")

    charts = dashboard.get("charts") or []
    orphan = [c.get("name") for c in charts if isinstance(c, dict) and c.get("tied_to_decision") is False]
    if orphan:
        add(MEDIUM, f"Chart(s) not tied to any decision: {', '.join(orphan)}.", "Remove or reframe so each chart drives an action.")

    if dashboard.get("colors_meaningful") is False:
        add(MEDIUM, "Color is used decoratively rather than to encode meaning.", "Reserve accent color for the primary signal only.")

    if dashboard.get("list_position") == "top":
        add(HIGH, "A detail list/table is exposed at the top before conclusions.", "Move detail lists to the bottom tier; lead with KPIs.")

    if dashboard.get("audience") == "mixed":
        add(HIGH, "Executive and operator content are mixed in one view.", "Split into an executive summary view and an operations view.")

    if dashboard.get("mobile_hierarchy_preserved") is False:
        add(MEDIUM, "Information hierarchy breaks down on mobile.", "Stack by tier; keep Tier 1 KPIs first on small screens.")

    order = {CRITICAL: 0, HIGH: 1, MEDIUM: 2}
    findings.sort(key=lambda f: order.get(f["severity"], 3))

    summary = {
        CRITICAL: sum(1 for f in findings if f["severity"] == CRITICAL),
        HIGH: sum(1 for f in findings if f["severity"] == HIGH),
        MEDIUM: sum(1 for f in findings if f["severity"] == MEDIUM),
    }
    return {
        "summary": summary,
        "findings": findings,
        "verdict": _verdict(summary),
    }


def _verdict(summary: Dict[str, int]) -> str:
    if summary[CRITICAL]:
        return "Not ready — critical hierarchy problems must be fixed first."
    if summary[HIGH]:
        return "Usable but needs work — resolve high-priority issues before shipping."
    if summary[MEDIUM]:
        return "Solid — minor polish recommended."
    return "No issues flagged by the heuristic checklist."
