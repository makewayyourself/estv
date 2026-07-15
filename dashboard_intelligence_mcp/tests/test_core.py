"""Offline unit tests for the Dashboard Intelligence core (no MCP dependency)."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dashboard_intelligence_mcp.core import (  # noqa: E402
    ReferenceStore,
    analyze_reference,
    audit_dashboard,
    compare_references,
    generate_blueprint,
    generate_build_prompt,
    search_references,
)


def _store():
    # Isolate the user archive to a temp file so tests never touch shipped data.
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    return ReferenceStore(user_path=tmp.name)


def test_seed_loads():
    store = _store()
    assert store.count() >= 8
    assert store.get("REF-101") is not None
    assert "cooperative" in store.industries()


def test_search_ranks_cooperative_executive_first():
    store = _store()
    query = {
        "business_domain": "consumer cooperative",
        "dashboard_type": "executive",
        "primary_user": "cooperative executive",
        "business_goal": "regional revenue and member profit monitoring",
        "main_metrics": ["total_sales", "active_members", "regional_sales", "inventory_turnover"],
        "information_density": "medium",
    }
    results = search_references(store, query, result_count=3)
    assert results
    top = results[0]
    # A cooperative executive reference should win, not a logistics/healthcare one.
    assert top["reference_id"] in {"REF-102", "REF-107"}
    assert top["score"] > results[-1]["score"] or len(results) == 1
    # Visual similarity must be capped low in the weighting.
    assert top["components"]["visual_similarity"] <= 100


def test_device_filters_low_mobile():
    store = _store()
    query = {"business_domain": "logistics operations", "device": "mobile"}
    results = search_references(store, query, result_count=10)
    ids = {r["reference_id"] for r in results}
    # REF-103 (logistics) and REF-108 (healthcare) are mobile_suitability=low.
    assert "REF-103" not in ids
    assert "REF-108" not in ids


def test_analyze_reference_structure():
    store = _store()
    analysis = analyze_reference(store.get("REF-102"))
    assert analysis["primary_kpi_count"] == 4
    assert analysis["hierarchy"]
    assert "top" in analysis["layout"]
    assert analysis["accessibility_risk"]["level"] in {"low", "medium", "high"}


def test_compare_picks_best_fit():
    store = _store()
    refs = [store.get("REF-102"), store.get("REF-103")]
    query = {
        "business_goal": "regional revenue and member profit",
        "primary_user": "cooperative executive",
        "dashboard_type": "executive",
        "main_metrics": ["regional_sales", "member_profit"],
    }
    result = compare_references(refs, query)
    assert result["recommended"] == "REF-102"
    assert "decision_fit" in result["matrix"]


def test_blueprint_tiers_and_primary_cap():
    data = [
        "total_sales", "gross_margin", "active_members", "member_profit",
        "monthly_trend", "regional_sales", "inventory_turnover",
        "member_settlement_list",
    ]
    bp = generate_blueprint(
        available_data=data,
        user_role="cooperative executive",
        business_goal="monitor regional revenue and member profit",
        references=[],
        max_primary_kpis=4,
    )
    tiers = {t["tier"]: t for t in bp["tiers"]}
    assert len(tiers[1]["widgets"]) <= 4
    # inventory_turnover and regional_sales route to cause/risk, not conclusion.
    conclusion_fields = {w["field"] for w in tiers[1]["widgets"]}
    assert "inventory_turnover" not in conclusion_fields


def test_blueprint_promotes_headline_when_conclusion_empty():
    # All metrics are regional breakdowns; Tier 1 must not be empty.
    bp = generate_blueprint(
        available_data=["regional_sales", "regional_members", "top_products", "inventory_turnover"],
        user_role="regional chapter lead",
        business_goal="judge region status in 5 seconds",
    )
    tier1 = [t for t in bp["tiers"] if t["tier"] == 1][0]
    fields = {w["field"] for w in tier1["widgets"]}
    assert fields, "Tier 1 should be promoted, not empty"
    assert "regional_sales" in fields
    # A pure risk field must not be promoted to a headline.
    assert "inventory_turnover" not in fields


def test_build_prompt_forbids_cloning():
    bp = generate_blueprint(
        available_data=["total_sales", "regional_sales", "inventory_turnover"],
        user_role="CEO",
        business_goal="revenue monitoring",
    )
    prompt = generate_build_prompt(bp, target_tool="lovable", framework="react")
    assert "clone" in prompt.lower()
    assert "react" in prompt.lower()

    prompt_ko = generate_build_prompt(bp, language="ko")
    assert "복제" in prompt_ko


def test_audit_flags_critical():
    dashboard = {
        "cards": [{"name": "a", "size": "m", "emphasis": "same"} for _ in range(6)],
        "kpi_count": 8,
        "top_widgets": 7,
        "filters": ["region", "region", "date"],
        "list_position": "top",
        "audience": "mixed",
        "colors_meaningful": False,
        "mobile_hierarchy_preserved": False,
    }
    result = audit_dashboard(dashboard)
    assert result["summary"]["Critical"] >= 1
    assert result["findings"][0]["severity"] == "Critical"
    assert "Not ready" in result["verdict"]


def test_register_reference_roundtrip():
    store = _store()
    before = store.count()
    stored = store.register(
        {
            "title": "My Custom Ops Board",
            "source_url": "https://example.com/x",
            "industry": "retail",
            "dashboard_type": "operations",
            "primary_user": "operations manager",
            "tags": ["ops", "custom"],
        }
    )
    assert stored["id"].startswith("REF-")
    assert store.count() == before + 1
    # Reload from disk proves persistence.
    store.reload()
    assert store.get(stored["id"]) is not None


def test_register_requires_title_and_source():
    store = _store()
    try:
        store.register({"source_url": "https://x"})
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        store.register({"title": "no source"})
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
