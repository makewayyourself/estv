# Dashboard Intelligence MCP

> 좋은 화면을 찾는 것을 넘어, 좋은 의사결정 구조를 설계하는 AI 디자인 시스템.
> Beyond finding pretty screens — a design-decision engine for dashboards.

An [MCP](https://modelcontextprotocol.io) server that turns a dashboard-reference
archive into a **design-decision engine**. Instead of "find a dashboard that
looks like this", it answers "find the dashboard whose *information structure*
fits this business goal, this user, and this data — then turn it into a build
spec."

It implements the five MVP capabilities from the project spec:

1. **Register** a reference (image/source URL + metadata).
2. **Analyze** its layout & information hierarchy.
3. **Search** references by natural business intent.
4. **Blueprint** — map a chosen reference's grammar onto the user's real data.
5. **Build prompt** — emit a prompt for AI coding tools (Lovable, Bolt, Claude
   Code, Cursor, Codex).

Plus `compare` and `audit` tools, MCP Resources, and MCP Prompts.

## Architecture

```
dashboard_intelligence_mcp/
├── core/                 # pure Python, no MCP dependency — unit tested offline
│   ├── store.py          # JSON-backed reference archive (seed + user)
│   ├── scoring.py        # intent-based ranking (visual similarity only 5%)
│   ├── analysis.py       # structure analysis + decision-fit comparison
│   ├── blueprint.py      # 4-tier data mapping + build-prompt rendering
│   └── audit.py          # heuristic UX/hierarchy audit
├── server.py             # thin MCP wrapper (FastMCP): tools/resources/prompts
├── data/
│   └── references.seed.json   # shipped reference archive (incl. cooperative/오너스마켓)
└── tests/test_core.py    # offline tests, no network/DB
```

The core is intentionally free of any MCP or database dependency so it stays
testable and can be reused directly (e.g. from the existing Streamlit `app.py`).
The reference archive is a JSON file today; the schema mirrors the spec's
`references` + `visual_analysis` tables and can migrate to PostgreSQL + a vector
DB later **without changing the tool contracts**.

## Ranking model

Search ranks by *decision fit*, deliberately down-weighting looks:

| criterion                  | weight |
| -------------------------- | -----: |
| business goal fit          |   30%  |
| user role fit              |   20%  |
| information hierarchy fit  |   20%  |
| data structure fit         |   15%  |
| implementation feasibility |   10%  |
| visual similarity          |    5%  |

## Tools

| tool | purpose |
| ---- | ------- |
| `search_dashboard_references` | rank references by business/decision intent |
| `analyze_dashboard_reference` | deconstruct one reference's structure |
| `compare_dashboard_references` | compare candidates on decision-fit, recommend a winner |
| `generate_dashboard_blueprint` | map a reference's grammar onto the user's data (4 tiers) |
| `generate_dashboard_build_prompt` | emit a build prompt for AI coding tools |
| `audit_existing_dashboard` | prioritized (Critical/High/Medium) UX audit |
| `register_dashboard_reference` | add a reference (records source + rights) |

## Resources

- `dashboard://reference/{id}`
- `dashboard://industry/{industry}`
- `dashboard://pattern/{slug}`
- `dashboard://guideline/information-hierarchy`

## Prompts

`executive-dashboard`, `operations-dashboard`, `member-analysis-dashboard`,
`reference-to-react`, `dashboard-ux-audit`.

## Run

```bash
pip install -r dashboard_intelligence_mcp/requirements.txt
# from the repo root (so the package is importable):
python -m dashboard_intelligence_mcp.server   # stdio transport
```

Register it with an MCP client using `mcp.config.example.json` (adjust `cwd`).
Set `DASHBOARD_MCP_DATA_DIR` to persist runtime-registered references outside the
package.

## Test

```bash
python dashboard_intelligence_mcp/tests/test_core.py
```

## 오너스마켓 specialization

The seed archive ships cooperative-specific references (`REF-102`, `REF-105`,
`REF-107`) so a query like *"오너스마켓 지역조합장용 대시보드 — 지역 매출, 조합원 수익,
매장 설립기금, 5초 안에 판단"* ranks the regional-chapter reference first and
blueprints those exact metrics.

## Rights / collection principle

The archive stores **source URL, attribution and layout grammar — not
pixel-copies**. `rights_status` defaults to `reference_only`. Extract
information hierarchy and component structure; keep provenance.
