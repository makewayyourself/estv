---
name: webtoon-orchestrator
description: "웹툰 한 회차를 트렌드 조사 → 대사 위주·고긴장·매 회차 반전 시나리오 → 캐릭터 레퍼런스 시트 선행 → 50+ 패널 렌더 → 오버레이 레터링(정확한 한글) 세로 스크롤 뷰어까지 만들어내는 통합 오케스트레이터(v2). 27개 전문 에이전트를 4개 단계별 팀으로 운영하되, 기본은 결정적 Workflow 스크립트(webtoon-episode)로 조율하고, Workflow 도구가 없으면 에이전트 팀(TeamCreate) 모드로 폴백한다. 트리거: '웹툰 만들어/제작', '웹툰 한 화/회차 만들어', '웹툰 시나리오부터 이미지까지', '웹툰 에피소드 제작', '웹툰 하네스 실행'. 후속: '다음 화 만들어', '이 회차 다시/수정', '반전 더 강하게', '패널 다시 그려', '특정 단계만 다시 실행'에도 반드시 이 스킬 사용. 단순 웹툰 추천/감상은 직접 응답."
---

# Webtoon Orchestrator v2 — 웹툰 제작 팀 조율

웹툰 한 회차를 트렌드 조사부터 완성 뷰어까지 만들어내는 통합 오케스트레이터. **반드시
먼저 `docs/DESIGN.md`를 읽어라** — 디렉터리/`_workspace` 경로(§3), 페르소나 계약(§4),
창작 게이트(§5), 어댑터/레터링/검증/캐시 계약(§6–§10)이 모든 동작의 기준이다.

## v2 핵심 (v1 대비 업그레이드)

| | 업그레이드 | 효과 |
|---|---|---|
| U1 | 하이브리드 **오버레이 레터링** | 한글 텍스트 깨짐 제거 — 실제 폰트로 DOM/SVG 렌더 |
| U2 | **결정적 Workflow 조율** | 재현·재개(resume)·예산 제어 가능, 팀 모드 폴백 |
| U3 | **플러그형 이미지 백엔드** | codex 5세션 병목 제거(동시성은 설정값) |
| U4 | **구조화 스토리 바이블/연속성 원장(JSON)** | 회차 간 모순을 기계 검증 |
| U5 | **객관적 C1 게이트**(유사도 점수) | 캐릭터 일관성 육안 검증 탈피 |
| U6 | **콘텐츠 주소 패널 캐시** | 재실행 시 변경분만 재렌더 |

## 실행 모드 A — Workflow (기본)

`Workflow` 도구가 있으면 **이 모드를 쓴다.** 결정적이고 재개 가능하다.

```
Workflow({ name: "webtoon-episode", args: {
  episode: <N>, request: "<사용자 요청>",
  backend: "codex|gpt-image|gemini|local-sd",  // 기본 codex
  concurrency: 5,                                // U3 — 병목 아님, 조절값
  mode: "initial|next|partial|new",             // 생략 시 Phase 0가 추론
  only: "<부분 재실행 대상>"                      // partial일 때
}})
```

스크립트(`.claude/workflows/webtoon-episode.js`)가 Phase 0 라우팅 → 준비 → 리서치
→ 시나리오 → 비주얼(렌더-검증 루프) → 조립의 모든 단계를 `agent(..., {agentType:'<페르소나>'})`로
스폰하며 진행한다. 실패 시 같은 인자로 다시 호출하면 변경되지 않은 단계는 캐시(resume)된다.

## 실행 모드 B — 에이전트 팀 (폴백)

Workflow 도구가 없는 환경에서만. 세션당 활성 팀 1개이므로 Phase마다 `TeamCreate`로 팀을
만들고 끝나면 `TeamDelete`로 정리한 뒤 다음 팀을 만든다. 모든 스폰에 `model: "opus"` 명시.
산출물은 `_workspace/`에 남아 다음 팀이 Read로 이어받는다. 팀 구성은 §팀 표 참조.

## 팀 구성 (27 페르소나, 4팀)

| 팀 | 페르소나 |
|---|---|
| **리서치(5)** | trend-scout, platform-ranker, audience-analyst, hook-analyst, trend-synthesizer |
| **시나리오(9)** | concept-architect, worldbuilder, character-designer, series-plotter, twist-master, tension-engineer, episode-outliner, dialogue-writer, script-editor |
| **비주얼(9)** | art-director, ref-sheet-artist, panel-director, letterer, prompt-smith, panel-artist-a/b/c, panel-validator |
| **조립검수(4)** | episode-compositor, quality-reviewer, continuity-manager, showrunner |

각 페르소나 정의는 `.claude/agents/<name>.md`. 도메인 방법론은 `webtoon-trend-research`,
`webtoon-scenario`, `webtoon-panel-breakdown`, `webtoon-panel-render`, `webtoon-lettering`,
`webtoon-assembly` 스킬에 있다.

## Phase 0: 실행 모드 판별 (항상 먼저)

`_workspace/` 존재 여부 + 사용자 요청으로 결정한다.
1. `_workspace/` 미존재 → **initial**(전체 실행).
2. 존재 + "다음 화" → **next**. {NN}++; `02_story`/style-bible/`refs/`/continuity 재사용(재렌더 금지), `03_episode`부터 신규.
3. 존재 + "OO만 다시" → **partial**. 해당 단계만 재구성·덮어쓰기, 영향받는 하위 단계만 재실행.
4. 존재 + 새 기획 → **new**. 기존 `_workspace/`를 `_workspace_archive/`로 이동(삭제 금지) 후 전체 실행.

## 워크플로우 요약

1. **준비** — `00_input/brief.md` 기록, `_workspace/` 하위 디렉터리 보장.
2. **리서치** — 4명 병렬 조사 → trend-synthesizer가 `trend-brief.md` 종합.
3. **시나리오** — concept→world→characters→series-arc→{twist-plan‖tension-curve}→beatsheet→script→script_final. §5 게이트(대사 주도/매 화 반전/50+ 패널) 강제. `story-bible.json` 유지(U4).
4. **비주얼** — art-director 스타일 바이블 → **레퍼런스 시트 선행**(refs/, 재사용 SSOT) → panel-director 샷리스트 ‖ letterer 오버레이 명세(`ep{NN}_panels.json`) → prompt-smith 프롬프트(빈 말풍선·여백) → 패널 렌더(어댑터+캐시, 동시성=설정값) ⇄ **panel-validator 6축 검증-재생성 루프(패널당 ≤3)**.
5. **조립·검수** — episode-compositor가 `overlay.mjs`로 아트 위에 말풍선 오버레이 → 세로 스크롤 `index.html` → quality-reviewer QA(PASS/FIX/REDO, ≤2루프) → continuity-manager가 `continuity-ledger.json` 정합 → showrunner 사인오프·`RELEASE/ep{NN}/`·다음 화 시드.
6. **마무리** — `_workspace/` 보존(감사 추적). 결과 요약(제목·로그라인·반전 한 줄·패널 수·뷰어 경로) + 피드백 요청.

## 에러 핸들링

| 상황 | 전략 |
|---|---|
| 페르소나 실패 | 해당 단계만 재실행(Workflow resume) 또는 대체 스폰 |
| 렌더 0바이트/손상 | 해당 패널만 재렌더(배치 전체 금지); panel_check.mjs C6 |
| 패널 md5 중복 | 중복 삭제 후 단독 재렌더(C6) |
| 배경 급변 | C2 REGEN → prompt-smith가 LOC_* 강화 후 그 패널만 |
| 캐릭터 외형 이탈 | C1 유사도 미달 REGEN → 레퍼런스 앵커 강조 재렌더 |
| 한글 텍스트 | 기본 오버레이라 깨질 일 없음. bake 패널만 C3 적용 |
| 백엔드 느림/한도 | `--concurrency` 낮추거나 `--backend` 교체(U3) |
| 50 패널 미만 | episode-outliner/panel-director에 비트 추가 분할 요청 |
| 반전 불명확 | 시나리오팀(twist-master/script-editor)으로 피드백 |
| 무한 재작업 | 패널 재생성 ≤3, 단계 재작업 ≤2; 초과 시 현 상태 진행 + 한계 보고 |
| 데이터 충돌 | 출처 병기, **삭제 금지** |

## 하네스 진화

같은 유형 피드백이 2회 이상 반복되면 해당 스킬/페르소나/스크립트 정의 개선을 제안한다.
단, `docs/DESIGN.md`의 "U1–U6 또는 정확성 수정이 아니면 추가하지 않는다" 원칙을 지켜
표면적을 작게 유지한다. 변경은 DESIGN 또는 README 변경 이력에 기록.
