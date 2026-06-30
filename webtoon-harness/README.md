# Webtoon Harness v2

한 줄 요청("웹툰 한 화 만들어줘")을 **완성된 세로 스크롤 웹툰 회차**로 바꿔주는 Claude Code
하니스. 트렌드 조사 → 대사 위주·고긴장·**매 회차 반전** 시나리오 → 캐릭터 레퍼런스 고정
작화 → 50+ 패널 렌더 → **정확한 한글 오버레이 레터링** 뷰어까지 27개 전문 에이전트가
4개 팀으로 협업한다.

> `revfactory/webtoon-harness`(v1)의 구조를 참고하되, **단순 복제가 아니라 6가지
> 실질적 업그레이드**를 더했다. 설계의 단일 기준은 [`docs/DESIGN.md`](docs/DESIGN.md).

## v1 대비 무엇이 업그레이드됐나

| | v1의 약점 | v2 업그레이드 | 위치 |
|---|---|---|---|
| **U1** | 이미지에 구운 **한글이 깨짐**(v1 #1 고질병) | **하이브리드 오버레이 레터링** — 실제 폰트로 DOM/SVG 텍스트 레이어. 항상 정확·수정/번역 자유. bake는 패널별 옵션 | `skills/webtoon-lettering`, `scripts/lettering/overlay.mjs`, `viewer/template.html` |
| **U2** | TeamCreate+SendMessage **비결정적 조율** | **결정적 Workflow 스크립트** — fan-out/pipeline, 재개(resume), 토큰 예산. 팀 모드 폴백 | `.claude/workflows/webtoon-episode.js` |
| **U3** | **codex CLI 하드락**(5세션 = 구조적 병목) | **플러그형 이미지 백엔드** — codex/gpt-image/gemini/local-sd 교체. 동시성은 설정값 | `scripts/imagegen/` |
| **U4** | 연속성을 **산문으로만** 추적 → 표류 | **구조화 스토리 바이블 + 연속성 원장(JSON)** — 스키마 검증, 회차 간 모순 차단 | `schemas/`, `continuity-manager` |
| **U5** | 캐릭터 일관성 **육안 판정** | **객관적 C1 게이트** — 레퍼런스 대비 유사도 점수 + md5/손상 검사 | `scripts/validate/panel_check.mjs` |
| **U6** | 재렌더 캐시 없음 | **콘텐츠 주소 패널 캐시** — 재실행 시 변경분만 재렌더 | `scripts/cache/panel_cache.mjs` |

설계 원칙: **U1–U6(또는 정확성 수정)이 아니면 추가하지 않는다.** 표면적을 작게 유지한다.

## 빠른 시작

1. 이 폴더의 `.claude/`를 대상 프로젝트 루트로 복사한다(스킬·에이전트·workflow가 활성화됨).
   `schemas/`, `scripts/`, `viewer/`, `assets/fonts/`도 함께 둔다.
2. 이미지 백엔드를 고른다:
   ```sh
   export WT_IMG_BACKEND=codex          # 또는 gpt-image / gemini / local-sd
   # gpt-image: OPENAI_API_KEY, gemini: GEMINI_API_KEY, local-sd: WT_SD_URL
   ```
3. Claude Code에서 자연어로 요청:
   - "트렌드 반영해서 웹툰 1화 만들어줘" → 오케스트레이터가 전체 파이프라인 실행
   - "다음 화 만들어" / "패널 23번 다시 그려" / "반전 더 강하게" → 부분/후속 실행
4. 결과: `_workspace/RELEASE/ep{NN}/index.html` (말풍선 포함 세로 스크롤 웹툰).

### Workflow로 직접 실행 (결정적·재개 가능)

```js
Workflow({ name: "webtoon-episode", args: {
  episode: 1,
  request: "회귀 스릴러, 매 화 반전",
  backend: "codex", concurrency: 5
}})
```
실패하면 같은 인자로 다시 호출 → 변경 안 된 단계는 캐시로 건너뛴다.

## 구조

```
webtoon-harness/
├── README.md                       이 문서
├── docs/DESIGN.md                  ★ 설계 계약(모든 동작의 기준)
├── .claude/
│   ├── skills/                     7개 스킬(orchestrator, lettering, trend-research,
│   │                               scenario, panel-breakdown, panel-render, assembly)
│   ├── agents/                     27개 페르소나 정의
│   └── workflows/webtoon-episode.js  결정적 오케스트레이션(U2)
├── schemas/                        story-bible · continuity-ledger · panel-manifest (U4)
├── scripts/
│   ├── imagegen/{adapter.md,render.sh,backends/*.sh}   플러그형 백엔드(U3)
│   ├── lettering/overlay.mjs        오버레이 레터링(U1)
│   ├── validate/panel_check.mjs     C1/C6 객관 검증(U5)
│   └── cache/panel_cache.mjs        패널 캐시(U6)
└── viewer/template.html            세로 스크롤 + 오버레이 텍스트 뷰어
```

## 작업 산출물(`_workspace/`)

회차 작업물은 **대상 프로젝트의 `_workspace/`**에 단계별로 쌓인다(감사 추적). 경로 규약은
`docs/DESIGN.md §3`. 하니스 자체는 이 폴더를 커밋하지 않는다.

```
_workspace/{00_input,01_research,02_story,03_episode,04_visual,05_panels,06_assembly,RELEASE}
```

## 27 페르소나 (4팀)

- **리서치(5)** trend-scout · platform-ranker · audience-analyst · hook-analyst · trend-synthesizer
- **시나리오(9)** concept-architect · worldbuilder · character-designer · series-plotter · twist-master · tension-engineer · episode-outliner · dialogue-writer · script-editor
- **비주얼(9)** art-director · ref-sheet-artist · panel-director · letterer · prompt-smith · panel-artist-a/b/c · panel-validator
- **조립검수(4)** episode-compositor · quality-reviewer · continuity-manager · showrunner

## 요구 사항

- Claude Code (Workflow 도구가 있으면 모드 A, 없으면 에이전트 팀 모드 B로 폴백)
- 선택한 이미지 백엔드의 CLI/API 자격
- Node.js(레터링·검증·캐시 스크립트, ESM `.mjs`, 외부 의존성 없음)
- 한글 폰트(`assets/fonts/NanumGothic.ttf` 번들)

## 라이선스/출처

구조적 아이디어는 `revfactory/webtoon-harness`(v1)에서 참고했으며, 본 v2는 위 U1–U6
업그레이드를 독자적으로 구현한 것이다.
