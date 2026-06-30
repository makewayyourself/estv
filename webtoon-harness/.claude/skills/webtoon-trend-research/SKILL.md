---
name: webtoon-trend-research
description: Runs the Research phase of the webtoon harness — five personas (trend-scout, platform-ranker, audience-analyst, hook-analyst, trend-synthesizer) fan out to scout genre/trope momentum, platform rankings & serialization structure, reader segments/drop-off, and hook/twist mechanisms, then synthesize a single planning brief for the scenario team. Use when starting a new episode and at the "리서치/트렌드 조사/시장 분석/장르 트렌드/독자 분석/기획 브리프" step, or whenever the user asks to research the webtoon market before writing a scenario.
---

# Webtoon Trend Research

웹툰 하네스의 1번째 단계(Research). 5개 페르소나가 시장을 조사해 `_workspace/01_research/`에 파싱 가능한 산출물을 남기고, 마지막에 단일 기획 브리프(`trend-brief.md`)로 종합한다. 이 브리프가 시나리오팀(02_story)의 입력이 된다.

## 언제 쓰나
- 새 에피소드 착수 직후, 시나리오 작성 전.
- "트렌드 조사", "시장 분석", "장르 트렌드", "독자 분석", "기획 브리프 만들어줘" 요청.

## 산출물 (고정 경로)
| 페르소나 | 파일 | 내용 |
|---|---|---|
| trend-scout | `_workspace/01_research/trend-scout.md` | 장르/트로프 모멘텀 표 |
| platform-ranker | `_workspace/01_research/platform-ranker.md` | 플랫폼 랭킹·연재 구조 권장 |
| audience-analyst | `_workspace/01_research/audience-analyst.md` | 독자 세그먼트·이탈/몰입 |
| hook-analyst | `_workspace/01_research/hook-analyst.md` | 훅/반전 패턴 카탈로그 |
| trend-synthesizer | `_workspace/01_research/trend-brief.md` | 종합 기획 브리프 |

## 실행 순서 (fan-out → synthesize)
1. **병렬(fan-out):** trend-scout · platform-ranker · audience-analyst · hook-analyst를 함께 실행한다. 넷 다 `_workspace/00_input/brief.md`를 읽는다. platform-ranker·audience-analyst·hook-analyst는 trend-scout 결과를 교차참조하므로, 의존을 엄격히 하려면 trend-scout를 먼저, 나머지 셋을 그다음 병렬로 돌린다.
2. **종합(synthesize):** 4종이 모두 존재하면 trend-synthesizer가 이를 읽어 `trend-brief.md`를 만든다. 입력 누락 시 중단·보고.
3. Workflow 모드에서는 `parallel()` 후 `pipeline()` 단계로 구성하고, 팀 폴백 모드에서는 4개 병렬 Task 후 종합 Task로 구성한다(DESIGN §9).

## 리서치 방법론 (5종 공통)

### 검색 (how to search)
- 다각도 질의: 장르 키워드(예: `웹툰 인기 장르 2026`, `회귀 빙의 환생`), 플랫폼+`랭킹`, 영어 병행(`webtoon trending tropes`).
- 한 번의 질의로 단정하지 말 것. 최소 2~3개 서로 다른 질의로 신호를 교차확인한다.
- 상위 결과는 WebSearch 스니펫만 믿지 말고 WebFetch로 본문을 열어 1차 신호를 확인한다.

### 출처 (how to source)
- 모든 주장 옆에 출처 URL + 조회일을 단다. 출처 없는 단정은 금지.
- 신뢰 우선순위: 플랫폼 공식 랭킹/공지 > 업계 기사 > 커뮤니티 반응. 커뮤니티 신호는 "반응"으로만 인용하고 사실로 승격하지 않는다.
- 추정은 추정으로 표기한다("추정:" 접두).

### 감사 추적 (audit trail)
- 산출물은 채팅이 아니라 **파일**로 남긴다(고정 헤딩 + 표). 다운스트림 페르소나가 파싱한다.
- 표 칼럼/헤딩 이름은 페르소나 파일에 정의된 그대로 유지한다(파서 안정성).
- 데이터 충돌은 삭제하지 말고 주석/`근거·트레이드오프`로 보존한다(DESIGN §9 "never delete conflicting data").

### 종합 (how the synthesizer combines)
- trend-synthesizer는 `Glob`으로 4종 존재를 확인하고, 표를 파싱한 뒤 단일 권장안으로 수렴한다.
- 충돌 해소 우선순위: **독자 리텐션 > 플랫폼 현실 > 트렌드 모멘텀 > 훅 다양성**.
- 브리프는 결정형 문장으로 쓴다(나열·모호어 금지). 컷 수는 ≥50을 명시해 50+컷 원칙(§5-4)과 정합시킨다.

## 완료 기준 (게이트)
- 5개 파일이 모두 고정 경로에 존재한다.
- 각 파일의 출처 주장에 URL이 있다.
- `trend-brief.md`가 단일 장르 방향·1순위 타깃 독자·훅 전략·긴장/반전 가이드·연재 포맷(≥50컷)을 모두 담는다.
- 시나리오팀이 추가 질문 없이 착수할 수 있을 만큼 구체적이다.
