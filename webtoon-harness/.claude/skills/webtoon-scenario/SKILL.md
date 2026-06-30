---
name: webtoon-scenario
description: >-
  웹툰 시나리오 팀 방법론 — 고긴장·대사 주도 스토리를 하이콘셉트에서 매 화 반전이 보장된 최종 대본까지 끌어내는 파이프라인.
  시나리오, 대본, 세계관, 캐릭터, 반전 설계 작업을 할 때 사용. Builds high-concept, dialogue-led webtoon
  scenarios with a guaranteed per-episode twist and 50+ panels, and maintains the structured story-bible.json (U4).
---

# Webtoon Scenario — 시나리오 팀 방법론

DESIGN.md(§4 페르소나 계약, §5 창작 게이트, §3 `_workspace/` 경로)를 단일 진실원으로 따른다.
크리에이티브 카피는 한국어, 코드/JSON 키는 영어.

## 무엇을 하는가
연구 결과(또는 brief)를 받아 9개 시나리오 페르소나를 순서대로 구동해
`_workspace/02_story/`와 `_workspace/03_episode/`에 파싱 가능한 산출물을 남긴다.
최종 산출물은 매 화 반전이 선명하게 착지하는 대사 주도 최종 대본
`ep{NN}_script_final.md`이다.

## 의존 파이프라인 (엄격한 순서)
```
concept-architect      → 02_story/concept.md
  → worldbuilder       → 02_story/world.md (+ story-bible.json: series/world/locations)
  → character-designer → 02_story/characters.md (+ story-bible.json: characters[])
  → series-plotter     → 02_story/series-arc.md (에피소드 맵)
  → [twist-master ∥ tension-engineer]   # 병렬
        twist-plan.md (화별 반전) / tension-curve.md (화별 곡선)
  → episode-outliner   → 03_episode/ep{NN}_beatsheet.md  (≥50 패널 강제)
  → dialogue-writer    → 03_episode/ep{NN}_script.md      (대사 주도)
  → script-editor      → 03_episode/ep{NN}_script_final.md (§5 게이트 통과)
```
각 페르소나는 채팅 산문이 아니라 **파일**을 쓴다. 후행 페르소나가 선행 산출물을 파싱한다.

## §5 창작 게이트 집행
- **대사 주도(§5.1)**: dialogue-writer가 내레이션을 소수로 제한, script-editor가 비율 검수.
- **고긴장(§5.2)**: tension-engineer가 상승 곡선을 설계, 매 화 종료 긴장 > 진입.
- **매 화 반전(§5.3)**: twist-master가 모든 화에 1:1 반전(복선→폭로) 배정,
  script-editor가 `script_final`에서 "반전이 선명히 착지"하는지 최종 검증.
- **50+ 패널(§5.4)**: episode-outliner가 비트를 분할해 패널 합 ≥50을 검산으로 강제,
  dialogue-writer가 하한 유지, 이후 panel-director가 절대 깨지 않음.
- **시리즈 연속성(§5.5)**: world/characters/locations를 재유도하지 않고 재사용.

## 50+ 패널이 강제되는 방식
episode-outliner는 비트 표에 비트별 예상 패널수를 매기고 합산한다.
합이 50 미만이면 긴장 비트를 더 잘게 분할(대화 왕복, 리액션 컷, 클로즈업/와이드 교차)해
재합산하며, `## 패널 합계 검산`이 통과할 때까지 반복한다. dialogue-writer와
script-editor는 이 하한을 검사로 재확인한다.

## story-bible.json 유지 방식 (U4)
- **소유자**: worldbuilder(`series`/`world_rules`/`locations[]`) + character-designer(`characters[]`).
- 스키마: `schemas/story-bible.schema.json` — 두 소유자 모두 작성 전 반드시 Read해 필드명을 일치시킨다.
- 구조(스키마 기준, 최상위 `additionalProperties:false`):
  `{ series{title,logline,genre[],tone,target_reader,art_style_ref},`
  `characters[]{id(^CHAR_),name,role(enum),appearance_tokens[],reference_sheet,personality,arc,speech_style},`
  `locations[]{token(^LOC_),name,description,lighting,recurring},`
  `world_rules[]{id,rule,established_in}, props[]{id,name,significance} }`
- **병합 규칙**: 항상 기존 파일을 읽고 자기 섹션만 병합한다. 다른 섹션·연속성 데이터를
  덮어쓰거나 삭제하지 않는다(§9: 충돌은 주석으로 표시). 화 종료 후
  continuity-manager가 `continuity-ledger.json`과 대조해 정합화한다.

## 다음 화 / 부분 재실행
다음 화 작업 시 concept/world/characters/story-bible는 재사용한다.
series-arc의 에피소드 맵에서 해당 화 행만 읽어 twist-plan·tension-curve의 해당 ep 블록을
참조하고, beatsheet→script→script_final만 새로 생성한다(§9 Phase 0 라우팅).
