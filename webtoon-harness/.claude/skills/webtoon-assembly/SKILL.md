---
name: webtoon-assembly
description: Runs the Assembly & Review phase of the webtoon harness — four personas (episode-compositor, quality-reviewer, continuity-manager, showrunner) assemble the vertical-scroll viewer with the overlay lettering layer, run episode-level QA gates, reconcile the structured continuity ledger, and package the signed-off release. Use after panels are rendered/validated, at the "조립/뷰어/QA/검수/연속성/릴리스/배포/패키징/다음 화" step, or whenever the user asks to assemble, review, or release a webtoon episode.
---

# Webtoon Assembly & Review

웹툰 하네스의 마지막 단계(Assembly). art-only 패널과 오버레이 레터링을 세로 스크롤 뷰어로 조립하고, 에피소드 전체를 QA한 뒤, 연속성 원장을 정합하고, 사인오프된 릴리스를 패키징한다. panel-validator(05_panels)가 끝난 뒤 시작한다.

## 언제 쓰나
- 패널 렌더·검증이 끝난 직후.
- "에피소드 조립", "뷰어 만들어줘", "QA/검수", "연속성 정리", "릴리스/배포/패키징", "다음 화 시드" 요청.

## 산출물 (고정 경로)
| 페르소나 | 파일 | 내용 |
|---|---|---|
| episode-compositor | `_workspace/06_assembly/ep{NN}/index.html` | 세로 스크롤 뷰어(art + 오버레이 레터링) |
| quality-reviewer | `_workspace/06_assembly/ep{NN}/qa_report.md` | G1–G7 게이트 + PASS/FIX/REDO |
| continuity-manager | `_workspace/06_assembly/continuity.md`, `_workspace/02_story/continuity-ledger.json` | 연속성 리포트 + 구조화 원장(U4) |
| showrunner | `_workspace/RELEASE/ep{NN}/` | 사인오프 패키지 + 다음 화 시드 |

## 1. 뷰어 조립 (오버레이 레터링 레이어, U1)
- 입력: `_workspace/05_panels/ep{NN}/panel_*.png`(art only, 텍스트 없음), `_workspace/04_visual/ep{NN}_panels.json`(`balloons[]`: text/speaker/box xywh %/tail/style).
- 베이스: `viewer/template.html`(세로 스크롤 셸). 말풍선은 `scripts/lettering/overlay.mjs`로 DOM/SVG 실시간 렌더 — 번들 폰트(NanumGothic)로 한글을 그린다.
- **불변식: 텍스트는 절대 이미지로 굽지 않는다.** 패널=art only, 텍스트=재식자 가능한 오버레이 레이어. (베이킹은 패널별 opt-in이며 그 경우 C3 검증 적용.)
- 컴포지터는 패널 간 간격·리듬(컷 여백, 장면 전환 큰 간격, 클라이맥스 호흡)을 설계한다.

## 2. QA 게이트
quality-reviewer는 `ep{NN}_validation.md`를 먼저 읽어 ACCEPT-FLAG 패널을 확인한 뒤 게이트를 검사한다:
- **G1** 패널 수 ≥ 50 (실측)
- **G2** 레퍼런스 외형 일관성(C1 교차)
- **G3** 배경 연속성(C2 교차)
- **G4** 오버레이 텍스트 정확·가독(이미지 베이킹 없음)
- **G5** 대사 흐름·화자 매칭(`script_final` 대조)
- **G6** 반전 전달(twist-plan/`script_final` 대조)
- **G7** 무결성(0바이트/손상 없음, md5 중복 없음)

판정: 전부 PASS→**PASS** / 국소 수정→**FIX** / 재작업→**REDO**. 라우팅: 패널=panel-artist+prompt-smith 리젠 루프(≤3회), 텍스트=letterer, 대사·반전=script-editor·dialogue-writer, 조립=episode-compositor.

## 3. 연속성 정합 (U4)
continuity-manager가 `continuity-ledger.json`을 정합한다(`schemas/continuity-ledger.schema.json` 존재 시 그 필드명; 없으면 DESIGN §3 구조: characters/worldFacts/plotThreads/foreshadowing). 이번 화의 새 외형 사실·세계관 사실·신규/해결 스레드·복선 회수를 반영하고, **충돌은 삭제하지 않고 주석으로 보존**한다. 사람이 읽는 요약은 `continuity.md`.

## 4. 릴리스 패키징
showrunner는 QA 판정이 PASS일 때만 `_workspace/RELEASE/ep{NN}/`에 `index.html`+참조 자산(PNG, overlay.mjs, 폰트)을 자기완결로 복사하고, `RELEASE_NOTES.md`(로그라인·반전·패널 수·QA·한계)와 `next-episode-seed.md`(클리프행어 핸드오프·열린 스레드·회수할 복선)를 작성한다.

## 실행 순서
1. episode-compositor → index.html (REGEN 미해결 시 보류)
2. quality-reviewer → qa_report.md (FIX/REDO면 라우팅 후 1·2 반복, rework ≤2회)
3. continuity-manager → continuity-ledger.json + continuity.md (PASS 후)
4. showrunner → RELEASE/ep{NN}/ (PASS 확인 후 사인오프)

## 업그레이드 책임
- U1: 오버레이 레터링(컴포지터, `overlay.mjs`+`template.html`).
- U4: 구조화 연속성 원장(continuity-manager).
- U5: 에피소드-레벨 QA 게이트(quality-reviewer)가 panel-validator 판정을 종합.
