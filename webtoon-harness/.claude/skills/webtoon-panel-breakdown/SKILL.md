---
name: webtoon-panel-breakdown
description: 웹툰 대본을 세로 스크롤 샷리스트(50컷 이상)로 분해하고, 스타일/장소/캐릭터 토큰화로 일관성을 설계하는 방법론. 컷 연출, 샷리스트 작성, 스타일 바이블·LOC/CHAR 토큰, 패널 분배가 필요할 때 사용한다.
---
# Webtoon Panel Breakdown — 샷리스트 & 토큰화 방법론

확정 대본을 일관성 있는 50컷 이상의 세로 스크롤 샷리스트로 분해하는 방법. art-director / panel-director / letterer / prompt-smith가 공유한다.

## 1. 토큰화로 일관성 고정 (재유도 금지)
- **LOC_* 장소 토큰**: 장소마다 `LOC_<SCREAMING_SNAKE>` + 한 줄 비주얼 정의(요소·시간대·색조). art-director가 style-bible에 정의, 모두가 그대로 인용.
- **CHAR_* 캐릭터 토큰**: 캐릭터마다 고정 외형값 + ref png 앵커. 패널 프롬프트의 일관성 닻.
- 다음 에피소드는 `continuity-ledger.json` / `refs/INDEX.md`의 기존 토큰을 **재사용**한다. 재정의 금지.

## 2. 씬 분할과 LOC 배정
- 대본을 비트→scene으로 묶고 `scene_id`(S01…) 부여.
- scene 하나에 정확히 하나의 `LOC_*`. 같은 scene의 모든 패널은 동일 LOC → C2(배경 연속성) 검증의 근거.

## 3. 50컷 이상 분해
- 비트를 패널로 쪼갠다. 부족하면 리액션 컷·앵글 전환·긴장 비트를 추가해 ≥50.
- 고정 표: `panel_id | scene_id | LOC | CHAR | 샷/앵글 | 연출 의도 | 대사 유무`.
- 세로 리듬: establishing → 가속 → 말미 임팩트(클리프행어).

## 4. 레터링 사양 (오버레이 기본, U1)
- 대사는 아트에 굽지 않는다. letterer가 `panels.json`의 `balloons[]`에 text·speaker·box(xywh %)·tail·style을 적는다.
- bake는 SFX 등 패널 단위 옵트인(`bake:true`)만.

## 5. 프롬프트 합성 입력 준비
- 패널 프롬프트 = 화풍 + LOC 정의 + CHAR(ref 앵커) + 샷/감정/구도.
- 비-bake 패널은 "빈 말풍선/여백" 요청 + 네거티브 `no English/gibberish/misspelled text`.
- 패널을 씬 그룹 A/B/C로 분배(가능하면 scene 경계 기준, 부하 균형).

## Definition of done
- 모든 패널에 panel_id + scene_id + 단일 LOC_* + (등장 시) CHAR_*가 있다.
- 패널 수 ≥ 50.
- LOC_*/CHAR_* 토큰이 style-bible/character-sheets에 정의돼 있다.
- 레터링은 오버레이 기본, bake는 명시 옵트인.
