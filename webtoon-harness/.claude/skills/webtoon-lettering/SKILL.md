---
name: webtoon-lettering
description: "웹툰 말풍선·대사를 이미지에 굽지 않고 실제 폰트의 HTML/SVG 오버레이 레이어로 렌더하는 하이브리드 레터링 방법론(U1). 한글 텍스트 깨짐을 원천 제거하고, 대사 수정·재배치·번역을 재렌더 없이 가능하게 한다. 말풍선 위치/꼬리/종류(speech·thought·narration·sfx·shout)를 ep{NN}_panels.json의 balloons[]로 명세하고 scripts/lettering/overlay.mjs + viewer/template.html로 합성한다. 트리거: '레터링', '말풍선', '대사 얹기', '한글 텍스트 정확하게', '대사 수정', '텍스트 오버레이', '번역본 만들기'. SFX 등 그림과 일체화돼야 하는 글자는 패널별 bake 옵션."
---

# Webtoon Lettering — 하이브리드(오버레이 우선) 레터링

v1의 #1 고질병은 **이미지에 구운 한글이 깨지는 것**이었다. v2는 대사를 **아트와 분리된
실제 폰트 레이어**로 패널 위에 얹는다. 한글이 항상 정확하고, 가독성이 보장되며, 수정·번역이
재렌더 없이 가능하다. 자세한 계약은 `docs/DESIGN.md §7`.

## 두 가지 모드

- **overlay (기본):** 패널 PNG는 **아트만**(빈 말풍선/여백 포함, 글자 없음). 대사는
  `overlay.mjs`가 DOM/SVG로 얹는다. 99%의 패널이 여기에 해당.
- **bake (패널별 옵션):** 그림과 떼어낼 수 없는 글자(통합 SFX, 간판 등)만 이미지에 직접
  생성. 이때만 panel-validator의 C3(텍스트) 검증이 적용된다.

## 데이터 명세 — `ep{NN}_panels.json`의 `balloons[]`

letterer가 패널마다 말풍선을 기록한다(스키마: `schemas/panel-manifest.schema.json`).
```jsonc
{
  "id": "panel_007", "lettering_mode": "overlay",
  "balloons": [
    { "speaker": "지후", "kind": "speech", "text": "...너였구나.",
      "box": { "x": 12, "y": 8, "w": 46, "h": 14 }, "tail": "down" },
    { "kind": "narration", "text": "그 순간 모든 게 무너졌다.",
      "box": { "x": 6, "y": 80, "w": 88, "h": 12 }, "tail": "none" }
  ]
}
```
- `box`는 패널 크기 대비 **백분율**(0–100) — 해상도 독립적.
- `kind`: speech / thought(생각, 둥근) / narration(나레이션 박스) / sfx(효과음) / shout(외침, 굵게).
- `text`는 짧고 한 호흡으로. 긴 대사는 여러 말풍선으로 나눈다.

## 합성 — `overlay.mjs`

```sh
node scripts/lettering/overlay.mjs \
  --manifest   _workspace/04_visual/ep{NN}_panels.json \
  --panels-dir _workspace/05_panels/ep{NN} \
  --template   viewer/template.html \
  --font       assets/fonts/NanumGothic.ttf \
  --title      "에피소드 제목" \
  --out        _workspace/06_assembly/ep{NN}/index.html
```
- 매니페스트를 읽어 패널을 `panel_*` 순으로 정렬, 각 패널을 `<figure>`(아트 `<img>` +
  절대 위치 말풍선 `<div>`)로 만들어 세로 스크롤 뷰어에 주입한다.
- 폰트는 상대 URL `@font-face`로 임베드 → 뷰어 폴더만 옮겨도 한글이 그대로 표시.
- `lettering_mode:"bake"` 패널은 오버레이를 생략(이미 글자가 그려져 있음).

## 작업 순서 (letterer ↔ episode-compositor)

1. **letterer**: 대본(`ep{NN}_script_final.md`)·샷리스트를 보고 패널별 말풍선을
   `ep{NN}_panels.json`에 기록 + 사람이 읽을 `ep{NN}_lettering.md` 작성.
2. **prompt-smith**: 프롬프트에 "빈 말풍선/여백 확보, `no English/gibberish/misspelled text`"를
   넣어 아트에 글자가 안 생기게 한다(bake 패널 제외).
3. **episode-compositor**: 렌더 통과 후 `overlay.mjs`로 합성 → `index.html`.

## 폰트

`assets/fonts/`의 NanumGothic(이미 번들). 다른 글꼴은 `--font`로 교체. 굵은 외침은
template.html의 `.shout`/`.sfx` 클래스가 처리하므로 별도 폰트 없이도 동작.

## 번역/수정 (오버레이의 보너스)

대사 텍스트만 고치고 `overlay.mjs`를 다시 돌리면 끝 — **패널 재렌더 불필요.** 같은 아트로
다국어판을 만들 수 있다(`text`만 교체).
