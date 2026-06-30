# Webtoon Harness v2 — 사용 설명서

웹툰 한 화를 한 줄 요청으로 완성하는 하니스의 실전 사용법. 설계 배경/계약은
[`DESIGN.md`](DESIGN.md), 개요는 [`../README.md`](../README.md) 참고.

---

## 1. 설치 (대상 프로젝트에 얹기)

이 하니스는 "복사해서 쓰는" 방식이다. 웹툰을 만들 프로젝트 루트에 복사한다.

```sh
# 웹툰 작업할 프로젝트 루트에서
cp -r webtoon-harness/.claude  ./    # 스킬·에이전트·workflow 활성화 (필수)
cp -r webtoon-harness/schemas  ./
cp -r webtoon-harness/scripts  ./
cp -r webtoon-harness/viewer   ./
cp -r webtoon-harness/assets   ./    # NanumGothic 폰트
cp -r webtoon-harness/docs     ./    # 설계 문서(선택)
```

> `.claude/`가 프로젝트 루트에 있어야 Claude Code가 스킬·에이전트·workflow를 인식한다.

## 2. 이미지 백엔드 선택 (U3)

그림을 어떤 엔진으로 그릴지 환경변수로 정한다. **하나만** 고른다.

```sh
export WT_IMG_BACKEND=codex        # 기본. codex CLI 로그인 필요(ChatGPT OAuth)
# export WT_IMG_BACKEND=gpt-image && export OPENAI_API_KEY=sk-...
# export WT_IMG_BACKEND=gemini    && export GEMINI_API_KEY=...
# export WT_IMG_BACKEND=local-sd  && export WT_SD_URL=http://127.0.0.1:7860
```

`local-sd`는 세션 한도가 없으니 `concurrency`를 5보다 크게 올려도 된다(원본 v1의 5세션
병목이 사라진 부분).

## 3. 실행 — 두 가지 방법

### 방법 A: 자연어 (가장 쉬움)

Claude Code에 그냥 말하면 오케스트레이터 스킬이 전체 파이프라인을 돈다.

```
"트렌드 반영해서 웹툰 1화 만들어줘"
"회귀 복수극으로 매 화 반전 있는 웹툰 1화 만들어줘"
```

### 방법 B: Workflow 직접 호출 (결정적·재개 가능, U2)

```js
Workflow({ name: "webtoon-episode", args: {
  episode: 1,
  request: "회귀 스릴러, 매 화 반전",
  backend: "codex",
  concurrency: 5,
  // mode: "initial|next|partial|new",  // 생략 시 Phase 0가 자동 추론
  // only: "패널 23"                      // partial일 때 대상
}})
```

중간에 실패하면 **같은 인자로 다시 호출**하면 된다 → 안 바뀐 단계는 캐시로 건너뛰고(U6)
실패 지점부터 이어서 진행한다.

## 4. 진행 단계 (자동 수행)

```
Phase 0 라우팅 → 준비
  → ① 리서치(5명)      trend-brief.md
  → ② 시나리오(9명)    script_final.md (+ story-bible.json, 매 화 반전·50+컷)
  → ③ 비주얼            레퍼런스 시트 먼저 → 50+ 패널 렌더 ⇄ 6축 검증 루프
  → ④ 조립             말풍선 오버레이 → QA → 연속성 정합 → 릴리스
```

## 5. 결과물

모든 산출물은 대상 프로젝트의 `_workspace/`에 단계별로 쌓인다(감사 추적).

```
_workspace/RELEASE/ep01/index.html        ← 최종 결과 (브라우저로 열기)
_workspace/03_episode/ep01_script_final.md ← 대본
_workspace/05_panels/ep01/panel_*.png      ← 패널 이미지(아트)
_workspace/04_visual/ep01_panels.json      ← 말풍선/패널 명세
```

`index.html`을 브라우저로 열면 **말풍선 한글이 정확히 박힌 세로 스크롤 웹툰**이 보인다.

## 6. 후속 작업 (처음부터 다시 안 만든다)

| 하고 싶은 것 | 이렇게 말하면 |
|---|---|
| 다음 화 | "다음 화 만들어" — 세계관/캐릭터/레퍼런스 재사용, 새 회차만 |
| 특정 패널 수정 | "패널 23번 다시 그려" — 그 패널만 재렌더+재검증 |
| 반전 강화 | "반전 더 강하게" — 시나리오만 손보고 영향 단계만 재실행 |
| 대사 수정/번역 | `ep{NN}_panels.json`의 `text`만 고치고 overlay 재실행 — **재렌더 불필요** |

### 대사/번역 수정 (재렌더 없이 — U1의 보너스)

```sh
node scripts/lettering/overlay.mjs \
  --manifest   _workspace/04_visual/ep01_panels.json \
  --panels-dir _workspace/05_panels/ep01 \
  --template   viewer/template.html --title "1화" \
  --out        _workspace/06_assembly/ep01/index.html
```

`text`만 다른 언어로 바꿔 다시 돌리면 같은 아트로 다국어판이 나온다.

## 7. 알아두면 좋은 핵심 개념

- **레퍼런스 우선**: 캐릭터 시트를 먼저 그려 `04_visual/refs/`에 고정 → 모든 패널이
  이걸 기준으로 일관성 유지(회차를 넘어가도 재사용).
- **오버레이 레터링**: 패널 그림엔 **빈 말풍선**만 그리고, 한글 대사는 나중에 실제
  폰트로 얹는다 → 한글이 깨질 일이 없고 수정이 자유롭다. (그림과 일체화돼야 하는 SFX
  등은 패널별로 `lettering_mode: "bake"` 옵션.)

## 8. 트러블슈팅

| 증상 | 대처 |
|---|---|
| 렌더가 느리다/한도 걸림 | `concurrency` 낮추거나 `backend` 교체(U3) |
| 패널이 중복/손상 | 자동으로 그 패널만 재렌더(C6 검증) |
| 캐릭터 외형이 흔들림 | C1 유사도 미달 자동 재렌더(U5) |
| 50컷 미만 | episode-outliner에 "비트 더 쪼개줘" 요청 |
| 한글이 이상함 | 기본 오버레이라 깨질 일 없음. bake 패널만 점검 |
| Workflow 도구 없는 환경 | 자동으로 에이전트 팀(TeamCreate) 모드로 폴백(B 모드) |

## 9. 유틸리티 스크립트 직접 쓰기 (선택)

```sh
# 캐시 키 계산 / 저장 / 조회 (U6)
node scripts/cache/panel_cache.mjs key   --prompt-file P.txt --backend codex --seed 7 --ref CHAR_A.png
node scripts/cache/panel_cache.mjs store --key <KEY> --src panel_001.png
node scripts/cache/panel_cache.mjs check --key <KEY> --out out.png

# 패널 객관 검증 (U5) — C1 유사도 + C6 md5/손상
node scripts/validate/panel_check.mjs --dir _workspace/05_panels/ep01 --refs _workspace/04_visual/refs --threshold 12

# 단일 패널 렌더 (어댑터 직접 호출, U3)
scripts/imagegen/render.sh --backend "$WT_IMG_BACKEND" --prompt-file prompt.txt \
  --out panel_001.png --ref refs/CHAR_A.png --seed 7 --concurrency 5
```

---

요청 한 줄("웹툰 1화 만들어줘")에서 `RELEASE/ep01/index.html`까지 — 그게 전부다.
