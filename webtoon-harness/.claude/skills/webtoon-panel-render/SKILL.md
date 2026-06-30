---
name: webtoon-panel-render
description: 이미지 어댑터로 웹툰 패널을 렌더하고, 콘텐츠 주소 캐시를 확인하며, 6축 검증→리젠 루프를 돌리는 방법. 패널 렌더링, 백엔드 어댑터 호출, 캐시 히트 처리, 패널 검증·재생성이 필요할 때 사용한다.
---
# Webtoon Panel Render — 어댑터 렌더 + 캐시 + 검증/리젠 루프

ref-sheet-artist / panel-artist-a·b·c / panel-validator가 공유하는 렌더링 실행 방법. 패널 PNG는 **아트 전용**(텍스트 없음)이며 레터링은 나중에 `scripts/lettering/overlay.mjs` + `viewer/template.html`로 오버레이된다. 예외는 `--bake` 패널뿐.

## 1. 캐시 먼저 (U6)
렌더 전 항상 확인:
```
node scripts/cache/panel_cache.mjs check   # 키: sha256(prompt + ref-ids + seed + backend)
```
- 히트 → 캐시 PNG를 출력 경로로 복사, 백엔드 호출 건너뜀.
- 미스 → 렌더로 진행.
캐시는 `_workspace/.cache/panels/`. **store는 ACCEPT 이후에만**.

## 2. 어댑터로 렌더 (U3) — 백엔드 직접 호출 금지
```
scripts/imagegen/render.sh --backend "$WT_IMG_BACKEND" \
  --prompt-file <panel.md> --out _workspace/05_panels/ep{NN}/panel_<id>.png \
  --ref <ref.png> [--seed N] [--concurrency K] [--bake]
```
- 백엔드는 `--backend`/`$WT_IMG_BACKEND`(기본 `codex`)로 결정, `backends/<name>.sh`로 라우팅.
- `--concurrency` 기본 5 — 조절 손잡이지 구조적 상한이 아님.
- `--ref`로 캐릭터 시트를 앵커링해 일관성 확보.
- `--bake`는 `panels.json` `bake:true` 패널에만. 그 외는 빈 말풍선/여백 아트.

## 3. 무결성 확인 후 검증
렌더 직후 PNG가 비제로·유효 헤더인지 본다. 이어 6축 검증:
```
node scripts/validate/panel_check.mjs --panels _workspace/05_panels/ep{NN} --refs _workspace/04_visual/refs
```
- **C1** 캐릭터 일관성(지각 해시/임베딩 vs refs, 객관) · **C6** 무결성+md5 중복 제거 → 스크립트.
- **C2** 배경/LOC 연속성 · **C4** 비트 충실도 · **C5** 읽기 순서 → 판단.
- **C3** 한글 텍스트 → `--bake` 패널만(오버레이는 항상 정확 → N/A).
- 판정: `ACCEPT` / `REGEN(사유)` / `ACCEPT-FLAG`.

## 4. 리젠 루프 (패널당 최대 3회)
REGEN → prompt-smith가 프롬프트 강화 → 담당 panel-artist가 *해당 패널만* 재렌더 → 재검사. 3회 초과 시 ACCEPT-FLAG로 한계 로그.

## 5. ACCEPT 후 캐시 store
```
node scripts/cache/panel_cache.mjs store
```
REGEN 산출물은 저장하지 않는다.

## Definition of done
- 모든 렌더가 `render.sh` 어댑터 경유(백엔드 직접 호출 0).
- 렌더 전 캐시 check, ACCEPT 후 store 수행.
- 비-bake 패널은 텍스트 없는 아트.
- 모든 패널이 6축 검증을 통과(ACCEPT) 또는 ACCEPT-FLAG로 한계 명시, 잔존 REGEN 0건.
