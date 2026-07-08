# 픽셀로지 오너스 마켓 — AI 재고 분류(Triage) 엔진 전산 설계서

> 이 문서는 심층 탐구(팟캐스트)에서 정의된 **"AI의 역할"** 을 실제 돌아가는 소프트웨어로
> 옮기기 위한 **1차 설계(스펙)** 입니다. 코드 구현 전에 입력 변수·계산식·판단 규칙·출력
> 포맷을 확정하는 것을 목표로 하며, `price_trend_formula.md`(가격 추이 계산식)와
> `#master_plan.md`(트리거/캠페인 엔진)의 문서 규약을 그대로 따릅니다.

---

## 0) 한 줄 정의 — 이 AI는 "돈 복사기"가 아니라 "응급실 분류 간호사"다

팟캐스트의 핵심 명제를 그대로 설계 원칙으로 고정합니다.

- ❌ **아님**: "가만히 있어도 매달 10% 보장" 같은 수익 보장 마법 (→ 유사수신 리스크)
- ✅ **맞음**: 창고로 쏟아져 들어오는 재고(=환자)를 **위급도별로 분류**하고,
  **어느 채널로 · 언제 · 어떤 마진에** 흘려보내야 손실을 막는지 판단하는 **리스크 필터**

따라서 이 엔진의 1차 목표 함수는 **"수익 극대화"가 아니라 "재고 사장(死藏)·평가손 최소화"** 입니다.
수익은 그 결과로 따라오는 2차 지표로 취급합니다.

---

## 1) 시스템 위치 — 3주체 거버넌스에서 엔진이 앉는 자리

팟캐스트의 3주체 구조 안에서 본 엔진은 **운영본부(픽셀로지)** 의 두뇌에 해당합니다.

| 주체 | 역할 | 엔진과의 관계 |
|---|---|---|
| 조합원 | 자본 출자 · 물품대금 위탁 · **실물 SKU 소유** | 엔진 판단의 **대상(입력)** 이자 정산의 귀속 주체 |
| 운영본부(픽셀로지) | AI 분석 · 보관 · 배송 · 정산 대행 | **본 Triage 엔진을 운영** |
| 공간 파트너 | 물리 매장 · 대형 물류창고(3PL) 제공 | 엔진이 선택하는 **처분 채널의 일부** |

엔진의 모든 판단·실행은 뒤(9절)의 **단일 원장(Single Ledger)** 에 로그로 남아
조합원 앱에 실시간 공개됩니다 — 팟캐스트의 "보이는 상품 / 보이는 창고 / 보이는 정산".

---

## 2) 엔진 파이프라인 개요

```
[수집] ──► [정규화] ──► [피처 생성] ──► [Triage 점수] ──► [채널·시점·마진 매칭] ──► [액션/리스크 로그] ──► [단일 원장]
  │            │             │                │                      │                        │
 2A/2B        3            4               5                      6·7                      8·9
```

- **2A 내부 데이터**: 조합원이 소유한 SKU 마스터 + 재고/원가/체류일 (CSV 업로드 또는 DB)
- **2B 외부 데이터(실연동)**: 네이버 데이터랩 검색/쇼핑 트렌드 + 네이버 쇼핑 경쟁가 (MCP)
- 이후 단계는 순수 계산 로직 → 결정론적으로 재현 가능해야 함(감사 대응)

---

## 3) 입력 데이터 (A) — 내부 SKU 마스터 · CSV 업로드 스키마

CSV 업로드와 DB 연동 **두 경로 모두** 아래 정규 스키마(`sku_master`)로 수렴시킵니다.
업로드 시 컬럼 매핑 UI로 헤더가 달라도 흡수합니다.

| 컬럼 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `sku_id` | str | ✅ | 실물 SKU 고유번호(바코드/일련번호). 조합원 소유의 "주민등록번호" |
| `owner_member_id` | str | ✅ | 이 SKU를 소유한 조합원 ID (정산 귀속) |
| `product_name` | str | ✅ | 상품명 |
| `brand` | str | | 브랜드 |
| `category_query` | str | ✅ | 카테고리 판별용 한글 키워드(예: "립스틱", "노트북") → 4절에서 `find_category`에 사용 |
| `qty_on_hand` | int | ✅ | 현재 재고 수량 |
| `unit_cost` | float | ✅ | 조합 도매 매입 단가(원) — **마진 계산의 기준선** |
| `inbound_date` | date | ✅ | 창고 입고일 → 체류일(aging) 계산 |
| `avg_daily_sellthrough` | float | | 최근 판매 속도(개/일). 없으면 카테고리 평균으로 대체 |
| `seasonal_peak_month` | int(1~12) | | 계절 수요 정점 월. 없으면 카테고리 시즌성으로 추정 |
| `condition_grade` | enum | | `new`/`refurb_A`/`refurb_B` (리퍼브 등급) |
| `min_channel` | str[] | | 판매 허용 채널 화이트리스트(선택) |

> **소싱 방어막 반영**: 팟캐스트의 "인코어 24년 소싱망 → 엑싯 가능한 정품만 편입" 원칙을
> 데이터 레벨에서 강제하려면, 업로드 단계에서 `unit_cost > 0`, `qty_on_hand > 0`,
> `condition_grade ∈ 허용집합` 검증을 통과하지 못한 행은 **반려**합니다.
> (창고 문을 아무 재고에나 열어주지 않는다 = 입력 게이트)

---

## 4) 입력 데이터 (B) — 외부 실연동 · 네이버 데이터랩/쇼핑 (MCP)

팟캐스트의 *"외부 포털 검색 트렌드 + 경쟁 플랫폼 과거·현재 가격 추이를 실시간으로 긁어 분석"* 을
아래 4개 MCP 툴로 실제 구현합니다. 호출 순서와 매핑을 고정합니다.

### 4-1. 카테고리 코드 확정 — `find_category`
```
입력:  query = sku_master.category_query   (예: "립스틱")
출력:  category code (예: "50000188")      → 이후 데이터랩 호출에 필수
```

### 4-2. 수요 모멘텀 — `datalab_shopping_keywords` (+ 보조 `datalab_search`)
카테고리 내 해당 상품/키워드의 **클릭 트렌드 시계열**을 가져와 수요 방향을 판단합니다.
```
입력:  category = (4-1의 코드)
       keyword  = [{ name: product_name, param: [핵심키워드…] }]
       startDate/endDate = 최근 12주,  timeUnit = "week"
출력:  주별 상대 클릭지수 시계열 ratios[]  → 5절 demand_momentum 계산
보조:  datalab_search 로 브랜드/일반명 검색량 추세를 교차검증 (검색 → 구매 선행지표)
```
> 절대량이 아니라 **상대지수 시계열**이 나오므로, 판단은 항상 "기울기/추세"로 합니다.

### 4-3. 경쟁가 침식 — `search_shop`
동일/유사 상품의 **현재 시장 최저가·가격 분포**를 수집합니다.
```
입력:  query = product_name (+ brand),  display = 40,  sort = "sim"
출력:  상품 리스트 [{ title, lprice(최저가), hprice, mallName, ... }]
집계:  comp_price_p10 = 하위10% 최저가대,  comp_price_median = 중앙값
       → 우리 원가 대비 마진 여력 + 경쟁 저가 압력(5절)
```
> **과거 추이**는 API가 스냅샷만 주므로, 호출 결과를 **단일 원장에 매일 적재**하여
> 시계열(`comp_price_history[sku_id][date]`)을 자체 축적합니다. 첫 실행일부터 추세가 쌓입니다.

### 4-4. 카테고리 시즌성 — `datalab_shopping_category`
계절 수요 정점 추정(`seasonal_peak_month` 미입력분 보완).
```
입력:  category = [{ name, param:[코드] }],  최근 24개월,  timeUnit="month"
출력:  월별 카테고리 클릭지수 → 정점 월(peak_month) 산출
```

### 4-5. 호출·캐시 정책
- 데이터랩은 상대지수라 **일 1회 배치**로 충분(과호출 방지). `search_shop`은 RED 후보만 실시간 재조회.
- 모든 외부 응답 원본(raw)을 `misc_data/` 규약에 맞춰 원장에 보존 → 판단 근거 감사 추적.
- 실연동 실패 시 **마지막 캐시값 + 저하 플래그**로 폴백(엔진이 멈추지 않게).

---

## 5) Triage 위급도 점수 (TUS, Triage Urgency Score) — 핵심 계산식

SKU마다 **0~100**의 위급도 점수를 산출합니다. **높을수록 빨리 처분해야 함**(=응급).
6개 신호를 각각 0~1로 정규화한 뒤 가중합합니다.

### 5-1. 신호 정의 (각 0~1, 1에 가까울수록 "위급")

```
# (a) 수요 모멘텀 하락  demand_drop
#   최근 4주 평균 대비 직전 4주 평균 변화율. 하락일수록 위급.
slope = (mean(ratios[-4:]) - mean(ratios[-8:-4])) / max(mean(ratios[-8:-4]), eps)
demand_drop = clip( -slope / DROP_NORM , 0, 1)          # slope<0(하락) → 값 상승

# (b) 경쟁가 침식  price_erosion
#   경쟁 최저가대가 우리 원가에 근접/하회할수록 위급.
margin_headroom = (comp_price_p10 - unit_cost) / max(unit_cost, eps)   # 여유 마진율
price_erosion = clip( (MARGIN_TARGET - margin_headroom) / MARGIN_TARGET , 0, 1)

# (c) 체류일(aging)
age_days = today - inbound_date
aging = clip( age_days / AGING_LIMIT , 0, 1)            # 예: 90일=1.0 (채화 기준)

# (d) 재고 깊이(overstock)  depth_risk
days_to_clear = qty_on_hand / max(avg_daily_sellthrough, eps)
depth_risk = clip( days_to_clear / DEPTH_LIMIT , 0, 1)  # 소진에 오래 걸릴수록 위급

# (e) 시즌 이탈  season_penalty
#   판매 정점까지의 거리. 정점이 이미 지났거나 멀수록 위급, 임박하면 낮음.
months_to_peak = ((seasonal_peak_month - current_month) mod 12)
season_penalty = 1 - gaussian(months_to_peak, mu=1, sigma=SEASON_SIGMA)
#   months_to_peak≈1(정점 임박) → season_penalty≈0 (보유 신호)

# (f) 마진 소멸 임박  margin_cliff  (즉시 손절 트리거)
margin_cliff = 1 if comp_price_p10 <= unit_cost * (1+MIN_MARGIN) else 0
```

### 5-2. 가중합 → TUS

```
raw = W_d*demand_drop + W_p*price_erosion + W_a*aging
    + W_o*depth_risk  + W_s*season_penalty

TUS = round( 100 * clip(raw, 0, 1) )
if margin_cliff == 1:
    TUS = max(TUS, 85)          # 원가 이하 경쟁 등장 → 강제 응급 승격
```

### 5-3. 기본 가중치·상수 (튜닝 대상, config로 분리)

| 상수 | 기본값 | 의미 |
|---|---|---|
| `W_d` 수요하락 | 0.25 | |
| `W_p` 경쟁가침식 | 0.25 | |
| `W_a` 체류일 | 0.20 | |
| `W_o` 재고깊이 | 0.15 | |
| `W_s` 시즌이탈 | 0.15 | (가중치 합 = 1.0) |
| `AGING_LIMIT` | 90일 | 채화(악성재고) 기준 |
| `DEPTH_LIMIT` | 120일 | 소진 목표 상한 |
| `MARGIN_TARGET` | 0.30 | 목표 여유 마진율 |
| `MIN_MARGIN` | 0.05 | 손익분기 최소 마진 |
| `DROP_NORM` | 0.5 | 수요 하락 정규화 스케일 |
| `SEASON_SIGMA` | 1.5 | 시즌 근접 허용폭(개월) |

> 이 상수 테이블은 `triage_config.json`으로 분리합니다(`tokenomics_config.json` 선례와 동일 패턴).
> 코드 수정 없이 대표가 임계값을 조정할 수 있어야 합니다.

---

## 6) Triage 등급 분류 (색상 트리아지)

| 등급 | TUS | 의미 | 처분 원칙 |
|---|---|---|---|
| 🔴 **RED** | ≥ 70 | 응급 — 지금 마진을 줄여서라도 처분 | 목표 처분 T+0~3일, 빠른 현금화 채널 우선 |
| 🟠 **AMBER** | 40~69 | 관찰 — 분할 처분/가격 실험 | T+7~14일, 채널 A/B 테스트 |
| 🟢 **GREEN** | < 40 | 보유/대기 — 시점 최적화가 이득 | 시즌 윈도우까지 보유, 마진 극대화 |

팟캐스트 예시 매핑:
- *"A 상품 = 쿠팡 가격경쟁 심함 → 오늘 마진 줄여 처분"* → **RED** (price_erosion·demand_drop 높음)
- *"B 상품 = 다음 달 계절 수요 폭발 → 자체몰 +30%"* → **GREEN** (season_penalty≈0, months_to_peak≈1)

---

## 7) 채널 · 시점 · 마진 매칭 규칙

### 7-1. 채널 마스터(속성 테이블, `channel_config.json`)

| 채널 | 현금화속도 | 수수료 | 마진잠재력 | 물량수용 | 비고 |
|---|---|---|---|---|---|
| 쿠팡/오픈마켓 | ★★★★★ | 높음 | 낮음 | 대 | 빠른 처분·가격경쟁 심함 |
| 자체몰(오너스마켓) | ★★★ | 낮음 | 높음 | 중 | 마진 극대화·조합원 링크샵 |
| 스마트스토어 | ★★★★ | 중 | 중 | 중 | 검색 유입 |
| 오프라인 매장(공간 파트너) | ★★ | 중 | 중 | 중 | 체험형·리퍼브 |
| B2B 대량/떨이 | ★★★★★ | 낮음 | 최저 | 특대 | 최후 손절(RED 잔량) |

각 채널은 `{speed, fee, margin_potential, capacity, allowed_condition}` 벡터로 정의.

### 7-2. 채널 선택 로직

```
후보 = 채널 중 (min_channel 화이트리스트 ∩ condition_grade 허용)

if 등급 == RED:
    # 속도 최우선, 손익분기 지킬 수 있는 가장 빠른 채널
    채널 = argmax(speed) among 후보 where 달성마진 >= MIN_MARGIN
    잔량이 커서 단일채널 수용 불가 → B2B 대량으로 오버플로우 분배
elif 등급 == AMBER:
    # 속도 × 마진 균형. 상위 2채널에 분할 배분(A/B)
    채널 = top2 by (0.5*norm(speed) + 0.5*norm(margin_potential))
else:  # GREEN
    # 마진 최우선(자체몰/스마트스토어), 시점은 시즌 윈도우
    채널 = argmax(margin_potential) among 후보
```

### 7-3. 판매 시점(timing)

```
if RED:    sell_window = [T+0, T+3]
if AMBER:  sell_window = [T+7, T+14]
if GREEN:  # 시즌 정점 직전으로 예약
    target_week = date_of(seasonal_peak_month) - LEAD_WEEKS   # 기본 LEAD_WEEKS=3
    sell_window = [target_week-1주, target_week+1주]
```

### 7-4. 목표 마진(target margin)

```
base_margin = (comp_price_median - unit_cost) / unit_cost      # 시장 중앙값 기준 여력
tier_mult   = { RED: 0.4, AMBER: 0.8, GREEN: 1.2 }[등급]        # 급할수록 마진 양보
target_margin = clip(base_margin * tier_mult, MIN_MARGIN, MARGIN_CAP)
list_price    = round_price( unit_cost * (1 + target_margin) )
# 단, RED에서도 list_price >= unit_cost*(1+MIN_MARGIN) 보장(손절이지 헐값투매 금지)
```

---

## 8) 출력 — 액션 로그 & 리스크 로그 (의사결정 산출물)

엔진은 SKU별 레코드와 포트폴리오 요약 두 층을 출력합니다.

### 8-1. SKU 액션 레코드 (`triage_action`)
```json
{
  "sku_id": "LP-2231-050",
  "owner_member_id": "M-00417",
  "tus": 82,
  "tier": "RED",
  "signals": { "demand_drop":0.71,"price_erosion":0.66,"aging":0.4,
               "depth_risk":0.3,"season_penalty":0.55,"margin_cliff":0 },
  "recommend": {
    "channel": ["쿠팡", "B2B대량"],
    "sell_window": ["2026-07-08","2026-07-11"],
    "list_price": 12900,
    "target_margin": 0.11,
    "reason": "경쟁 최저가 급락(-18%)+검색 수요 4주 연속 하락 → 오늘 마진 축소 처분"
  },
  "evidence": { "comp_price_p10":11800,"comp_price_median":13500,
                "demand_slope":-0.14,"source":"naver_datalab+shop","asof":"2026-07-08" }
}
```

### 8-2. 포트폴리오 리스크 로그 (`risk_log`) — 조합원 앱 노출용
- 등급 분포(🔴/🟠/🟢 개수·평가금액)
- **"이번 주 방치 시 예상 평가손"** = Σ(RED 미처분 × 일일 가격침식률 × 잔여일)
- 시즌 캘린더(GREEN 대기 물량이 언제 풀리는지)
- **손실 가능성 투명 고지**: "이번 달 채널 경쟁 심화로 X SKU 미판매 재고 발생 가능" (숨기지 않음)

> 팟캐스트 원칙: *수익률을 부풀리는 대신 리스크를 투명 공유* → 리스크 로그가 1급 산출물.

---

## 9) 단일 원장(Single Ledger) 연동 — "보이는 정산"

엔진의 모든 판단·실행 이벤트는 조작 불가능한 append-only 원장에 기록됩니다.

```
event: { ts, sku_id, owner_member_id, type, payload, engine_version, input_hash }
type ∈ { INGEST, DATALAB_FETCH, TRIAGE_SCORE, RECOMMEND, LISTED,
         PRICE_CHANGE, SOLD, SETTLE }
```
- `input_hash` = 그날 입력(SKU + 외부데이터)의 해시 → **동일 입력이면 동일 판단** 재현 검증.
- 조합원 앱의 원장 뷰는 "파주 창고 렉3-2열 립스틱이 어제 5만원에 결제→오늘 출고" 로그를
  **실시간 동기화**. 감정 호소가 아니라 검증 가능한 데이터로 신뢰 확보.

---

## 10) 컴플라이언스 가드레일 (유사수신·수익보장 금지)

`MOA.md`(위·수탁/무보장 원칙)와 정합하도록 엔진에 하드 룰을 박습니다.

- **무보장 변동정산**: 엔진은 예상 마진을 "추천"할 뿐 **확정 수익률을 산출·표시하지 않음**.
- 출력 문구는 "보장/확정/월 O%" 등의 표현을 **금칙어 필터**로 차단(컴플라이언스 모드).
- 자금 성격 3분리(출자금/위탁금/비용)는 정산 레이어에서 유지 — 엔진은 **실물 판매 실현 시에만**
  정산 이벤트(SETTLE) 발생시킴("팔려야 돈이 발생").
- 모든 추천에 **손실 가능성·근거 데이터**를 동반 표기(면책이 아니라 투명성).

---

## 11) 이용고 배당 훅 (거버넌스 연결) — 엔진이 남겨야 할 지표

협동조합의 이용고 배당(출자액이 아니라 **생태계 기여도**에 비례 환원)을 계산하려면,
엔진이 원장에 **기여 지표**를 함께 적재해야 합니다.

```
contribution[member] += Σ (실현매출액 * w1)                # 위탁 판매 실현
                      + Σ (셀프판매 링크 전환매출 * w2)     # 개인 링크샵 판매
                      + (채널 리밸런싱 참여/의결 참여 * w3) # 생태계 활동
```
→ 연말 잉여이익 배분의 기준 데이터. (엔진은 계산 소스만 제공, 배당 확정은 조합 의결)

---

## 12) 모듈 / 파일 구조 (구현 단계 청사진)

`#master_plan.md`가 제안한 `models/` 패턴을 그대로 확장합니다.

```
triage/
  config/
    triage_config.json      # 5절 가중치·임계 상수
    channel_config.json     # 7절 채널 마스터
  ingest/
    csv_loader.py           # 3절 CSV 업로드 + 컬럼 매핑 + 검증 게이트
    naver_datalab.py        # 4절 MCP 4종 호출·캐시·폴백
  features/
    signals.py              # 5-1 신호 6종 계산
    scoring.py              # 5-2 TUS 가중합
  decide/
    triage.py               # 6절 등급 분류
    matcher.py              # 7절 채널·시점·마진 매칭
  output/
    action_log.py           # 8-1 SKU 레코드
    risk_log.py             # 8-2 포트폴리오 리스크
    ledger.py               # 9절 단일 원장 append-only
  compliance/
    guardrails.py           # 10절 금칙어·무보장 룰
  ui/
    triage_page.py          # (선택) Streamlit 탭 or 독립 데모 UI
```

- 순수 계산부(`features/`, `decide/`)는 **외부 의존 없이 단위테스트** 가능해야 함(결정론).
- `naver_datalab.py`만 MCP/네트워크에 의존 → 나머지는 목업 데이터로도 완전 동작.

---

## 13) 구현 로드맵 (스펙 → 코드)

| 단계 | 산출물 | 데이터 | 검증 |
|---|---|---|---|
| **P0 (본 문서)** | 설계 확정 | — | 변수·계산식·규칙 합의 |
| **P1** | 목업으로 엔진 로직 | 가상 SKU + 가상 트렌드 | TUS/등급/매칭 단위테스트 |
| **P2** | CSV 업로드 경로 | 실제 자사 재고 CSV | 컬럼매핑·검증게이트 통과 |
| **P3** | 네이버 데이터랩 실연동 | `find_category`→`datalab_*`→`search_shop` | 실트렌드로 RED/GREEN 검증 |
| **P4** | 단일 원장 + 리스크 로그 UI | 축적 시계열 | 재현성(input_hash)·투명 노출 |
| **P5** | 컴플라이언스·이용고배당 훅 | 원장 지표 | 금칙어 필터·기여도 산출 |

---

## 14) 미결 결정 사항 (확정 필요)

1. **가중치 초기값**(5-3)을 자사 과거 재고 데이터로 백테스트해 캘리브레이션할지?
2. **채널 마스터**(7-1)에 실제 운영 채널·수수료율을 무엇으로 채울지(쿠팡/자체몰/스마트스토어 외 추가?).
3. **원장 저장소**: 파일 기반(append-only JSONL) 시작 → 이후 DB/온체인 앵커링 여부.
4. **UI 형태**: 기존 `app.py` Streamlit 탭 통합 vs 독립 페이지(1차 답변은 "스펙 먼저"였으므로 P1에서 재결정).
5. **데이터랩 매핑 정밀도**: 상품명↔카테고리 자동 매핑 정확도가 낮은 SKU의 수동 오버라이드 방식.

---

### 부록 A) 팟캐스트 개념 → 전산 매핑 요약

| 팟캐스트 표현 | 전산 구현 |
|---|---|
| 응급실 분류 간호사 | TUS 점수 + RED/AMBER/GREEN 등급(5·6절) |
| 검색 트렌드·경쟁가 실시간 분석 | 네이버 데이터랩/쇼핑 MCP 실연동(4절) |
| 최적 채널·시기 매칭 | 채널·시점·마진 매처(7절) |
| 악성재고 → "발견되지 않은 가치" | 입력 게이트 통과 재고를 자산으로 재정의(3절) |
| 보이는 상품/창고/정산, 단일 원장 | append-only Ledger + 재현성 해시(9절) |
| 수익보장 아님·리스크 투명 공유 | 컴플라이언스 가드레일 + 리스크 로그(8·10절) |
| 이용고 배당(기여 비례) | 원장 기여 지표 훅(11절) |
