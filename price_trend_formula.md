# ESTV 가격 변동 추이 계산식 (요약)

이 문서는 `app.py` 시뮬레이션 엔진에서 **가격 변동 추이**가 어떻게 계산되는지, 사용되는 **모든 주요 변수**와 **계산 흐름**을 정리한 설명입니다.

---

## 1) 기본 상수 및 상태 변수

- `TOTAL_SUPPLY`: 총 공급량 (기본 1,000,000,000)
- `LISTING_PRICE`: 상장가 (기본 0.50 USD)
- `pool_token`: 유동성 풀 토큰 잔고
- `pool_usdt`: 유동성 풀 USDT 잔고
- `price = pool_usdt / pool_token`
- `k_constant = pool_token * pool_usdt` (AMM 모델에서의 k)

---

## 2) 입력 변수 (입력 UI & 시뮬레이션 인자)

### 공급/락업 관련
- `initial_circulating_percent` : 초기 유통량 비율 (%)
- `unbonding_days` : 언본딩 기간(일)
- `sell_pressure_ratio` : 락업 해제 시 매도율
- `initial_investor_allocation` : 초기 투자자 락업 물량 비율 및 일정
  - `percent`, `cliff`, `vesting`, `interval`
- `initial_investor_sell_ratio` : 초기 투자자 해제 매도율
- `initial_investor_sell_usdt_schedule` : 초기 투자자 월간 매도액(일 단위 분해)

### 수요/유입 관련
- `monthly_buy_volume` : 총 월간 매수 유입(기본 + 유저 유입 합산)
- `base_monthly_buy_volume` : 기본 월간 유입
- `base_daily_buy_schedule` : 월간 유입을 일 단위로 나눈 스케줄
- `daily_user_buy_schedule` : 기존 회원 유입(일 단위)
- `turnover_ratio` : 신규 유입 회전율(총합)
- `turnover_buy_share` : 회전율 매수 비중

### 변동성 완화 관련
- `steps_per_month` : 월간 유입 분할 단위(30/7일)
- `lp_growth_rate` : LP 성장률 (월 단위)
- `max_buy_usdt_ratio` : 매수 캡 (풀 대비 USDT 한도)
- `max_sell_token_ratio` : 매도 캡 (풀 대비 토큰 한도)

### 가격 모델 관련
- `price_model` : `"AMM" | "CEX" | "HYBRID"`
- `depth_usdt_1pct`, `depth_usdt_2pct` : CEX 오더북 깊이
- `depth_growth_rate` : HYBRID 모델 깊이 증가율

### 마케팅/캠페인 관련
- `use_marketing_contract_scenario` : 마케팅 덤핑 시나리오 활성화
- `campaigns` : 캠페인 리스트 (구간별 매수/매도/소각/바이백 변화)
- `triggers` : 조건 발동형 캠페인
- `enable_triggers` : 트리거 자동 가동 여부

### 소각/바이백
- `burn_fee_rate` : 매수 시 소각 비율
- `monthly_buyback_usdt` : 월간 바이백 금액

---

## 3) 락업 해제(베스팅) 물량 계산

기본 토크노믹스 + 초기 투자자 락업 분을 합산하여 월별 언락을 계산합니다.

```
monthly_unlock_allocation = TOTAL_SUPPLY * allocation.percent

if current_month < allocation.cliff:
    monthly_unlock = 0
elif allocation.vesting == 0:
    monthly_unlock = monthly_unlock_allocation
elif current_month >= allocation.cliff + allocation.vesting:
    monthly_unlock = 0
else:
    if allocation.interval > 1:
        # 분기/반기 등 주기 해제
        releases = max(1, allocation.vesting // allocation.interval)
        monthly_unlock = monthly_unlock_allocation / releases
    else:
        monthly_unlock = monthly_unlock_allocation / allocation.vesting
```

일 단위 분해:

```
daily_unlock = monthly_unlock / steps_per_month
```

언본딩 반영:

```
sell_queue[target_day] += daily_unlock * sell_pressure_ratio
sell_queue[target_day] += daily_initial_unlock * initial_investor_sell_ratio
```

---

## 4) 일 단위 매수/매도 흐름

### 기본 매수

```
base_daily_buy = remaining_buy / steps_per_month
# base_daily_buy_schedule가 있으면 해당 값으로 대체
daily_user_buy = daily_user_buy_schedule[day_index]

# 매수 증폭은 "유저 유입"에만 적용
step_buy = base_daily_buy + (daily_user_buy * buy_multiplier)
```

### 회전율(신규 유입의 추가 매도/매수)

```
turnover_sell = monthly_buy_volume * turnover_ratio * (1 - turnover_buy_share)
turnover_buy  = monthly_buy_volume * turnover_ratio * turnover_buy_share

step_turnover_sell = turnover_sell / steps_per_month
step_turnover_buy  = turnover_buy / steps_per_month
```

### 초기 투자자 월간 추가 매도액(USDT)

```
extra_sell_usdt = initial_investor_sell_usdt_schedule[day_index]
extra_sell_token = extra_sell_usdt / current_price
```

실제 매도 반영(캠페인 억제 적용):

```
effective_sell_pressure = max(0, sell_pressure_ratio - sell_suppression_delta)
sell_ratio_scale = 1.0
if sell_pressure_ratio > 0:
    sell_ratio_scale = effective_sell_pressure / sell_pressure_ratio
step_sell = remaining_sell * sell_ratio_scale
```

마케팅 덤핑(옵션):

```
if use_marketing_contract_scenario and current_price >= 0.10:
    dump_today = marketing_remaining * 0.005
    marketing_remaining -= dump_today
    step_sell += dump_today
```

최종 매도량:

```
total_sell = step_sell + step_turnover_sell
```

---

## 5) 캠페인/트리거 효과 적용

각 캠페인 활성 구간에서 아래 변수가 누적 조정됩니다.

```
buy_multiplier
sell_suppression_delta
burn_rate_delta
buyback_usdt_delta
max_sell_token_ratio_delta
```

최종 적용:

```
effective_buy_usdt = buy_usdt * buy_multiplier
effective_sell_pressure = max(0, sell_pressure_ratio - sell_suppression_delta)
effective_burn_rate = max(0, burn_fee_rate + burn_rate_delta)
effective_buyback = monthly_buyback_usdt + buyback_usdt_delta
effective_max_sell_ratio = max(0, max_sell_token_ratio + max_sell_token_ratio_delta)
```

---

## 6) 매수/매도 캡 (슬리피지 상한)

```
effective_max_sell_ratio = max(0, max_sell_token_ratio - max_sell_token_ratio_delta)
max_sell_token = pool_token * effective_max_sell_ratio
max_buy_usdt = pool_usdt * max_buy_usdt_ratio

total_sell = min(total_sell, max_sell_token)
step_buy = min(step_buy, max_buy_usdt)
```

---

## 7) 가격 모델별 가격 계산

### AMM 모델 (x*y=k)

```
total_buy = step_buy + step_turnover_buy

pool_token += total_sell
usdt_out = pool_usdt - (k_constant / pool_token)
pool_usdt -= usdt_out
pool_usdt += total_buy
token_out = pool_token - (k_constant / pool_usdt)
pool_token -= token_out
price = pool_usdt / pool_token
```

### CEX 모델 (오더북 깊이 기반)

```
impact_for_usdt(volume_usdt):
  if volume_usdt <= depth_usdt_1pct:
      impact = 0.01 * (volume_usdt / depth_usdt_1pct)
  else:
      extra = volume_usdt - depth_usdt_1pct
      impact = 0.01 + 0.01 * min(extra / (depth_usdt_2pct - depth_usdt_1pct), 1.0)

price_after = price * (1 + buy_impact - sell_impact)
buy_token_out = total_buy / price_after
sell_usdt_out = total_sell * price_after

pool_usdt = pool_usdt + total_buy - sell_usdt_out
pool_token = pool_token + total_sell - buy_token_out
pool_usdt = pool_token * price_after
```

### HYBRID 모델

```
if day_index % steps_per_month == 0:
    depth_usdt_1pct *= (1 + depth_growth_rate)
    depth_usdt_2pct *= (1 + depth_growth_rate)
```

---

## 8) LP 성장 반영

가격 상승 시 LP 유입을 반영합니다.

```
if new_price > prev_step_price:
    add_usdt = pool_usdt * (lp_growth_rate / steps_per_month)
    add_token = add_usdt / new_price
    pool_usdt += add_usdt
    pool_token += add_token
    k_constant = pool_token * pool_usdt
```

---

## 9) 소각/바이백 반영

```
if price_model in ["CEX", "HYBRID"]:
    token_out = total_buy / current_price
trade_volume_tokens = total_sell + token_out
burn_tokens = trade_volume_tokens * effective_burn_rate
pool_token -= burn_tokens
k_constant = pool_token * pool_usdt

total_buyback = monthly_buyback_usdt + (buyback_usdt_delta * steps_per_month)
step_buyback = total_buyback / steps_per_month
if price_model in ["CEX", "HYBRID"]:
    pool_usdt += step_buyback
    pool_token -= (step_buyback / current_price)
else:
    pool_usdt += step_buyback
    token_out_buyback = pool_token - (k_constant / pool_usdt)
    pool_token -= token_out_buyback
```

---

## 10) 최종 가격 추이 저장

```
daily_price_history.append(price)
```

그래프는 `daily_price_history`를 기반으로 시각화됩니다.

---

## 11) 가격 변동에 영향을 주는 모든 변수 요약

- 공급: `initial_circulating_percent`, 락업/베스팅(`cliff`, `vesting`, `interval`), `unbonding_days`
- 매도 압력: `sell_pressure_ratio`, `initial_investor_sell_ratio`, `initial_investor_sell_usdt_schedule`
- 매수 유입: `base_monthly_buy_volume`, `daily_user_buy_schedule`, `monthly_buy_volume`
- 변동성 완화: `steps_per_month`, `turnover_ratio`, `turnover_buy_share`, `lp_growth_rate`, `max_buy_usdt_ratio`, `max_sell_token_ratio`
- 가격 모델: `price_model`, `depth_usdt_1pct`, `depth_usdt_2pct`, `depth_growth_rate`
- 마케팅/캠페인: `use_marketing_contract_scenario`, `campaigns`, `triggers`, `enable_triggers`
- 소각/바이백: `burn_fee_rate`, `monthly_buyback_usdt`

---

필요하면 이 문서를 `app.py` 코드와 1:1 대응되는 **수식 기반 상세버전**으로도 확장해 줄게요.*** End Patch"}와 continuation tool call? Mist. Need valid JSON. I used apply_patch with JSON but included raw patch. Should call apply_patch with FREEFORM. Let's redo. This tool not in parallel. Return patch. Need proper. Let's call apply_patch properly. (No JSON). Let's do. Make sure file path. We'll reissue. We'll ignore previous; tool didn't run? It did not because invalid format? Actually I used JSON wrapper incorrectly. Must call apply_patch. Let's do now.```
