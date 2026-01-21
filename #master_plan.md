## What

지금 올려주신 `app.py`는 “공급(언락→매도) + 수요(월 매수 유입) + AMM 가격 반응”까지는 이미 구현되어 있습니다. 
하지만 대표님이 만든 **Master Plan(1.6억 전환 → 실매수 유입, 가격 하락 전/후 자동 마케팅 트리거)**을 “코드로 편입”하려면, 현재 코드 구조에서 몇 가지를 **모듈화/분리**하고, “마케팅이 가격에 영향을 주는 경로”를 **시간축/이벤트/트리거 엔진**으로 추가해야 합니다. 

---

## Why

지금 엔진은 입력값(`monthly_buy_volume`, `sell_pressure_ratio`, `burn_fee_rate` 등)을 한 번 정하면 “그대로 끝까지” 달리기 때문에,
Master Plan의 핵심인 “가격이 내려가기 전에는 A를 준비하고, 내려가면 즉시 B를 가동한다”가 **구조적으로 구현될 수 없습니다**. 

즉, **마케팅을 ‘고정 파라미터’가 아니라 ‘상황 반응형 정책(Policy)’으로 만들어야** 시뮬레이션이 현실적으로 작동합니다.

---

## Summary

코드화하려면 보완 포인트는 크게 6개입니다.

1. 토크노믹스/락업 스케줄을 “데이터(표) 기반 설정”으로 분리
2. 업비트 환경을 반영한 “CEX 오더북/깊이 기반 가격 충격 모델” 추가(AMM만으로는 한계)
3. Master Plan을 “캠페인 객체”로 만들고, 시간축(Phase 1/2/3)에 따라 매수·매도·소각·보유지표를 바꾸게 하기
4. 가격/드로다운/거래량/언락 이벤트를 감지하는 “트리거 엔진” 추가
5. 트리거에 따라 자동 추천·자동 적용되는 “마케팅 플레이북(권고/가동)” 레이어 추가
6. 시뮬레이션 출력에 “무엇을 언제 실행해야 하는지”를 결과로 함께 내보내기(리스크 로그를 액션 로그로 확장)

---

## Body

### 1) 토크노믹스/락업 설정이 코드에 하드코딩되어 있어 “현실 반영”이 막힙니다

현재 `TokenSimulationEngine.base_allocations`가 코드에 직접 박혀 있고, 최신 토크노믹스 표/합의와 수치가 달라질 가능성이 큽니다. 
예를 들어 지금 코드에는 `Team_Advisors: 20%`, `Treasury: cliff 24, vesting 1`처럼 “최종 합의”와 다르게 해석될 여지가 있는 값들이 있습니다. 

**보완 방법(필수)**

* `tokenomics_config.json`(또는 YAML)로 분리
* 각 트랜치에 대해 `percent, cliff_months, vesting_months, release_interval, lock_type(vesting/timelock/none)`를 데이터로 정의
* 엔진은 “설정 파일을 읽어 계산”만 담당
  이렇게 해야 대표님이 토크노믹스를 바꾸거나(36~48개월 선택형) 시나리오를 늘려도 코드 수정 없이 대응됩니다.

---

### 2) Upbit(=CEX) 현실을 반영하려면 “AMM 상수곱 모델만”으로는 부족합니다

현재 엔진은 `pool_token`, `pool_usdt`, `k_constant`로 AMM 가격을 움직입니다. 
하지만 업비트는 오더북 기반(CEX)이고, 핵심은 “호가 깊이(±x%)에 얼마가 쌓였는지”입니다. AMM은 “대체 모델”로는 유용하지만, 업비트 상장 시뮬레이션의 진단 정확도를 올리려면 CEX 모델이 필요합니다.

**보완 방법(필수)**
가격 모델을 인터페이스로 분리하세요.

* `PriceModelAMM` (지금 구현)
* `PriceModelOrderBookDepth` (추가)

  * 입력: `bid_depth_usd_1pct, bid_depth_usd_2pct, ask_depth_usd_1pct, ...` 또는 “깊이 곡선” 파라미터
  * 로직: `sell_usd`가 depth를 얼마나 소모하는지로 슬리피지/가격하락률 추정
* `PriceModelHybrid` (선택): 초기에는 얕은 오더북 → MM/유동성 증가로 깊이 파라미터가 월별로 커지도록

이게 들어가야 “업비트 관점에서 필요한 매수세(Real Buying Pressure)”를 현실적으로 역산할 수 있습니다.

---

### 3) Master Plan을 코드로 넣으려면 “캠페인 객체화”가 필요합니다

Master Plan에는 Buy & Verify, Scarcity FOMO, Holding Challenge, Pay & Burn, VIP Tier 등 “행동을 바꾸는 장치”가 들어있습니다. 
그런데 현재 코드는 유입을 `conversion_rate, avg_ticket`로만 단순화하고 있고, 캠페인이 “언제 시작되어 언제 끝나는지”, “매수/매도/소각에 각각 어떤 영향이 있는지”가 구조화되어 있지 않습니다. 

**보완 방법(필수)**
`Campaign`이라는 1급 객체를 추가하세요.

* 속성 예시

  * `name`, `phase`, `start_day`, `end_day`
  * `buy_multiplier` (실매수 유발 계수)
  * `sell_suppression_delta` (매도율 억제)
  * `burn_rate_delta` (소각/흡수 강화)
  * `conversion_delta` (전환율 상승)
  * `ticket_delta` (평균 매수액 상승)
  * `budget_usd`, `roi_model`(선택)

* 시뮬레이터에서는 매일(또는 스텝마다) 활성화된 캠페인들의 효과를 합산해
  `effective_daily_buy`, `effective_sell_pressure`, `effective_burn_fee`를 계산하게 만듭니다.

이렇게 해야 “공급은 막고 수요는 펌핑”이라는 전략이 숫자로 연결됩니다. 

---

### 4) “가격이 내려가기 전/내려가면 즉시”를 만들려면 트리거 엔진이 필요합니다

현재는 -20%, -50% 같은 경고 로그만 남깁니다. 
하지만 대표님이 원하는 건 **경고가 아니라 실행**입니다.

**보완 방법(필수)**
`TriggerEngine`를 추가하고, 트리거가 발동하면 `Campaign`을 자동 활성화하거나 “권고”를 출력하세요.

* 트리거 입력 지표(추천)

  * `drawdown_from_high` (고점 대비 하락률)
  * `drawdown_from_listing` (상장가 대비)
  * `volume_spike` (거래량 급증)
  * `sell_pressure_spike` (언락/언본딩 큐 증가)
  * `liquidity_depth_drop` (오더북 깊이 감소)
  * `cliff_event_window` (클리프/언락 이벤트 접근)

* 트리거 예시(대표님 전략과 정합)

  * 고점 대비 -20%: Buy & Verify 가동 권고/자동 가동
  * -30%: Holding Challenge 시즌2, 보상 상향
  * -40%: Pay & Burn 할인율 확대 + VIP 기준 완화(락인 강화)
  * 클리프 D-30: “대형 유틸리티 이벤트/파트너십 발표 캠페인” 가동

이 구조를 넣어야 “즉각 반응하는 회사”를 시뮬레이션이 대신합니다. 

---

### 5) “투자자 설득용”과 “내부 리스크 관리용”을 동시에 만족하려면 출력이 바뀌어야 합니다

지금은 최종 가격/ROI 중심입니다. 
투자자 설득과 내부 운영에는 다음 출력이 추가로 있어야 “의사결정 도구”가 됩니다.

**출력 보완(필수)**

* “이번 달 필요한 최소 순매수(USD)” = 예상 매도액 + 안전버퍼
* “가격 방어 실패 시 자동 추천 액션” (캠페인/예산/기간)
* “캠페인별 효과 기여도” (가격 상승의 원인이 무엇인지 분해)
* “리스크 캘린더” (언락/언본딩/클리프/마케팅 계약 위험 구간)
* “컴플라이언스 모드” 토글: 자극적 문구 대신 ‘유틸리티/혜택 전환’ 중심으로만 액션 노출

---

### 6) 지금 코드에서 즉시 손봐야 할 구체 포인트(실전 체크리스트)

`app.py` 기준으로 바로 적용할 수 있는 수정 포인트만 짚겠습니다. 

* `base_allocations`를 파일 기반으로 로딩하도록 변경
* `Liquidity_MM`가 vesting 0인데 “초기 유통량 3%”와의 관계가 애매합니다. “초기 유통 = 어떤 지갑/어떤 트랜치가 포함되는지”를 명시적으로 분리해야 합니다. (유통 정의 레이어 필요)
* `Treasury`가 `cliff 24, vesting 1`로 사실상 24개월 후 일괄 100%에 가까운 언락처럼 보일 수 있습니다. 이건 “재단 준비금 락 없음/24개월 락업” 같은 조건을 정확히 반영할 수 있도록 트랜치 정의를 재정리해야 합니다.
* `daily_user_buy_schedule`는 “유입=매수”로 단순 가정입니다. Master Plan의 Buy & Verify처럼 “매수 인증→혜택”은 실매수 계수를 높이는 대신, KYC/송금/인증 지연(라그)이 생깁니다. 지연/이탈을 모델링해야 현실성이 올라갑니다. 
* `use_marketing_contract_scenario`는 ‘덤핑 1회’만 반영합니다. 실제론 여러 번/분할/조건부로 나올 수 있어 “덤핑 정책(Policy)”로 일반화해야 합니다.

---

원하시면 다음 단계로, 제가 대표님 코드 구조를 기준으로 “추가해야 할 클래스/파일 구조”를 딱 정해드릴게요. 예를 들면:

* `models/tokenomics.py` (트랜치/스케줄 계산)
* `models/price_models.py` (AMM/OrderBook/Hybrid)
* `models/campaigns.py` (Buy&Verify, Holding, Burn, VIP…)
* `models/triggers.py` (drawdown/volume/cliff 감지)
* `models/recommender.py` (권고 엔진)
* `ui/dashboard.py` (액션 로그/캘린더)

그리고 “Master Plan의 각 캠페인이 어떤 파라미터를 얼마나 움직이는지”를 **기본값 테이블**로 만들어서 바로 코드에 넣을 수 있게 드리겠습니다.
