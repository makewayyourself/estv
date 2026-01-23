# [Specification] ESTV 시뮬레이터 UX/UI 개편: 튜토리얼 위자드(Wizard) 도입

## 1. 개요 (Objective)
기존의 복잡한 원페이지 설정 방식을 탈피하여, 초보 사용자(경영진/기획자)가 **전략적 사고의 흐름(Goal → Supply → Demand → Defend)**에 따라 자연스럽게 시뮬레이션을 설정할 수 있도록 **단계별 튜토리얼 모드(Wizard Mode)**를 도입한다.

## 2. 핵심 UI 구조 변경 (Architecture)

### 2.1. 세션 상태 관리 (State Management)
Streamlit의 `session_state`를 사용하여 두 가지 모드를 구분하고 단계를 추적한다.
```python
if 'mode' not in st.session_state:
    st.session_state['mode'] = 'tutorial'  # 'tutorial' or 'expert'
if 'tutorial_step' not in st.session_state:
    st.session_state['tutorial_step'] = 1


2.2. 사이드바 레이아웃
최상단: 모드 전환 토글 (🔰 튜토리얼 모드 <-> ⚙️ 전문가 모드)

튜토리얼 모드 시: 진행률(Progress Bar)과 현재 단계(Step X/5)에 해당하는 입력 위젯만 표시. [이전] / [다음] 버튼 제공.

전문가 모드 시: 기존처럼 모든 설정이 아코디언(Expander) 형태로 나열되되, 중요도 순으로 재정렬.

3. 튜토리얼 단계별 상세 기획 (Step-by-Step Logic)
각 단계는 헤더(제목) → 가이드(설명) → 입력(위젯) → 인사이트(실시간 피드백) 순서로 배치한다.

Step 1: 목표 설정 (Goal Setting)
"우리의 목적지는 어디인가?"

UI 요소:

헤더: 🎯 Step 1. 목표 설정

가이드: "시뮬레이션의 기준을 정합니다. 목표 가격이 높을수록 더 정교한 공급 통제와 수요 견인이 필요합니다."

입력: * target_price: 목표 가격 (예: $5.00)

contract_mode: 계약 시나리오 (기존 계약서 vs 변동 계약서 vs 사용자 조정)

로직 매핑:

contract_mode 선택 시 관련된 기본값(Supply, Unbonding 등)을 session_state에 프리로드.

Step 2: 공급 통제 (Supply Control)
"댐의 수문을 얼마나 열 것인가?"

UI 요소:

헤더: 📉 Step 2. 공급 제한 (Risk 관리)

가이드: "시장에 물건(토큰)이 너무 많이 풀리면 가격은 오르지 않습니다. 초기 유통량을 3% 이하로 억제하는 것이 핵심입니다."

입력:

input_supply: 초기 유통량 (Slider, 0~10%)

input_unbonding: 언본딩 기간 (Slider, 0~60일)

경고(Validation): input_supply > 3.0일 경우 st.error("🚨 법적 리스크 발생: 초기 유통량은 3%를 초과할 수 없습니다.") 표시.

Step 3: 수요 확보 (Demand Generation)
"얼마나 강력한 펌프를 달 것인가?"

UI 요소:

헤더: 📈 Step 3. 수요 견인 (Growth)

가이드: "물건이 귀해도 사는 사람이 없으면 의미가 없습니다. 1.6억 명의 유저 중 몇 %를 데려올 수 있습니까?"

입력:

conversion_rate: 거래소 유입 전환율 (Slider, 0.01% ~ 2.0%)

avg_ticket: 1인당 평균 매수액 (Number, 기본 $100)

실시간 계산: * 예상 월간 유입액 = 1.6억명 * 전환율 * 객단가 / 12개월

st.metric(label="월간 매수 파워", value=f"${calculated_inflow:,.0f}")

Step 4: 시장 구조 (Market Structure)
"그릇(시장)의 크기는 적절한가?"

UI 요소:

헤더: 🏗️ Step 4. 시장 깊이 (Volatility)

가이드: "오더북이 얇으면 적은 매도에도 가격이 폭락합니다. 가격 방어를 위한 시장의 기초 체력을 설정하세요."

입력:

market_depth_level: 오더북 체력 (Selectbox: "약함", "보통", "강함")

로직 매핑:

"약함" -> depth_usdt_1pct = 300,000

"보통" -> depth_usdt_1pct = 1,000,000

"강함" -> depth_usdt_1pct = 3,000,000

Step 5: 방어 및 실행 (Defense & Action)
"위기 시 사용할 치트키는 무엇인가?"

UI 요소:

헤더: 🛡️ Step 5. 방어 정책 및 실행

가이드: "가격이 급락할 때 회사가 개입할 예산과 정책을 준비합니다."

입력:

monthly_buyback_usdt: 월간 바이백 예산 ($)

burn_fee_rate: 소각 수수료율 (%)

실행 버튼: [🚀 시뮬레이션 결과 확인하기] (누르면 메인 화면 차트 업데이트)

4. 전문가 모드 UI 재구성 (Expert Mode Refactoring)
튜토리얼 모드가 꺼졌을 때는 기존 사이드바를 유지하되, **전략적 계층 구조(Hierarchy)**로 그룹핑하여 가독성을 높인다.

Group 1: 🎯 시나리오 & 목표 (Contract Mode, Target Price)

Group 2: ⚖️ 펀더멘탈 (Supply & Demand) (Supply, Unbonding, Inflow, Conversion)

Group 3: 🏗️ 시장 구조 (Price Model, Depth, LP Growth) - 복잡한 변수는 Expander로 숨김

Group 4: 🛡️ 정책 및 개입 (Buyback, Burn, Master Plan, Triggers)

Group 5: 📊 분석 도구 (Confidence, Upbit Compare)

5. 결과 화면 (Dashboard) 개선
5.1. XAI (설명 가능한 AI) 주석 강화
차트에 표시되는 주석을 더 직관적으로 만든다.

패닉 셀(Panic Sell): 빨간색 역삼각형(▼) 마커 + 툴팁 "심리적 지지선 붕괴"

마케팅 효과(Marketing): 초록색 삼각형(▲) 마커 + 툴팁 "캠페인 유입 효과"

고래 덤핑(Whale Dump): 주황색 원(●) 마커 + 툴팁 "초기 투자자 락업 해제"

5.2. 액션 제안 (Prescriptive Analytics)
결과 하단에 st.info 또는 st.warning을 사용하여 구체적인 행동을 지시한다.

IF final_price < target_price: "📉 목표가 미달입니다. 전환율을 0.1% 더 올리거나, 언본딩 기간을 10일 더 늘리세요."

IF status == "CRITICAL": "🚨 위험! 가격 방어가 불가능합니다. 바이백 예산을 $50,000 이상 편성해야 합니다."

6. 개발 가이드 (To Cursor)
Backend Logic 유지: 기존 TokenSimulationEngine 클래스와 calculate_dynamic_sell_pressure 등의 함수는 그대로 유지한다.

Frontend Logic 교체: st.sidebar 부분을 전면 재작성하여 if st.session_state['mode'] == 'tutorial': 분기 처리를 구현한다.

Mapping: 튜토리얼의 간소화된 입력값(예: '오더북 강함')을 엔진이 이해하는 수치(예: 'depth=3M')로 변환하는 매핑 로직을 run_simulation 직전에 추가한다.

### 💡 Cursor에게 내릴 프롬프트 (Copy & Paste)

이 파일을 바탕으로 Cursor에게 이렇게 지시하세요.

> **"이 `tutorial_ux_spec.md` 명세서를 바탕으로 현재의 `app.py`를 리팩토링해줘. 기존의 시뮬레이션 계산 로직은 건드리지 말고, Streamlit 사이드바 UI를 '튜토리얼 모드'와 '전문가 모드'로 분리해서 구현해줘. 특히 튜토리얼 모드에서는 사용자가 5단계를 순서대로 진행하며 각 변수의 의미를 이해할 수 있도록 `st.info`로 설명을 추가해줘."**
