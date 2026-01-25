# app_merged.py의 UI/UX 및 핵심 기능을 통합
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
from datetime import datetime

st.set_page_config(
    page_title="ESTV Strategic AI Advisor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stProgress > div > div > div > div {
        background-color: #4f46e5;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

class TokenSimulationEngine:
    def __init__(self):
        pass
    def run(self, inputs):
        initial_price = inputs.get('initial_price', 0.1)
        days = inputs.get('days', 365)
        buy_volume = inputs.get('monthly_buy_volume', 50000)
        liquidity_level = inputs.get('liquidity_level', 3)
        volatility = inputs.get('volatility', 1.0)
        prices = [initial_price]
        current_price = initial_price
        liquidity_constant = 1000000 * (liquidity_level ** 1.5)
        daily_prices = []
        for day in range(days):
            daily_buy_base = buy_volume / 30
            buy_noise = np.random.uniform(0.8, 1.2)
            daily_buy = daily_buy_base * buy_noise
            profit_ratio = max(0.05, (current_price - initial_price) / initial_price * 0.1)
            sell_pressure_factor = volatility
            daily_sell = daily_buy * np.random.uniform(0.8, 1.2) * (1 + profit_ratio * sell_pressure_factor)
            net_flow = daily_buy - daily_sell
            impact = net_flow / liquidity_constant
            market_noise = (np.random.random() - 0.5) * 0.02 * volatility
            current_price = current_price * (1 + impact + market_noise)
            if current_price < 0.001: current_price = 0.001
            daily_prices.append(current_price)
        return {
            'final_price': current_price,
            'daily_price_trend': daily_prices
        }

def generate_ai_strategy_report(success_rate, var_95, median_price, target_price, inputs):
    buy_vol = inputs['monthly_buy_volume']
    liquidity = inputs['liquidity_level']
    report = {}
    if success_rate >= 80:
        report['sentiment'] = "🚀 매우 긍정적 (Strong Bullish)"
        report['color'] = "green"
        report['action'] = "현재 모멘텀 유지 및 생태계 확장 주력"
        report['detail'] = "현재의 유동성과 매수 유입은 목표 달성에 이상적인 비율입니다. 가격 상승에 따른 자연스러운 매도 물량을 충분히 소화하고 있습니다."
    elif success_rate >= 50:
        report['sentiment'] = "⚖️ 중립/신중 (Cautious Optimism)"
        report['color'] = "orange"
        report['action'] = "마케팅 강도 상향 또는 유동성 보강 필요"
        report['detail'] = f"목표 달성 확률이 반반입니다. 성공 확률을 80%대로 높이려면 월간 매수 유입을 약 {int(buy_vol * 1.3):,} Unit까지 늘리거나, 유동성 레벨을 한 단계 높여 하락 변동성을 줄여야 합니다."
    else:
        report['sentiment'] = "⚠️ 위험 (Bearish Risk)"
        report['color'] = "red"
        report['action'] = "공격적 확장 중단 및 방어선(LP) 구축 최우선"
        report['detail'] = f"현재 구조로는 목표가(${target_price}) 도달이 어렵습니다. 특히 하방 리스크(VaR)가 ${var_95:.3f}로 매우 취약합니다. 마케팅보다는 유동성 풀(LP) 인센티브를 강화하여 가격 방어력을 높이는 것이 시급합니다."
    return report

def main():
    with st.sidebar:
        st.header("⚙️ 전문가 시나리오 설정")
        target_price = st.number_input(
            "목표 가격 ($)", value=0.5, step=0.05,
            help="1년 후 달성하고자 하는 토큰 목표 가격입니다."
        )
        initial_price = st.number_input(
            "초기 가격 ($)", value=0.1, step=0.01, format="%.3f",
            help="시뮬레이션 시작 시점의 토큰 가격입니다."
        )
        days = st.slider(
            "시뮬레이션 기간 (일)", 30, 730, 365, step=30,
            help="시뮬레이션을 진행할 전체 기간(일 단위)입니다."
        )
        monthly_buy_volume = st.slider(
            "월간 매수 유입 (Unit)", 10000, 500000, 50000, step=5000,
            help="월별로 시장에 유입되는 신규 매수량(토큰 단위)입니다."
        )
        new_user_rate = st.slider(
            "신규 유입률 (%/월)", 0, 100, 10, step=1,
            help="월별 신규 투자자(지갑) 증가율입니다."
        )
        marketing_budget = st.slider(
            "마케팅 예산 ($/월)", 0, 100000, 10000, step=1000,
            help="월별 마케팅/프로모션에 투입되는 예산입니다."
        )
        liquidity_level = st.slider(
            "유동성 깊이 (Liquidity)", 1, 10, 3,
            help="DEX/거래소에 공급된 유동성 풀의 상대적 깊이(1=얕음, 10=매우 깊음)입니다."
        )
        liquidity_type = st.selectbox(
            "유동성 풀 구조", ["고정형", "가변형"],
            help="고정형: 유동성 풀 크기 고정, 가변형: 시뮬레이션 중 유동성 변화 허용"
        )
        lockup_ratio = st.slider(
            "락업 비율 (%)", 0, 100, 20, step=5,
            help="전체 토큰 중 락업(출금불가) 상태의 비율입니다."
        )
        volatility = st.slider(
            "시장 변동성 (Panic/FOMO)", 0.5, 3.0, 1.0, step=0.1,
            help="시장 가격의 일간 변동성(1=보통, 3=매우 높음)입니다."
        )
        buy_tax = st.slider(
            "매수 거래세 (%)", 0, 10, 1, step=1,
            help="매수 시 부과되는 거래세율입니다."
        )
        sell_tax = st.slider(
            "매도 거래세 (%)", 0, 10, 1, step=1,
            help="매도 시 부과되는 거래세율입니다."
        )
        holder_ratio = st.slider(
            "홀더 비율 (%)", 0, 100, 60, step=5,
            help="전체 투자자 중 장기 보유자(홀더)의 비율입니다."
        )
        trader_ratio = st.slider(
            "트레이더 비율 (%)", 0, 100, 30, step=5,
            help="전체 투자자 중 단기 매매자(트레이더)의 비율입니다."
        )
        bot_ratio = st.slider(
            "봇/스나이퍼 비율 (%)", 0, 100, 10, step=5,
            help="전체 투자자 중 자동매매/스나이퍼봇의 비율입니다."
        )
        st.markdown("---")
        st.subheader("고급 이벤트/정책")
        big_sell_event = st.checkbox(
            "대규모 매도 이벤트 발생", value=False,
            help="특정 시점에 대규모 매도(투매) 이벤트가 발생할 수 있습니다."
        )
        big_sell_prob = st.slider(
            "대규모 매도 확률 (%)", 0, 100, 5, step=1,
            help="시뮬레이션 기간 중 대규모 매도 이벤트가 발생할 확률입니다."
        )
        pump_event = st.checkbox(
            "펌프(급등) 이벤트 발생", value=False,
            help="특정 시점에 가격이 급등(펌프)하는 이벤트가 발생할 수 있습니다."
        )
        pump_prob = st.slider(
            "펌프 확률 (%)", 0, 100, 3, step=1,
            help="시뮬레이션 기간 중 펌프 이벤트가 발생할 확률입니다."
        )
        fund_inflow = st.slider(
            "외부 펀드 유입 ($/월)", 0, 100000, 0, step=1000,
            help="외부 투자자/기관 등에서 유입되는 추가 자금 규모입니다."
        )
        inflation_policy = st.selectbox(
            "인플레이션 정책", ["없음", "연 2%", "연 5%", "연 10%"],
            help="토큰 공급량 증가(인플레이션) 정책을 선택합니다."
        )
        ai_strategy = st.selectbox(
            "AI 전략 모드", ["공격적", "중립적", "방어적"],
            help="AI가 시뮬레이션에서 적용할 전략적 성향입니다."
        )
        scenario_preset = st.selectbox(
            "시나리오 프리셋", ["사용자 정의", "보수적", "공격적", "혼합형"],
            help="자주 쓰는 시나리오 조합을 빠르게 불러올 수 있습니다."
        )
        iterations = st.slider(
            "시뮬레이션 횟수 (Monte Carlo)", 10, 500, 50,
            help="Monte Carlo 반복 횟수(시나리오 샘플 개수)입니다."
        )
        run_btn = st.button("🚀 AI 시뮬레이션 실행", type="primary", use_container_width=True)
    st.title("ESTV Strategic AI Advisor")
    st.caption("Chaos Labs Benchmark Engine v2.5 | 전문가용 시나리오 시뮬레이터")
    if run_btn:
        with st.spinner("AI가 수백 가지 시나리오를 시뮬레이션 중입니다..."):
            engine = TokenSimulationEngine()
            inputs = {
                'initial_price': initial_price,
                'days': days,
                'monthly_buy_volume': monthly_buy_volume,
                'liquidity_level': liquidity_level,
                'liquidity_type': liquidity_type,
                'lockup_ratio': lockup_ratio,
                'volatility': volatility,
                'buy_tax': buy_tax,
                'sell_tax': sell_tax,
                'holder_ratio': holder_ratio,
                'trader_ratio': trader_ratio,
                'bot_ratio': bot_ratio,
                'new_user_rate': new_user_rate,
                'marketing_budget': marketing_budget,
                'big_sell_event': big_sell_event,
                'big_sell_prob': big_sell_prob,
                'pump_event': pump_event,
                'pump_prob': pump_prob,
                'fund_inflow': fund_inflow,
                'inflation_policy': inflation_policy,
                'ai_strategy': ai_strategy,
                'scenario_preset': scenario_preset,
                'target_price': target_price
            }
            all_final_prices = []
            all_trends = []
            success_count = 0
            for _ in range(iterations):
                res = engine.run(inputs)
                # 결과값 유효성 검사
                final_price = res['final_price']
                daily_price_trend = res['daily_price_trend']
                if not np.isfinite(final_price):
                    final_price = 0.1
                daily_price_trend = [p if np.isfinite(p) else 0.1 for p in daily_price_trend]
                all_final_prices.append(final_price)
                all_trends.append(daily_price_trend)
                if final_price >= target_price:
                    success_count += 1
            # 전체 결과 유효성 검사
            if not all_final_prices or not all(np.isfinite(all_final_prices)):
                all_final_prices = [0.1] * iterations
            if not all_trends or not all([all(np.isfinite(trend)) for trend in all_trends]):
                all_trends = [[0.1] * 365 for _ in range(iterations)]
            success_rate = (success_count / iterations) * 100
            median_trend = np.median(all_trends, axis=0)
            p10_trend = np.percentile(all_trends, 10, axis=0)
            p90_trend = np.percentile(all_trends, 90, axis=0)
            median_final_price = np.median(all_final_prices)
            var_95 = np.percentile(all_final_prices, 5)
            # 그래프 데이터 유효성 검사
            if not np.isfinite(median_final_price):
                median_final_price = 0.1
            if not np.isfinite(var_95):
                var_95 = 0.1
            if not all(np.isfinite(median_trend)):
                median_trend = np.full(365, 0.1)
            if not all(np.isfinite(p10_trend)):
                p10_trend = np.full(365, 0.1)
            if not all(np.isfinite(p90_trend)):
                p90_trend = np.full(365, 0.1)
            strategy = generate_ai_strategy_report(success_rate, var_95, median_final_price, target_price, inputs)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("성공 확률 (Win Rate)", f"{success_rate:.1f}%", delta=f"Target ${target_price}")
            with col2:
                st.metric("예상 최종가 (Median)", f"${median_final_price:.3f}")
            with col3:
                st.metric("리스크 (VaR 95%)", f"${var_95:.3f}", delta="-Worst Case", delta_color="inverse")
            st.markdown("---")
            st.subheader("🤖 AI 전략 분석 리포트")
            if strategy['color'] == 'green':
                st.success(f"**{strategy['sentiment']}**\n\n📌 **Action:** {strategy['action']}\n\n{strategy['detail']}")
            elif strategy['color'] == 'orange':
                st.warning(f"**{strategy['sentiment']}**\n\n📌 **Action:** {strategy['action']}\n\n{strategy['detail']}")
            else:
                st.error(f"**{strategy['sentiment']}**\n\n📌 **Action:** {strategy['action']}\n\n{strategy['detail']}")
            st.markdown("---")
            col_chart1, col_chart2 = st.columns([2, 1])
            with col_chart1:
                st.subheader("📈 시나리오별 가격 경로 (365일)")
                fig_traj = go.Figure()
                days_axis = list(range(365))
                fig_traj.add_trace(go.Scatter(
                    x=days_axis + days_axis[::-1],
                    y=list(p90_trend) + list(p10_trend)[::-1],
                    fill='toself',
                    fillcolor='rgba(200, 200, 200, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='예측 범위 (80% Confidence)',
                    showlegend=True
                ))
                fig_traj.add_trace(go.Scatter(
                    x=days_axis,
                    y=median_trend,
                    line=dict(color='#4f46e5', width=3),
                    name='중위값 (Median Path)'
                ))
                fig_traj.add_hline(y=target_price, line_dash="dash", line_color="green", annotation_text="Target")
                fig_traj.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Days",
                    yaxis_title="Price ($)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_traj, width='stretch')
            with col_chart2:
                st.subheader("📊 최종 가격 분포")
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=all_final_prices,
                    nbinsx=15,
                    marker_color='#6366f1',
                    opacity=0.75
                ))
                fig_dist.add_vline(x=target_price, line_dash="dash", line_color="green")
                fig_dist.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Final Price ($)",
                    yaxis_title="Frequency",
                    showlegend=False
                )
                st.plotly_chart(fig_dist, width='stretch')
            st.markdown("### 💾 분석 기록 저장")
            col_save1, col_save2 = st.columns(2)
            with col_save1:
                snapshot = {
                    "timestamp": datetime.now().isoformat(),
                    "inputs": inputs,
                    "results": {
                        "success_rate": success_rate,
                        "median_price": median_final_price,
                        "var_95": var_95
                    }
                }
                json_snapshot = json.dumps(snapshot, indent=2, default=str)
                st.download_button(
                    label="📥 현재 분석결과 다운로드 (JSON)",
                    data=json_snapshot,
                    file_name="estv_strategy_report.json",
                    mime="application/json"
                )
    else:
        st.info("👈 좌측 사이드바에서 시나리오 변수를 설정하고 '시뮬레이션 실행' 버튼을 눌러주세요.")
        st.markdown("""
        ### 사용 가이드
        1. **목표 가격 설정**: 달성하고자 하는 토큰의 가격입니다.
        2. **매수 유입 & 유동성**: 마케팅 예산과 LP 풀의 크기를 조절합니다.
        3. **시장 변동성**: 시장 상황(불장/하락장)에 따른 민감도를 테스트합니다.
        4. **AI 전략 확인**: 시뮬레이션 후 AI가 제시하는 구체적인 액션 플랜을 확인하세요.
        """)

if __name__ == "__main__":
    main()
