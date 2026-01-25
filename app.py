# 전략적 개입 에이전트 (StrategicInterventionAgent)
class StrategicInterventionAgent:
    """
    Benchmarks: Gauntlet's Dynamic Risk Engine
    역할: 단순 규칙 수행이 아니라, '자원 효율성'을 계산하여 개입 여부를 판단함.
    """
    def __init__(self, total_budget_usdt, strategy_mode="DEFENSIVE"):
        self.budget = total_budget_usdt
        self.strategy_mode = strategy_mode  # DEFENSIVE, AGGRESSIVE, BALANCED
        self.intervention_history = []

    def evaluate(self, market_state):
        """
        AI 판단 로직:
        1. 현재 가격 추세(Momentum)가 하락세인가?
        2. 오더북 깊이(Depth)가 얇아져서 개입 효과가 극대화되는 시점인가?
        3. 남은 예산으로 방어가 가능한가?
        """
        price = market_state['price']
        roi = market_state['roi']
        volatility = market_state['volatility']
        depth_health = market_state['depth_ratio']
        
        # 판단 스코어링 (0.0 ~ 1.0)
        urgency_score = 0.0
        
        # 로직 1: 하락 가속도 감지 (떨어지는 칼날 잡지 않기 vs 지지선 방어)
        if roi < -20 and volatility > 0.1: 
            urgency_score += 0.4  # 급락 시 경계 태세
            
        # 로직 2: 유동성 고갈 감지 (이때가 개입 효율이 가장 높음 - 적은 돈으로 가격 올리기)
        if depth_health < 0.6:
            urgency_score += 0.3
            
        # 로직 3: 전략 모드에 따른 가중치
        if self.strategy_mode == "DEFENSIVE":
            if roi < -10: urgency_score += 0.2
        elif self.strategy_mode == "AGGRESSIVE":
            if volatility > 0.05: urgency_score += 0.2 # 변동성 있으면 공격적 개입

        # 행동 결정
        action = "HOLD"
        amount = 0.0
        
        if urgency_score >= 0.7 and self.budget > 0:
            action = "BUYBACK"
            # 예산의 10% ~ 30%를 동적으로 할당 (급할수록 많이)
            allocation_ratio = min(0.3, (urgency_score - 0.5)) 
            amount = self.budget * allocation_ratio
            self.budget -= amount
            
        return action, amount, urgency_score
# app.py 파일에 이 내용을 복사해 넣으세요
import streamlit as st
from dotenv import load_dotenv
import os


# .env에서 OpenAI API 키 불러오기
load_dotenv()
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")



# --- 입력된 키는 st.session_state["openai_api_key"]로 사용 가능 ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib
import math
import json
import os
import time
from fpdf import FPDF
from openai import OpenAI

# [STRATEGIC KNOWLEDGE BASE]
# 업로드된 4개 파일의 핵심 전략을 AI에게 Context로 주입합니다.


# [STRATEGIC KNOWLEDGE BASE: ESTV OFFICIAL STRATEGY]
# 업로드된 4개 파일(리스크, 마케팅, P2P, 설계)의 핵심 전략을 통합한 기준 데이터입니다.

ESTV_STRATEGIC_CONTEXT = """
[1. Project Identity: ESTV Nexus]
- Vision: Web3 Media Protocol & DePIN-based P2P Mesh Network.
- Core Asset: 160M+ connected devices (Samsung TV Plus, LG, Roku, etc.).
- Value Model: 'Watch & Earn' 2.0 + 'Host & Earn' (DePIN Node).

[2. Critical Risk Management (출처: 코인 상장 리스크 및 회피 전략.pdf)]
- Risk Factor: Private Sale ($0.05) vs Listing Price ($0.50) -> 10x Gap causes dumping risk.
- 3-Layer Defense Strategy:
    1. Legal: SAFT contains 'No-OTC' & 'Anti-Hedging' clauses.
    2. Technical: 'KPI-based Dynamic Vesting' (Unlock pauses if Price < $0.80 or MAU < 1M).
    3. Economic: 'Soft Lock-up' (High APY Staking to induce voluntary holding).
- Liquidity Target: Minimum $500,000 depth (Tier 2 Standard) to absorb shock.

[3. Marketing Roadmap (출처: ESTV 코인 상장 후 마케팅 전략.pdf)]
- Total Budget: $1,000,000 (Phase 1: 40%, Phase 2: 30%, Phase 3: 30%).
- Key Phases:
    - Phase 1 (D-7 ~ D+30): Wallet Abstraction, Airdrop for 'Proof of Engagement'.
    - Phase 2 (D+31 ~ D+90): Staking Open (APR 15%), Influencer Campaign.
    - Phase 3 (Post-TGE): 'Real Yield' Disclosure (30% of Ad Revenue used for Buyback).
- Goal: Secure 50k Active Holders.

[4. P2P DePIN Strategy (출처: ESTV P2P 통합 전략.pdf)]
- Concept: Users act as CDN nodes (Host) to reduce server costs.
- Flywheel: More Users -> Lower Cost -> Higher Buyback from Savings -> Token Price Up.
"""

RUN_SIM_BUTTON_LABEL = "🚀 시뮬레이션 결과 확인하기"
STEP0_SAVE_PATH = os.path.join(os.path.dirname(__file__), "step0_saved.json")
FULL_HISTORY_DIR = os.path.join(os.path.dirname(__file__), "analysis_history")
STEP0_KEYS = [
    "project_symbol",
    "project_total_supply",
    "project_pre_circulated",
    "project_unlocked",
    "project_unlocked_vesting",
    "project_holders",
    "target_tier",
    "project_type",
    "audit_status",
    "concentration_ratio",
    "has_legal_opinion",
    "has_whitepaper"
]

FULL_SNAPSHOT_KEYS = [
    "mode",
    "mode_selector",
    "tutorial_step",
    "step0_completed",
    "contract_mode_label",
    "contract_mode",
    "input_supply",
    "input_unbonding",
    "input_sell_ratio",
    "input_buy_volume",
    "simulation_unit",
    "simulation_value",
    "scenario_preset",
    "conversion_rate",
    "avg_ticket",
    "use_buy_inflow_pattern",
    "pattern_month4_avg_krw",
    "enable_dual_pipeline",
    "migration_target",
    "migration_ramp_months",
    "acquisition_target",
    "acquisition_ramp_months",
    "use_phase_inflow",
    "phase2_days",
    "phase2_multiplier",
    "prelisting_days",
    "prelisting_multiplier",
    "prelisting_release_days",
    "volume_volatility",
    "volatility_project_type",
    "weekend_dip",
    "price_model",
    "depth_usdt_1pct",
    "depth_usdt_2pct",
    "depth_growth_rate",
    "steps_per_month",
    "turnover_ratio",
    "turnover_buy_share",
    "lp_growth_rate",
    "max_buy_usdt_ratio",
    "max_sell_token_ratio",
    "use_master_plan",
    "use_triggers",
    "buy_verify_boost",
    "holding_suppress",
    "payburn_delta",
    "buyback_daily",
    "monthly_buyback_usdt",
    "burn_fee_rate",
    "initial_investor_lock_months",
    "initial_investor_locked_tokens",
    "initial_investor_vesting_months",
    "initial_investor_release_percent",
    "initial_investor_release_interval",
    "initial_investor_sell_ratio",
    "panic_sensitivity",
    "fomo_sensitivity",
    "private_sale_price",
    "profit_taking_multiple",
    "arbitrage_threshold",
    "min_depth_ratio",
    "project_symbol",
    "project_total_supply",
    "project_pre_circulated",
    "project_unlocked",
    "project_unlocked_vesting",
    "project_holders",
    "target_tier",
    "project_type",
    "audit_status",
    "concentration_ratio",
    "has_legal_opinion",
    "has_whitepaper",
    "marketing_dashboard_url",
    "show_upbit_baseline",
    "enable_confidence"
]

COIN_TYPE_VOLATILITY = {
    "New Listing (신규 상장)": {
        "default": 1.6,
        "range": "1.2~2.2",
        "desc": "상장 초기 변동성이 높아 급등락이 잦습니다."
    },
    "Major (비트/이더)": {
        "default": 0.6,
        "range": "0.3~1.0",
        "desc": "유동성이 깊어 상대적으로 안정적인 움직임을 보입니다."
    },
    "Major Alts (메이저 알트)": {
        "default": 1.0,
        "range": "0.6~1.6",
        "desc": "중간 수준의 변동성을 가진 대표 알트 구간입니다."
    },
    "Meme/Low Cap (밈/잡코인)": {
        "default": 2.3,
        "range": "1.6~3.0",
        "desc": "유동성 얕고 투기성이 강해 변동성이 극단적입니다."
    }
}

SENTIMENT_DEFAULTS = {
    "New Listing (신규 상장)": {"panic": 1.6, "fomo": 1.8},
    "Meme/Low Cap (밈/잡코인)": {"panic": 2.5, "fomo": 3.0},
    "Major Alts (메이저 알트)": {"panic": 1.1, "fomo": 1.2},
    "Major (비트/이더)": {"panic": 0.6, "fomo": 0.7}
}

STRATEGY_PLAYBOOK = {
    "KPI_BREACH": {
        "title": "🚨 기관 물량 베스팅 긴급 유예(Deferral) 발동 권고",
        "condition": "가격이 목표가($0.8) 하회 시",
        "action_plan": """
        1. [거버넌스] 긴급 이사회 소집 후 '가격 안정화 협약' 의결을 강력히 권고합니다.
        2. [SAFT 수정] 초기 투자자 상위 3인과 협의하여, 금월 해제 물량의 80%를 3개월 뒤로 미루는 'Voluntary Lock-up' 체결을 권고합니다.
        3. [보상안] 락업 연장 동의자에게 연 15% 추가 APY(토큰 보상) 제공을 검토하십시오.
        """
    },
    "LIQUIDITY_CRISIS": {
        "title": "💧 유동성 공급(LP) 비상 확충 계획 수립",
        "condition": "오더북 깊이가 위험 수준일 때",
        "action_plan": """
        1. [MM 계약] 지정된 마켓 메이킹(MM) 파트너사에게 'Bid Wall(매수벽) 강화'를 요청하십시오.
        2. [재원 마련] 마케팅 예산의 30%를 USDT로 전환해 오더북 투입을 권고합니다.
        3. [커뮤니티] 'LP 스테이킹 프로그램' 런칭으로 자발적 유동성 공급을 유도하십시오.
        """
    }
}

# NOTE: Streamlit Cloud redeploy trigger (no functional change)

RESET_DEFAULTS = {
    "mode": "tutorial",
    "mode_selector": "초보자",
    "tutorial_step": 0,
    "step0_completed": False,
    "show_user_manual": False,
    "contract_mode_applied": None,
    "contract_mode_label": "사용자 조정",
    "apply_target_scenario": False,
    "apply_reverse_scenario": False,
    "apply_upbit_baseline": False,
    "reverse_result": None,
    "reverse_apply_payload": None,
    "reverse_apply_pending": False,
    "optimized_result": None,
    "optimized_inputs": None,
    "optimized_notes": None,
    "recommended_notes": None,
    "ai_strategy_report": None,
    "ai_tune_banner_ts": None,
    "simulation_active": False,
    "simulation_active_requested": False,
    "simulation_active_force": False,
    "step0_load_pending": False,
    "step0_load_payload": None,
    "full_load_pending": False,
    "full_load_payload": None,
    "loaded_result": None,
    "loaded_inputs": None,
    "reverse_target_price": 5.0,
    "reverse_basis": "전환율 조정",
    "reverse_volatility_mode": "완화",
    "reverse_auto_price_model": True,
    "project_symbol": "ESTV",
    "project_total_supply": 1_000_000_000,
    "project_pre_circulated": 0.0,
    "project_unlocked": 0.0,
    "project_unlocked_vesting": 0,
    "project_holders": 0,
    "target_tier": "Tier 2 (Bybit, Gate.io, KuCoin) - Hard",
    "project_type": "New Listing (신규 상장)",
    "audit_status": "미진행",
    "concentration_ratio": 0.0,
    "has_legal_opinion": False,
    "has_whitepaper": False,
    "tutorial_target_price": 0.0,
    "contract_mode": "사용자 조정",
    "input_supply": 3.0,
    "input_unbonding": 30,
    "input_sell_ratio": 30,
    "input_buy_volume": 200000,
    "simulation_unit": "월",
    "simulation_value": 1,
    "scenario_preset": "직접 입력",
    "conversion_rate": 0.10,
    "avg_ticket": 100.0,
    "use_buy_inflow_pattern": False,
    "pattern_month4_avg_krw": 50,
    "enable_dual_pipeline": False,
    "migration_target": 50_000,
    "migration_ramp_months": 3,
    "acquisition_target": 10_000,
    "acquisition_ramp_months": 12,
    "use_phase_inflow": False,
    "phase2_days": 30,
    "phase2_multiplier": 2.0,
    "prelisting_days": 30,
    "prelisting_multiplier": 1.5,
    "prelisting_release_days": 7,
    "volume_volatility": COIN_TYPE_VOLATILITY["New Listing (신규 상장)"]["default"],
    "volatility_project_type": "New Listing (신규 상장)",
    "weekend_dip": True,
    "price_model": "AMM",
    "depth_usdt_1pct": 1_000_000,
    "depth_usdt_2pct": 3_000_000,
    "depth_growth_rate": 2.0,
    "steps_per_month": 30,
    "turnover_ratio": 5.0,
    "turnover_buy_share": 50.0,
    "lp_growth_rate": 1.0,
    "max_buy_usdt_ratio": 5.0,
    "max_sell_token_ratio": 5.0,
    "use_master_plan": False,
    "use_triggers": True,
    "buy_verify_boost": 0.5,
    "holding_suppress": 0.1,
    "payburn_delta": 0.002,
    "buyback_daily": 0,
    "monthly_buyback_usdt": 0,
    "burn_fee_rate": 0.3,
    "initial_investor_lock_months": 12,
    "initial_investor_locked_tokens": 0.0,
    "initial_investor_vesting_months": 12,
    "initial_investor_release_percent": 10.0,
    "initial_investor_release_interval": 1,
    "initial_investor_sell_ratio": 50,
    "panic_sensitivity": 1.5,
    "fomo_sensitivity": 1.2,
    "sentiment_project_type": "New Listing (신규 상장)",
    "private_sale_price": 0.05,
    "profit_taking_multiple": 5.0,
    "arbitrage_threshold": 2.0,
    "min_depth_ratio": 0.3,
    "show_upbit_baseline": False,
    "enable_confidence": False,
    "confidence_runs": 300,
    "confidence_uncertainty": 10.0,
    "confidence_mape": 15.0,
    "krw_per_usd": 1300,
    "marketing_dashboard_url": "http://localhost:5173"
}

# ==========================================
# 1. 시뮬레이션 엔진 클래스 (핵심 로직)
# ==========================================
def calculate_dynamic_sell_pressure(base_ratio, current_price, price_history, config):
    if len(price_history) < 7:
        return base_ratio
    ma_7 = sum(price_history[-7:]) / 7
    if ma_7 <= 0:
        return base_ratio
    trend_delta = (current_price - ma_7) / ma_7
    if trend_delta < 0:
        panic_factor = 1 + (abs(trend_delta) * config.get('panic_sensitivity', 1.5))
        return min(1.0, base_ratio * panic_factor)
    lock_factor = 1 - (trend_delta * 0.5)
    return max(0.0, base_ratio * lock_factor)


def get_investor_decision(daily_unlock, current_price, config):
    private_sale_price = max(config.get('private_sale_price', 0.05), 1e-9)
    roi = current_price / private_sale_price
    if roi < 1.1:
        return daily_unlock * 0.1
    if roi > config.get('profit_taking_multiple', 5.0):
        return daily_unlock * 2.0
    return daily_unlock


def adjust_depth_by_volatility(base_depth, price_history, config):
    if len(price_history) < 3:
        return base_depth
    recent_volatility = float(np.std(price_history[-3:])) / max(price_history[-1], 1e-9)
    depth_multiplier = 1 / (1 + (recent_volatility * 10))
    final_multiplier = max(depth_multiplier, config.get('min_depth_ratio', 0.3))
    return base_depth * final_multiplier


def apply_fomo_buy(base_buy, current_price, prev_price, config):
    if current_price > prev_price:
        growth_rate = (current_price - prev_price) / max(prev_price, 1e-9)
        fomo_volume = base_buy * growth_rate * config.get('fomo_sensitivity', 1.2)
        return base_buy + fomo_volume
    return base_buy


def calculate_holder_score(holders, target_tier):
    thresholds = {
        "Tier 1": 10000,
        "Tier 2": 3000,
        "Tier 3": 500,
        "DEX": 0
    }
    required = thresholds.get(target_tier, 500)
    if target_tier == "DEX":
        return 100, "DEX는 홀더 수 제한이 없습니다."
    if holders >= required:
        return 100, "✅ 합격 안정권입니다."
    if holders >= required * 0.5:
        score = int((holders / required) * 100)
        return score, f"⚠️ 부족합니다. {target_tier} 기준 {required:,}명 이상 권장됩니다."
    return 0, f"🚨 [광탈 확정] {target_tier} 최소 기준({required:,}명)에 턱없이 부족합니다."


def check_comprehensive_red_flags(inputs):
    warnings = []
    safe_supply = max(float(inputs.get("total_supply", 1.0)), 1.0)
    pre_circulated = float(inputs.get("pre_circulated", 0.0))
    unlocked = float(inputs.get("unlocked", 0.0))
    unlocked_vesting_months = int(inputs.get("unlocked_vesting_months", 0))
    holders = int(inputs.get("holders", 0))
    target_tier = inputs.get("target_tier", "Tier 3")
    circ_ratio = (pre_circulated / safe_supply) * 100.0
    if circ_ratio > 30:
        warnings.append({
            "level": "CRITICAL",
            "msg": f"🚨 초기 유통량({circ_ratio:.1f}%) 과다! 거래소는 15% 미만을 선호합니다."
        })
    unlock_ratio = (unlocked / pre_circulated * 100.0) if pre_circulated > 0 else 0.0
    vesting_months = max(1, unlocked_vesting_months)
    effective_monthly_dump = unlocked / vesting_months
    monthly_dump_ratio = (effective_monthly_dump / pre_circulated * 100.0) if pre_circulated > 0 else 0.0
    if unlocked_vesting_months == 0 and unlock_ratio > 20:
        warnings.append({
            "level": "DANGER",
            "msg": f"💣 오버행 경고: 기유통 물량의 {unlock_ratio:.1f}%가 '즉시 매도' 가능 상태입니다. 급락 위험이 매우 큽니다."
        })
    elif unlocked_vesting_months > 0 and monthly_dump_ratio > 10:
        warnings.append({
            "level": "WARNING",
            "msg": f"⚠️ 매도 압력 주의: 언락 물량이 매월 유통량의 {monthly_dump_ratio:.1f}%씩 쏟아집니다. (기간: {vesting_months}개월)"
        })
    holder_score, holder_msg = calculate_holder_score(holders, target_tier)
    if holder_score < 50:
        warnings.append({
            "level": "CRITICAL",
            "msg": holder_msg
        })
    elif holder_score < 100:
        warnings.append({
            "level": "WARNING",
            "msg": holder_msg
        })
    audit_status = inputs.get("audit_status", "미진행")
    if audit_status == "미진행":
        warnings.append({
            "level": "CRITICAL",
            "msg": "❌ 보안 요건 미달: Audit 리포트가 필수입니다."
        })
    if not inputs.get("has_legal_opinion", False):
        warnings.append({
            "level": "CRITICAL",
            "msg": "❌ 법적 리스크: 증권성 검토 의견서가 없으면 심사 접수조차 불가합니다."
        })
    if inputs.get("concentration_ratio", 0) > 80:
        warnings.append({
            "level": "DANGER",
            "msg": "💣 중앙화 리스크: 상위 홀더 물량이 과도합니다. 공정 분배 위반 소지."
        })
    if inputs.get("project_type", "").startswith("Meme") and holders < 10000:
        warnings.append({
            "level": "WARNING",
            "msg": "⚠️ 밈코인은 압도적인 커뮤니티 화력이 필수입니다."
        })
    if not inputs.get("has_whitepaper", False):
        warnings.append({
            "level": "CRITICAL",
            "msg": "❌ 필수 서류 누락: 백서와 유통량 계획표는 필수입니다."
        })
    return warnings


def create_realistic_schedule(
    target_users,
    ramp_months,
    total_months,
    avg_ticket,
    volatility,
    use_weekend_effect=True
):
    daily_schedule = []
    days_per_month = 30
    total_days = max(1, int(total_months * days_per_month))
    safe_ramp = max(1, int(ramp_months))
    safe_target = max(0.0, float(target_users))
    safe_ticket = max(0.0, float(avg_ticket))
    safe_volatility = max(0.0, float(volatility))

    for day in range(total_days):
        current_month = day / days_per_month
        if current_month < safe_ramp:
            growth_factor = (current_month + 1) / safe_ramp
            monthly_users = safe_target * growth_factor
        else:
            monthly_users = safe_target

        base_daily_usd = (monthly_users * safe_ticket) / days_per_month

        noise = np.random.normal(loc=1.0, scale=safe_volatility)
        noise = max(0.1, noise)

        weekend_factor = 1.0
        if use_weekend_effect:
            day_of_week = day % 7
            if day_of_week >= 5:
                weekend_factor = np.random.uniform(0.6, 0.75)
            else:
                weekend_factor = np.random.uniform(1.0, 1.1)

        final_daily_usd = base_daily_usd * noise * weekend_factor
        daily_schedule.append(final_daily_usd)

    return daily_schedule


def build_optimized_inputs(base_inputs, sim_log):
    adjusted = dict(base_inputs)
    notes = []
    reasons = sim_log.get("reason_code", [])
    reason_texts = sim_log.get("reason", [])
    depth_series = sim_log.get("liquidity_depth_ratio", [])

    has_panic = "PANIC_SELL" in reasons
    has_whale = "WHALE_DUMP" in reasons
    has_liquidity = any("LIQUIDITY_DRAIN" in r for r in reason_texts) or any(
        d < 0.5 for d in depth_series
    )

    if has_panic:
        adjusted["monthly_buyback_usdt"] = max(
            adjusted.get("monthly_buyback_usdt", 0.0),
            adjusted.get("monthly_buy_volume", 0.0) * 0.05
        )
        adjusted["sell_pressure_ratio"] = max(0.0, adjusted.get("sell_pressure_ratio", 0.0) * 0.85)
        adjusted["unbonding_days"] = max(adjusted.get("unbonding_days", 0), 14)
        notes.append("공포 투매 완화: 바이백 확대, 매도율 완화, 언본딩 강화")

    if has_liquidity:
        adjusted["price_model"] = "CEX" if adjusted.get("price_model") == "AMM" else adjusted.get("price_model")
        adjusted["depth_usdt_1pct"] = max(adjusted.get("depth_usdt_1pct", 0.0) * 1.5, 800_000)
        adjusted["depth_usdt_2pct"] = max(adjusted.get("depth_usdt_2pct", 0.0) * 1.5, 2_000_000)
        adjusted["lp_growth_rate"] = max(adjusted.get("lp_growth_rate", 0.0), 0.015)
        notes.append("유동성 보강: 오더북 깊이/LP 성장률 상향")

    if has_whale:
        adjusted["initial_investor_sell_ratio"] = max(
            0.0,
            adjusted.get("initial_investor_sell_ratio", 0.0) * 0.8
        )
        adjusted["max_sell_token_ratio"] = max(
            0.0,
            adjusted.get("max_sell_token_ratio", 0.0) * 0.9
        )
        notes.append("대량 매도 완화: 초기 투자자 매도율/일 매도 캡 축소")

    if not notes:
        notes.append("현재 리스크가 낮아 보수적 미세 조정만 적용")
    return adjusted, notes


class TokenSimulationEngine:
    def __init__(self):
        self.TOTAL_SUPPLY = 1_000_000_000
        self.LISTING_PRICE = 0.50
        self.base_allocations = {
            "Private_Sale": {"percent": 0.20, "cliff": 0,  "vesting": 12},
            "Ecosystem_Rewards": {"percent": 0.30, "cliff": 0,  "vesting": 48},
            "Team_Advisors": {"percent": 0.20, "cliff": 12, "vesting": 36},
            "Partners_Growth": {"percent": 0.10, "cliff": 0, "vesting": 12, "interval": 3},
            "Liquidity_MM": {"percent": 0.10, "cliff": 0, "vesting": 0},
            "Treasury": {"percent": 0.10, "cliff": 24, "vesting": 1},
        }

    def _calculate_monthly_unlock(self, allocation, current_month):
        total_amount = self.TOTAL_SUPPLY * allocation['percent']
        if current_month < allocation['cliff']:
            return 0
        elif allocation.get('vesting', 0) == 0:
            return total_amount
        elif current_month >= allocation['cliff'] + allocation['vesting']:
            return 0 
        else:
            interval = allocation.get('interval', 1)
            if interval > 1:
                offset = current_month - allocation['cliff']
                if offset % interval != 0:
                    return 0
                releases = max(1, allocation['vesting'] // interval)
                return total_amount / releases
            return total_amount / allocation['vesting']

    def _apply_orderbook_trade(self, pool_token, pool_usdt, buy_usdt, sell_token, depth_usdt_1pct, depth_usdt_2pct):
        """
        단순 CEX 오더북 모델:
        - 1% 깊이까지는 depth_usdt_1pct, 2%까지는 depth_usdt_2pct로 선형 소비
        - 매수는 가격 상승, 매도는 가격 하락으로 반영
        """
        price = pool_usdt / pool_token
        one_pct_depth = max(depth_usdt_1pct, 1.0)
        two_pct_depth = max(depth_usdt_2pct, one_pct_depth)

        def impact_for_usdt(volume_usdt):
            if volume_usdt <= one_pct_depth:
                return 0.01 * (volume_usdt / one_pct_depth)
            extra = volume_usdt - one_pct_depth
            extra_depth = max(two_pct_depth - one_pct_depth, 1.0)
            return 0.01 + 0.01 * min(extra / extra_depth, 1.0)

        buy_impact = impact_for_usdt(buy_usdt)
        sell_impact = impact_for_usdt(sell_token * price)
        price_after = price * (1 + buy_impact - sell_impact)

        buy_token_out = buy_usdt / max(price_after, 1e-9)
        sell_usdt_out = sell_token * price_after

        pool_usdt = max(pool_usdt + buy_usdt - sell_usdt_out, 1e-9)
        pool_token = max(pool_token + sell_token - buy_token_out, 1e-9)
        pool_usdt = pool_token * price_after
        return pool_token, pool_usdt, price_after

    def run(self, inputs):
        self.TOTAL_SUPPLY = float(inputs.get("total_supply", self.TOTAL_SUPPLY))
        steps_per_month = max(1, int(inputs.get('steps_per_month', 30)))
        total_days = int(inputs.get('simulation_days', steps_per_month * int(inputs.get('simulation_months', 24))))
        total_days = max(total_days, 1)
        price_history = [self.LISTING_PRICE]
        daily_price_history = [self.LISTING_PRICE]
        daily_events = []
        risk_log = []
        burned_total = 0.0
        action_logs = []
        simulation_log = {
            "day": [],
            "price": [],
            "reason_code": [],
            "action_needed": [],
            "reason": [],
            "action": [],
            "sentiment_index": [],
            "sell_pressure_vol": [],
            "buy_power_vol": [],
            "liquidity_depth_ratio": [],
            "marketing_trigger": [],
            "whale_sell_volume": [],
            "normal_buy_volume": [],
            "sell_sources": [],
            "sell_source_text": [],
            "action_amount_usdt": [],
            "action_message": []
        }
        
        initial_supply = self.TOTAL_SUPPLY * (inputs['initial_circulating_percent'] / 100.0)
        pool_token = max(initial_supply * 0.2, 1e-9)
        pool_usdt = pool_token * self.LISTING_PRICE
        k_constant = pool_token * pool_usdt
        amm_pool_token = pool_token
        amm_pool_usdt = pool_usdt
        amm_k = k_constant

        delay_days = int(inputs['unbonding_days'])
        sell_queue = [0.0] * (total_days + delay_days + 5)
        sell_queue_initial = [0.0] * (total_days + delay_days + 5)
        unlocked_queue = [0.0] * total_days

        marketing_cost_basis = 0.05
        marketing_supply = 100_000_000
        marketing_remaining = marketing_supply

        turnover_ratio = inputs.get('turnover_ratio', 0.0)
        lp_growth_rate = inputs.get('lp_growth_rate', 0.0)
        daily_user_buy_schedule = inputs.get('daily_user_buy_schedule', [])
        max_buy_usdt_ratio = inputs.get('max_buy_usdt_ratio', 0.0)
        max_sell_token_ratio = inputs.get('max_sell_token_ratio', 0.0)
        step_lp_growth_rate = lp_growth_rate / steps_per_month
        burn_fee_rate = inputs.get('burn_fee_rate', 0.0)
        monthly_buyback_usdt = inputs.get('monthly_buyback_usdt', 0.0)
        unlocked_amount = float(inputs.get("unlocked", 0.0))
        unlocked_vesting_months = int(inputs.get("unlocked_vesting_months", 0))
        price_model = inputs.get('price_model', "AMM")
        depth_usdt_1pct = inputs.get('depth_usdt_1pct', 1_000_000.0)
        depth_usdt_2pct = inputs.get('depth_usdt_2pct', 3_000_000.0)
        depth_growth_rate = inputs.get('depth_growth_rate', 0.0)
        market_cfg = inputs.get('market_sentiment_config', {})
        target_tier = inputs.get("target_tier", "Tier 3")
        panic_sensitivity = market_cfg.get('panic_sensitivity', 1.5)
        fomo_sensitivity = market_cfg.get('fomo_sensitivity', 1.2)
        private_sale_price = market_cfg.get('private_sale_price', 0.05)
        profit_taking_multiple = market_cfg.get('profit_taking_multiple', 5.0)
        arbitrage_threshold = market_cfg.get('arbitrage_threshold', 0.02)
        min_depth_ratio = market_cfg.get('min_depth_ratio', 0.3)
        campaigns = inputs.get('campaigns', [])
        triggers = inputs.get('triggers', [])
        enable_triggers = inputs.get('enable_triggers', False)
        triggered_flags = set()
        high_price = self.LISTING_PRICE
        kpi_target = float(inputs.get("kpi_target_price", 0.8))
        kpi_warning_triggered = False
        kpi_breach_day = None
        kpi_breach_price = None

        allocations = dict(self.base_allocations)
        initial_investor_alloc = inputs.get("initial_investor_allocation")
        if initial_investor_alloc:
            allocations["Initial_Investors"] = initial_investor_alloc
        initial_investor_remaining = 0.0
        if initial_investor_alloc:
            initial_investor_remaining = self.TOTAL_SUPPLY * initial_investor_alloc.get("percent", 0.0)

        initial_investor_sell_ratio = inputs.get("initial_investor_sell_ratio", inputs.get("sell_pressure_ratio", 0.0))
        initial_investor_sell_usdt_schedule = inputs.get("initial_investor_sell_usdt_schedule", [])

        if unlocked_amount > 0:
            if unlocked_vesting_months <= 0:
                unlocked_queue[0] += unlocked_amount
            else:
                vesting_days = max(1, int(unlocked_vesting_months * steps_per_month))
                daily_unlocked = unlocked_amount / vesting_days
                for d in range(min(total_days, vesting_days)):
                    unlocked_queue[d] += daily_unlocked

        # [NEW] 전략 에이전트 초기화
        ai_agent = StrategicInterventionAgent(
            total_budget_usdt=inputs.get('monthly_buyback_usdt', 0) * 12, # 1년치 예산 가정
            strategy_mode="DEFENSIVE"
        )

        for day_index in range(total_days):
            day_reasons = []
            day_actions = []

            def log_reason_action(reason, action):
                day_reasons.append(reason)
                day_actions.append(action)

            prev_day_price = daily_price_history[-1]
            if len(daily_price_history) >= 7:
                ma_7 = float(np.mean(daily_price_history[-7:]))
            else:
                ma_7 = prev_day_price
            if price_model == "HYBRID" and day_index > 0 and day_index % steps_per_month == 0:
                depth_usdt_1pct *= (1.0 + depth_growth_rate)
                depth_usdt_2pct *= (1.0 + depth_growth_rate)

            current_price = pool_usdt / pool_token
            price_change_ratio = (current_price - prev_day_price) / max(prev_day_price, 1e-9)
            liquidity_depth_ratio = 1.0
            if price_model in ["CEX", "HYBRID"] and price_change_ratio < 0:
                liquidity_depth_ratio = max(min_depth_ratio, 1.0 - (panic_sensitivity * abs(price_change_ratio)))

            # [NEW] 시장 상태 진단 (State Observation)
            market_state = {
                'price': current_price,
                'roi': (current_price - self.LISTING_PRICE) / self.LISTING_PRICE * 100,
                'volatility': abs(price_change_ratio),
                'depth_ratio': liquidity_depth_ratio
            }

            # [NEW] AI의 전략적 판단 호출
            ai_action, ai_amount, urgency = ai_agent.evaluate(market_state)

            if ai_action == "BUYBACK":
                # 결정된 금액만큼 즉시 시장가 매수 집행
                if price_model in ["CEX", "HYBRID"]:
                    pool_usdt += ai_amount
                    # 오더북에서 토큰을 걷어감 (가격 상승)
                    buyback_impact = ai_amount / max(depth_usdt_1pct * liquidity_depth_ratio, 1.0) * 0.01
                    current_price = current_price * (1 + buyback_impact)
                log_reason_action(f"AI_INTERVENTION (Score {urgency:.2f})", f"${ai_amount:,.0f} BUYBACK")
                action_logs.append({
                    "day": day_index + 1,
                    "action": "🤖 AI 전략 개입",
                    "reason": f"긴급도 {urgency:.2f} >= 0.7 (유동성 {market_state['depth_ratio']:.2f})"
                })
            # ...기존 로직 계속...
        final_price = daily_price_history[-1]
        roi = (final_price - self.LISTING_PRICE) / self.LISTING_PRICE * 100
        
        status = "STABLE"
        if roi < -30: status = "UNSTABLE"
        if roi < -60: status = "CRITICAL"
        
        legal_check = True
        if inputs['initial_circulating_percent'] > 3.0:
            legal_check = False
            status = "ILLEGAL"

        return {
            "inputs": inputs,
            "final_price": final_price,
            "roi": roi,
            "status": status,
            "legal_check": legal_check,
            "risk_logs": risk_log,
            "price_trend": price_history,
            "daily_price_trend": daily_price_history,
            "daily_events": daily_events,
            "action_logs": action_logs,
            "burned_total": burned_total,
            "simulation_log": simulation_log,
            "kpi_warning_triggered": kpi_warning_triggered,
            "kpi_breach_day": kpi_breach_day,
            "kpi_breach_price": kpi_breach_price
        }


def estimate_required_monthly_buy(engine, base_inputs, target_price, max_iter=20):
    low = 0.0
    high = max(1_000_000.0, base_inputs["monthly_buy_volume"] * 20)
    best = high
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        test_inputs = dict(base_inputs)
        test_inputs["monthly_buy_volume"] = mid
        result = run_sim_with_cache(test_inputs)
        if result["final_price"] >= target_price:
            best = mid
            high = mid
        else:
            low = mid
    return best


@st.cache_data(show_spinner=False)
def _run_sim_cached(inputs_json):
    inputs = json.loads(inputs_json)
    engine = TokenSimulationEngine()
    return engine.run(inputs)


def run_sim_with_cache(inputs):
    inputs_json = json.dumps(inputs, sort_keys=True, ensure_ascii=False)
    return _run_sim_cached(inputs_json)


def build_reset_result(inputs, total_days):
    zero_series = [0.0] * max(1, int(total_days))
    empty_log = {
        "day": [],
        "price": [],
        "reason_code": [],
        "action_needed": [],
        "reason": [],
        "action": [],
        "sentiment_index": [],
        "sell_pressure_vol": [],
        "buy_power_vol": [],
        "liquidity_depth_ratio": [],
        "marketing_trigger": [],
        "whale_sell_volume": [],
        "normal_buy_volume": [],
        "sell_sources": [],
        "sell_source_text": [],
        "action_amount_usdt": [],
        "action_message": []
    }
    return {
        "inputs": inputs,
        "final_price": 0.0,
        "roi": 0.0,
        "status": "RESET",
        "legal_check": True,
        "risk_logs": [],
        "price_trend": zero_series,
        "daily_price_trend": zero_series,
        "daily_events": [],
        "action_logs": [],
        "burned_total": 0.0,
        "simulation_log": empty_log
    }


@st.cache_data(show_spinner=False)
def _run_confidence_cached(inputs_json, runs, noise_pct, mape_threshold):
    base_inputs = json.loads(inputs_json)
    engine = TokenSimulationEngine()
    base_result = engine.run(base_inputs)
    base_trend = np.array(base_result["daily_price_trend"], dtype=float)
    base_trend = np.maximum(base_trend, 1e-9)

    rng = np.random.default_rng(42)
    target_keys = [
        "initial_circulating_percent",
        "unbonding_days",
        "sell_pressure_ratio",
        "monthly_buy_volume",
        "turnover_ratio",
        "lp_growth_rate",
        "max_buy_usdt_ratio",
        "max_sell_token_ratio",
        "burn_fee_rate",
        "monthly_buyback_usdt",
        "depth_usdt_1pct",
        "depth_usdt_2pct",
        "depth_growth_rate"
    ]
    int_keys = {"unbonding_days"}
    mape_list = []
    good = 0

    for _ in range(max(1, runs)):
        sim_inputs = dict(base_inputs)
        for key in target_keys:
            if key not in sim_inputs:
                continue
            val = sim_inputs[key]
            if val is None:
                continue
            noise = rng.uniform(-noise_pct, noise_pct)
            new_val = val * (1 + noise)
            if key in int_keys:
                new_val = int(round(new_val))
            if key in ["initial_circulating_percent"]:
                new_val = min(max(new_val, 0.0), 100.0)
            elif key in ["sell_pressure_ratio", "turnover_ratio", "lp_growth_rate", "max_buy_usdt_ratio", "max_sell_token_ratio", "burn_fee_rate", "depth_growth_rate"]:
                new_val = max(new_val, 0.0)
            else:
                new_val = max(new_val, 0.0)
            sim_inputs[key] = new_val

        sim_result = engine.run(sim_inputs)
        sim_trend = np.array(sim_result["daily_price_trend"], dtype=float)
        n = min(len(base_trend), len(sim_trend))
        mape = float(np.mean(np.abs(sim_trend[:n] - base_trend[:n]) / base_trend[:n]) * 100)
        mape_list.append(mape)
        if mape <= mape_threshold:
            good += 1

    confidence = (good / max(1, runs)) * 100
    mape_array = np.array(mape_list, dtype=float)
    return {
        "confidence": confidence,
        "avg_mape": float(np.mean(mape_array)),
        "p10_mape": float(np.percentile(mape_array, 10)),
        "p90_mape": float(np.percentile(mape_array, 90))
    }


def run_confidence_with_cache(inputs, runs, noise_pct, mape_threshold):
    inputs_json = json.dumps(inputs, sort_keys=True, ensure_ascii=False)
    return _run_confidence_cached(inputs_json, runs, noise_pct, mape_threshold)


def filter_recommended_settings(payload):
    return dict(payload), []


def generate_strategy_guide(current_price, target_price, period_months, suggested_inflow, suggested_supply):
    required_growth = (target_price - current_price) / current_price
    monthly_intensity = required_growth / period_months

    strategy_title = ""
    tactics = []

    if monthly_intensity < 0.5:
        strategy_title = "🌱 [Level 1] 오가닉 성장 전략 (Organic Growth)"
        tactics = [
            "**커뮤니티 결속:** 디스코드/텔레그램 AMA를 주 1회 개최하여 홀더 신뢰를 쌓으세요.",
            "**콘텐츠 마케팅:** 블로그와 유튜브를 통해 프로젝트의 기술적 진보를 알리세요.",
            "**공급 관리:** 별도의 강제 락업보다는 스테이킹 리워드(APR 5~10%)로 자발적 보유를 유도하세요."
        ]
    elif monthly_intensity < 2.0:
        strategy_title = "🚀 [Level 2] 부스팅 전략 (Aggressive Boosting)"
        tactics = [
            f"**자금 집중:** 월 **${suggested_inflow:,.0f}** 규모의 유입을 위해 유료 광고(Ads) 집행이 필수입니다.",
            "**인플루언서(KOL):** Tier 2급 유튜버/인플루언서 3명 이상과 계약하여 화제성을 만드세요.",
            "**이벤트:** 거래소와 연계한 '순매수 이벤트'나 '트레이딩 대회'를 개최하세요."
        ]
    else:
        strategy_title = "🔥 [Level 3] 공급 쇼크 전략 (Supply Shock Operation)"
        tactics = [
            f"**극단적 락업:** 현재 유통량인 {suggested_supply:.1f}%를 제외한 **모든 물량을 재단이 회수/락업**해야 합니다.",
            "**시장가 매수:** MM 팀을 통해 매도벽을 강제로 뚫어버리는 **'시장가 매수(Market Buy)'**가 필요합니다.",
            "**뉴스 호재:** '대형 파트너십'이나 '메인넷 런칭'급의 초대형 호재 없이는 이 가격 유지가 불가능합니다."
        ]

    guide_text = f"""
### {strategy_title}
사장님, **{period_months}개월 내 ${target_price}** 달성을 위한 AI 전략 제안입니다.

#### 📋 실행 과제 (Action Items)
1. {tactics[0]}
2. {tactics[1]}
3. {tactics[2]}

#### ⚙️ 시스템 자동 조정 내역
* **자금 투입:** 월 ${suggested_inflow:,.0f} 로 상향
* **유통량 제한:** {suggested_supply:.1f}% 로 축소
"""
    return guide_text


def generate_strategic_imperative(inputs, series):
    depth_1pct = float(inputs.get("depth_usdt_1pct", 0.0))
    init_circ = float(inputs.get("initial_circulating_percent", 0.0))
    unbonding_days = int(inputs.get("unbonding_days", 0))
    monthly_buy = float(inputs.get("monthly_buy_volume", 0.0))
    target_tier = inputs.get("target_tier", "Tier 3")

    if depth_1pct < 500_000:
        return {
            "title": "합격 조건: 오더북 깊이 $500k 이상 확보",
            "content": (
                "상장 심사 통과를 위해 **1% 구간 오더북 유동성**을 최소 $500k 이상으로 확보하세요. "
                "유동성 방어가 확보되면 초기 급락과 슬리피지를 크게 줄일 수 있습니다."
            )
        }
    if init_circ > 5.0:
        return {
            "title": "합격 조건: 초기 유통량 5% 이하로 조정",
            "content": (
                "상장 직후 과도한 유통 물량은 즉각적인 차익 실현을 유발합니다. "
                "**초기 유통량을 5% 이하**로 제한해 가격 방어력을 확보하세요."
            )
        }
    if unbonding_days < 30:
        return {
            "title": "합격 조건: 언본딩 30일 이상 확보",
            "content": (
                "언본딩 기간이 짧으면 단기 매도 압력이 집중됩니다. "
                "**언본딩 30일 이상** 확보가 안정적 가격 형성에 필수입니다."
            )
        }
    if monthly_buy < 500_000:
        return {
            "title": "합격 조건: 월간 매수 유입 $500k 이상 확보",
            "content": (
                "심사 통과를 위해 월간 매수 유입이 최소 $500k 이상 필요합니다. "
                "유입이 늘수록 유동성 방어와 가격 안정성이 개선됩니다."
            )
        }
    if series and max(series) < 0.6:
        return {
            "title": "합격 조건: 가격 안정 구간 유지",
            "content": (
                "목표 거래소 등급("
                f"{target_tier}) 기준으로 가격 안정 구간을 유지해야 합니다. "
                "캠페인/유동성 정책을 유지해 추세적 하락을 방지하세요."
            )
        }
    return {
        "title": "합격 조건: 현재 구조 유지 및 확장",
        "content": (
            "핵심 리스크 지표가 안정 범위에 있습니다. "
            "현재 구조를 유지하면서 유입·유동성을 점진적으로 강화하세요."
        )
    }


KOREAN_FONT_FILES = {
    "regular": os.path.join("assets", "fonts", "NanumGothic.ttf"),
    "bold": os.path.join("assets", "fonts", "NanumGothic-Bold.ttf"),
    "extra": os.path.join("assets", "fonts", "NanumGothic-ExtraBold.ttf")
}


def resolve_korean_fonts():
    regular = KOREAN_FONT_FILES["regular"]
    bold = KOREAN_FONT_FILES["bold"]
    if os.path.exists(regular) and os.path.exists(bold):
        return {"regular": regular, "bold": bold}
    return None


def generate_insight_text(result, inputs):
    score = result.get("final_score", 0)
    liquidity = float(inputs.get("depth_usdt_1pct", 0))
    target_price = float(inputs.get("target_price", inputs.get("reverse_target_price", 0)))

    score_messages = {
        "high": [
            "전반적인 토크노믹스 설계가 매우 견고하며, Tier 1 심사 기준을 상회합니다.",
            "심사 통과 가능성이 높습니다. 다만 과열 구간 관리가 중요합니다."
        ],
        "mid": [
            "상장은 가능하나, 상장 후 1개월 내 가격 변동성 리스크가 큽니다.",
            "핵심 지표는 통과선이지만 유동성/수요 보강이 필요합니다."
        ],
        "low": [
            "현재 구조로는 상장 심사 탈락이 확정적입니다. 전면 재설계가 요구됩니다.",
            "리스크가 과도합니다. 즉시 구조 개선 없이는 상장 불가 수준입니다."
        ]
    }
    if score >= 80:
        grade = "S (즉시 상장 가능)"
        summary = score_messages["high"][0]
    elif score >= 60:
        grade = "B (보완 필요)"
        summary = score_messages["mid"][0]
    else:
        grade = "D (상장 불가)"
        summary = score_messages["low"][0]

    liquidity_messages = {
        "low": [
            f"현재 오더북 두께(${(liquidity / 1000):.1f}k)는 방어 불가 수준입니다. 이대로면 상장 폐지 리스크가 큽니다.",
            "유동성이 지나치게 얕습니다. 즉시 $200k 이상으로 보강하지 않으면 급락이 반복됩니다."
        ],
        "mid": [
            "오더북 깊이가 기준선은 넘지만, 대규모 매도 방어에는 부족합니다.",
            "현재 유동성은 방어선 수준입니다. 상장 직후 2배 이상의 보강이 필요합니다."
        ],
        "high": [
            "유동성은 충분하지만 자본 효율성이 떨어질 수 있습니다. 운영 비용과 효과를 점검하세요.",
            "오더북이 과도하게 두꺼워졌습니다. 효율적 재배분으로 ROI를 최적화하세요."
        ]
    }
    if liquidity < 100000:
        liq_msg = liquidity_messages["low"][0]
    elif liquidity < 300000:
        liq_msg = liquidity_messages["mid"][0]
    else:
        liq_msg = liquidity_messages["high"][0]

    target_messages = {
        "low": "목표가가 낮아 안전성은 높지만, 투자자 모멘텀 확보가 어렵습니다.",
        "mid": "목표가가 현실적입니다. 공급 통제와 유입 계획을 유지하세요.",
        "high": "목표가가 높아졌습니다. 유동성/매수 유입을 과감히 증액해야 합니다."
    }
    if target_price <= 1.0:
        target_msg = target_messages["low"]
    elif target_price <= 5.0:
        target_msg = target_messages["mid"]
    else:
        target_msg = target_messages["high"]

    return grade, summary, liq_msg + "\n" + target_msg


def generate_ai_consulting_report(result, inputs):
    recommendations = []

    if result.get("kpi_warning_triggered"):
        breach_day = result.get("kpi_breach_day")
        breach_price = result.get("kpi_breach_price")
        rec = STRATEGY_PLAYBOOK["KPI_BREACH"]
        msg = f"""
        **[진단]** Day {breach_day}에 가격이 ${breach_price:.2f}로 하락하며 KPI 방어선이 붕괴되었습니다.
        이 상태에서 예정된 물량이 출회되면 가격은 추가 하락할 가능성이 큽니다.

        **[경영진 권고]**
        {rec['title']}

        **[구체적 실행 계획 (Action Items)]**
        {rec['action_plan']}
        """
        recommendations.append(msg.strip())

    liquidity_depth = float(inputs.get("depth_usdt_1pct", 0))
    depth_ratio_series = result.get("simulation_log", {}).get("liquidity_depth_ratio", [])
    min_depth_ratio = min(depth_ratio_series) if depth_ratio_series else 1.0
    if liquidity_depth < 200000 or min_depth_ratio < 0.5:
        rec = STRATEGY_PLAYBOOK["LIQUIDITY_CRISIS"]
        msg = f"""
        **[진단]** 오더북 깊이가 위험 수준으로 추정됩니다. (1% 깊이 ${liquidity_depth:,.0f}, 최소 심리 깊이 {min_depth_ratio:.2f})

        **[경영진 권고]**
        {rec['title']}

        **[구체적 실행 계획 (Action Items)]**
        {rec['action_plan']}
        """
        recommendations.append(msg.strip())

    return recommendations



def get_real_ai_insight(api_key, inputs, result, score, series):
    if not api_key:
        return None

    # 1. 시뮬레이션 데이터 추출 (Data Extraction)
    max_price = max(series) if series else 0.0
    final_price = result.get('final_price', 0.0)
    liquidity_1pct = inputs.get('depth_usdt_1pct', 0)
    monthly_buy = inputs.get('monthly_buy_volume', 0)
    worst_day = result.get('worst_day', 'N/A')
    
    # 2. 프롬프트 구성 (Persona + Context + Data + Instruction)
    system_prompt = f"""
    You are the Chief Strategy Officer (CSO) of ESTV. 
    Your role is to rigorously evaluate the token simulation results against our official strategy documents.
    
    [Strategic Standards (Our Playbook)]
    {ESTV_STRATEGIC_CONTEXT}
    
    [Current Simulation Result]
    - Final Score: {score}/100
    - Max Price: ${max_price:.2f} (Target: $5.0)
    - Final Price: ${final_price:.2f}
    - Liquidity Depth (1%): ${liquidity_1pct:,.0f}
    - Monthly Buy Pressure: ${monthly_buy:,.0f}
    - KPI Vesting Triggered: {result.get('kpi_warning_triggered', False)} (Means price dropped below target)
    """

    user_prompt = """
    Write a 'Strategic Alignment Report' in Korean based on the data above.
    
    **Output Structure (Strictly follow this):**
    
    **1. 🛡️ 전략 정합성 진단 (Strategy Alignment)**
    - Compare the Liquidity Depth (${liquidity_1pct:,.0f}) against our 'Risk Strategy' target ($500k).
    - Did the 'KPI-based Dynamic Vesting' work? (Check if KPI Vesting was triggered).
    - Is the Monthly Buy Pressure sufficient to support the 'Phase 2 Staking' plan?
    
    **2. ⚠️ 발견된 괴리 및 위험 (Gap Analysis)**
    - Identify specific gaps between our 'DePIN Growth Vision' and the actual simulation outcome.
    - If the score is low ({score}), explain WHY based on the '3-Layer Defense Strategy'.
    - Mention if the 'Marketing Budget' seems insufficient for the observed sell pressure.
    
    **3. 💊 AI 실행 권고 (Action Items)**
    - Provide 3 concrete actions aligned with our Roadmap.
    - Example: "Increase Buyback allocation from Ad Revenue", "Enforce stricter SAFT clauses", "Boost Phase 1 Marketing".
    - Use specific terms like 'Host & Earn', 'Real Yield', 'Soft Lock-up'.
    
    **Tone:** Professional, Insightful, Executive-level. Be critical if the score is low.
    """

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o", # 또는 gpt-4-turbo
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI 분석 중 오류가 발생했습니다: {str(e)}"


class AdvancedReport(FPDF):
    def __init__(self, title="AI Report"):
        super().__init__()
        self.title = title
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, self.title, ln=True, align="C")
        self.ln(10)

    def chapter_title(self, title):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, title, ln=True, align="L")
        self.ln(4)

    def chapter_body(self, body):
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 8, body)
        self.ln()

    def add_section(self, title, body):
        self.add_page()
        self.chapter_title(title)
        self.chapter_body(body)

def create_full_report(inputs, series, score, target_price):
    pdf = AdvancedReport()
    pdf.add_page()

    max_price = max(series) if series else 0.0
    worst_day = "N/A"
    if series and len(series) > 2:
        diffs = [series[i] - series[i - 1] for i in range(1, len(series))]
        min_idx = diffs.index(min(diffs))
        worst_day = f"{min_idx + 1}"

    result_summary = {
        "final_score": score,
        "max_price": max_price,
        "worst_day": worst_day
    }
    grade, summary, liq_msg = generate_insight_text(result_summary, inputs)

    pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 20, pdf._safe_text("전략 시뮬레이션 결과 보고서"), 0, 1, "C")
    pdf.ln(10)

    pdf.chapter_title("1. AI CSO 종합 진단 (Powered by GPT-4)")
    # OpenAI API Key를 세션에서 가져옴
    openai_api_key = ""
    try:
        openai_api_key = st.session_state.get("openai_api_key", "")
    except Exception:
        pass
    real_ai_text = None
    if openai_api_key:
        try:
            real_ai_text = get_real_ai_insight(openai_api_key, inputs, result_summary, score, series)
        except Exception as e:
            real_ai_text = None
    if real_ai_text:
        pdf.body_text(real_ai_text)
    else:
        if "D" in grade:
            pdf.set_text_color(255, 0, 0)
        pdf.body_text(f"■ 종합 등급: {grade}")
        pdf.set_text_color(0, 0, 0)
        pdf.body_text(f"■ 진단 요약:\n{summary}")

    pdf.chapter_title("2. 핵심 리스크 및 대응 전략")
    pdf.body_text(f"■ 유동성 리스크:\n{liq_msg}")
    pdf.body_text(f"■ 최대 낙폭 구간:\n시뮬레이션 상 Day {worst_day}에 가장 큰 하락이 예상됩니다. 이 시기에 맞춰 마케팅 자금을 집중 투하해야 합니다.")

    pdf.chapter_title("3. 주요 시뮬레이션 지표")
    metrics = {
        "목표 가격": f"${target_price:,.2f}",
        "최대 도달 가격": f"${max_price:,.2f}",
        "필요 초기 자금 (LP)": f"${inputs.get('depth_usdt_1pct', 0) * 2:,.0f}",
        "월간 마케팅 예산": f"${inputs.get('monthly_buy_volume', 0):,.0f}"
    }
    pdf.add_metric_table(metrics)

    pdf.chapter_title("4. 설정 기록 (Inputs Snapshot)")
    settings_snapshot = {
        "코인 심볼": inputs.get("project_symbol", "ESTV"),
        "총 발행량": f"{inputs.get('total_supply', 0):,.0f}",
        "초기 유통량(%)": f"{inputs.get('initial_circulating_percent', 0):.2f}",
        "언본딩 기간(일)": f"{inputs.get('unbonding_days', 0)}",
        "락업 해제 매도율(%)": f"{inputs.get('sell_pressure_ratio', 0) * 100:.1f}",
        "월간 매수 유입($)": f"{inputs.get('monthly_buy_volume', 0):,.0f}",
        "오더북 깊이(1%)": f"${inputs.get('depth_usdt_1pct', 0):,.0f}",
        "패닉 민감도": f"{inputs.get('panic_sensitivity', 0):.2f}",
        "FOMO 민감도": f"{inputs.get('fomo_sensitivity', 0):.2f}",
        "차익거래 임계값(%)": f"{inputs.get('arbitrage_threshold', 0) * 100:.1f}",
        "패닉 깊이 하한": f"{inputs.get('min_depth_ratio', 0):.2f}"
    }
    pdf.add_metric_table(settings_snapshot)

    pdf.add_page()
    pdf.chapter_title("5. AI 전략 컨설팅 및 실행 계획")
    ai_advice_list = generate_ai_consulting_report(result_summary, inputs)
    if ai_advice_list:
        for advice in ai_advice_list:
            clean_text = advice.replace("**", "").strip()
            pdf.set_font(pdf.font_name, "", 11)
            pdf.multi_cell(0, 8, pdf._safe_text(clean_text))
            pdf.ln(5)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)
    else:
        pdf.body_text("✅ 현재 시뮬레이션 상 중대한 전략적 위험이 감지되지 않았습니다. 기존 계획대로 진행하십시오.")

    return pdf.output(dest="S").encode("latin-1", "replace")

# ==========================================
# 2. Streamlit UI 구성
# ==========================================
st.set_page_config(page_title="ESTV 토큰 시뮬레이터", layout="wide")

def hard_reset_session():
    st.cache_data.clear()
    keep_keys = {"hard_reset_pending"}
    for k in list(st.session_state.keys()):
        if k not in keep_keys:
            del st.session_state[k]
    st.session_state.update(RESET_DEFAULTS)
    st.session_state["reset_triggered"] = True
    st.session_state["hard_reset_pending"] = False
    if os.path.exists(STEP0_SAVE_PATH):
        try:
            os.remove(STEP0_SAVE_PATH)
        except OSError:
            pass


def save_step0_snapshot():
    payload = {key: st.session_state.get(key, RESET_DEFAULTS.get(key)) for key in STEP0_KEYS}
    with open(STEP0_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_step0_snapshot():
    if not os.path.exists(STEP0_SAVE_PATH):
        return False
    with open(STEP0_SAVE_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    st.session_state["step0_load_payload"] = payload
    st.session_state["step0_load_pending"] = True
    return True


def apply_step0_snapshot():
    payload = st.session_state.get("step0_load_payload")
    if not payload:
        return
    for key, value in payload.items():
        st.session_state[key] = value
    st.session_state["step0_load_pending"] = False
    st.session_state["step0_load_payload"] = None


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def build_full_snapshot(inputs, result):
    payload = {
        "version": 1,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_state": {key: st.session_state.get(key, RESET_DEFAULTS.get(key)) for key in FULL_SNAPSHOT_KEYS},
        "inputs": to_jsonable(inputs),
        "result": to_jsonable(result)
    }
    return payload


def ensure_history_dir():
    os.makedirs(FULL_HISTORY_DIR, exist_ok=True)


def save_full_snapshot_to_history(payload):
    ensure_history_dir()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_symbol = str(payload.get("session_state", {}).get("project_symbol", "ESTV")).replace("/", "_")
    filename = f"analysis_{safe_symbol}_{timestamp}.json"
    path = os.path.join(FULL_HISTORY_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return filename


def list_history_files():
    if not os.path.exists(FULL_HISTORY_DIR):
        return []
    files = [f for f in os.listdir(FULL_HISTORY_DIR) if f.endswith(".json")]
    files.sort(reverse=True)
    return files


def load_history_file(filename):
    path = os.path.join(FULL_HISTORY_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_full_snapshot(payload):
    st.session_state["full_load_payload"] = payload
    st.session_state["full_load_pending"] = True


def apply_full_snapshot():
    payload = st.session_state.get("full_load_payload")
    if not payload:
        return
    for key, value in payload.get("session_state", {}).items():
        st.session_state[key] = value
    st.session_state["loaded_inputs"] = payload.get("inputs")
    st.session_state["loaded_result"] = payload.get("result")
    st.session_state["simulation_active"] = True
    st.session_state["full_load_pending"] = False
    st.session_state["full_load_payload"] = None

if st.session_state.get("hard_reset_pending"):
    hard_reset_session()
    st.rerun()

if st.session_state.get("step0_load_pending"):
    apply_step0_snapshot()
    st.rerun()

if st.session_state.get("full_load_pending"):
    apply_full_snapshot()
    st.rerun()

ai_banner_ts = st.session_state.get("ai_tune_banner_ts")
if ai_banner_ts and (time.time() - ai_banner_ts) <= 3.0:
    st.success("✅ AI가 조정한 목표가로 각 설정들을 자동 조정하고 있습니다.")
elif ai_banner_ts:
    st.session_state["ai_tune_banner_ts"] = None

manual_path = os.path.abspath("user_manual.md")
if st.session_state.get("show_user_manual"):
    if os.path.exists(manual_path):
        with open(manual_path, "r", encoding="utf-8") as manual_file:
            manual_text = manual_file.read()
        with st.expander("📘 사용설명서", expanded=True):
            st.markdown(manual_text)
    else:
        st.info("사용설명서 파일을 찾을 수 없습니다.")



# 사이드바: 사용자 입력 컨트롤
step0_visible = st.session_state.get("tutorial_step", 0) == 0 and not st.session_state.get("step0_completed", False)
def should_show_kyc_warnings():
    kyc_keys = [
        "project_symbol",
        "project_total_supply",
        "project_pre_circulated",
        "project_unlocked",
        "project_unlocked_vesting",
        "project_holders",
        "target_tier",
        "project_type",
        "audit_status",
        "concentration_ratio",
        "has_legal_opinion",
        "has_whitepaper"
    ]
    for key in kyc_keys:
        if st.session_state.get(key, RESET_DEFAULTS.get(key)) != RESET_DEFAULTS.get(key):
            return True
    return False

legal_supply = st.session_state.get("input_supply", 3.0)
if legal_supply > 3.0 and step0_visible:
    st.sidebar.error("🚨 [Legal Check] 초기 유통량 3% 초과")

def toggle_user_manual():
    st.session_state["show_user_manual"] = not st.session_state.get("show_user_manual", False)

top_controls = st.sidebar.columns([1, 1])
with top_controls[0]:
    manual_button_label = "📘 사용설명서 닫기" if st.session_state.get("show_user_manual") else "📘 사용설명서 열기"
    st.button(manual_button_label, on_click=toggle_user_manual)
with top_controls[1]:
    if st.button("🔄 전체 초기화"):
        st.session_state["hard_reset_pending"] = True
        st.rerun()

st.sidebar.header("🎯 시나리오 & 목표 설정")
if step0_visible:
    st.sidebar.subheader("📝 Step 0. 프로젝트 기본 정보")
    symbol = st.sidebar.text_input(
        "코인 심볼",
        value=st.session_state.get("project_symbol", "ESTV"),
        key="project_symbol",
        help="거래소에서 사용할 코인 심볼(티커)입니다."
    )
    total_supply_input = st.sidebar.number_input(
        "총 발행량 (Total Supply)",
        min_value=1.0,
        value=float(st.session_state.get("project_total_supply", 1_000_000_000)),
        step=1_000_000.0,
        key="project_total_supply",
        help="프로젝트의 총 발행량입니다."
    )
    pre_circulated = st.sidebar.number_input(
        "현재 유통량 (Pre-circulated)",
        min_value=0.0,
        value=float(st.session_state.get("project_pre_circulated", 0.0)),
        step=1_000_000.0,
        key="project_pre_circulated",
        help="재단 지갑을 떠나 외부로 나간 물량입니다."
    )
    unlocked = st.sidebar.number_input(
        "언락 물량 (Unlocked)",
        min_value=0.0,
        value=float(st.session_state.get("project_unlocked", 0.0)),
        step=1_000_000.0,
        key="project_unlocked",
        help="현재 유통량 중 즉시 매도 가능한 물량입니다."
    )
    unlocked_vesting_months = st.sidebar.number_input(
        "언락 물량 해제 기간 (개월)",
        min_value=0,
        max_value=60,
        value=int(st.session_state.get("project_unlocked_vesting", 0)),
        step=1,
        key="project_unlocked_vesting",
        help="해당 언락 물량이 시장에 전량 매도되기까지 걸리는 예상 기간입니다. 0이면 즉시 매도로 간주합니다."
    )
    holders = st.sidebar.number_input(
        "보유자 수 (Holders)",
        min_value=0,
        value=int(st.session_state.get("project_holders", 0)),
        step=100,
        key="project_holders",
        help="현재 코인을 보유한 지갑 수입니다."
    )
    target_tier = st.sidebar.selectbox(
        "목표로 하는 거래소 등급은 무엇입니까?",
        options=[
            "Tier 1 (Binance, Upbit, Coinbase) - Hell",
            "Tier 2 (Bybit, Gate.io, KuCoin) - Hard",
            "Tier 3 (Small CEX) - Normal",
            "DEX (Uniswap only) - Easy"
        ],
        index=1,
        key="target_tier",
        help="목표 거래소 등급에 따라 보유자/유통 기준이 달라집니다."
    )
    project_type = st.sidebar.selectbox(
        "프로젝트 유형",
        [
            "New Listing (신규 상장)",
            "Major (비트/이더)",
            "Major Alts (메이저 알트)",
            "Meme/Low Cap (밈/잡코인)"
        ],
        index=0,
        key="project_type",
        help="프로젝트 유형에 따라 추천 변동성 기본값이 달라집니다."
    )
    audit_status = st.sidebar.selectbox(
        "보안 감사(Audit) 여부",
        ["완료 (Tier 1 - CertiK 등)", "완료 (Tier 2)", "진행 중", "미진행"],
        index=3,
        key="audit_status",
        help="감사 완료 여부는 상장 심사 핵심 체크 항목입니다."
    )
    concentration_ratio = st.sidebar.slider(
        "상위 10인 지갑 보유 비중 (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state.get("concentration_ratio", 0.0)),
        step=1.0,
        key="concentration_ratio",
        help="지갑 집중도가 높을수록 리스크 경고가 강화됩니다."
    )
    has_legal_opinion = st.sidebar.checkbox(
        "증권성 검토 법률 의견서 보유",
        value=bool(st.session_state.get("has_legal_opinion", False)),
        key="has_legal_opinion",
        help="법률 의견서 미보유 시 상장 리스크가 커집니다."
    )
    has_whitepaper = st.sidebar.checkbox(
        "백서 및 유통량 계획표 완비",
        value=bool(st.session_state.get("has_whitepaper", False)),
        key="has_whitepaper",
        help="백서/유통 계획이 없으면 심사 리스크가 커집니다."
    )
    st.sidebar.markdown("### 💾 Step 0 저장")
    save_cols = st.sidebar.columns(2)
    with save_cols[0]:
        if st.button("저장"):
            save_step0_snapshot()
            st.sidebar.success("Step 0 저장 완료")
    with save_cols[1]:
        if st.button("불러오기"):
            if load_step0_snapshot():
                st.sidebar.success("Step 0 불러오기 완료")
                st.rerun()
            else:
                st.sidebar.info("저장된 Step 0 정보가 없습니다.")
else:
    symbol = st.session_state.get("project_symbol", "ESTV")
    total_supply_input = float(st.session_state.get("project_total_supply", 1_000_000_000))
    pre_circulated = float(st.session_state.get("project_pre_circulated", 0.0))
    unlocked = float(st.session_state.get("project_unlocked", 0.0))
    unlocked_vesting_months = int(st.session_state.get("project_unlocked_vesting", 0))
    holders = int(st.session_state.get("project_holders", 0))
    target_tier = st.session_state.get("target_tier", "Tier 2 (Bybit, Gate.io, KuCoin) - Hard")
    project_type = st.session_state.get("project_type", "New Listing (신규 상장)")
    audit_status = st.session_state.get("audit_status", "미진행")
    concentration_ratio = float(st.session_state.get("concentration_ratio", 0.0))
    has_legal_opinion = bool(st.session_state.get("has_legal_opinion", False))
    has_whitepaper = bool(st.session_state.get("has_whitepaper", False))
if target_tier.startswith("Tier 1"):
    target_tier_key = "Tier 1"
elif target_tier.startswith("Tier 2"):
    target_tier_key = "Tier 2"
elif target_tier.startswith("Tier 3"):
    target_tier_key = "Tier 3"
else:
    target_tier_key = "DEX"
pre_circ_ratio = (pre_circulated / total_supply_input * 100.0) if total_supply_input > 0 else 0.0
if step0_visible:
    red_flag_inputs = {
        "total_supply": total_supply_input,
        "pre_circulated": pre_circulated,
        "unlocked": unlocked,
        "unlocked_vesting_months": unlocked_vesting_months,
        "target_tier": target_tier_key,
        "holders": holders,
        "project_type": project_type,
        "audit_status": audit_status,
        "concentration_ratio": concentration_ratio,
        "has_legal_opinion": has_legal_opinion,
        "has_whitepaper": has_whitepaper
    }
    show_kyc_alerts = st.session_state.get("step0_completed", False) or should_show_kyc_warnings()
    if show_kyc_alerts:
        for warn in check_comprehensive_red_flags(red_flag_inputs):
            if warn["level"] == "CRITICAL":
                st.sidebar.error(warn["msg"])
            elif warn["level"] == "DANGER":
                st.sidebar.warning(warn["msg"])
            else:
                st.sidebar.warning(warn["msg"])

        score = 100.0
        if pre_circ_ratio > 10:
            score -= (pre_circ_ratio - 10) * 1.0
        unlock_ratio = (unlocked / pre_circulated * 100.0) if pre_circulated > 0 else 0.0
        if unlock_ratio > 20:
            score -= (unlock_ratio - 20) * 2.0
        holder_score, holder_msg = calculate_holder_score(int(holders), target_tier_key)
        score -= (100 - holder_score) * 0.2
        if audit_status == "미진행":
            score -= 30
        if not has_legal_opinion:
            score -= 30
        if not has_whitepaper:
            score -= 30
        score = max(0.0, min(100.0, score))
        st.session_state["listing_score"] = score
        if score >= 80:
            grade = "양호"
        elif score >= 60:
            grade = "주의"
        else:
            grade = "거절 위험"
        scorecard_help = (
            "거래소는 수수료보다 신뢰를 먼저 봅니다. 신뢰가 무너지면 뱅크런이 발생합니다.\n"
            "즉시 거절되는 3대 리스크: 덤핑 구조(과도한 초기 유통/물량 집중), "
            "유동성 고갈(거래량·오더북 약함), 법적 리스크(증권성/AML).\n"
            "내부 심사는 덤핑 테스트/유동성 스트레스 테스트로 진행되며 회복 불가 판정이면 거절·상폐됩니다.\n"
            "이 점수는 거절 위험의 사전 경고등입니다. 경고/위험 구간에서의 상장 신청은 사실상 거절 신청서입니다.\n"
            "목표: Status: Stable + Legal Check: Pass 유지 후 그 설정값을 상장 서류에 반영."
        )
        score_cols = st.sidebar.columns([5, 1])
        score_cols[0].metric("상장 적합성 점수", f"{score:.0f} / 100")
        with score_cols[1].popover("?", use_container_width=True):
            st.markdown(scorecard_help)

        score_msg = f"귀하의 프로젝트 상장 적합도는 [ {score:.0f}점 / 100점 ] 입니다. ({grade})"
        if grade == "거절 위험":
            st.sidebar.error(score_msg)
        elif grade == "주의":
            st.sidebar.warning(score_msg)
        else:
            st.sidebar.info(score_msg)

if "mode" not in st.session_state:
    st.session_state["mode"] = "tutorial"
if "tutorial_step" not in st.session_state:
    st.session_state["tutorial_step"] = 0
if "step0_completed" not in st.session_state:
    st.session_state["step0_completed"] = False
if not st.session_state["step0_completed"]:
    st.session_state["tutorial_step"] = 0
prev_mode = st.session_state.get("mode")
is_expert = st.session_state.get("mode") == "expert"
is_tutorial = not is_expert

if st.session_state.get("apply_target_scenario"):
    target_payload = {
        "input_supply": 3.0,
        "input_unbonding": 30,
        "input_sell_ratio": 30,
        "conversion_rate": 0.50,
        "avg_ticket": 100,
        "input_buy_volume": 0,
        "scenario_preset": "Scenario B (현실적)",
        "steps_per_month": 30,
        "turnover_ratio": 5.0,
        "lp_growth_rate": 1.0,
        "max_buy_usdt_ratio": 5.0,
        "max_sell_token_ratio": 5.0
    }
    filtered_payload, recommended_notes = filter_recommended_settings(target_payload)
    st.session_state.update(filtered_payload)
    if recommended_notes:
        st.session_state["recommended_notes"] = recommended_notes
    st.session_state["apply_target_scenario"] = False

if st.session_state.get("apply_reverse_scenario"):
    payload = st.session_state.get("reverse_apply_payload", {})
    if payload:
        filtered_payload, recommended_notes = filter_recommended_settings(payload)
        st.session_state.update(filtered_payload)
        if recommended_notes:
            st.session_state["recommended_notes"] = recommended_notes
    st.session_state["apply_reverse_scenario"] = False

if st.session_state.get("apply_upbit_baseline"):
    krw_rate = st.session_state.get("krw_per_usd", 1300)
    upbit_payload = {
        "input_supply": 45.0,
        "input_unbonding": 14,
        "input_sell_ratio": 15,
        "input_buy_volume": int(3_500_000_000 / max(krw_rate, 1)),
        "scenario_preset": "직접 입력",
    }
    filtered_payload, recommended_notes = filter_recommended_settings(upbit_payload)
    st.session_state.update(filtered_payload)
    if recommended_notes:
        st.session_state["recommended_notes"] = recommended_notes
    st.session_state["apply_upbit_baseline"] = False


st.sidebar.markdown("---")
total_steps = 6
current_step = int(st.session_state.get("tutorial_step", 0))
current_step = max(0, min(total_steps - 1, current_step))
if not st.session_state.get("step0_completed"):
    current_step = 0
st.session_state["tutorial_step"] = current_step
step0_preview = current_step == 0

if current_step == 0:
    st.session_state["step0_completed"] = False
    st.sidebar.subheader("📝 Step 0. 프로젝트 기본 정보")
    st.sidebar.info(
        "상장 심사에서 가장 먼저 보는 기본 요건입니다. "
        "정량(유통/언락)과 정성(Audit/법률/백서)을 먼저 체크합니다."
    )
    st.sidebar.caption("필수 서류가 미준비면 심사 접수 자체가 불가능합니다.")
    st.sidebar.button(
        "⏭️ Step 0 건너뛰기",
        on_click=lambda: (
            st.session_state.__setitem__("step0_completed", True),
            st.session_state.__setitem__("tutorial_step", 1)
        )
    )
    if st.sidebar.button("다음 ➡"):
        st.session_state["step0_completed"] = True
        st.session_state["tutorial_step"] = 1
        st.rerun()
else:
    mode = st.sidebar.radio(
        "모드 선택",
        options=["초보자", "전문가"],
        index=0 if st.session_state.get("mode") == "tutorial" else 1,
        help="초보자는 핵심 7개만, 전문가는 상세 설정까지 봅니다.",
        key="mode_selector"
    )
    st.session_state["mode"] = "tutorial" if mode == "초보자" else "expert"
    is_expert = st.session_state["mode"] == "expert"
    is_tutorial = not is_expert

    st.sidebar.progress((current_step + 1) / total_steps)
    st.sidebar.caption(f"Step {current_step} / {total_steps - 1}")

    if is_tutorial:
        st.sidebar.info(
            "🔰 초보자 모드 시작 안내\n"
            "- 진행 순서: 프로젝트 기본 → 목표 → 공급 → 수요 → 시장 → 방어\n"
            "- 핵심 7개만 설정: 목표 가격, 계약 시나리오, 초기 유통량, "
            "언본딩 기간, 전환율, 평균 매수액, 월간 바이백 예산\n"
            "- 나머지(오더북/회전율/캡/심리 등)는 안정적인 기본값으로 자동 적용됩니다."
        )

        if current_step == 1:
            st.sidebar.subheader("🎯 Step 1. 목표 설정 & 시나리오")
            st.sidebar.info(
                "시뮬레이션의 기준을 정합니다. 목표가가 높을수록 "
                "공급 통제(유통량/언본딩)와 수요 견인(전환율/객단가)이 더 중요해집니다."
            )
            contract_mode_label = st.sidebar.selectbox(
                "시나리오 모드 선택",
                ["사용자 조정", "목표가 조정"],
                index=0,
                key="contract_mode_label"
            )
            st.session_state["contract_mode"] = "사용자 조정"
            if contract_mode_label == "사용자 조정":
                st.sidebar.info("ℹ️ 가이드: 각 설정값을 사용자가 직접 정하면, 실시간으로 AI가 그에 따른 결과값을 계산하여 보여줍니다.")

            st.sidebar.markdown("---")
            if contract_mode_label == "목표가 조정":
                target_price = st.sidebar.number_input(
                    "목표가 조정 ($)",
                    value=float(st.session_state.get("tutorial_target_price", 0.0)),
                    step=0.5,
                    key="tutorial_target_price",
                    help="목표가격이란 사용자가 자동으로 올리고 싶은 가격대를 선택하면, AI 가 각 설정값(유입량, 공급제한 등)의 필요값을 도출하여 보여드리는 시스템입니다."
                )
                if st.sidebar.button("🪄 조정 (AI 최적화 실행)"):
                    with st.spinner("AI가 최적 시나리오를 연산 중입니다..."):
                        time.sleep(1.0)
                    required_inflow_base = 200_000
                    multiplier = max(target_price / 0.5, 0.1)
                    st.session_state["input_buy_volume"] = required_inflow_base * multiplier * 0.5
                    st.session_state["input_supply"] = 1.0
                    st.session_state["input_unbonding"] = 60
                    st.session_state["input_sell_ratio"] = 20
                    simulation_unit = st.session_state.get("simulation_unit", "월")
                    simulation_value = int(st.session_state.get("simulation_value", 1))
                    if simulation_unit == "일":
                        period_months = max(1, int(math.ceil(simulation_value / 30)))
                    elif simulation_unit == "년":
                        period_months = max(1, simulation_value * 12)
                    else:
                        period_months = max(1, simulation_value)
                    guide_msg = generate_strategy_guide(
                        current_price=0.5,
                        target_price=target_price,
                        period_months=period_months,
                        suggested_inflow=st.session_state["input_buy_volume"],
                        suggested_supply=st.session_state["input_supply"]
                    )
                    st.session_state["ai_strategy_report"] = guide_msg
                    st.session_state["ai_tune_banner_ts"] = time.time()

                st.sidebar.caption(f"현재 시뮬레이션 목표: **${target_price:.2f}**")
        elif current_step == 2:
            st.sidebar.subheader("📉 Step 2. 공급 제한 (Risk 관리)")
            st.sidebar.info(
                "시장에 풀리는 물량을 제한해야 가격을 방어할 수 있습니다. "
                "초기 유통량 3% 이하 + 언본딩 지연이 핵심입니다."
            )
            input_supply = st.sidebar.slider(
                "초기 유통량 (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.get("input_supply", 3.0)),
                step=0.5,
                key="input_supply",
                help="초기 유통량이 높을수록 가격 방어가 어려워집니다."
            )
            if input_supply > 3.0:
                st.sidebar.error("🚨 법적 리스크 발생: 초기 유통량은 3%를 초과할 수 없습니다.")
            input_unbonding = st.sidebar.slider(
                "언본딩 기간 (일)",
                min_value=0,
                max_value=60,
                value=int(st.session_state.get("input_unbonding", 30)),
                step=5,
                key="input_unbonding",
                help="언본딩 기간이 길수록 매도 지연 효과가 큽니다."
            )
            input_sell_ratio = st.session_state.get("input_sell_ratio", 30)
        elif current_step == 3:
            st.sidebar.subheader("📈 Step 3. 수요 견인 (Growth)")
            st.sidebar.info(
                "유입 전환율과 객단가가 월간 매수 파워를 결정합니다. "
                "기본 매수 유입은 튜토리얼에서 자동값을 사용합니다."
            )
            conversion_rate = st.sidebar.slider(
                "거래소 유입 전환율 (%)",
                min_value=0.01,
                max_value=2.00,
                value=float(st.session_state.get("conversion_rate", 0.10)),
                step=0.01,
                format="%.2f%%",
                key="conversion_rate",
                help="기존 회원 중 실제 매수로 전환되는 비율입니다."
            )
            avg_ticket = st.sidebar.number_input(
                "1인당 평균 매수액 ($)",
                value=float(st.session_state.get("avg_ticket", 100.0)),
                step=10.0,
                key="avg_ticket",
                help="회원 1명이 평균적으로 매수하는 금액입니다."
            )
            estv_total_users = 160_000_000
            calculated_inflow = (estv_total_users * (conversion_rate / 100.0) * avg_ticket) / 12.0
            st.sidebar.metric("월간 매수 파워", f"${calculated_inflow:,.0f}")
        elif current_step == 4:
            st.sidebar.subheader("🏗️ Step 4. 시장 깊이 (Volatility)")
            st.sidebar.info(
                "오더북이 얇으면 작은 매도에도 가격이 크게 흔들립니다. "
                "튜토리얼에서는 오더북 체력을 기본값(보통)으로 자동 설정합니다."
            )
            st.sidebar.caption("전문가 모드에서 오더북 깊이를 직접 조정할 수 있습니다.")
        else:
            st.sidebar.subheader("🛡️ Step 5. 방어 정책 및 실행")
            st.sidebar.info(
                "급락 시 사용할 바이백 예산을 설정합니다. "
                "소각 수수료율 등 세부 정책은 기본값으로 자동 적용됩니다."
            )
            monthly_buyback_usdt = st.sidebar.number_input(
                "월간 바이백 예산($)",
                value=int(st.session_state.get("monthly_buyback_usdt", 0)),
                step=100000,
                key="monthly_buyback_usdt",
                help="시장 방어를 위한 월간 바이백 예산입니다."
            )
            if st.sidebar.button(RUN_SIM_BUTTON_LABEL):
                st.session_state["simulation_active"] = True
                st.session_state["simulation_active_requested"] = True
                st.session_state["simulation_active_force"] = True
                st.session_state["loaded_result"] = None
                st.session_state["loaded_inputs"] = None
                st.rerun()

    nav_cols = st.sidebar.columns(2)
    with nav_cols[0]:
        if st.button("⬅ 이전"):
            st.session_state["tutorial_step"] = max(0, current_step - 1)
            st.rerun()
    with nav_cols[1]:
        if st.button("다음 ➡", disabled=current_step == total_steps - 1):
            st.session_state["tutorial_step"] = current_step + 1
            st.rerun()

    # Tutorial defaults for hidden fields
    contract_mode = st.session_state.get("contract_mode", "사용자 조정")
    input_supply = st.session_state.get("input_supply", 3.0)
    input_unbonding = st.session_state.get("input_unbonding", 30)
    input_sell_ratio = st.session_state.get("input_sell_ratio", 30)
    input_buy_volume = st.session_state.get("input_buy_volume", 200000)
    conversion_rate = st.session_state.get("conversion_rate", 0.10)
    avg_ticket = st.session_state.get("avg_ticket", 100.0)
    simulation_unit = "월"
    simulation_value = 1
    total_days = 30
    simulation_months = 1
    onboarding_months = 12
    krw_rate = 1300
    use_buy_inflow_pattern = False
    base_daily_buy_schedule = []
    total_new_buyers = 160_000_000 * (conversion_rate / 100.0)
    total_inflow_money = total_new_buyers * avg_ticket
    monthly_user_buy_volume = total_inflow_money / onboarding_months
    total_inflow_days = onboarding_months * 30
    base_daily_user_buy = total_inflow_money / max(total_inflow_days, 1)
    selected_type = st.session_state.get("project_type", "New Listing (신규 상장)")
    ref_data = COIN_TYPE_VOLATILITY.get(selected_type, COIN_TYPE_VOLATILITY["New Listing (신규 상장)"])
    schedule_volatility = float(st.session_state.get("volume_volatility", ref_data["default"]))
    schedule_weekend = bool(st.session_state.get("weekend_dip", True))
    monthly_user_target = total_new_buyers / max(onboarding_months, 1)
    daily_user_buy_schedule = create_realistic_schedule(
        monthly_user_target,
        onboarding_months,
        simulation_months,
        avg_ticket,
        schedule_volatility,
        schedule_weekend
    )[:total_days]
    use_phase_inflow = False
    phase2_days = 30
    prelisting_days = 30
    prelisting_release_days = 7
    market_depth_level = st.session_state.get("market_depth_level", "보통")
    depth_map = {
        "약함": (300_000, 800_000),
        "보통": (1_000_000, 3_000_000),
        "강함": (3_000_000, 9_000_000)
    }
    depth_usdt_1pct, depth_usdt_2pct = depth_map[market_depth_level]
    price_model = "CEX"
    depth_growth_rate = 0.0
    steps_per_month = 30
    turnover_ratio = 5.0
    turnover_buy_share = 50.0
    lp_growth_rate = 1.0
    max_buy_usdt_ratio = 5.0
    max_sell_token_ratio = 5.0
    use_master_plan = False
    use_triggers = False
    buy_verify_boost = 0.5
    holding_suppress = 0.1
    payburn_delta = 0.002
    buyback_daily = 0.0
    monthly_buyback_usdt = st.session_state.get("monthly_buyback_usdt", 0)
    burn_fee_rate = st.session_state.get("burn_fee_rate", 0.3)
    panic_sensitivity = 1.5
    fomo_sensitivity = 1.2
    private_sale_price = 0.05
    profit_taking_multiple = 5.0
    arbitrage_threshold = 2.0
    min_depth_ratio = 0.3
    market_sentiment_config = {
        "panic_sensitivity": panic_sensitivity,
        "fomo_sensitivity": fomo_sensitivity,
        "private_sale_price": private_sale_price,
        "profit_taking_multiple": profit_taking_multiple,
        "arbitrage_threshold": arbitrage_threshold / 100.0,
        "min_depth_ratio": min_depth_ratio
    }
    initial_investor_lock_months = 0
    initial_investor_locked_tokens = 0.0
    initial_investor_vesting_months = 0
    initial_investor_release_percent = 10.0
    initial_investor_release_interval = 1
    initial_investor_sell_ratio = 0.0
    derived_vesting_months = 1
    initial_investor_locked_percent = 0.0
    campaigns = []
    triggers = []
    enable_confidence = False
    show_upbit_baseline = False
    krw_per_usd = 1300
if step0_preview:
    contract_mode = st.session_state.get("contract_mode", "사용자 조정")
    input_supply = st.session_state.get("input_supply", 3.0)
    input_unbonding = st.session_state.get("input_unbonding", 30)
    input_sell_ratio = st.session_state.get("input_sell_ratio", 30)
    input_buy_volume = st.session_state.get("input_buy_volume", 200000)
    conversion_rate = st.session_state.get("conversion_rate", 0.10)
    avg_ticket = st.session_state.get("avg_ticket", 100.0)
    simulation_unit = "월"
    simulation_value = 1
    total_days = 30
    simulation_months = 1
    onboarding_months = 12
    krw_rate = 1300
    use_buy_inflow_pattern = False
    base_daily_buy_schedule = []
    total_new_buyers = 160_000_000 * (conversion_rate / 100.0)
    total_inflow_money = total_new_buyers * avg_ticket
    monthly_user_buy_volume = total_inflow_money / onboarding_months
    total_inflow_days = onboarding_months * 30
    base_daily_user_buy = total_inflow_money / max(total_inflow_days, 1)
    selected_type = st.session_state.get("project_type", "New Listing (신규 상장)")
    ref_data = COIN_TYPE_VOLATILITY.get(selected_type, COIN_TYPE_VOLATILITY["New Listing (신규 상장)"])
    schedule_volatility = float(st.session_state.get("volume_volatility", ref_data["default"]))
    schedule_weekend = bool(st.session_state.get("weekend_dip", True))
    monthly_user_target = total_new_buyers / max(onboarding_months, 1)
    daily_user_buy_schedule = create_realistic_schedule(
        monthly_user_target,
        onboarding_months,
        simulation_months,
        avg_ticket,
        schedule_volatility,
        schedule_weekend
    )[:total_days]
    use_phase_inflow = False
    phase2_days = 30
    prelisting_days = 30
    prelisting_release_days = 7
    market_depth_level = st.session_state.get("market_depth_level", "보통")
    depth_map = {
        "약함": (300_000, 800_000),
        "보통": (1_000_000, 3_000_000),
        "강함": (3_000_000, 9_000_000)
    }
    depth_usdt_1pct, depth_usdt_2pct = depth_map[market_depth_level]
    price_model = "CEX"
    depth_growth_rate = 0.0
    steps_per_month = 30
    turnover_ratio = 5.0
    turnover_buy_share = 50.0
    lp_growth_rate = 1.0
    max_buy_usdt_ratio = 5.0
    max_sell_token_ratio = 5.0
    use_master_plan = False
    use_triggers = False
    buy_verify_boost = 0.5
    holding_suppress = 0.1
    payburn_delta = 0.002
    buyback_daily = 0.0
    monthly_buyback_usdt = st.session_state.get("monthly_buyback_usdt", 0)
    burn_fee_rate = st.session_state.get("burn_fee_rate", 0.3)
    panic_sensitivity = 1.5
    fomo_sensitivity = 1.2
    private_sale_price = 0.05
    profit_taking_multiple = 5.0
    arbitrage_threshold = 2.0
    min_depth_ratio = 0.3
    market_sentiment_config = {
        "panic_sensitivity": panic_sensitivity,
        "fomo_sensitivity": fomo_sensitivity,
        "private_sale_price": private_sale_price,
        "profit_taking_multiple": profit_taking_multiple,
        "arbitrage_threshold": arbitrage_threshold / 100.0,
        "min_depth_ratio": min_depth_ratio
    }
    initial_investor_lock_months = 0
    initial_investor_locked_tokens = 0.0
    initial_investor_vesting_months = 0
    initial_investor_release_percent = 10.0
    initial_investor_release_interval = 1
    initial_investor_sell_ratio = 0.0
    derived_vesting_months = 1
    initial_investor_locked_percent = 0.0
    campaigns = []
    triggers = []
    enable_confidence = False
    show_upbit_baseline = False
    krw_per_usd = 1300
if is_expert and current_step > 0:
    st.sidebar.info(
        "⚙️ 전문가 모드 안내\n"
        "- 모든 변수를 직접 조정합니다.\n"
        "- 공급/수요/시장 구조/방어 정책/분석 도구까지 세부 튜닝 가능합니다."
    )
    st.sidebar.subheader("🎯 Step 1. 목표 설정 & 시나리오")
    contract_mode_label = st.sidebar.selectbox(
        "시나리오 모드 선택",
        ["사용자 조정", "목표가 조정"],
        index=0,
        key="contract_mode_label",
        help="시뮬레이션 방식을 먼저 선택합니다."
    )
    st.session_state["contract_mode"] = "사용자 조정"
    if contract_mode_label == "사용자 조정":
        st.sidebar.info("ℹ️ 가이드: 각 설정값을 사용자가 직접 정하면, 실시간으로 AI가 그에 따른 결과값을 계산하여 보여줍니다.")

    st.sidebar.markdown("---")
    if contract_mode_label == "목표가 조정":
        target_price = st.sidebar.number_input(
            "목표가 조정 ($)",
            value=float(st.session_state.get("tutorial_target_price", 0.0)),
            step=0.5,
            key="tutorial_target_price",
            help="목표가격이란 사용자가 자동으로 올리고 싶은 가격대를 선택하면, AI 가 각 설정값(유입량, 공급제한 등)의 필요값을 도출하여 보여드리는 시스템입니다."
        )
        if st.sidebar.button("🪄 조정 (AI 최적화 실행)"):
            with st.spinner("AI가 최적 시나리오를 연산 중입니다..."):
                time.sleep(1.0)
            required_inflow_base = 200_000
            multiplier = max(target_price / 0.5, 0.1)
            st.session_state["input_buy_volume"] = required_inflow_base * multiplier * 0.5
            st.session_state["input_supply"] = 1.0
            st.session_state["input_unbonding"] = 60
            st.session_state["input_sell_ratio"] = 20
            simulation_unit = st.session_state.get("simulation_unit", "월")
            simulation_value = int(st.session_state.get("simulation_value", 1))
            if simulation_unit == "일":
                period_months = max(1, int(math.ceil(simulation_value / 30)))
            elif simulation_unit == "년":
                period_months = max(1, simulation_value * 12)
            else:
                period_months = max(1, simulation_value)
            guide_msg = generate_strategy_guide(
                current_price=0.5,
                target_price=target_price,
                period_months=period_months,
                suggested_inflow=st.session_state["input_buy_volume"],
                suggested_supply=st.session_state["input_supply"]
            )
            st.session_state["ai_strategy_report"] = guide_msg
            st.session_state["ai_tune_banner_ts"] = time.time()

        st.sidebar.caption(f"현재 시뮬레이션 목표: **${target_price:.2f}**")

    st.sidebar.subheader("🎯 $5.00 달성 목표 시나리오")
    with st.sidebar.expander("시나리오 설명", expanded=is_expert):
        st.markdown("""
    - 공급 통제: 초기 유통량 3.0%, 언본딩 30일, 매도율 30%
    - 수요 폭발: 1.6억명 × 0.5% 전환율 × $100 = 월 $6.6M 유입
    - 리스크 제거: 마케팅 덤핑 시나리오 비활성화
    """)
    with st.sidebar.expander("KPI 체크리스트 & 예상 흐름", expanded=is_expert):
        st.markdown("""
    **2. 조건별 달성 목표 (KPI Checklist)**  
    시뮬레이션 결과가 현실이 되기 위한 실제 KPI입니다.
    """)

    def apply_target_scenario():
        st.session_state["apply_target_scenario"] = True

    st.sidebar.button("목표 시나리오 적용", on_click=apply_target_scenario)

    st.sidebar.markdown("---")
    st.sidebar.header("⚖️ 펀더멘탈: 공급과 수요")

    st.sidebar.subheader("📉 공급 부담(매도 리스크)")
    input_supply = st.sidebar.slider(
        "1. 초기 유통량 (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state.get("input_supply", 3.0)),
        step=0.5,
        help="초기 유통되는 토큰 비율입니다. 높을수록 시장 유통 물량이 많아져 가격 방어가 어려울 수 있습니다.",
        key="input_supply"
    )
    if input_supply > 3.0:
        st.sidebar.error("🚨 특약 제5조 위반! (3% 초과)")

    supply_expander = st.sidebar.expander("📉 공급 상세 (언본딩/매도율)", expanded=is_expert)
    input_unbonding = supply_expander.slider(
        "2. 언본딩 기간 (일)",
        min_value=0,
        max_value=90,
        value=30,
        step=10,
        help="언본딩 대기 기간입니다. 길수록 매도 지연이 커져 단기 하락 압력이 완화됩니다.",
        key="input_unbonding"
    )
    if input_unbonding < 30:
        supply_expander.warning("⚠️ 특약 권장 사항 미달 (<30일)")

    input_sell_ratio = supply_expander.slider(
        "3. 락업 해제 시 매도율 (%)",
        10,
        100,
        50,
        help="락업 해제 물량 중 실제로 매도되는 비율입니다. 높을수록 가격 하방 압력이 커집니다.",
        key="input_sell_ratio"
    )

    investor_expander = st.sidebar.expander("🔒 초기 투자자 상세 베스팅", expanded=is_expert)
    initial_investor_lock_months = investor_expander.slider(
        "3-1. 초기 투자자 락업 기간 (개월)",
        min_value=0,
        max_value=60,
        value=12,
        step=1,
        help="초기 투자자 물량이 시장에 풀리기 전까지 묶이는 기간입니다.",
        key="initial_investor_lock_months"
    )
    initial_investor_locked_tokens = investor_expander.number_input(
        "3-2. 락업 물량 (토큰 수)",
        min_value=0.0,
        value=0.0,
        step=1_000_000.0,
        help="초기 투자자에게 배정된 락업 토큰 수량입니다. 0이면 미적용됩니다.",
        key="initial_investor_locked_tokens"
    )
    initial_investor_vesting_months = investor_expander.slider(
        "3-3. 베스팅 기간 (개월)",
        min_value=0,
        max_value=60,
        value=12,
        step=1,
        help="락업 종료 후 몇 개월에 걸쳐 해제할지 선택합니다.",
        key="initial_investor_vesting_months"
    )
    initial_investor_release_percent = investor_expander.slider(
        "3-4. 월별 해제 비율 (%)",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        help="락업 물량 중 매월 해제되는 비율입니다. 설정값에 따라 실제 베스팅 기간이 자동 보정됩니다.",
        key="initial_investor_release_percent"
    )
    initial_investor_release_interval = investor_expander.slider(
        "3-5. 해제 주기 (개월)",
        min_value=1,
        max_value=12,
        value=1,
        step=1,
        help="해제 주기를 설정합니다. 예: 3개월이면 분기 단위로 해제됩니다.",
        key="initial_investor_release_interval"
    )
    initial_investor_sell_ratio = investor_expander.slider(
        "3-6. 초기 투자자 해제 매도율 (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        help="언락된 물량 중 실제로 매도되는 비율입니다. 공격적일수록 높게 설정하세요.",
        key="initial_investor_sell_ratio"
    )

    TOTAL_SUPPLY = float(total_supply_input)
    initial_investor_locked_percent = (initial_investor_locked_tokens / TOTAL_SUPPLY) * 100.0 if initial_investor_locked_tokens > 0 else 0.0
    if initial_investor_locked_percent > 100.0:
        investor_expander.error("락업 물량이 총 공급량을 초과했습니다.")

    derived_vesting_months = max(1, int(math.ceil(100.0 / max(initial_investor_release_percent, 1.0))))
    if initial_investor_vesting_months > 0 and initial_investor_vesting_months != derived_vesting_months:
        investor_expander.info(f"월별 해제 비율 기준으로 베스팅 기간이 {derived_vesting_months}개월로 보정됩니다.")
    if initial_investor_locked_tokens > 0:
        estimated_lock_value = initial_investor_locked_tokens * 0.50
        vesting_months_used = derived_vesting_months if initial_investor_vesting_months > 0 else 1
        safe_months = max(1, vesting_months_used)
        monthly_unlock_theoretical = initial_investor_locked_tokens / safe_months
        final_monthly_sell = monthly_unlock_theoretical * (initial_investor_sell_ratio / 100.0)
        st.session_state["calculated_monthly_sell_pressure"] = final_monthly_sell

        investor_expander.markdown("---")
        investor_expander.subheader("📉 매도 압력 자동 산출 (Auto-Calculated)")
        c1, c2 = investor_expander.columns(2)
        c1.metric(
            label="월간 언락 물량 (Max)",
            value=f"{monthly_unlock_theoretical:,.0f} 개",
            help="베스팅 스케줄에 따라 매월 풀리는 최대 물량입니다."
        )
        c2.metric(
            label="실제 예상 매도 압력",
            value=f"{final_monthly_sell:,.0f} 개",
            delta=f"매도율 {initial_investor_sell_ratio:.0f}% 적용",
            delta_color="inverse",
            help="시뮬레이션에 반영되는 월간 매도 수량입니다."
        )
        investor_expander.caption(
            f"락업 물량: {int(initial_investor_locked_tokens):,}개 "
            f"(총 공급의 {initial_investor_locked_percent:.2f}%) / "
            f"예상 평가액: ${estimated_lock_value:,.0f}"
        )

    st.sidebar.subheader("📈 수요 힘(매수 유입)")
    input_buy_volume = st.sidebar.number_input(
        "4. 월간 매수 유입 자금 ($)",
        value=200000,
        step=50000,
        help="월간 기본 매수 유입 자금입니다. 클수록 매수 압력이 증가해 가격 상승 요인이 됩니다.",
        key="input_buy_volume"
    )
    inflow_expander = st.sidebar.expander("📌 유입 상세(전환율/패턴/기간)", expanded=is_expert)
    use_buy_inflow_pattern = inflow_expander.checkbox(
        "월간 매수 유입 시계열 패턴 사용",
        value=False,
        help="월별 매수 유입을 패턴(초기 급증→조정→안정)으로 반영합니다.",
        key="use_buy_inflow_pattern"
    )
    pattern_month4_avg_krw = inflow_expander.slider(
        "월 4+ 평균 유입(억 KRW)",
        min_value=40,
        max_value=60,
        value=50,
        step=5,
        help="월 4 이후 장기 평균 유입 규모(억 원)입니다.",
        key="pattern_month4_avg_krw"
    )
    simulation_unit = inflow_expander.selectbox(
        "4-1. 시뮬레이션 기간 단위",
        options=["일", "월", "년"],
        index=1,
        help="기간 단위를 선택합니다. 월 단위는 30일 기준으로 환산됩니다.",
        key="simulation_unit"
    )
    simulation_value = inflow_expander.number_input(
        "4-2. 시뮬레이션 기간 값",
        min_value=1,
        value=1 if simulation_unit == "월" else 30,
        step=1,
        help="선택한 단위에 맞는 기간 값을 입력합니다.",
        key="simulation_value"
    )
    if simulation_unit == "일":
        total_days = simulation_value
    elif simulation_unit == "년":
        total_days = simulation_value * 365
    else:
        total_days = simulation_value * 30
    simulation_months = max(1, int(math.ceil(total_days / 30)))

    krw_rate = st.session_state.get("krw_per_usd", 1300)
    base_daily_buy_schedule = []
    if use_buy_inflow_pattern:
        monthly_krw_series = [
            30_000_000_000,
            15_000_000_000,
            8_000_000_000
        ]
        month4_krw = pattern_month4_avg_krw * 100_000_000
        total_months = max(1, int(math.ceil(total_days / 30)))
        while len(monthly_krw_series) < total_months:
            monthly_krw_series.append(month4_krw)
        for day in range(total_days):
            month_idx = min(day // 30, len(monthly_krw_series) - 1)
            monthly_usd = monthly_krw_series[month_idx] / max(krw_rate, 1)
            base_daily_buy_schedule.append(monthly_usd / 30.0)

    inflow_expander.markdown("---")
    inflow_expander.subheader("👥 기존 회원 유입 (Demand Side)")
    estv_total_users = 160_000_000
    inflow_expander.caption("기존 회원 수는 보수적으로 1억 6천만 명 기준을 사용합니다.")
    inflow_help = inflow_expander.expander("ℹ️ 유입 시나리오 도움말", expanded=is_expert)
    inflow_help.markdown("""
**1억 6천만명 유입 퍼널**
1. 인지(Awareness): 플랫폼 토큰 상장 인지 (약 30~50%)
2. 관심(Interest): 관심을 갖는 비율 (약 10~20%)
3. 행동(Action - KYC): 계좌 개설/인증까지 도달 (약 5%)
4. 구매(Purchase): 실제 매수 전환 (최종 타깃)

**시나리오별 추천 값(월간 매수 압력 추정)**
| 시나리오 | 전환율 | 1인당 매수액 | 특징 |
|---|---:|---:|---|
| A (보수적) | 0.05% | $50 | 유기적 유입 |
| B (현실적) | 0.50% | $100 | 기본값 권장 |
| C (공격적) | 2.00% | $200 | 공격적 캠페인 |
""")

    preset_map = {
        "직접 입력": None,
        "Scenario A (보수적)": {"conversion_rate": 0.05, "avg_ticket": 50},
        "Scenario B (현실적)": {"conversion_rate": 0.50, "avg_ticket": 100},
        "Scenario C (공격적)": {"conversion_rate": 2.00, "avg_ticket": 200},
    }
    def apply_preset():
        preset = st.session_state.get("scenario_preset", "직접 입력")
        if preset_map.get(preset):
            st.session_state["conversion_rate"] = preset_map[preset]["conversion_rate"]
            st.session_state["avg_ticket"] = preset_map[preset]["avg_ticket"]

    scenario_preset = inflow_expander.selectbox(
        "시나리오 프리셋",
        options=list(preset_map.keys()),
        index=0,
        key="scenario_preset",
        on_change=apply_preset,
        help="전환율/객단가를 빠르게 설정하는 프리셋입니다."
    )

    conversion_rate = inflow_expander.slider(
        "5. 회원 거래소 유입 전환율 (%)",
        min_value=0.01,
        max_value=2.00,
        value=0.10,
        step=0.01,
        format="%.2f%%",
        key="conversion_rate",
        help="기존 회원 중 거래소로 유입되는 비율입니다. 높을수록 신규 유입 매수 자금이 커집니다."
    )

    avg_ticket = inflow_expander.number_input(
        "6. 1인당 평균 매수 금액 ($)",
        value=50,
        step=10,
        key="avg_ticket",
        help="신규 유입 1인당 평균 매수 금액입니다. 클수록 월간 추가 매수세가 증가합니다."
    )

    enable_dual_pipeline = inflow_expander.checkbox(
        "듀얼 파이프라인 유입 사용",
        value=False,
        key="enable_dual_pipeline",
        help="기존 회원/신규 회원 유입을 서로 다른 속도로 선형 증가시키는 방식입니다."
    )
    migration_target = 50_000
    migration_ramp_months = 3
    acquisition_target = 10_000
    acquisition_ramp_months = 12
    if enable_dual_pipeline:
        migration_target = inflow_expander.number_input(
            "기존 회원 목표(명/월)",
            min_value=0,
            value=50_000,
            step=1000,
            key="migration_target",
            help="기존 회원 유입 목표치를 월 기준으로 설정합니다."
        )
        migration_ramp_months = inflow_expander.slider(
            "기존 회원 도달 기간(개월)",
            min_value=1,
            max_value=12,
            value=3,
            step=1,
            key="migration_ramp_months",
            help="기존 회원 유입 목표에 도달하는 기간입니다."
        )
        acquisition_target = inflow_expander.number_input(
            "신규 회원 목표(명/월)",
            min_value=0,
            value=10_000,
            step=1000,
            key="acquisition_target",
            help="신규 회원 유입 목표치를 월 기준으로 설정합니다."
        )
        acquisition_ramp_months = inflow_expander.slider(
            "신규 회원 도달 기간(개월)",
            min_value=1,
            max_value=24,
            value=12,
            step=1,
            key="acquisition_ramp_months",
            help="신규 회원 유입 목표에 도달하는 기간입니다."
        )

    onboarding_months = 12
    total_new_buyers = estv_total_users * (conversion_rate / 100.0)
    total_inflow_money = total_new_buyers * avg_ticket
    monthly_user_buy_volume = total_inflow_money / onboarding_months
    total_inflow_days = onboarding_months * 30
    base_daily_user_buy = total_inflow_money / max(total_inflow_days, 1)

    use_phase_inflow = inflow_expander.checkbox(
        "유입 스케줄(Phase) 적용",
        value=False,
        help="Master MD의 Phase 흐름을 반영해 초기 30일 유입을 강화합니다.",
        key="use_phase_inflow"
    )
    phase2_days = 30
    phase2_multiplier = 2.0
    prelisting_days = 30
    prelisting_multiplier = 1.5
    prelisting_release_days = 7
    if use_phase_inflow:
        phase2_days = inflow_expander.slider(
            "Phase 2 기간(일)",
            min_value=7,
            max_value=60,
            value=30,
            step=1,
            key="phase2_days",
            help="상장 직후 집중 유입이 유지되는 기간입니다."
        )
        phase2_multiplier = inflow_expander.slider(
            "Phase 2 유입 배수",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.1,
            key="phase2_multiplier",
            help="상장 직후 유입을 몇 배로 증폭할지 설정합니다."
        )
        prelisting_days = inflow_expander.slider(
            "Phase 1 대기 기간(일)",
            min_value=7,
            max_value=60,
            value=30,
            step=1,
            key="prelisting_days",
            help="상장 전 유입이 대기(잠재 수요로 누적)되는 기간입니다."
        )
        prelisting_multiplier = inflow_expander.slider(
            "Phase 1 대기 수요 배수",
            min_value=1.0,
            max_value=5.0,
            value=1.5,
            step=0.1,
            key="prelisting_multiplier",
            help="대기 수요가 상장 직후 유입될 때의 증폭 정도입니다."
        )
        prelisting_release_days = inflow_expander.slider(
            "Phase 1 방출 기간(일)",
            min_value=1,
            max_value=30,
            value=7,
            step=1,
            key="prelisting_release_days",
            help="대기 수요가 상장 후 며칠에 걸쳐 분산 방출되는지 설정합니다."
        )

    total_sim_months = simulation_months
    if enable_dual_pipeline:
        schedule_volatility = float(st.session_state.get("volume_volatility", 0.5))
        schedule_weekend = bool(st.session_state.get("weekend_dip", True))
        schedule_migration = create_realistic_schedule(
            migration_target,
            migration_ramp_months,
            total_sim_months,
            avg_ticket,
            schedule_volatility,
            schedule_weekend
        )
        schedule_acquisition = create_realistic_schedule(
            acquisition_target,
            acquisition_ramp_months,
            total_sim_months,
            avg_ticket,
            schedule_volatility,
            schedule_weekend
        )
        final_daily_buy_schedule = [
            a + b for a, b in zip(schedule_migration, schedule_acquisition)
        ]
        daily_user_buy_schedule = final_daily_buy_schedule[:total_days]
        total_inflow_days = max(1, len(daily_user_buy_schedule))
        total_inflow_money = float(sum(daily_user_buy_schedule))
        monthly_user_buy_volume = float(sum(daily_user_buy_schedule[:min(30, total_inflow_days)]))
        base_daily_user_buy = monthly_user_buy_volume / 30.0
        use_phase_inflow = False
    else:
        phase2_days = min(phase2_days, total_inflow_days)
        prelisting_days = min(prelisting_days, total_inflow_days)
        prelisting_release_days = max(1, min(prelisting_release_days, total_inflow_days))
        prelisting_daily = base_daily_user_buy * prelisting_multiplier
        prelisting_total = prelisting_daily * prelisting_days
        phase2_daily = base_daily_user_buy * phase2_multiplier
        phase2_total = phase2_daily * phase2_days
        remaining_total = max(total_inflow_money - prelisting_total - phase2_total, 0.0)
        remaining_days = max(total_inflow_days - prelisting_days - phase2_days, 1)
        phase3_daily = remaining_total / remaining_days

        selected_type = st.session_state.get("project_type", "New Listing (신규 상장)")
        ref_data = COIN_TYPE_VOLATILITY.get(selected_type, COIN_TYPE_VOLATILITY["New Listing (신규 상장)"])
        schedule_volatility = float(st.session_state.get("volume_volatility", ref_data["default"]))
        schedule_weekend = bool(st.session_state.get("weekend_dip", True))
        monthly_user_target = total_new_buyers / max(onboarding_months, 1)

        if use_phase_inflow:
            daily_user_buy_schedule = []
            for d in range(total_days):
                if d < total_inflow_days:
                    if d < prelisting_days:
                        daily_user_buy_schedule.append(0.0)
                    elif d < prelisting_days + phase2_days:
                        release_day = d - prelisting_days
                        release_ratio = min((release_day + 1) / prelisting_release_days, 1.0)
                        daily_user_buy_schedule.append(phase2_daily + (prelisting_daily * release_ratio))
                    else:
                        daily_user_buy_schedule.append(phase3_daily)
                else:
                    daily_user_buy_schedule.append(0.0)
        else:
            daily_user_buy_schedule = create_realistic_schedule(
                monthly_user_target,
                onboarding_months,
                simulation_months,
                avg_ticket,
                schedule_volatility,
                schedule_weekend
            )[:total_days]
            total_inflow_days = max(1, len(daily_user_buy_schedule))
            total_inflow_money = float(sum(daily_user_buy_schedule))
            monthly_user_buy_volume = float(sum(daily_user_buy_schedule[:min(30, total_inflow_days)]))
            base_daily_user_buy = monthly_user_buy_volume / 30.0

    if enable_dual_pipeline:
        inflow_expander.info(
            "📊 **유입 분석 결과 (듀얼 파이프라인)**\n"
            f"- 기존 회원 목표: {int(migration_target):,}명/월 (도달 {migration_ramp_months}개월)\n"
            f"- 신규 회원 목표: {int(acquisition_target):,}명/월 (도달 {acquisition_ramp_months}개월)\n"
            f"- **월간 추가 매수세(첫 달 기준): +${int(monthly_user_buy_volume):,}**"
        )
        inflow_expander.caption("듀얼 파이프라인 사용 시 Phase 유입 스케줄은 적용되지 않습니다.")
    else:
        inflow_expander.info(f"""
📊 **유입 분석 결과**
- 신규 유입 인원: {int(total_new_buyers):,}명
- 총 매수 대기 자금: ${int(total_inflow_money):,}
- **월간 추가 매수세: +${int(monthly_user_buy_volume):,}**
""")
    if use_phase_inflow:
        inflow_expander.caption(
            f"Phase 1 대기(상장 전 {prelisting_days}일): 유입 대기 → "
            f"상장 직후 {prelisting_release_days}일 완화 방출 / "
            f"상장 직후 일 ${int(phase2_daily + prelisting_daily):,} 유입 / "
            f"Phase 3 이후: 일 ${int(phase3_daily):,} 유입"
        )

    st.sidebar.markdown("---")
    st.sidebar.header("🏗️ 시장 구조/유동성")
    market_expander = st.sidebar.expander("가격 모델 & 오더북", expanded=is_expert)
    selected_type = st.session_state.get("project_type", "New Listing (신규 상장)")
    ref_data = COIN_TYPE_VOLATILITY.get(selected_type, COIN_TYPE_VOLATILITY["New Listing (신규 상장)"])
    if st.session_state.get("volatility_project_type") != selected_type:
        st.session_state["volume_volatility"] = float(ref_data["default"])
        st.session_state["volatility_project_type"] = selected_type
    volume_volatility = market_expander.slider(
        "📊 거래량 변동성 (Volatility)",
        min_value=0.1,
        max_value=3.0,
        value=float(st.session_state.get("volume_volatility", ref_data["default"])),
        step=0.1,
        help=f"선택하신 '{selected_type}'의 권장 변동성은 {ref_data['range']} 입니다.\n({ref_data['desc']})",
        key="volume_volatility"
    )
    weekend_dip = market_expander.checkbox(
        "주말 거래량 감소 반영",
        value=bool(st.session_state.get("weekend_dip", True)),
        key="weekend_dip",
        help="주말 거래량 감소를 반영해 일시적 수요 약화를 시뮬레이션합니다."
    )
    price_model = market_expander.selectbox(
        "가격 모델",
        options=["AMM", "CEX", "HYBRID"],
        index=0,
        help="AMM은 풀의 상수곱(x*y=k)로 가격을 계산합니다. CEX는 오더북 깊이에 따라 체결 슬리피지를 반영합니다. HYBRID는 CEX 방식에 월별 오더북 깊이 증가를 더해 유동성 확장을 모사합니다.",
        key="price_model"
    )
    depth_usdt_1pct = market_expander.number_input(
        "오더북 1% 깊이($)",
        value=1_000_000,
        step=100_000,
        help="CEX 모델에서 ±1% 구간의 매수/매도 깊이입니다.",
        key="depth_usdt_1pct"
    )
    depth_usdt_2pct = market_expander.number_input(
        "오더북 2% 깊이($)",
        value=3_000_000,
        step=100_000,
        help="CEX 모델에서 ±2% 구간의 매수/매도 깊이입니다.",
        key="depth_usdt_2pct"
    )
    depth_growth_rate = market_expander.slider(
        "오더북 깊이 성장률(월, %)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="HYBRID 모델에서 월별 오더북 깊이 증가율입니다.",
        key="depth_growth_rate"
    )
    steps_per_month = market_expander.selectbox(
        "거래 분할 단위",
        options=[30, 7],
        index=0,
        format_func=lambda x: f"{x}일 분할",
        help="월간 매수/매도를 일/주 단위로 분할해 변동성을 완화합니다.",
        key="steps_per_month"
    )
    turnover_ratio = market_expander.slider(
        "신규 유입 회전율(총합, %)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5,
        help="신규 유입 매수·매도 총 회전율입니다. 비대칭 비율로 매수/매도 분배합니다.",
        key="turnover_ratio"
    )
    turnover_buy_share = market_expander.slider(
        "회전율 매수 비중(%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=5.0,
        help="회전율 중 매수로 반영되는 비중입니다. 나머지는 매도로 반영됩니다.",
        key="turnover_buy_share"
    )
    lp_growth_rate = market_expander.slider(
        "LP 성장률(월 기준, %)",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help=(
            "LP는 Liquidity Pool(유동성 풀)의 약자입니다. "
            "가격이 오를 때 LP에 유입되는 유동성 비율을 뜻합니다. "
            "값이 높을수록 풀의 깊이가 커져 슬리피지가 줄고 급등락이 완화됩니다."
        ),
        key="lp_growth_rate"
    )
    max_buy_usdt_ratio = market_expander.slider(
        "매수 캡(풀 USDT 대비, %)",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help=(
            "풀 USDT는 유동성 풀에 쌓여 있는 USDT 잔액을 뜻합니다. "
            "풀 USDT 대비 1회 매수 상한을 제한하며, 낮을수록 대규모 매수가 분할되어 "
            "가격 급등이 완만해집니다."
        ),
        key="max_buy_usdt_ratio"
    )
    max_sell_token_ratio = market_expander.slider(
        "매도 캡(풀 토큰 대비, %)",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help=(
            "풀 토큰은 유동성 풀에 쌓여 있는 토큰 잔액을 뜻합니다. "
            "풀 토큰 대비 1회 매도 상한을 제한하며, 낮을수록 급격한 덤핑을 제한해 "
            "가격 하락 폭을 줄입니다."
        ),
        key="max_sell_token_ratio"
    )

    st.sidebar.markdown("---")
    st.sidebar.header("🛡️ 방어·부양 정책")
    st.sidebar.subheader("🚀 Master Plan 모드")
    use_master_plan = st.sidebar.checkbox(
        "Master Plan 캠페인 활성화",
        value=False,
        help="Buy & Verify, Holding Challenge, Pay & Burn을 캠페인/트리거로 반영합니다.",
        key="use_master_plan"
    )
    use_triggers = False
    buy_verify_boost = 0.5
    holding_suppress = 0.1
    payburn_delta = 0.002
    buyback_daily = 0.0
    if use_master_plan:
        campaign_expander = st.sidebar.expander("🔥 캠페인 및 트리거 상세", expanded=is_expert)
        use_triggers = campaign_expander.checkbox(
            "트리거 자동 가동",
            value=True,
            key="use_triggers",
            help="가격 하락 시 사전에 정의된 캠페인을 자동 재가동하여 급락을 완화하기 위해 사용합니다."
        )
        buy_verify_boost = campaign_expander.slider(
            "Buy & Verify 매수 증폭(+)",
            0.0,
            2.0,
            0.5,
            0.1,
            key="buy_verify_boost",
            help="매수 유인을 강화해 상장 초반 수요를 끌어올립니다."
        )
        holding_suppress = campaign_expander.slider(
            "Holding 매도 억제(-)",
            0.0,
            0.3,
            0.1,
            0.01,
            key="holding_suppress",
            help="매도 심리를 억제해 단기 급락을 완화합니다."
        )
        payburn_delta = campaign_expander.slider(
            "Pay & Burn 소각 증폭(+)",
            0.0,
            0.01,
            0.002,
            0.001,
            key="payburn_delta",
            help="소각을 강화해 유통량 감소 효과를 높입니다."
        )
        buyback_daily = campaign_expander.number_input(
            "캠페인 일일 바이백($)",
            value=0,
            step=10000,
            key="buyback_daily",
            help="캠페인 기간에 실행하는 일일 바이백 예산입니다."
        )

    st.sidebar.subheader("💰 바이백/소각")
    monthly_buyback_usdt = st.sidebar.number_input(
        "월간 바이백 예산($)",
        value=0,
        step=100000,
        help="광고/NFT/수수료 등 사업 수익으로 토큰을 시장에서 매수해 소각하는 예산입니다.",
        key="monthly_buyback_usdt"
    )
    burn_expander = st.sidebar.expander("🔥 소각 상세", expanded=is_expert)
    burn_fee_rate = burn_expander.slider(
        "거래 수수료 소각률(%)",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state.get("burn_fee_rate", 0.3)),
        step=0.5,
        help="거래 수수료 중 일부를 토큰으로 소각합니다. 높을수록 유통량이 줄어 가격 상승 압력이 생깁니다.",
        key="burn_fee_rate"
    )

    sentiment_expander = st.sidebar.expander("🧠 시장 심리/비선형", expanded=is_expert)
    p_type = st.session_state.get("project_type", "New Listing (신규 상장)")
    defaults = SENTIMENT_DEFAULTS.get(p_type, SENTIMENT_DEFAULTS["New Listing (신규 상장)"])
    if st.session_state.get("sentiment_project_type") != p_type:
        st.session_state["panic_sensitivity"] = defaults["panic"]
        st.session_state["fomo_sensitivity"] = defaults["fomo"]
        st.session_state["sentiment_project_type"] = p_type

    sentiment_cols = sentiment_expander.columns(2)
    panic_sensitivity = sentiment_cols[0].slider(
        "😱 패닉 민감도 (Panic)",
        min_value=0.5,
        max_value=3.0,
        value=float(st.session_state.get("panic_sensitivity", defaults["panic"])),
        step=0.1,
        help="하락장에서 매도세가 증폭되는 정도입니다. 신규 상장은 1.5 이상이 현실적입니다.",
        key="panic_sensitivity"
    )
    fomo_sensitivity = sentiment_cols[1].slider(
        "🤩 FOMO 민감도 (Greed)",
        min_value=0.5,
        max_value=5.0,
        value=float(st.session_state.get("fomo_sensitivity", defaults["fomo"])),
        step=0.1,
        help="상승장에서 추격 매수가 붙는 정도입니다. 밈코인은 3.0 이상까지 치솟습니다.",
        key="fomo_sensitivity"
    )
    private_sale_price = sentiment_expander.number_input(
        "초기 투자자 평단가($)",
        value=0.05,
        step=0.01,
        help="초기 투자자의 평균 매입 단가입니다. 이 가격 이하에서는 매도가 둔화됩니다.",
        key="private_sale_price"
    )
    profit_taking_multiple = sentiment_expander.slider(
        "이익실현 목표 배수",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
        help="초기 투자자가 평단가 대비 몇 배 상승 시 이익실현 매도를 강화할지 설정합니다.",
        key="profit_taking_multiple"
    )
    arbitrage_threshold = sentiment_expander.slider(
        "차익거래 임계값(%)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="가격 변동률이 이 값을 넘으면 차익거래 유입을 가정합니다.",
        format="%.1f%%",
        key="arbitrage_threshold"
    )
    min_depth_ratio = sentiment_expander.slider(
        "패닉 시 오더북 깊이 하한",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("min_depth_ratio", 0.3)),
        step=0.01,
        help="패닉 국면에서 오더북 깊이가 줄어드는 최소 비율입니다.",
        key="min_depth_ratio"
    )

    market_sentiment_config = {
        "panic_sensitivity": panic_sensitivity,
        "fomo_sensitivity": fomo_sensitivity,
        "private_sale_price": private_sale_price,
        "profit_taking_multiple": profit_taking_multiple,
        "arbitrage_threshold": arbitrage_threshold / 100.0,
        "min_depth_ratio": min_depth_ratio
    }

    campaigns = []
    triggers = []

contract_mode = st.session_state.get("contract_mode", "사용자 조정")
use_master_plan = bool(st.session_state.get("use_master_plan", False))

if st.session_state.get("step0_completed", False) and use_master_plan:
    phase2_start = prelisting_days
    phase2_end = min(prelisting_days + phase2_days, total_days)
    campaigns.extend([
        {
            "name": "Buy & Verify",
            "start_day": phase2_start,
            "end_day": phase2_end,
            "buy_multiplier": buy_verify_boost,
            "sell_suppression_delta": 0.0,
            "burn_rate_delta": 0.0,
            "buyback_usdt_delta": 0.0,
            "max_sell_token_ratio_delta": 0.0
        },
        {
            "name": "Holding Challenge",
            "start_day": phase2_start,
            "end_day": phase2_end,
            "buy_multiplier": 0.0,
            "sell_suppression_delta": holding_suppress,
            "burn_rate_delta": 0.0,
            "buyback_usdt_delta": 0.0,
            "max_sell_token_ratio_delta": 0.0
        },
        {
            "name": "Pay & Burn",
            "start_day": phase2_end,
            "end_day": total_days,
            "buy_multiplier": 0.0,
            "sell_suppression_delta": 0.0,
            "burn_rate_delta": payburn_delta,
            "buyback_usdt_delta": buyback_daily,
            "max_sell_token_ratio_delta": 0.0
        }
    ])

    triggers = [
        {
            "name": "D31-Guard: Buy&Verify Season2 Warmup",
            "day_start": 24,
            "duration_days": 14,
            "buy_multiplier": 0.35,
            "sell_suppression_delta": 0.0,
            "burn_rate_delta": 0.0,
            "buyback_usdt_delta": 0.0,
            "max_sell_token_ratio_delta": 0.0
        },
        {
            "name": "D31-Guard: Holding Extension (31~60 Lock-in)",
            "day_start": 27,
            "duration_days": 21,
            "buy_multiplier": 0.0,
            "sell_suppression_delta": 0.12,
            "burn_rate_delta": 0.0,
            "buyback_usdt_delta": 0.0,
            "max_sell_token_ratio_delta": 0.05
        },
        {
            "name": "D31-Guard: Liquidity Buffer",
            "day_start": 29,
            "duration_days": 10,
            "buy_multiplier": 0.0,
            "sell_suppression_delta": 0.0,
            "burn_rate_delta": 0.0,
            "buyback_usdt_delta": 20000,
            "max_sell_token_ratio_delta": 0.0
        },
        {
            "name": "Buy & Verify 재가동",
            "drawdown": 0.20,
            "duration_days": 14,
            "buy_multiplier": 0.3,
            "sell_suppression_delta": 0.0,
            "burn_rate_delta": 0.0,
            "buyback_usdt_delta": buyback_daily,
            "max_sell_token_ratio_delta": 0.0
        },
        {
            "name": "Holding Challenge 시즌2",
            "drawdown": 0.30,
            "duration_days": 14,
            "buy_multiplier": 0.0,
            "sell_suppression_delta": 0.1,
            "burn_rate_delta": 0.0,
            "buyback_usdt_delta": 0.0,
            "max_sell_token_ratio_delta": 0.05
        },
        {
            "name": "Pay & Burn 강화",
            "drawdown": 0.40,
            "duration_days": 30,
            "buy_multiplier": 0.0,
            "sell_suppression_delta": 0.0,
            "burn_rate_delta": 0.003,
            "buyback_usdt_delta": buyback_daily,
            "max_sell_token_ratio_delta": 0.05
        }
    ]

if is_expert and current_step > 0:
    st.sidebar.markdown("---")
    st.sidebar.header("📊 분석/비교")
    st.sidebar.subheader("✅ 가격 변동추이 신뢰도")
    enable_confidence = st.sidebar.checkbox(
        "신뢰도 계산 활성화",
        value=False,
        help="입력값에 불확실성을 부여해 여러 번 시뮬레이션하고, 기준 추이와 유사한 비율을 신뢰도로 계산합니다.",
        key="enable_confidence"
    )
    confidence_runs = st.sidebar.slider(
        "시뮬레이션 횟수",
        min_value=100,
        max_value=1000,
        value=300,
        step=50,
        help="횟수가 많을수록 안정적이지만 계산 시간이 늘어납니다.",
        key="confidence_runs"
    )
    confidence_uncertainty = st.sidebar.slider(
        "입력값 불확실성(±%)",
        min_value=0.0,
        max_value=30.0,
        value=10.0,
        step=1.0,
        help="주요 입력값에 랜덤 변동을 주는 범위입니다.",
        key="confidence_uncertainty"
    )
    confidence_mape = st.sidebar.slider(
        "허용 변동폭(평균 오차, %)",
        min_value=5.0,
        max_value=30.0,
        value=15.0,
        step=1.0,
        help="기준 추이와 평균 오차가 이 값 이하인 시뮬레이션의 비율을 신뢰도로 계산합니다.",
        key="confidence_mape"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🇰🇷 Upbit 평균 시나리오")
    show_upbit_baseline = st.sidebar.checkbox(
        "Upbit 평균 그래프 표시",
        value=False,
        help="한국 주요 거래소의 평균 추정치를 기준으로 그래프를 비교 표시합니다.",
        key="show_upbit_baseline"
    )
    def apply_upbit_baseline_clicked():
        st.session_state["apply_upbit_baseline"] = True

    apply_upbit_baseline = st.sidebar.button("Upbit 평균값 적용", on_click=apply_upbit_baseline_clicked)
    krw_per_usd = st.sidebar.number_input(
        "KRW/USD 환율",
        value=1300,
        step=50,
        help="KRW 기준 월간 유입을 USD로 환산하기 위한 환율입니다.",
        key="krw_per_usd"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 마케팅 대시보드")
    default_dashboard_url = os.getenv("MARKETING_DASHBOARD_URL", "http://localhost:5173")
    dashboard_url = st.sidebar.text_input(
        "대시보드 URL",
        value=default_dashboard_url,
        key="marketing_dashboard_url",
        help="Streamlit Cloud에서는 로컬 주소가 아니라 배포된 URL을 입력해야 합니다."
    )
    st.sidebar.link_button("마케팅 대시보드 열기", dashboard_url)
    if dashboard_url.startswith("http://localhost") or dashboard_url.startswith("http://127.0.0.1"):
        st.sidebar.info("Streamlit Cloud에서는 로컬 주소로 접속할 수 없습니다. 배포된 URL로 변경하세요.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 OpenAI 연동 (선택)")
    openai_api_key = st.sidebar.text_input(
        "🔑 OpenAI API Key 입력 (GPT-4 리포트)",
        type="password",
        key="openai_api_key",
        help="키를 입력하면 AI가 실시간으로 전략 리포트를 작성합니다. (GPT-4, gpt-3.5-turbo 등 지원)"
    )
    st.sidebar.markdown("---")
    apply_btn = st.sidebar.button(
        RUN_SIM_BUTTON_LABEL,
        type="primary",
        use_container_width=True
    )
    if apply_btn:
        st.session_state["simulation_active"] = True
        st.session_state["simulation_active_requested"] = True
        st.session_state["simulation_active_force"] = True
        st.session_state["loaded_result"] = None
        st.session_state["loaded_inputs"] = None
        st.rerun()

# 메인 화면 로직 분기
if st.session_state.get("simulation_active_requested"):
    st.session_state["simulation_active"] = True
    st.session_state["simulation_active_requested"] = False
if st.session_state.get("simulation_active_force") and not st.session_state.get("simulation_active", False):
    st.session_state["simulation_active"] = True
    st.session_state["simulation_active_force"] = False
if not st.session_state.get("simulation_active", False):
    st.title(f"📊 {st.session_state.get('project_symbol', 'ESTV')} 토큰 상장 리스크 & 수급 시뮬레이터")
    st.markdown(
        "계약 시나리오와 토크노믹스 입력(유통·언본딩·유입·유동성·방어 정책)을 바탕으로 "
        "**가격 추이와 리스크를 시뮬레이션**합니다."
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("최종 가격", "$0.000", "0.0%")
    col2.metric("상태 진단", "READY", help="시뮬레이션 대기 중입니다.")
    col3.metric("법적 리스크", "CHECKING..")
    col4.metric("경고 발생", "-")
    st.info(
        "### 👋 시뮬레이션 준비 완료\n"
        "좌측 사이드바에서 **목표, 공급, 수요, 시장 변수**를 설정하세요.\n"
        f"설정이 완료되면 하단의 **[{RUN_SIM_BUTTON_LABEL}]** 버튼을 눌러 결과를 확인하세요."
    )
    st.subheader("📈 가격 변동 추이 (대기 중)")
    empty_chart_data = pd.DataFrame(
        {"Price": [0.5] * 30, "Day": range(30)}
    )
    st.line_chart(empty_chart_data, x="Day", y="Price")
    st.caption("시뮬레이션을 실행하면 이곳에 예측 그래프가 표시됩니다.")
    st.stop()

# 시뮬레이션 결과 화면
st.title(f"📊 {st.session_state.get('project_symbol', 'ESTV')} 토큰 상장 리스크 & 수급 시뮬레이터")
st.markdown(
    "계약 시나리오와 토크노믹스 입력(유통·언본딩·유입·유동성·방어 정책)을 바탕으로 "
    "**가격 추이와 리스크를 시뮬레이션**합니다."
)

# 초기 투자자 락업/베스팅 적용 값 구성
initial_investor_allocation = None
initial_investor_sell_usdt_schedule = [0.0] * total_days
if initial_investor_locked_tokens > 0 and initial_investor_locked_percent <= 100.0:
    vesting_months_used = 0 if initial_investor_vesting_months == 0 else derived_vesting_months
    initial_investor_allocation = {
        "percent": max(0.0, min(1.0, initial_investor_locked_tokens / TOTAL_SUPPLY)),
        "cliff": int(initial_investor_lock_months),
        "vesting": int(vesting_months_used),
        "interval": int(initial_investor_release_interval),
    }

# 시뮬레이션 실행
engine = TokenSimulationEngine()
inputs = {
    'target_tier': target_tier_key,
    'total_supply': total_supply_input,
    'pre_circulated': pre_circulated,
    'unlocked': unlocked,
    'unlocked_vesting_months': unlocked_vesting_months,
    'initial_circulating_percent': input_supply,
    'unbonding_days': input_unbonding,
    'sell_pressure_ratio': input_sell_ratio / 100.0,
    'monthly_buy_volume': input_buy_volume + monthly_user_buy_volume,
    'base_monthly_buy_volume': input_buy_volume,
    'base_daily_buy_schedule': base_daily_buy_schedule,
    'daily_user_buy_schedule': daily_user_buy_schedule,
    'use_marketing_contract_scenario': False,
    'simulation_months': simulation_months,
    'simulation_days': total_days,
    'steps_per_month': steps_per_month,
    'turnover_ratio': turnover_ratio / 100.0,
    'turnover_buy_share': turnover_buy_share / 100.0,
    'lp_growth_rate': lp_growth_rate / 100.0,
    'max_buy_usdt_ratio': max_buy_usdt_ratio / 100.0,
    'max_sell_token_ratio': max_sell_token_ratio / 100.0,
    'burn_fee_rate': burn_fee_rate / 100.0,
    'monthly_buyback_usdt': monthly_buyback_usdt,
    'market_sentiment_config': market_sentiment_config,
    'volume_volatility': st.session_state.get("volume_volatility"),
    'weekend_dip': st.session_state.get("weekend_dip"),
    'initial_investor_allocation': initial_investor_allocation,
    'initial_investor_sell_ratio': initial_investor_sell_ratio / 100.0,
    'initial_investor_sell_usdt_schedule': initial_investor_sell_usdt_schedule,
    'price_model': price_model,
    'depth_usdt_1pct': depth_usdt_1pct,
    'depth_usdt_2pct': depth_usdt_2pct,
    'depth_growth_rate': depth_growth_rate / 100.0,
    'krw_per_usd': krw_per_usd,
    'campaigns': campaigns,
    'triggers': triggers,
    'enable_triggers': use_triggers
}
contract_notes = []
reset_triggered = bool(st.session_state.get("reset_triggered", False))
loaded_result = st.session_state.get("loaded_result")
loaded_inputs = st.session_state.get("loaded_inputs")
if reset_triggered:
    result = build_reset_result(inputs, total_days)
    upbit_baseline_result = None
    st.session_state["reset_triggered"] = False
elif loaded_result:
    result = loaded_result
    if loaded_inputs:
        inputs = loaded_inputs
    upbit_baseline_result = None
else:
    result = run_sim_with_cache(inputs)
    upbit_baseline_result = None
    if show_upbit_baseline:
        upbit_monthly_buy = 3_500_000_000 / max(krw_per_usd, 1)
        upbit_inputs = dict(inputs)
        upbit_inputs.update({
            "initial_circulating_percent": 45.0,
            "unbonding_days": 14,
            "sell_pressure_ratio": 0.15,
            "monthly_buy_volume": upbit_monthly_buy,
            "base_monthly_buy_volume": upbit_monthly_buy,
            "daily_user_buy_schedule": [upbit_monthly_buy / 30] * total_days,
            "use_marketing_contract_scenario": False,
            "use_master_plan": False,
            "campaigns": [],
            "triggers": [],
            "enable_triggers": False
        })
        upbit_baseline_result = run_sim_with_cache(upbit_inputs)

# 결과 표시 (대시보드)
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    f"최종 가격 ({simulation_value}{simulation_unit} 후)",
    f"${result['final_price']:.3f}",
    f"{result['roi']:.1f}%"
)
col2.metric("상태 진단", result['status'], delta_color="off")
col3.metric("법적 리스크", "통과" if result['legal_check'] else "위반(Illegal)")
col4.metric("경고 발생 횟수", f"{len(result['risk_logs'])} 회")
if contract_notes:
    st.info("계약 적용: " + ", ".join(contract_notes))
ai_strategy_report = st.session_state.get("ai_strategy_report")
if ai_strategy_report:
    with st.expander("🧭 AI 전략 가이드", expanded=True):
        st.markdown(ai_strategy_report)

ai_consulting = generate_ai_consulting_report(result, inputs)
series = result.get('daily_price_trend', [])
if ai_consulting:
    with st.expander("🧠 AI 컨설팅 리포트", expanded=True):
        for item in ai_consulting:
            st.markdown(item)
        openai_key = st.session_state.get("openai_api_key", "")
        if openai_key:
            if st.button("🧠 AI 실시간 정밀 분석"):
                if not openai_key:
                    st.error("OpenAI API 키가 필요합니다.")
                else:
                    with st.spinner("ESTV 전략 문서를 분석하여 리포트를 작성 중입니다..."):
                        # series(가격 데이터)를 result에서 꺼내는 코드 추가
                        series = result.get('simulation_log', {}).get('price', [])
                        ai_report = get_real_ai_insight(
                            openai_key,
                            inputs,
                            result,
                            float(st.session_state.get("listing_score", 0.0)),
                            series
                        )
                        if ai_report:
                            st.markdown(ai_report)
                            st.session_state['ai_insight_text'] = ai_report
        real_insight = st.session_state.get("ai_real_insight")
        if real_insight:
            st.markdown("---")
            st.markdown(real_insight)

# 가격 변동 추이 시각화 추가
if series and len(series) > 2:
    chart_data = pd.DataFrame({"Price": series, "Day": range(1, len(series)+1)})
    st.subheader("📈 가격 변동 추이")
    st.line_chart(chart_data, x="Day", y="Price")
    st.caption("시뮬레이션 기간 동안의 가격 변동 추이입니다.")

if enable_confidence and not reset_triggered:
    confidence_result = run_confidence_with_cache(
        inputs,
        confidence_runs,
        confidence_uncertainty / 100.0,
        confidence_mape
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("가격 변동추이 신뢰도", f"{confidence_result['confidence']:.1f}%")
    c2.metric("평균 오차(MAPE)", f"{confidence_result['avg_mape']:.1f}%")
    c3.metric("오차 범위(10~90%)", f"{confidence_result['p10_mape']:.1f}% ~ {confidence_result['p90_mape']:.1f}%")
    st.caption("신뢰도는 입력값 불확실성 범위 내에서 기준 추이와 유사한 시뮬레이션 비율입니다.")

with st.expander("🎯 역산 목표 가격 시뮬레이션", expanded=False):
    target_price = st.number_input(
        "목표 최종 가격 ($)",
        min_value=0.1,
        value=5.0,
        step=0.1,
        help="목표 최종 가격을 입력하면 역산 로직이 필요한 유입/설정을 계산합니다.",
        key="reverse_target_price"
    )
    reverse_basis = st.selectbox(
        "역산 기준",
        options=["전환율 조정", "평균 매수액 조정", "전환율+매수액 균등"],
        index=0,
        help="목표가 달성을 위해 어떤 변수를 우선 조정할지 선택합니다.",
        key="reverse_basis"
    )
    volatility_mode = st.selectbox(
        "변동성 적용 방식",
        options=["완화", "중립", "공격"],
        index=0,
        help="목표가를 맞출 때 변동성을 줄이거나(완화), 유지(중립), 높이는(공격) 방향으로 설정합니다.",
        key="reverse_volatility_mode"
    )
    auto_price_model = st.checkbox(
        "가격 모델/오더북 자동 조정",
        value=True,
        help="역산 계산 시 가격 모델과 오더북 깊이도 함께 조정합니다.",
        key="reverse_auto_price_model"
    )

    if st.button("역산 계산"):
        target_monthly_buy = estimate_required_monthly_buy(engine, inputs, target_price)
        required_monthly_user_buy = max(0.0, target_monthly_buy - input_buy_volume)
        required_total_buyers = (required_monthly_user_buy * onboarding_months) / max(avg_ticket, 1)
        required_conversion_rate = (required_total_buyers / estv_total_users) * 100.0
        current_total_new_buyers = estv_total_users * (conversion_rate / 100.0)
        required_avg_ticket = (
            (required_monthly_user_buy * onboarding_months) / current_total_new_buyers
            if current_total_new_buyers > 0 else 0.0
        )

        st.session_state["reverse_result"] = {
            "target_price": target_price,
            "target_monthly_buy": target_monthly_buy,
            "required_monthly_user_buy": required_monthly_user_buy,
            "required_conversion_rate": required_conversion_rate,
            "required_avg_ticket": required_avg_ticket,
            "reverse_basis": reverse_basis,
            "volatility_mode": volatility_mode
        }
        st.session_state["reverse_apply_pending"] = True

    reverse_result = st.session_state.get("reverse_result")
    if reverse_result:
        target_monthly_buy = reverse_result["target_monthly_buy"]
        required_monthly_user_buy = reverse_result["required_monthly_user_buy"]
        required_conversion_rate = reverse_result["required_conversion_rate"]
        required_avg_ticket = reverse_result["required_avg_ticket"]
        reverse_basis = reverse_result["reverse_basis"]
        volatility_mode = reverse_result["volatility_mode"]

        st.markdown("**역산 결과**")
        st.metric("필요 월간 총 매수 유입 ($)", f"{target_monthly_buy:,.0f}")
        st.metric("필요 월간 유저 유입 ($)", f"{required_monthly_user_buy:,.0f}")
        st.metric("필요 전환율 (%)", f"{required_conversion_rate:.2f}")
        st.metric("전환율 고정 시 필요 평균 매수액 ($)", f"{required_avg_ticket:,.0f}")

        st.markdown("**왜 이 수치인가?**")
        st.write(
            "1) 목표 최종 가격을 달성하기 위한 월간 총 매수 유입을 역산했습니다.\n"
            f"2) 월간 유저 유입 = 필요 월간 총 매수 유입 - 기본 매수 유입 (${input_buy_volume:,.0f}).\n"
            f"3) 전환율 = (월간 유저 유입 × {onboarding_months}개월) / (회원수 × 평균 매수액)."
        )
        if use_phase_inflow:
            st.write(
                f"4) Phase 1 대기 수요는 상장 직후 {prelisting_release_days}일에 걸쳐 "
                "점진적으로 방출됩니다."
            )

        st.markdown("**변동성 설정(현재값)**")
        st.write(
            f"- 분할 단위: {steps_per_month}일\n"
            f"- 회전율: {turnover_ratio:.2f}% (매수 비중 {turnover_buy_share:.0f}%)\n"
            f"- LP 성장률: {lp_growth_rate:.2f}%/월\n"
            f"- 매수 캡: {max_buy_usdt_ratio:.2f}%\n"
            f"- 매도 캡: {max_sell_token_ratio:.2f}%\n"
            f"- 거래 수수료 소각률: {burn_fee_rate:.2f}%\n"
            f"- 월간 바이백 예산: ${monthly_buyback_usdt:,.0f}"
        )

        apply_payload = {
            "scenario_preset": "직접 입력",
            "input_buy_volume": 0
        }

        if reverse_basis == "전환율 조정":
            capped_conversion = min(2.0, required_conversion_rate)
            apply_payload["conversion_rate"] = capped_conversion
            if required_conversion_rate > 2.0:
                adjusted_avg = (
                    (required_monthly_user_buy * onboarding_months) /
                    (estv_total_users * (capped_conversion / 100.0))
                )
                apply_payload["avg_ticket"] = max(adjusted_avg, 1.0)
            else:
                apply_payload["avg_ticket"] = avg_ticket
        elif reverse_basis == "평균 매수액 조정":
            apply_payload["conversion_rate"] = conversion_rate
            apply_payload["avg_ticket"] = max(required_avg_ticket, 1.0)
        else:
            current_total = max(total_inflow_money, 1.0)
            scale = (required_monthly_user_buy * onboarding_months) / current_total
            base_conv = conversion_rate * math.sqrt(max(scale, 0.0))
            capped_conversion = min(2.0, base_conv)
            apply_payload["conversion_rate"] = capped_conversion
            adjusted_avg = (
                (required_monthly_user_buy * onboarding_months) /
                (estv_total_users * (capped_conversion / 100.0))
            )
            apply_payload["avg_ticket"] = max(adjusted_avg, 1.0)

        if volatility_mode == "완화":
            apply_payload.update({
                "steps_per_month": 30,
                "turnover_ratio": 3.0,
                "turnover_buy_share": 60.0,
                "lp_growth_rate": 1.5,
                "max_buy_usdt_ratio": 4.0,
                "max_sell_token_ratio": 4.0,
                "burn_fee_rate": 0.3,
                "monthly_buyback_usdt": max(0, int(target_monthly_buy * 0.05))
            })
        elif volatility_mode == "공격":
            apply_payload.update({
                "steps_per_month": 7,
                "turnover_ratio": 8.0,
                "turnover_buy_share": 40.0,
                "lp_growth_rate": 0.5,
                "max_buy_usdt_ratio": 8.0,
                "max_sell_token_ratio": 8.0,
                "burn_fee_rate": 0.1,
                "monthly_buyback_usdt": 0
            })
        else:
            apply_payload.update({
                "steps_per_month": 30,
                "turnover_ratio": 5.0,
                "turnover_buy_share": 50.0,
                "lp_growth_rate": 1.0,
                "max_buy_usdt_ratio": 5.0,
                "max_sell_token_ratio": 5.0,
                "burn_fee_rate": 0.3,
                "monthly_buyback_usdt": max(0, int(target_monthly_buy * 0.03))
            })

        apply_payload.update({
            "use_master_plan": True,
            "use_triggers": True,
            "buy_verify_boost": max(0.0, buy_verify_boost),
            "holding_suppress": max(0.0, holding_suppress),
            "payburn_delta": max(0.0, payburn_delta),
            "buyback_daily": max(0.0, buyback_daily),
            "use_phase_inflow": True,
            "phase2_days": phase2_days,
            "phase2_multiplier": max(1.0, phase2_multiplier),
            "prelisting_days": prelisting_days,
            "prelisting_multiplier": max(1.0, prelisting_multiplier),
            "prelisting_release_days": prelisting_release_days
        })

        if auto_price_model:
            if volatility_mode == "완화":
                apply_payload.update({
                    "price_model": "HYBRID",
                    "depth_usdt_1pct": max(1_000_000, depth_usdt_1pct),
                    "depth_usdt_2pct": max(3_000_000, depth_usdt_2pct),
                    "depth_growth_rate": max(2.0, depth_growth_rate)
                })
            elif volatility_mode == "공격":
                apply_payload.update({
                    "price_model": "CEX",
                    "depth_usdt_1pct": max(300_000, depth_usdt_1pct),
                    "depth_usdt_2pct": max(800_000, depth_usdt_2pct),
                    "depth_growth_rate": max(0.0, depth_growth_rate)
                })
            else:
                apply_payload.update({
                    "price_model": "CEX",
                    "depth_usdt_1pct": max(800_000, depth_usdt_1pct),
                    "depth_usdt_2pct": max(2_000_000, depth_usdt_2pct),
                    "depth_growth_rate": max(1.0, depth_growth_rate)
                })

        if st.session_state.get("reverse_apply_pending"):
            st.session_state["reverse_apply_payload"] = apply_payload
            st.session_state["apply_reverse_scenario"] = True
            st.session_state["reverse_apply_pending"] = False
            st.rerun()

# 경고 메시지 박스
if result['status'] == "ILLEGAL":
    st.error("⛔ [CRITICAL] 초기 유통량이 법적 한도(3%)를 초과했습니다. 프로젝트 중단 사유가 됩니다.")
elif result['status'] == "CRITICAL":
    st.error("🔥 [DANGER] 가격이 -60% 이상 폭락했습니다. 뱅크런 위험이 있습니다.")
elif result['status'] == "UNSTABLE":
    st.warning("⚠️ [WARNING] 가격 방어선이 불안정합니다. 매수 자금을 늘리거나 언본딩을 강화하세요.")
else:
    st.success("✅ [SAFE] 안정적인 가격 흐름을 유지하고 있습니다.")

# Master Plan 카드
if use_master_plan:
    card_path = os.path.join(os.path.dirname(__file__), "master_plan_card.md")
    try:
        with open(card_path, "r", encoding="utf-8") as f:
            master_plan_md = f.read().strip()
    except Exception:
        master_plan_md = "**Master Plan 요약을 불러오지 못했습니다.**"

    st.markdown("""
<div style="background-color:#0f172a; border:1px solid #1f2937; padding:16px; border-radius:12px; margin-bottom:12px;">
  <div style="font-size:18px; font-weight:700; margin-bottom:8px;">🚀 Master Plan 요약 카드</div>
  <div style="font-size:14px; line-height:1.6;">
""", unsafe_allow_html=True)
    st.markdown(master_plan_md)
    st.markdown(f"""
**설정값 요약**
- Phase 1 대기: {prelisting_days}일, 배수 {prelisting_multiplier:.1f}, 방출 {prelisting_release_days}일
- Phase 2 기간: {phase2_days}일, 배수 {phase2_multiplier:.1f}
- Buy & Verify 증폭: {buy_verify_boost:.2f}
- Holding 억제: {holding_suppress:.2f}
- Pay & Burn 증폭: {payburn_delta:.4f}
- 일일 바이백: ${buyback_daily:,.0f}
""")
    st.markdown("</div>", unsafe_allow_html=True)

# 차트 그리기
st.subheader("📈 가격 변동 추이 (Interactive)")
series = result['daily_price_trend']

if st.button("🤖 AI 최적화 제안"):
    optimized_inputs, optimized_notes = build_optimized_inputs(result["inputs"], result.get("simulation_log", {}))
    st.session_state["optimized_notes"] = optimized_notes
    st.session_state["optimized_result"] = run_sim_with_cache(optimized_inputs)
    st.session_state["optimized_inputs"] = optimized_inputs

opt_result = st.session_state.get("optimized_result")
opt_notes = st.session_state.get("optimized_notes", [])
if opt_result:
    st.caption("AI 최적화 시나리오를 점선으로 함께 표시합니다.")
    if opt_notes:
        st.info(" · ".join(opt_notes))

go = None
try:
    go = importlib.import_module("plotly.graph_objects")
except Exception:
    go = None

if go is not None:
    days = list(range(len(series)))
    make_subplots = importlib.import_module("plotly.subplots").make_subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.08
    )
    turnover_pct = result["inputs"].get("turnover_ratio", 0.0) * 100
    lp_growth_pct = result["inputs"].get("lp_growth_rate", 0.0) * 100
    max_buy_pct = result["inputs"].get("max_buy_usdt_ratio", 0.0) * 100
    max_sell_pct = result["inputs"].get("max_sell_token_ratio", 0.0) * 100
    steps_per_month = result["inputs"].get("steps_per_month", 30)
    split_label = f"{steps_per_month}일 분할"
    lp_daily_label = "예" if lp_growth_pct > 0 else "아니오"
    main_line_color = "#00F0FF" if result['legal_check'] else "#FF4D4D"
    fig.add_trace(go.Scatter(
        x=days,
        y=series,
        mode="lines",
        name="ESTV Price ($)",
        line=dict(color=main_line_color, width=3),
        fill="tozeroy",
        fillcolor="rgba(0, 240, 255, 0.12)" if result['legal_check'] else "rgba(255, 77, 77, 0.12)",
        hovertemplate="<b>Day %{x}</b><br>Price: $%{y:.4f}<extra></extra>"
    ), row=1, col=1)
    if opt_result:
        opt_series = opt_result.get("daily_price_trend", [])
        opt_days = list(range(len(opt_series)))
        fig.add_trace(go.Scatter(
            x=opt_days,
            y=opt_series,
            mode="lines",
            name="AI 최적화 시나리오",
            line=dict(color="purple", dash="dot")
        ), row=1, col=1)
    if upbit_baseline_result:
        up_series = upbit_baseline_result["daily_price_trend"]
        up_days = list(range(len(up_series)))
        fig.add_trace(go.Scatter(
            x=up_days,
            y=up_series,
            mode="lines",
            name="Upbit 평균 시나리오",
            line=dict(color="gray", dash="dash")
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0, len(series) - 1],
        y=[0.5, 0.5],
        mode="lines",
        name="Listing Price ($0.50)",
        line=dict(color="gray", dash="dot")
    ), row=1, col=1)
    support_line = float(np.percentile(series, 20)) if series else 0.5
    fig.add_trace(go.Scatter(
        x=[0, len(series) - 1],
        y=[support_line, support_line],
        mode="lines",
        name="Support Line",
        line=dict(color="rgba(120,120,120,0.6)", dash="dash")
    ), row=1, col=1)

    log = result.get("simulation_log")
    if log:
        reason_days = []
        reason_prices = []
        reason_texts = []
        for i, reason in enumerate(log.get("reason_code", [])):
            if reason == "NORMAL":
                continue
            if i >= len(series):
                break
            reason_days.append(log.get("day", [])[i] if i < len(log.get("day", [])) else i)
            reason_prices.append(log.get("price", [])[i] if i < len(log.get("price", [])) else series[i])
            reason_texts.append(reason)
        if reason_days:
            reason_colors = [
                "#00FF88" if ("FOMO" in text or "MARKETING" in text) else "#FF5555"
                for text in reason_texts
            ]
            fig.add_trace(go.Scatter(
                x=reason_days,
                y=reason_prices,
                mode="markers",
                name="중요 이벤트",
                marker=dict(size=10, color=reason_colors, symbol="diamond-open", line=dict(width=2)),
                text=reason_texts,
                hovertemplate="<b>%{text}</b><br>Day %{x}<br>Price $%{y:.4f}<extra></extra>"
            ), row=1, col=1)

        reason_colors = {
            "PANIC_SELL": "red",
            "WHALE_DUMP": "orange",
            "FOMO_RALLY": "green"
        }
        xai_days = []
        xai_prices = []
        xai_reason = []
        xai_action = []
        xai_sentiment = []
        xai_sell = []
        xai_buy = []
        xai_source = []
        xai_action_msg = []
        for i, reason in enumerate(log.get("reason_code", [])):
            if reason == "NORMAL":
                continue
            xai_days.append(log["day"][i])
            xai_prices.append(log["price"][i])
            xai_reason.append(reason)
            xai_action.append(log["action_needed"][i])
            xai_sentiment.append(log["sentiment_index"][i])
            xai_sell.append(log["sell_pressure_vol"][i])
            xai_buy.append(log["buy_power_vol"][i])
            xai_source.append(log.get("sell_source_text", [""])[i])
            xai_action_msg.append(log.get("action_message", [""])[i])
        if xai_days:
            fig.add_trace(go.Scatter(
                x=xai_days,
                y=xai_prices,
                mode="markers",
                name="원인/대응",
                marker=dict(
                    color=[reason_colors.get(r, "gray") for r in xai_reason],
                    size=9,
                    symbol="circle-open"
                ),
                customdata=list(zip(xai_reason, xai_action, xai_sentiment, xai_sell, xai_buy, xai_source, xai_action_msg)),
                hovertemplate=(
                    "Day %{x}<br>"
                    "Price $%{y:.4f}<br>"
                    "원인 %{customdata[0]}<br>"
                    "대응 %{customdata[1]}<br>"
                    "심리 지수 %{customdata[2]:.2f}<br>"
                    "매도 압력 %{customdata[3]:,.0f}<br>"
                    "매수 지지력 %{customdata[4]:,.0f}<br>"
                    "매도 출처 %{customdata[5]}<br>"
                    "%{customdata[6]}"
                    "<extra></extra>"
                )
            ), row=1, col=1)

        narrative_annotations = []
        whale_volumes = log.get("whale_sell_volume", [])
        if whale_volumes:
            whale_threshold = max(1_000_000, float(np.percentile(whale_volumes, 90)))
        else:
            whale_threshold = 1_000_000
        max_log_len = min(
            len(log.get("sentiment_index", [])),
            len(log.get("whale_sell_volume", [])),
            len(log.get("liquidity_depth_ratio", [])),
            len(log.get("marketing_trigger", [])),
            len(log.get("buy_power_vol", [])),
            len(log.get("normal_buy_volume", []))
        )
        max_idx = min(len(series), max_log_len)
        def collect_annotations(drop_thresh, rise_thresh, depth_thresh, fomo_multiplier):
            items = []
            for i in range(1, max_idx):
                prev_price = series[i - 1]
                if prev_price <= 0:
                    continue
                price_change = (series[i] - prev_price) / prev_price
                sentiment = log["sentiment_index"][i]
                whale_sell = log["whale_sell_volume"][i]
                liquidity_depth = log["liquidity_depth_ratio"][i]
                marketing_trigger = log["marketing_trigger"][i]
                buy_volume = log["buy_power_vol"][i]
                normal_buy = max(log["normal_buy_volume"][i], 1e-9)
                tag = None
                if price_change <= -drop_thresh and sentiment < 0.8:
                    tag = "📉 공포 투매 (Panic Sell)"
                elif price_change <= -drop_thresh and whale_sell > whale_threshold:
                    tag = "🐋 고래 덤핑 (Whale Dump)"
                elif price_change <= -0.03 and liquidity_depth < depth_thresh:
                    tag = "💧 유동성 고갈 (Slippage Spike)"
                elif price_change >= rise_thresh and marketing_trigger:
                    tag = "🚀 마케팅 효과 (Campaign)"
                elif price_change >= rise_thresh and buy_volume > normal_buy * fomo_multiplier:
                    tag = "🔥 FOMO 유입"
                if tag:
                    items.append({
                        "day": i,
                        "price": series[i],
                        "tag": tag,
                        "score": abs(price_change)
                    })
            return items

        narrative_annotations = collect_annotations(
            drop_thresh=0.05,
            rise_thresh=0.05,
            depth_thresh=0.5,
            fomo_multiplier=2.0
        )
        if not narrative_annotations:
            narrative_annotations = collect_annotations(
                drop_thresh=0.03,
                rise_thresh=0.03,
                depth_thresh=0.7,
                fomo_multiplier=1.5
            )
        # Always annotate explicit PANIC_SELL events from log if present
        reason_list = log.get("reason", [])
        for i in range(1, min(max_idx, len(reason_list))):
            if "PANIC_SELL" in reason_list[i]:
                narrative_annotations.append({
                    "day": i,
                    "price": series[i],
                    "tag": "📉 공포 투매 (Panic Sell)",
                    "score": 1.0
                })

        narrative_annotations = sorted(narrative_annotations, key=lambda x: x["score"], reverse=True)[:12]
        if narrative_annotations:
            y_offset = max(series) * 0.05
            for idx, ann in enumerate(narrative_annotations):
                fig.add_annotation(
                    x=ann["day"],
                    y=ann["price"] + (max(series) * 0.05 if series else 0),
                    text=ann["tag"],
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    ax=0,
                    ay=-20 - (idx % 3) * 10,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    row=1,
                    col=1
                )
            st.caption(f"스토리텔링 주석 {len(narrative_annotations)}개 표시됨")
        else:
            st.caption("스토리텔링 주석 조건에 맞는 구간이 없어 표시되지 않았습니다.")

        # Zone coloring based on sentiment index
        sentiment_series = log.get("sentiment_index", [])
        if sentiment_series:
            zones = []
            current_zone = None
            start_idx = 0
            for i, sentiment in enumerate(sentiment_series[:len(series)]):
                if sentiment < 0.9:
                    zone = "RED"
                elif sentiment > 1.1:
                    zone = "GREEN"
                else:
                    zone = "GREY"
                if current_zone is None:
                    current_zone = zone
                    start_idx = i
                    continue
                if zone != current_zone:
                    zones.append((start_idx, i, current_zone))
                    current_zone = zone
                    start_idx = i
            zones.append((start_idx, len(sentiment_series), current_zone))
            zone_colors = {
                "RED": "rgba(255, 0, 0, 0.08)",
                "GREEN": "rgba(0, 180, 0, 0.08)",
                "GREY": "rgba(120, 120, 120, 0.04)"
            }
            for start, end, zone in zones:
                fig.add_vrect(
                    x0=start,
                    x1=max(start + 1, end),
                    fillcolor=zone_colors[zone],
                    opacity=0.6,
                    line_width=0,
                    row=1,
                    col=1
                )

        # Upbit-style volume bars (buy + sell)
        sell_vols = log.get("sell_pressure_vol", [])
        buy_vols = log.get("buy_power_vol", [])
        if sell_vols and buy_vols:
            min_len = min(len(sell_vols), len(buy_vols), len(series))
            vol_days = list(range(min_len))
            total_vols = []
            vol_colors = []
            for i in range(min_len):
                total_vols.append(sell_vols[i] + buy_vols[i])
                if i > 0 and series[i] >= series[i - 1]:
                    vol_colors.append("rgba(0, 255, 136, 0.6)")
                elif i > 0:
                    vol_colors.append("rgba(255, 60, 60, 0.6)")
                else:
                    vol_colors.append("rgba(150, 150, 150, 0.5)")
            fig.add_trace(go.Bar(
                x=vol_days,
                y=total_vols,
                name="Volume",
                marker_color=vol_colors,
                hovertemplate="<b>Day %{x}</b><br>Total Vol: %{y:,.0f}<extra></extra>"
            ), row=2, col=1)
            fig.update_layout(
                barmode="group",
                bargap=0.2
            )

    if len(series) > 2:
        diffs = [series[i] - series[i - 1] for i in range(1, len(series))]
        segment_count = 5
        seg_size = max(1, len(diffs) // segment_count)
        segments = []
        for i in range(0, len(diffs), seg_size):
            segments.append(range(i, min(i + seg_size, len(diffs))))
        segments = segments[:segment_count]

        up_days = []
        down_days = []
        for seg in segments:
            seg_list = list(seg)
            if not seg_list:
                continue
            max_idx = max(seg_list, key=lambda i: diffs[i])
            min_idx = min(seg_list, key=lambda i: diffs[i])
            up_days.append(max_idx + 1)
            down_days.append(min_idx + 1)

        up_days = list(dict.fromkeys(up_days))
        down_days = list(dict.fromkeys(down_days))
        up_customdata = [[turnover_pct, lp_growth_pct, max_buy_pct, max_sell_pct, split_label, lp_daily_label]] * len(up_days)
        down_customdata = [[turnover_pct, lp_growth_pct, max_buy_pct, max_sell_pct, split_label, lp_daily_label]] * len(down_days)

        fig.add_trace(go.Scatter(
            x=up_days,
            y=[series[d] for d in up_days],
            mode="markers",
            name="급등 구간",
            marker=dict(color="green", size=8),
            customdata=up_customdata,
            hovertemplate=(
                "급등 구간<br>"
                "Day %{x}<br>"
                "Price $%{y:.4f}<br>"
                "회전율 %{customdata[0]:.2f}%<br>"
                "LP 성장률 %{customdata[1]:.2f}%/월<br>"
                "매수 캡 %{customdata[2]:.2f}%<br>"
                "매도 캡 %{customdata[3]:.2f}%<br>"
                "분할 단위 %{customdata[4]}<br>"
                "LP 일단위 적용 %{customdata[5]}"
                "<extra></extra>"
            )
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=down_days,
            y=[series[d] for d in down_days],
            mode="markers",
            name="급락 구간",
            marker=dict(color="red", size=8),
            customdata=down_customdata,
            hovertemplate=(
                "급락 구간<br>"
                "Day %{x}<br>"
                "Price $%{y:.4f}<br>"
                "회전율 %{customdata[0]:.2f}%<br>"
                "LP 성장률 %{customdata[1]:.2f}%/월<br>"
                "매수 캡 %{customdata[2]:.2f}%<br>"
                "매도 캡 %{customdata[3]:.2f}%<br>"
                "분할 단위 %{customdata[4]}<br>"
                "LP 일단위 적용 %{customdata[5]}"
                "<extra></extra>"
            )
        ), row=1, col=1)

        event_days = []
        event_prices = []
        event_amounts = []
        event_customdata = []
        for event in result.get("daily_events", []):
            if event["type"] == "MarketingDump":
                event_days.append(event["day"])
                event_prices.append(event["price"])
                event_amounts.append(event.get("amount", 0))
                event_customdata.append([split_label, lp_daily_label])

        if event_days:
            fig.add_trace(go.Scatter(
                x=event_days,
                y=event_prices,
                mode="markers",
                name="마케팅 덤핑",
                marker=dict(color="orange", size=9, symbol="circle"),
                customdata=list(zip(event_amounts, event_customdata)),
                hovertemplate=(
                    "마케팅 덤핑 발생<br>"
                    "Day %{x}<br>"
                    "Price $%{y:.4f}<br>"
                    "덤핑 수량 %{customdata[0]:,.0f}개<br>"
                    "분할 단위 %{customdata[1][0]}<br>"
                    "LP 일단위 적용 %{customdata[1][1]}"
                    "<extra></extra>"
                )
            ), row=1, col=1)

    fig.update_layout(
        title="📈 ESTV Price Simulation (Interactive)",
        template="plotly_dark",
        xaxis=dict(
            title="Timeline (Days)",
            showgrid=False,
            rangeslider=dict(visible=True)
        ),
        hovermode="x unified",
        height=560,
        margin=dict(l=10, r=10, t=45, b=10),
        barmode="overlay"
    )
    fig.update_yaxes(title_text="Price (USDT)", dtick=0.25, row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
    if log:
        last_idx = min(len(series), len(log.get("sell_pressure_vol", [])), len(log.get("buy_power_vol", []))) - 1
        if last_idx >= 0:
            last_price = series[last_idx]
            last_sell = log["sell_pressure_vol"][last_idx]
            last_buy = log["buy_power_vol"][last_idx]
            signal_icons = []
            if last_price < support_line and last_buy < last_sell:
                signal_icons.append("🛡️ 바이백 권장")
            if last_sell > last_buy * 1.2:
                signal_icons.append("📢 호재 뉴스 필요")
            if signal_icons:
                st.markdown("**액션 제안:** " + " · ".join(signal_icons))
        with st.expander("🧭 상승/급락 원인 & 대응 가이드", expanded=True):
            safe_len = min(
                len(series),
                len(log.get("price", [])),
                len(log.get("reason_code", [])),
                len(log.get("action_needed", [])),
                len(log.get("sentiment_index", [])),
                len(log.get("sell_pressure_vol", [])),
                len(log.get("buy_power_vol", [])),
                len(log.get("liquidity_depth_ratio", [])),
                len(log.get("marketing_trigger", [])),
                len(log.get("reason", []))
            )
            if safe_len < 2:
                st.write("설명 가능한 데이터가 아직 충분하지 않습니다.")
            else:
                price_changes = [0.0] + [
                    (series[i] - series[i - 1]) / max(series[i - 1], 1e-9)
                    for i in range(1, safe_len)
                ]
                minor_thresh = 0.02
                major_thresh = 0.05
                rise_idx = [i for i in range(1, safe_len) if price_changes[i] >= minor_thresh]
                drop_idx = [i for i in range(1, safe_len) if price_changes[i] <= -minor_thresh]
                major_rise_idx = [i for i in rise_idx if price_changes[i] >= major_thresh]
                major_drop_idx = [i for i in drop_idx if price_changes[i] <= -major_thresh]

                rise_fomo = sum(1 for i in rise_idx if log["reason_code"][i] == "FOMO_RALLY")
                rise_marketing = sum(1 for i in rise_idx if log["marketing_trigger"][i])
                rise_buy_support = sum(1 for i in rise_idx if log["buy_power_vol"][i] > log["sell_pressure_vol"][i])

                drop_panic = sum(1 for i in drop_idx if log["reason_code"][i] == "PANIC_SELL")
                drop_whale = sum(1 for i in drop_idx if log["reason_code"][i] == "WHALE_DUMP")
                drop_liquidity = sum(1 for i in drop_idx if log["liquidity_depth_ratio"][i] < 0.7)
                drop_sell_over = sum(1 for i in drop_idx if log["sell_pressure_vol"][i] > log["buy_power_vol"][i])

                st.markdown("**상승 원인 요약 (기준: +2% 이상)**")
                st.write(
                    f"- FOMO/추격매수: {rise_fomo}회\n"
                    f"- 마케팅/캠페인 효과: {rise_marketing}회\n"
                    f"- 매수 지지력이 매도보다 큼: {rise_buy_support}회\n"
                    f"- 급등(+5% 이상): {len(major_rise_idx)}회"
                )
                st.markdown("**급락 원인 요약 (기준: -2% 이하)**")
                st.write(
                    f"- 공포 투매: {drop_panic}회\n"
                    f"- 대량 매도(고래/이익실현/마케팅 덤핑): {drop_whale}회\n"
                    f"- 유동성 얕음(슬리피지 확대): {drop_liquidity}회\n"
                    f"- 매도 압력이 매수보다 큼: {drop_sell_over}회\n"
                    f"- 급락(-5% 이하): {len(major_drop_idx)}회"
                )

                reason_label = {
                    "PANIC_SELL": "공포 투매",
                    "WHALE_DUMP": "대량 매도",
                    "FOMO_RALLY": "FOMO 매수",
                    "ARBITRAGE_SWAP": "차익거래 스왑",
                    "NORMAL": "일반 구간"
                }
                source_label = {
                    "investor_unlock": "초기 투자자 물량",
                    "marketing_dump": "마케팅 물량",
                    "turnover_sell": "회전율 매도",
                    "panic_sell": "심리 매도",
                    "unlocked_overhang": "언락 오버행"
                }
                action_label = {
                    "NEED_BUYBACK": "바이백/매수 방어",
                    "MARKETING_OP": "마케팅/캠페인 강화",
                    "STABILIZE_PRICE": "가격 괴리 안정화",
                    "ADD_LIQUIDITY": "유동성 공급",
                    "NONE": "모니터링"
                }
                guide_map = {
                    "PANIC_SELL": "매도율 상향·심리 악화가 원인입니다. 대응: 바이백 확대, 매도 캡 강화, 언본딩/락업 연장.",
                    "WHALE_DUMP": "대량 물량 출회(마케팅 덤핑/이익실현)가 원인입니다. 대응: 락업/베스팅 재설계, OTC 분할 매도.",
                    "FOMO_RALLY": "상승 추세에 따른 추격 매수 유입입니다. 대응: 과열 경보, 분할 매도 계획.",
                    "ARBITRAGE_SWAP": "CEX/DEX 괴리로 가격이 재조정되었습니다. 대응: 오더북 깊이/LP 균형.",
                    "NORMAL": "특정 이벤트 없이 수급이 균형인 구간입니다."
                }

                drop_candidates = []
                for i in drop_idx:
                    drop_candidates.append((price_changes[i], i))
                drop_candidates.sort(key=lambda x: x[0])
                top_drops = drop_candidates[:3]
                if top_drops:
                    rows = []
                    for change, i in top_drops:
                        source_text = log.get("sell_source_text", [""])[i]
                        if "investor_unlock" in source_text:
                            source_text = source_text.replace("investor_unlock", source_label["investor_unlock"])
                        if "marketing_dump" in source_text:
                            source_text = source_text.replace("marketing_dump", source_label["marketing_dump"])
                        if "turnover_sell" in source_text:
                            source_text = source_text.replace("turnover_sell", source_label["turnover_sell"])
                        if "panic_sell" in source_text:
                            source_text = source_text.replace("panic_sell", source_label["panic_sell"])
                        rows.append({
                            "Day": i + 1,
                            "변동률": f"{change * 100:.1f}%",
                            "원인": reason_label.get(log["reason_code"][i], log["reason_code"][i]),
                            "세부": log["reason"][i],
                            "권장 대응": action_label.get(log["action_needed"][i], log["action_needed"][i]),
                            "매도 출처": source_text
                        })
                    st.markdown("**최근 급락 Top 3 상세**")
                    st.table(pd.DataFrame(rows))

                st.markdown("**문제 해결 가이드**")
                for key in ["PANIC_SELL", "WHALE_DUMP", "LIQUIDITY_DRAIN", "FOMO_RALLY", "ARBITRAGE_SWAP", "NORMAL"]:
                    if key == "LIQUIDITY_DRAIN":
                        st.write("- 유동성 고갈: 오더북 깊이/LP 성장률 상향, 대형 매도 분할 유도.")
                        continue
                    st.write(f"- {reason_label.get(key, key)}: {guide_map.get(key, '')}")

                action_messages = []
                for i in range(safe_len):
                    msg = log.get("action_message", [""])[i]
                    if msg:
                        action_messages.append({
                            "Day": i + 1,
                            "처방": msg
                        })
                if action_messages:
                    st.markdown("**정량 처방 로그**")
                    st.table(pd.DataFrame(action_messages).head(10))
else:
    st.info("툴팁 표시를 위해 plotly가 필요합니다. `pip install plotly` 후 다시 실행해 주세요.")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series, label='ESTV Price ($)', color='blue' if result['legal_check'] else 'red')
    ax.axhline(y=0.5, color='gray', linestyle=':', label='Listing Price ($0.50)')
    ax.set_xlabel("Day")
    ax.set_ylabel("Price")
    ax.legend()
    ax.set_yticks([i * 0.25 for i in range(int(max(series) / 0.25) + 2)])
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

if len(series) > 2:
    diffs = [series[i] - series[i - 1] for i in range(1, len(series))]
    min_idx = diffs.index(min(diffs))
    drop_day = min_idx + 1

    reasons = []

    # A. 마케팅 덤핑 체크
    for event in result.get("daily_events", []):
        if event["type"] == "MarketingDump" and abs(event["day"] - drop_day) <= 2:
            reasons.append("마케팅 덤핑(물량 투하) 발생")
            break

    # B. 초기 투자자 락업 해제 체크 (Cliff)
    allocation = inputs.get('initial_investor_allocation', {})
    if not isinstance(allocation, dict):
        allocation = {}
    investor_cliff_months = allocation.get('cliff', 12)
    investor_cliff_days = investor_cliff_months * 30
    if drop_day >= investor_cliff_days and drop_day <= investor_cliff_days + 7:
        reasons.append(f"초기 투자자 락업 해제(D+{investor_cliff_days}) 물량 출회")

    # C. Day 1~7 초기 급락
    if drop_day <= 7:
        if inputs["initial_circulating_percent"] > 5.0:
            reasons.append("초기 유통량 과다(5% 초과)로 인한 차익 실현")
        elif inputs["depth_usdt_1pct"] < 500_000:
            reasons.append("초기 오더북 유동성 부족(슬리피지 심화)")
        elif inputs["turnover_ratio"] > 0.1:
            reasons.append("신규 유입자의 높은 단타 회전율(Panic Sell)")
        else:
            reasons.append("매수세 부족 대비 초기 유통 물량 매도 우위")

    # D. 스테이킹/언본딩 이후 매도
    unbonding_days = inputs.get("unbonding_days", 0)
    if drop_day > unbonding_days + 30 and drop_day > 7:
        reasons.append("스테이킹/보상 물량 언본딩 이후 매도 압력")

    # E. 심리적 요인
    log = result.get("simulation_log", {})
    if log:
        idx = drop_day - 1
        if idx < len(log.get("reason_code", [])):
            code = log["reason_code"][idx]
            if code == "PANIC_SELL":
                reasons.append("시장 심리 악화로 인한 공포 투매(Panic Sell)")
            elif code == "WHALE_DUMP":
                reasons.append("고래(대량 보유자)의 일시적 덤핑")

    reasons = list(set(reasons))
    if not reasons:
        reasons.append("매수세 실종 및 자연스러운 차익 실현 매물 소화")

    source_note = ""
    if log:
        src_list = log.get("sell_source_text", [])
        if drop_day - 1 < len(src_list):
            raw_text = src_list[drop_day - 1]
            clean_text = raw_text.replace("investor_unlock", "투자자 물량") \
                                 .replace("marketing_dump", "마케팅 물량") \
                                 .replace("turnover_sell", "신규 단타 매도") \
                                 .replace("panic_sell", "심리적 투매")
            source_note = f" (상세 비중: {clean_text.split(': ')[-1]})"

    st.info(
        f"📉 **최대 급락 발생일: Day {drop_day}**\n"
        f"- **주요 원인:** {', '.join(reasons)}\n"
        f"- **매도 구성:** {source_note}"
    )

# 로그 테이블
if result['risk_logs']:
    st.subheader("📜 리스크 발생 로그")
    st.table(pd.DataFrame(result['risk_logs']))
if result.get("action_logs"):
    st.subheader("📌 캠페인 액션 로그")
    st.table(pd.DataFrame(result["action_logs"]))

# 전략 리포트 다운로드
if st.session_state.get("simulation_active", False):
    listing_score = float(st.session_state.get("listing_score", 0.0))
    target_price_value = float(st.session_state.get("tutorial_target_price", 0.0))
    pdf_bytes = create_full_report(inputs, series, listing_score, target_price_value)
    log_data = result.get("simulation_log", {})
    log_df = pd.DataFrame(log_data) if log_data else pd.DataFrame()
    log_json = log_df.to_json(orient="records", force_ascii=False, indent=2) if not log_df.empty else "[]"
    log_csv = log_df.to_csv(index=False) if not log_df.empty else ""

    st.download_button(
        label="📥 전략 리포트 다운로드 (PDF)",
        data=pdf_bytes,
        file_name="ESTV_Listing_Strategy_Report.pdf",
        mime="application/pdf",
        help="상장 심사 제출용 근거 자료 및 상세 전략이 포함된 리포트입니다."
    )
    download_cols = st.columns(2)
    with download_cols[0]:
        st.download_button(
            label="📥 전체 기록 다운로드 (CSV)",
            data=log_csv,
            file_name="ESTV_Simulation_Log.csv",
            mime="text/csv",
            disabled=log_df.empty,
            help="시뮬레이션 전체 로그를 CSV로 저장합니다."
        )
    with download_cols[1]:
        st.download_button(
            label="📥 전체 기록 다운로드 (JSON)",
            data=log_json,
            file_name="ESTV_Simulation_Log.json",
            mime="application/json",
            disabled=log_df.empty,
            help="시뮬레이션 전체 로그를 JSON으로 저장합니다."
        )
    st.markdown("---")
    st.subheader("💾 전체 분석 저장/불러오기")
    full_snapshot = build_full_snapshot(inputs, result)
    full_snapshot_json = json.dumps(full_snapshot, ensure_ascii=False, indent=2, default=str)
    st.download_button(
        label="💾 전체 분석 저장 (JSON)",
        data=full_snapshot_json,
        file_name="ESTV_Full_Analysis.json",
        mime="application/json",
        help="설정 + 결과 + 로그를 포함한 전체 분석을 저장합니다."
    )
    if st.button("🗂️ 지난 기록 저장"):
        saved_name = save_full_snapshot_to_history(full_snapshot)
        st.success(f"저장 완료: {saved_name}")
    history_files = list_history_files()
    if history_files:
        selected_history = st.selectbox(
            "지난 기록 열기",
            options=history_files,
            index=0
        )
        if st.button("📂 선택 기록 불러오기"):
            payload = load_history_file(selected_history)
            if payload:
                load_full_snapshot(payload)
                st.success("선택한 기록을 불러오는 중입니다.")
                st.rerun()
            else:
                st.info("선택한 기록을 불러오지 못했습니다.")
    uploaded_snapshot = st.file_uploader(
        "전체 분석 불러오기 (JSON)",
        type=["json"],
        key="full_snapshot_file"
    )

    # --- 처음으로 돌아가기 버튼 ---
    def reset_to_start():
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.session_state.update({
            "tutorial_step": 0,
            "step0_completed": False,
            "simulation_active": False
        })
        st.success("초기화되었습니다. 처음 화면으로 돌아갑니다.")
        st.rerun()

    st.markdown("---")
    if st.button("🏠 처음으로 돌아가기", help="모든 입력과 결과를 초기화하고 첫 화면으로 이동합니다."):
        reset_to_start()
