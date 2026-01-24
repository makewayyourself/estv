# app.py 파일에 이 내용을 복사해 넣으세요
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib
import math
import json
import os
import time

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

# NOTE: Streamlit Cloud redeploy trigger (no functional change)

RESET_DEFAULTS = {
    "mode": "tutorial",
    "mode_selector": "초보자",
    "tutorial_step": 0,
    "step0_completed": False,
    "show_user_manual": False,
    "contract_mode_applied": None,
    "contract_mode_label": "사용자 조정 (Manual)",
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
    "ai_tune_banner_ts": None,
    "reverse_target_price": 5.0,
    "reverse_basis": "전환율 조정",
    "reverse_volatility_mode": "완화",
    "reverse_auto_price_model": True,
    "project_symbol": "ESTV",
    "project_total_supply": 1_000_000_000,
    "project_pre_circulated": 0.0,
    "project_unlocked": 0.0,
    "project_holders": 0,
    "target_tier": "Tier 2 (Bybit, Gate.io, KuCoin) - Hard",
    "project_type": "New Listing (신규 상장)",
    "audit_status": "미진행",
    "concentration_ratio": 0.0,
    "has_legal_opinion": False,
    "has_whitepaper": False,
    "tutorial_target_price": 5.0,
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
    "initial_investor_monthly_sell_usdt": 0.0,
    "panic_sensitivity": 1.5,
    "fomo_sensitivity": 1.2,
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
    holders = int(inputs.get("holders", 0))
    target_tier = inputs.get("target_tier", "Tier 3")
    circ_ratio = (pre_circulated / safe_supply) * 100.0
    if circ_ratio > 30:
        warnings.append({
            "level": "CRITICAL",
            "msg": f"🚨 초기 유통량({circ_ratio:.1f}%) 과다! 거래소는 15% 미만을 선호합니다."
        })
    unlock_ratio = (unlocked / pre_circulated * 100.0) if pre_circulated > 0 else 0.0
    if unlock_ratio > 50:
        warnings.append({
            "level": "DANGER",
            "msg": f"💣 기유통 물량의 {unlock_ratio:.1f}%가 언락 상태입니다. 상장 직후 투매가 발생합니다."
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

        allocations = dict(self.base_allocations)
        initial_investor_alloc = inputs.get("initial_investor_allocation")
        if initial_investor_alloc:
            allocations["Initial_Investors"] = initial_investor_alloc
        initial_investor_remaining = 0.0
        if initial_investor_alloc:
            initial_investor_remaining = self.TOTAL_SUPPLY * initial_investor_alloc.get("percent", 0.0)

        initial_investor_sell_ratio = inputs.get("initial_investor_sell_ratio", inputs.get("sell_pressure_ratio", 0.0))
        initial_investor_sell_usdt_schedule = inputs.get("initial_investor_sell_usdt_schedule", [])

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

            month_index = (day_index // steps_per_month) + 1
            monthly_new_unlock = 0
            monthly_initial_unlock = 0
            for name, algo in allocations.items():
                unlock_amount = self._calculate_monthly_unlock(algo, month_index)
                if name == "Initial_Investors":
                    monthly_initial_unlock += unlock_amount
                else:
                    monthly_new_unlock += unlock_amount

            daily_unlock = monthly_new_unlock / steps_per_month
            daily_initial_unlock = monthly_initial_unlock / steps_per_month
            target_day = day_index + delay_days
            if target_day < len(sell_queue):
                sell_queue[target_day] += daily_unlock * inputs['sell_pressure_ratio']
                sell_queue_initial[target_day] += daily_initial_unlock

            remaining_sell = sell_queue[day_index]
            remaining_initial_sell = sell_queue_initial[day_index]
            remaining_buy = inputs['base_monthly_buy_volume']
            turnover_buy_share = inputs.get('turnover_buy_share', 0.5)
            turnover_sell_share = 1.0 - turnover_buy_share
            remaining_turnover_sell = inputs['monthly_buy_volume'] * turnover_ratio * turnover_sell_share
            remaining_turnover_buy = inputs['monthly_buy_volume'] * turnover_ratio * turnover_buy_share

            current_price = pool_usdt / pool_token
            price_change_ratio = (current_price - prev_day_price) / max(prev_day_price, 1e-9)
            depth_ratio = 1.0
            if price_model in ["CEX", "HYBRID"] and price_change_ratio < 0:
                depth_ratio = max(min_depth_ratio, 1.0 - (panic_sensitivity * abs(price_change_ratio)))
            if current_price > high_price:
                high_price = current_price

            active_campaigns = []
            for c in campaigns:
                if c["start_day"] <= day_index < c["end_day"]:
                    active_campaigns.append(c)

            for t in triggers:
                if t.get("day_start") is not None:
                    if day_index == t["day_start"] and t["name"] not in triggered_flags:
                        triggered_flags.add(t["name"])
                        activation = {
                            "name": t["name"],
                            "start_day": day_index,
                            "end_day": min(day_index + t["duration_days"], total_days),
                            "buy_multiplier": t.get("buy_multiplier", 0.0),
                            "sell_suppression_delta": t.get("sell_suppression_delta", 0.0),
                            "burn_rate_delta": t.get("burn_rate_delta", 0.0),
                            "buyback_usdt_delta": t.get("buyback_usdt_delta", 0.0),
                            "max_sell_token_ratio_delta": t.get("max_sell_token_ratio_delta", 0.0)
                        }
                        campaigns.append(activation)
                        action_logs.append({
                            "day": day_index + 1,
                            "action": t["name"],
                            "reason": "Day-window 사전 가동"
                        })

            if enable_triggers and high_price > 0:
                drawdown = (high_price - current_price) / high_price
                for t in triggers:
                    if t.get("drawdown") is None:
                        continue
                    if drawdown >= t["drawdown"] and t["name"] not in triggered_flags:
                        triggered_flags.add(t["name"])
                        activation = {
                            "name": t["name"],
                            "start_day": day_index,
                            "end_day": min(day_index + t["duration_days"], total_days),
                            "buy_multiplier": t.get("buy_multiplier", 0.0),
                            "sell_suppression_delta": t.get("sell_suppression_delta", 0.0),
                            "burn_rate_delta": t.get("burn_rate_delta", 0.0),
                            "buyback_usdt_delta": t.get("buyback_usdt_delta", 0.0),
                            "max_sell_token_ratio_delta": t.get("max_sell_token_ratio_delta", 0.0)
                        }
                        campaigns.append(activation)
                        action_logs.append({
                            "day": day_index + 1,
                            "action": t["name"],
                            "reason": f"고점 대비 {drawdown*100:.1f}% 하락"
                        })

            buy_multiplier = 1.0
            sell_suppression_delta = 0.0
            burn_rate_delta = 0.0
            buyback_usdt_delta = 0.0
            max_sell_token_ratio_delta = 0.0
            for c in active_campaigns:
                buy_multiplier += c.get("buy_multiplier", 0.0)
                sell_suppression_delta += c.get("sell_suppression_delta", 0.0)
                burn_rate_delta += c.get("burn_rate_delta", 0.0)
                buyback_usdt_delta += c.get("buyback_usdt_delta", 0.0)
                max_sell_token_ratio_delta += c.get("max_sell_token_ratio_delta", 0.0)

            # Step 2: 변수 동적 조정
            base_sell_ratio = inputs['sell_pressure_ratio']
            dynamic_sell_ratio = calculate_dynamic_sell_pressure(
                base_sell_ratio,
                current_price,
                daily_price_history,
                market_cfg
            )
            if price_model in ["CEX", "HYBRID"]:
                depth_usdt_1pct = adjust_depth_by_volatility(depth_usdt_1pct, daily_price_history, market_cfg)
                depth_usdt_2pct = adjust_depth_by_volatility(depth_usdt_2pct, daily_price_history, market_cfg)
            effective_sell_pressure = max(0.0, dynamic_sell_ratio - sell_suppression_delta)
            sell_ratio_scale = 1.0
            if base_sell_ratio > 0:
                sell_ratio_scale = effective_sell_pressure / base_sell_ratio
            step_sell = remaining_sell * sell_ratio_scale
            # Step 3: 물량 결정
            if day_index < len(initial_investor_sell_usdt_schedule) and current_price > private_sale_price:
                extra_sell_usdt = initial_investor_sell_usdt_schedule[day_index]
                if extra_sell_usdt > 0:
                    remaining_initial_sell += extra_sell_usdt / current_price
            investor_sell = get_investor_decision(
                remaining_initial_sell * initial_investor_sell_ratio,
                current_price,
                market_cfg
            )
            investor_sell = min(investor_sell, initial_investor_remaining)
            step_sell += investor_sell
            initial_investor_remaining = max(initial_investor_remaining - investor_sell, 0.0)
            daily_user_buy = 0.0
            if day_index < len(daily_user_buy_schedule):
                daily_user_buy = daily_user_buy_schedule[day_index]
            base_daily_buy = remaining_buy / steps_per_month
            base_daily_buy_schedule = inputs.get('base_daily_buy_schedule', [])
            if day_index < len(base_daily_buy_schedule):
                base_daily_buy = base_daily_buy_schedule[day_index]
            step_buy = base_daily_buy + (daily_user_buy * buy_multiplier)
            base_step_buy = step_buy
            step_buy = apply_fomo_buy(step_buy, current_price, prev_day_price, market_cfg)
            step_turnover_sell = remaining_turnover_sell / steps_per_month
            step_turnover_buy = remaining_turnover_buy / steps_per_month
            base_turnover_buy = step_turnover_buy
            step_turnover_buy = apply_fomo_buy(step_turnover_buy, current_price, prev_day_price, market_cfg)

            normal_buy_volume = base_step_buy + base_turnover_buy

            marketing_dump_today = False
            dump_today = 0.0
            if inputs.get('use_marketing_contract_scenario') and marketing_remaining > 0:
                if current_price >= marketing_cost_basis * 2.0:
                    dump_today = marketing_remaining * 0.005
                    marketing_remaining = max(marketing_remaining - dump_today, 0.0)
                    step_sell += dump_today
                    marketing_dump_today = True
                    log_reason_action("MARKETING_DUMP", "NEED_BUYBACK")
                    action_logs.append({
                        "day": day_index + 1,
                        "action": "마케팅 덤핑(지속)",
                        "reason": f"가격 ${current_price:.2f} 도달, 잔여 {int(marketing_remaining):,}개"
                    })

            profit_dump_today = False
            profit_dump = 0.0
            if initial_investor_remaining > 0 and current_price >= private_sale_price * profit_taking_multiple:
                profit_dump = initial_investor_remaining * 0.01
                initial_investor_remaining = max(initial_investor_remaining - profit_dump, 0.0)
                step_sell += profit_dump
                profit_dump_today = True
                log_reason_action("PROFIT_TAKING", "NEED_BUYBACK")
                action_logs.append({
                    "day": day_index + 1,
                    "action": "초기 투자자 이익실현",
                    "reason": f"목표가 {profit_taking_multiple:.1f}x 도달, 잔여 {int(initial_investor_remaining):,}개"
                })

            prev_step_price = current_price

            raw_total_sell = step_sell + step_turnover_sell
            effective_max_sell_ratio = max(0.0, max_sell_token_ratio - max_sell_token_ratio_delta)
            if effective_max_sell_ratio > 0:
                sell_cap = pool_token * effective_max_sell_ratio
                total_sell = min(raw_total_sell, sell_cap)
            else:
                total_sell = raw_total_sell

            base_sell_component = remaining_sell
            panic_extra = max(0.0, step_sell - (base_sell_component + investor_sell + dump_today))
            sell_sources = {
                "investor_unlock": max(investor_sell, 0.0),
                "marketing_dump": max(dump_today, 0.0),
                "turnover_sell": max(step_turnover_sell, 0.0),
                "panic_sell": max(panic_extra, 0.0)
            }
            source_total = sum(sell_sources.values())
            if source_total > 0 and total_sell < raw_total_sell:
                scale = total_sell / max(raw_total_sell, 1e-9)
                for k in sell_sources:
                    sell_sources[k] *= scale
                source_total = sum(sell_sources.values())

            if max_buy_usdt_ratio > 0:
                buy_cap = pool_usdt * max_buy_usdt_ratio
                step_buy = min(step_buy, buy_cap)

            total_buy = step_buy + step_turnover_buy
            if target_tier == "Tier 1" and current_price > prev_day_price * 1.05:
                total_sell *= 1.5
                if effective_max_sell_ratio > 0:
                    sell_cap = pool_token * effective_max_sell_ratio
                    total_sell = min(total_sell, sell_cap)
                log_reason_action("KIMCHI_PREMIUM", "INCREASE_SELL_PRESSURE")
                action_logs.append({
                    "day": day_index + 1,
                    "action": "김치 프리미엄 역풍",
                    "reason": "해외 대비 과열 가격 가정"
                })
            # Shadow AMM price for arbitrage reference
            amm_pool_token += total_sell
            amm_usdt_out = amm_pool_usdt - (amm_k / max(amm_pool_token, 1e-9))
            amm_pool_usdt -= amm_usdt_out
            amm_pool_usdt += total_buy
            amm_token_out = amm_pool_token - (amm_k / max(amm_pool_usdt, 1e-9))
            amm_pool_token -= amm_token_out
            amm_price = amm_pool_usdt / max(amm_pool_token, 1e-9)
            amm_k = amm_pool_token * amm_pool_usdt

            # Step 4: 거래 체결
            token_out = 0.0
            if price_model in ["CEX", "HYBRID"]:
                pool_token, pool_usdt, _ = self._apply_orderbook_trade(
                    pool_token,
                    pool_usdt,
                    buy_usdt=total_buy,
                    sell_token=total_sell,
                    depth_usdt_1pct=depth_usdt_1pct * depth_ratio,
                    depth_usdt_2pct=depth_usdt_2pct * depth_ratio
                )
            else:
                pool_token += total_sell
                usdt_out = pool_usdt - (k_constant / pool_token)
                pool_usdt -= usdt_out
                pool_usdt += total_buy
                token_out = pool_token - (k_constant / pool_usdt)
                pool_token -= token_out

            current_price = pool_usdt / pool_token
            # Step 5: 차익거래 체크 (CEX/HYBRID)
            if price_model in ["CEX", "HYBRID"]:
                deviation = abs(current_price - amm_price) / max(amm_price, 1e-9)
                if deviation >= arbitrage_threshold:
                    pool_usdt = max(pool_token * amm_price, 1e-9)
                    k_constant = pool_token * pool_usdt
                    current_price = pool_usdt / pool_token
                    log_reason_action("ARBITRAGE_SWAP", "STABILIZE_PRICE")
                    action_logs.append({
                        "day": day_index + 1,
                        "action": "차익거래 스왑",
                        "reason": f"CEX-DEX 괴리 {deviation*100:.2f}%"
                    })

            if price_model in ["CEX", "HYBRID"]:
                token_out = (step_buy + step_turnover_buy) / max(current_price, 1e-9)
            trade_volume_tokens = total_sell + token_out
            effective_burn_rate = max(0.0, burn_fee_rate + burn_rate_delta)
            if effective_burn_rate > 0:
                burn_tokens = trade_volume_tokens * effective_burn_rate
                pool_token = max(pool_token - burn_tokens, 1e-9)
                burned_total += burn_tokens
                k_constant = pool_token * pool_usdt

            total_buyback = monthly_buyback_usdt + (buyback_usdt_delta * steps_per_month)
            if total_buyback > 0:
                step_buyback = total_buyback / steps_per_month
                if price_model in ["CEX", "HYBRID"]:
                    token_out_buyback = step_buyback / max(current_price, 1e-9)
                    pool_usdt += step_buyback
                    pool_token = max(pool_token - token_out_buyback, 1e-9)
                else:
                    pool_usdt += step_buyback
                    token_out_buyback = pool_token - (k_constant / pool_usdt)
                    pool_token -= token_out_buyback
                burned_total += token_out_buyback
            
            new_price = pool_usdt / pool_token
            if step_lp_growth_rate > 0 and new_price > prev_step_price:
                add_usdt = pool_usdt * step_lp_growth_rate
                add_token = add_usdt / new_price
                pool_usdt += add_usdt
                pool_token += add_token
                new_price = pool_usdt / pool_token
                k_constant = pool_token * pool_usdt

            if marketing_dump_today:
                pool_usdt *= 0.8
                new_price = pool_usdt / pool_token
                log_reason_action("SUPPLY_SHOCK", "RESTORE_TRUST")
                action_logs.append({
                    "day": day_index + 1,
                    "action": "유통량 쇼크 패널티",
                    "reason": "마케팅 덤핑으로 유통량 계획 위반"
                })

            panic_triggered = dynamic_sell_ratio > base_sell_ratio * 1.1 and price_change_ratio < 0
            fomo_triggered = (step_buy > base_step_buy) or (step_turnover_buy > base_turnover_buy)
            if marketing_dump_today or profit_dump_today:
                reason_code = "WHALE_DUMP"
            elif panic_triggered:
                reason_code = "PANIC_SELL"
            elif fomo_triggered:
                reason_code = "FOMO_RALLY"
            else:
                reason_code = "NORMAL"
            if reason_code in ["PANIC_SELL", "WHALE_DUMP"]:
                action_needed = "NEED_BUYBACK"
            elif reason_code == "FOMO_RALLY":
                action_needed = "MARKETING_OP"
            else:
                action_needed = "NONE"
            if panic_triggered:
                log_reason_action("PANIC_SELL", "NEED_BUYBACK")
            if fomo_triggered:
                log_reason_action("FOMO_RALLY", "MARKETING_OP")
            sentiment_index = max(0.5, min(1.5, 1.0 + (price_change_ratio * fomo_sensitivity)))
            liquidity_depth_ratio = depth_ratio if price_model in ["CEX", "HYBRID"] else 1.0
            marketing_trigger = marketing_dump_today or (buy_multiplier > 1.0)
            whale_sell_volume = dump_today + profit_dump
            if liquidity_depth_ratio < 0.5 and price_change_ratio < 0:
                log_reason_action("LIQUIDITY_DRAIN", "ADD_LIQUIDITY")

            support_line = float(np.percentile(daily_price_history, 20)) if daily_price_history else self.LISTING_PRICE
            action_amount_usdt = 0.0
            action_message = ""
            if "LIQUIDITY_DRAIN" in day_reasons:
                sell_usdt = total_sell * max(current_price, 1e-9)
                if price_model in ["CEX", "HYBRID"]:
                    one_pct_depth = max(depth_usdt_1pct * depth_ratio, 1.0)
                    needed_lp = max(0.0, sell_usdt - one_pct_depth)
                else:
                    needed_lp = max(0.0, sell_usdt - (pool_usdt * 0.01))
                action_amount_usdt = needed_lp
                if needed_lp > 0:
                    action_message = f"⚠️ 유동성 부족! 슬리피지 안정(1%)을 위해 ${needed_lp:,.0f} 추가 LP 필요."
            if "PANIC_SELL" in day_reasons and current_price < support_line:
                buyback_needed = max(0.0, (support_line - current_price) / max(current_price, 1e-9)) * pool_usdt
                action_amount_usdt = max(action_amount_usdt, buyback_needed)
                if buyback_needed > 0:
                    action_message = f"📉 지지선 이탈! 복귀를 위해 ${buyback_needed:,.0f} 바이백 권장."

            simulation_log["day"].append(day_index + 1)
            simulation_log["price"].append(new_price)
            simulation_log["reason_code"].append(reason_code)
            action_needed_display = action_needed
            if action_message:
                action_needed_display = f"{action_needed} | {action_message}"
            simulation_log["action_needed"].append(action_needed_display)
            simulation_log["reason"].append(", ".join(day_reasons) if day_reasons else "NONE")
            simulation_log["action"].append(", ".join(day_actions) if day_actions else "NONE")
            simulation_log["sentiment_index"].append(sentiment_index)
            simulation_log["sell_pressure_vol"].append(total_sell)
            simulation_log["buy_power_vol"].append(total_buy)
            simulation_log["liquidity_depth_ratio"].append(liquidity_depth_ratio)
            simulation_log["marketing_trigger"].append(marketing_trigger)
            simulation_log["whale_sell_volume"].append(whale_sell_volume)
            simulation_log["normal_buy_volume"].append(normal_buy_volume)
            simulation_log["sell_sources"].append(sell_sources)
            source_label = {
                "investor_unlock": "초기 투자자 물량",
                "marketing_dump": "마케팅 물량",
                "turnover_sell": "회전율 매도",
                "panic_sell": "심리 매도"
            }
            if source_total > 0:
                major_source = max(sell_sources.items(), key=lambda x: x[1])
                major_ratio = (major_source[1] / source_total) * 100
                ratio_parts = []
                for key, val in sell_sources.items():
                    if val <= 0:
                        continue
                    ratio_parts.append(f"{source_label.get(key, key)} {val / source_total * 100:.0f}%")
                ratio_text = ", ".join(ratio_parts) if ratio_parts else "비중 계산 불가"
                source_text = (
                    f"오늘 하락의 주범: {source_label.get(major_source[0], major_source[0])}"
                    f" ({major_ratio:.0f}%) · 비중: {ratio_text}"
                )
            else:
                source_text = "오늘 하락의 주범: 데이터 없음"
            simulation_log["sell_source_text"].append(source_text)
            simulation_log["action_amount_usdt"].append(action_amount_usdt)
            simulation_log["action_message"].append(action_message)

            daily_price_history.append(new_price)
            price_history.append(new_price)
            
            current_drop = (new_price - self.LISTING_PRICE) / self.LISTING_PRICE * 100
            if current_drop < -20 and "Warning" not in [x['level'] for x in risk_log]:
                risk_log.append({"month": month_index, "level": "Warning", "msg": f"가격 -20% 돌파 (${new_price:.2f})"})
            if current_drop < -50 and "Danger" not in [x['level'] for x in risk_log]:
                risk_log.append({"month": month_index, "level": "Danger", "msg": f"가격 반토막 (${new_price:.2f})"})
                
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
            "simulation_log": simulation_log
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

if st.session_state.get("hard_reset_pending"):
    hard_reset_session()
    st.rerun()

st.title("📊 ESTV 토큰 상장 리스크 & 수급 시뮬레이터")
st.markdown(
    "계약 시나리오와 토크노믹스 입력(유통·언본딩·유입·유동성·방어 정책)을 바탕으로 "
    "**가격 추이와 리스크를 시뮬레이션**합니다."
)
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
else:
    symbol = st.session_state.get("project_symbol", "ESTV")
    total_supply_input = float(st.session_state.get("project_total_supply", 1_000_000_000))
    pre_circulated = float(st.session_state.get("project_pre_circulated", 0.0))
    unlocked = float(st.session_state.get("project_unlocked", 0.0))
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
                ["사용자 조정 (Manual)", "기존 계약서 (Legacy)", "변동 계약서 (Dynamic)"],
                index=0,
                key="contract_mode_label"
            )
            contract_mode = "사용자 조정"
            if "기존 계약서" in contract_mode_label:
                contract_mode = "기존 계약서"
            elif "변동 계약서" in contract_mode_label:
                contract_mode = "변동 계약서"
            st.session_state["contract_mode"] = contract_mode
            if "사용자 조정" in contract_mode_label:
                st.sidebar.info("ℹ️ 가이드: 각 설정값을 사용자가 직접 정하면, 실시간으로 AI가 그에 따른 결과값을 계산하여 보여줍니다.")

            st.sidebar.markdown("---")
            target_price = st.sidebar.number_input(
                "목표가 조정 ($)",
                value=float(st.session_state.get("tutorial_target_price", 5.0)),
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
                max_value=max(0.1, min(100.0, pre_circ_ratio)),
                value=min(st.session_state.get("input_supply", 3.0), 10.0),
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
            st.sidebar.button("🚀 시뮬레이션 결과 확인하기")

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
    initial_investor_monthly_sell_usdt = 0.0
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
    initial_investor_monthly_sell_usdt = 0.0
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
        ["사용자 조정 (Manual)", "기존 계약서 (Legacy)", "변동 계약서 (Dynamic)"],
        index=0,
        key="contract_mode_label",
        help="시뮬레이션 방식을 먼저 선택합니다."
    )
    contract_mode = "사용자 조정"
    if "기존 계약서" in contract_mode_label:
        contract_mode = "기존 계약서"
    elif "변동 계약서" in contract_mode_label:
        contract_mode = "변동 계약서"
    st.session_state["contract_mode"] = contract_mode
    if "사용자 조정" in contract_mode_label:
        st.sidebar.info("ℹ️ 가이드: 각 설정값을 사용자가 직접 정하면, 실시간으로 AI가 그에 따른 결과값을 계산하여 보여줍니다.")

    st.sidebar.markdown("---")
    target_price = st.sidebar.number_input(
        "목표가 조정 ($)",
        value=float(st.session_state.get("tutorial_target_price", 5.0)),
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
        max_value=max(0.1, min(100.0, pre_circ_ratio)),
        value=3.0,
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
        help="초기 투자자 해제 물량 중 실제로 매도되는 비율입니다.",
        key="initial_investor_sell_ratio"
    )
    initial_investor_monthly_sell_usdt = investor_expander.number_input(
        "3-7. 초기 투자자 월간 판매 금액 ($)",
        min_value=0.0,
        value=0.0,
        step=50_000.0,
        help="락업 해제 기간 동안 월간 추가 매도 금액(USDT 기준)을 반영합니다.",
        key="initial_investor_monthly_sell_usdt"
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
        max_value=2.0,
        value=0.3,
        step=0.1,
        help="거래 수수료 중 일부를 토큰으로 소각합니다. 높을수록 유통량이 줄어 가격 상승 압력이 생깁니다.",
        key="burn_fee_rate"
    )

    sentiment_expander = st.sidebar.expander("🧠 시장 심리/비선형", expanded=is_expert)
    panic_sensitivity = sentiment_expander.slider(
        "패닉 민감도",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="가격 하락 시 매도 압력을 증폭시키는 강도입니다.",
        key="panic_sensitivity"
    )
    fomo_sensitivity = sentiment_expander.slider(
        "FOMO 민감도",
        min_value=1.0,
        max_value=2.0,
        value=1.2,
        step=0.1,
        help="가격 상승 시 추격 매수를 증폭시키는 강도입니다.",
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
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05,
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
    if initial_investor_monthly_sell_usdt > 0:
        lock_days = int(initial_investor_lock_months * steps_per_month)
        vesting_days = max(1, int(vesting_months_used * steps_per_month)) if vesting_months_used > 0 else 1
        daily_sell_usdt = initial_investor_monthly_sell_usdt / max(steps_per_month, 1)
        end_day = min(lock_days + vesting_days, total_days)
        for d in range(lock_days, end_day):
            initial_investor_sell_usdt_schedule[d] = daily_sell_usdt

# 시뮬레이션 실행
engine = TokenSimulationEngine()
inputs = {
    'target_tier': target_tier_key,
    'total_supply': total_supply_input,
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
if reset_triggered:
    result = build_reset_result(inputs, total_days)
    upbit_baseline_result = None
    st.session_state["reset_triggered"] = False
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

if enable_confidence and not reset_triggered:
    confidence_result = run_confidence_with_cache(
        adjusted_inputs,
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
st.subheader("📈 가격 변동 추이 (일 단위)")
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
    fig.add_trace(go.Scatter(
        x=days,
        y=series,
        mode="lines",
        name="ESTV Price ($)",
        line=dict(color="blue" if result['legal_check'] else "red")
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

        # Battlefield view: buy vs sell bars
        sell_vols = log.get("sell_pressure_vol", [])
        buy_vols = log.get("buy_power_vol", [])
        if sell_vols and buy_vols:
            bar_days = list(range(min(len(sell_vols), len(buy_vols), len(series))))
            fig.add_trace(go.Bar(
                x=bar_days,
                y=[sell_vols[i] for i in bar_days],
                name="매도 압력",
                marker_color="rgba(255, 0, 0, 0.6)"
            ), row=2, col=1)
            fig.add_trace(go.Bar(
                x=bar_days,
                y=[buy_vols[i] for i in bar_days],
                name="매수 지지력",
                marker_color="rgba(0, 180, 0, 0.6)"
            ), row=2, col=1)

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
        xaxis_title="Day",
        hovermode="closest",
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        barmode="overlay"
    )
    fig.update_yaxes(title_text="Price", dtick=0.25, row=1, col=1)
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
                    "panic_sell": "심리 매도"
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
    for event in result.get("daily_events", []):
        if event["type"] == "MarketingDump" and abs(event["day"] - drop_day) <= 2:
            reasons.append("마케팅 덤핑 발생 영향")
            break
    unbonding_days = inputs.get("unbonding_days", 0)
    cliff_days = [
        alloc["cliff"] * inputs["steps_per_month"]
        for alloc in engine.base_allocations.values()
        if alloc.get("cliff", 0) > 0
    ]
    cliff_sell_days = [d + unbonding_days for d in cliff_days]
    if cliff_sell_days and any(drop_day >= d and abs(drop_day - d) <= inputs["steps_per_month"] // 2 for d in cliff_sell_days):
        reasons.append("클리프 해제 이후 언본딩 경과 매도 증가")
    if inputs["sell_pressure_ratio"] > 0.3 and drop_day >= unbonding_days:
        reasons.append("락업 해제 매도율이 높음(언본딩 이후)")
    if inputs["turnover_ratio"] > 0:
        reasons.append("신규 유입 회전율로 추가 매도 발생")
    if not reasons:
        reasons.append("유동성 대비 거래량이 커 가격 민감도가 높음")

    source_note = ""
    log = result.get("simulation_log", {})
    if log:
        src_list = log.get("sell_source_text", [])
        if drop_day - 1 < len(src_list):
            source_note = f" (출처: {src_list[drop_day - 1]})"
    st.info(f"가장 큰 급락은 Day {drop_day}에 발생. 원인 추정: " + ", ".join(reasons) + source_note)

# 로그 테이블
if result['risk_logs']:
    st.subheader("📜 리스크 발생 로그")
    st.table(pd.DataFrame(result['risk_logs']))
if result.get("action_logs"):
    st.subheader("📌 캠페인 액션 로그")
    st.table(pd.DataFrame(result["action_logs"]))
