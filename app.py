# app.py 파일에 이 내용을 복사해 넣으세요
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib
import math
import json
import os

# NOTE: Streamlit Cloud redeploy trigger (no functional change)

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
            "sentiment_index": [],
            "sell_pressure_vol": [],
            "buy_power_vol": []
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

            marketing_dump_today = False
            if inputs.get('use_marketing_contract_scenario') and marketing_remaining > 0:
                if current_price >= marketing_cost_basis * 2.0:
                    dump_today = marketing_remaining * 0.005
                    marketing_remaining = max(marketing_remaining - dump_today, 0.0)
                    step_sell += dump_today
                    marketing_dump_today = True
                    action_logs.append({
                        "day": day_index + 1,
                        "action": "마케팅 덤핑(지속)",
                        "reason": f"가격 ${current_price:.2f} 도달, 잔여 {int(marketing_remaining):,}개"
                    })

            profit_dump_today = False
            if initial_investor_remaining > 0 and current_price >= private_sale_price * profit_taking_multiple:
                profit_dump = initial_investor_remaining * 0.01
                initial_investor_remaining = max(initial_investor_remaining - profit_dump, 0.0)
                step_sell += profit_dump
                profit_dump_today = True
                action_logs.append({
                    "day": day_index + 1,
                    "action": "초기 투자자 이익실현",
                    "reason": f"목표가 {profit_taking_multiple:.1f}x 도달, 잔여 {int(initial_investor_remaining):,}개"
                })

            prev_step_price = current_price

            total_sell = step_sell + step_turnover_sell
            effective_max_sell_ratio = max(0.0, max_sell_token_ratio - max_sell_token_ratio_delta)
            if effective_max_sell_ratio > 0:
                sell_cap = pool_token * effective_max_sell_ratio
                total_sell = min(total_sell, sell_cap)

            if max_buy_usdt_ratio > 0:
                buy_cap = pool_usdt * max_buy_usdt_ratio
                step_buy = min(step_buy, buy_cap)

            total_buy = step_buy + step_turnover_buy
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
            sentiment_index = max(0.5, min(1.5, 1.0 + (price_change_ratio * fomo_sensitivity)))

            simulation_log["day"].append(day_index + 1)
            simulation_log["price"].append(new_price)
            simulation_log["reason_code"].append(reason_code)
            simulation_log["action_needed"].append(action_needed)
            simulation_log["sentiment_index"].append(sentiment_index)
            simulation_log["sell_pressure_vol"].append(total_sell)
            simulation_log["buy_power_vol"].append(total_buy)

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


def apply_contract_inputs(base_inputs, mode):
    adjusted = dict(base_inputs)
    notes = []
    if mode == "기존 계약서":
        adjusted["initial_circulating_percent"] = 10.0
        adjusted["unbonding_days"] = 0
        adjusted["use_marketing_contract_scenario"] = True
        krw_rate = adjusted.get("krw_per_usd", 1300)
        upbit_monthly_buy = 3_500_000_000 / max(krw_rate, 1)
        adjusted["monthly_buy_volume"] = upbit_monthly_buy
        adjusted["base_monthly_buy_volume"] = upbit_monthly_buy
        adjusted["daily_user_buy_schedule"] = [upbit_monthly_buy / 30] * max(adjusted.get("simulation_days", 30), 1)
        adjusted["price_model"] = "CEX"
        adjusted["depth_usdt_1pct"] = 300_000
        adjusted["depth_usdt_2pct"] = 800_000
        adjusted["depth_growth_rate"] = 0.0
        notes.append("기존 계약서 기준 자동 적용")
    elif mode == "변동 계약서":
        adjusted["initial_circulating_percent"] = 3.0
        adjusted["unbonding_days"] = 20
    return adjusted, notes


def filter_recommended_settings(payload):
    return dict(payload), []

# ==========================================
# 2. Streamlit UI 구성
# ==========================================
st.set_page_config(page_title="ESTV 토큰 시뮬레이터", layout="wide")

st.title("📊 ESTV 토큰 상장 리스크 시뮬레이터")
st.markdown("특약 계약서(Legal)와 토크노믹스(Design) 변수를 조정하여 **미래 가격**을 예측합니다.")

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
st.sidebar.header("🛠 시나리오 설정")
def toggle_user_manual():
    st.session_state["show_user_manual"] = not st.session_state.get("show_user_manual", False)

manual_button_label = "📘 사용설명서 닫기" if st.session_state.get("show_user_manual") else "📘 사용설명서 열기"
st.sidebar.button(manual_button_label, on_click=toggle_user_manual)

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
st.sidebar.header("📜 계약 시나리오")
contract_mode = st.sidebar.radio(
    "계약 시나리오 선택",
    options=["사용자 조정", "기존 계약서", "변동 계약서", "역산목표가격"],
    index=0,
    help="기본은 사용자 조정이며, 다른 옵션은 계약/역산 기준으로 자동 적용됩니다."
)

if st.session_state.get("contract_mode_applied") != contract_mode:
    if contract_mode == "기존 계약서":
        krw_rate = st.session_state.get("krw_per_usd", 1300)
        st.session_state.update({
            "input_supply": 10.0,
            "input_unbonding": 0,
            "input_sell_ratio": 15,
            "input_buy_volume": int(3_500_000_000 / max(krw_rate, 1)),
            "scenario_preset": "직접 입력",
            "simulation_unit": "일",
            "simulation_value": 30,
            "price_model": "CEX",
            "depth_usdt_1pct": 300_000,
            "depth_usdt_2pct": 800_000,
            "depth_growth_rate": 0.0
        })
    st.session_state["contract_mode_applied"] = contract_mode

input_supply = st.sidebar.slider(
    "1. 초기 유통량 (%)",
    min_value=0.0,
    max_value=100.0,
    value=3.0,
    step=0.5,
    help="초기 유통되는 토큰 비율입니다. 높을수록 시장 유통 물량이 많아져 가격 방어가 어려울 수 있습니다.",
    key="input_supply"
)
if input_supply > 3.0:
    st.sidebar.error("🚨 특약 제5조 위반! (3% 초과)")

input_unbonding = st.sidebar.slider(
    "2. 언본딩 기간 (일)",
    min_value=0,
    max_value=90,
    value=30,
    step=10,
    help="언본딩 대기 기간입니다. 길수록 매도 지연이 커져 단기 하락 압력이 완화됩니다.",
    key="input_unbonding"
)
if input_unbonding < 30:
    st.sidebar.warning("⚠️ 특약 권장 사항 미달 (<30일)")

input_sell_ratio = st.sidebar.slider(
    "3. 락업 해제 시 매도율 (%)",
    10,
    100,
    50,
    help="락업 해제 물량 중 실제로 매도되는 비율입니다. 높을수록 가격 하방 압력이 커집니다.",
    key="input_sell_ratio"
)

st.sidebar.markdown("---")
st.sidebar.header("🔒 초기 투자자 락업/베스팅")
initial_investor_lock_months = st.sidebar.slider(
    "3-1. 초기 투자자 락업 기간 (개월)",
    min_value=0,
    max_value=60,
    value=12,
    step=1,
    help="초기 투자자 물량이 시장에 풀리기 전까지 묶이는 기간입니다."
)
initial_investor_locked_tokens = st.sidebar.number_input(
    "3-2. 락업 물량 (토큰 수)",
    min_value=0.0,
    value=0.0,
    step=1_000_000.0,
    help="초기 투자자에게 배정된 락업 토큰 수량입니다. 0이면 미적용됩니다."
)
initial_investor_vesting_months = st.sidebar.slider(
    "3-3. 베스팅 기간 (개월)",
    min_value=0,
    max_value=60,
    value=12,
    step=1,
    help="락업 종료 후 몇 개월에 걸쳐 해제할지 선택합니다."
)
initial_investor_release_percent = st.sidebar.slider(
    "3-4. 월별 해제 비율 (%)",
    min_value=1.0,
    max_value=100.0,
    value=10.0,
    step=1.0,
    help="락업 물량 중 매월 해제되는 비율입니다. 설정값에 따라 실제 베스팅 기간이 자동 보정됩니다."
)
initial_investor_release_interval = st.sidebar.slider(
    "3-5. 해제 주기 (개월)",
    min_value=1,
    max_value=12,
    value=1,
    step=1,
    help="해제 주기를 설정합니다. 예: 3개월이면 분기 단위로 해제됩니다."
)
initial_investor_sell_ratio = st.sidebar.slider(
    "3-6. 초기 투자자 해제 매도율 (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=5,
    help="초기 투자자 해제 물량 중 실제로 매도되는 비율입니다."
)
initial_investor_monthly_sell_usdt = st.sidebar.number_input(
    "3-7. 초기 투자자 월간 판매 금액 ($)",
    min_value=0.0,
    value=0.0,
    step=50_000.0,
    help="락업 해제 기간 동안 월간 추가 매도 금액(USDT 기준)을 반영합니다."
)

TOTAL_SUPPLY = 1_000_000_000
initial_investor_locked_percent = (initial_investor_locked_tokens / TOTAL_SUPPLY) * 100.0 if initial_investor_locked_tokens > 0 else 0.0
if initial_investor_locked_percent > 100.0:
    st.sidebar.error("락업 물량이 총 공급량을 초과했습니다.")

derived_vesting_months = max(1, int(math.ceil(100.0 / max(initial_investor_release_percent, 1.0))))
if initial_investor_vesting_months > 0 and initial_investor_vesting_months != derived_vesting_months:
    st.sidebar.info(f"월별 해제 비율 기준으로 베스팅 기간이 {derived_vesting_months}개월로 보정됩니다.")
if initial_investor_locked_tokens > 0:
    estimated_lock_value = initial_investor_locked_tokens * 0.50
    st.sidebar.caption(
        f"락업 물량: {int(initial_investor_locked_tokens):,}개 "
        f"(총 공급의 {initial_investor_locked_percent:.2f}%) / "
        f"예상 평가액: ${estimated_lock_value:,.0f}"
    )
input_buy_volume = st.sidebar.number_input(
    "4. 월간 매수 유입 자금 ($)",
    value=200000,
    step=50000,
    help="월간 기본 매수 유입 자금입니다. 클수록 매수 압력이 증가해 가격 상승 요인이 됩니다.",
    key="input_buy_volume"
)
use_buy_inflow_pattern = st.sidebar.checkbox(
    "월간 매수 유입 시계열 패턴 사용",
    value=False,
    help="월별 매수 유입을 패턴(초기 급증→조정→안정)으로 반영합니다."
)
pattern_month4_avg_krw = st.sidebar.slider(
    "월 4+ 평균 유입(억 KRW)",
    min_value=40,
    max_value=60,
    value=50,
    step=5,
    help="월 4 이후 장기 평균 유입 규모(억 원)입니다."
)
simulation_unit = st.sidebar.selectbox(
    "4-1. 시뮬레이션 기간 단위",
    options=["일", "월", "년"],
    index=1,
    help="기간 단위를 선택합니다.",
    key="simulation_unit"
)
simulation_value = st.sidebar.number_input(
    "4-2. 시뮬레이션 기간 값",
    min_value=1,
    value=24 if simulation_unit == "월" else 1,
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

st.sidebar.markdown("---")
st.sidebar.header("👥 기존 회원 유입 (Demand Side)")

estv_total_users = 160_000_000
st.sidebar.caption("기존 회원 수는 보수적으로 1억 6천만 명 기준을 사용합니다.")

with st.sidebar.expander("ℹ️ 유입 시나리오 도움말", expanded=False):
    st.markdown("""
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

scenario_preset = st.sidebar.selectbox(
    "시나리오 프리셋",
    options=list(preset_map.keys()),
    index=0,
    key="scenario_preset",
    on_change=apply_preset
)

conversion_rate = st.sidebar.slider(
    "5. 회원 거래소 유입 전환율 (%)",
    min_value=0.01,
    max_value=2.00,
    value=0.10,
    step=0.01,
    format="%.2f%%",
    key="conversion_rate",
    help="기존 회원 중 거래소로 유입되는 비율입니다. 높을수록 신규 유입 매수 자금이 커집니다."
)

avg_ticket = st.sidebar.number_input(
    "6. 1인당 평균 매수 금액 ($)",
    value=50,
    step=10,
    key="avg_ticket",
    help="신규 유입 1인당 평균 매수 금액입니다. 클수록 월간 추가 매수세가 증가합니다."
)

onboarding_months = 12

total_new_buyers = estv_total_users * (conversion_rate / 100.0)
total_inflow_money = total_new_buyers * avg_ticket
monthly_user_buy_volume = total_inflow_money / onboarding_months
total_inflow_days = onboarding_months * 30
base_daily_user_buy = total_inflow_money / max(total_inflow_days, 1)

use_phase_inflow = st.sidebar.checkbox(
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
    phase2_days = st.sidebar.slider(
        "Phase 2 기간(일)",
        min_value=7,
        max_value=60,
        value=30,
        step=1,
        key="phase2_days",
        help="상장 직후 집중 유입이 유지되는 기간입니다."
    )
    phase2_multiplier = st.sidebar.slider(
        "Phase 2 유입 배수",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.1,
        key="phase2_multiplier",
        help="상장 직후 유입을 몇 배로 증폭할지 설정합니다."
    )
    prelisting_days = st.sidebar.slider(
        "Phase 1 대기 기간(일)",
        min_value=7,
        max_value=60,
        value=30,
        step=1,
        key="prelisting_days",
        help="상장 전 유입이 대기(잠재 수요로 누적)되는 기간입니다."
    )
    prelisting_multiplier = st.sidebar.slider(
        "Phase 1 대기 수요 배수",
        min_value=1.0,
        max_value=5.0,
        value=1.5,
        step=0.1,
        key="prelisting_multiplier",
        help="대기 수요가 상장 직후 유입될 때의 증폭 정도입니다."
    )
    prelisting_release_days = st.sidebar.slider(
        "Phase 1 방출 기간(일)",
        min_value=1,
        max_value=30,
        value=7,
        step=1,
        key="prelisting_release_days",
        help="대기 수요가 상장 후 며칠에 걸쳐 분산 방출되는지 설정합니다."
    )

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

daily_user_buy_schedule = []
for d in range(total_days):
    if d < total_inflow_days:
        if use_phase_inflow:
            if d < prelisting_days:
                daily_user_buy_schedule.append(0.0)
            elif d < prelisting_days + phase2_days:
                release_day = d - prelisting_days
                release_ratio = min((release_day + 1) / prelisting_release_days, 1.0)
                daily_user_buy_schedule.append(phase2_daily + (prelisting_daily * release_ratio))
            else:
                daily_user_buy_schedule.append(phase3_daily)
        else:
            daily_user_buy_schedule.append(base_daily_user_buy)
    else:
        daily_user_buy_schedule.append(0.0)

st.sidebar.info(f"""
📊 **유입 분석 결과**
- 신규 유입 인원: {int(total_new_buyers):,}명
- 총 매수 대기 자금: ${int(total_inflow_money):,}
- **월간 추가 매수세: +${int(monthly_user_buy_volume):,}**
""")
if use_phase_inflow:
    st.sidebar.caption(
        f"Phase 1 대기(상장 전 {prelisting_days}일): 유입 대기 → "
        f"상장 직후 {prelisting_release_days}일 완화 방출 / "
        f"상장 직후 일 ${int(phase2_daily + prelisting_daily):,} 유입 / "
        f"Phase 3 이후: 일 ${int(phase3_daily):,} 유입"
    )

st.sidebar.markdown("---")
st.sidebar.header("🚀 Master Plan 모드")
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
    use_triggers = st.sidebar.checkbox(
        "트리거 자동 가동",
        value=True,
        key="use_triggers",
        help="가격 하락 시 사전에 정의된 캠페인을 자동 재가동하여 급락을 완화하기 위해 사용합니다."
    )
    buy_verify_boost = st.sidebar.slider(
        "Buy & Verify 매수 증폭(+)",
        0.0,
        2.0,
        0.5,
        0.1,
        key="buy_verify_boost",
        help="매수 유인을 강화해 상장 초반 수요를 끌어올립니다."
    )
    holding_suppress = st.sidebar.slider(
        "Holding 매도 억제(-)",
        0.0,
        0.3,
        0.1,
        0.01,
        key="holding_suppress",
        help="매도 심리를 억제해 단기 급락을 완화합니다."
    )
    payburn_delta = st.sidebar.slider(
        "Pay & Burn 소각 증폭(+)",
        0.0,
        0.01,
        0.002,
        0.001,
        key="payburn_delta",
        help="소각을 강화해 유통량 감소 효과를 높입니다."
    )
    buyback_daily = st.sidebar.number_input(
        "캠페인 일일 바이백($)",
        value=0,
        step=10000,
        key="buyback_daily",
        help="캠페인 기간에 실행하는 일일 바이백 예산입니다."
    )

st.sidebar.markdown("---")
st.sidebar.header("📊 마케팅 대시보드")
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

campaigns = []
triggers = []
if use_master_plan:
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

st.sidebar.markdown("---")
st.sidebar.header("📉 변동성 완화 설정")
price_model = st.sidebar.selectbox(
    "가격 모델",
    options=["AMM", "CEX", "HYBRID"],
    index=0,
    help="AMM은 풀의 상수곱(x*y=k)로 가격을 계산합니다. CEX는 오더북 깊이에 따라 체결 슬리피지를 반영합니다. HYBRID는 CEX 방식에 월별 오더북 깊이 증가를 더해 유동성 확장을 모사합니다.",
    key="price_model"
)
depth_usdt_1pct = st.sidebar.number_input(
    "오더북 1% 깊이($)",
    value=1_000_000,
    step=100_000,
    help="CEX 모델에서 ±1% 구간의 매수/매도 깊이입니다.",
    key="depth_usdt_1pct"
)
depth_usdt_2pct = st.sidebar.number_input(
    "오더북 2% 깊이($)",
    value=3_000_000,
    step=100_000,
    help="CEX 모델에서 ±2% 구간의 매수/매도 깊이입니다.",
    key="depth_usdt_2pct"
)
depth_growth_rate = st.sidebar.slider(
    "오더북 깊이 성장률(월, %)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.5,
    help="HYBRID 모델에서 월별 오더북 깊이 증가율입니다.",
    key="depth_growth_rate"
)

st.sidebar.markdown("---")
st.sidebar.header("✅ 가격 변동추이 신뢰도")
enable_confidence = st.sidebar.checkbox(
    "신뢰도 계산 활성화",
    value=False,
    help="입력값에 불확실성을 부여해 여러 번 시뮬레이션하고, 기준 추이와 유사한 비율을 신뢰도로 계산합니다."
)
confidence_runs = st.sidebar.slider(
    "시뮬레이션 횟수",
    min_value=100,
    max_value=1000,
    value=300,
    step=50,
    help="횟수가 많을수록 안정적이지만 계산 시간이 늘어납니다."
)
confidence_uncertainty = st.sidebar.slider(
    "입력값 불확실성(±%)",
    min_value=0.0,
    max_value=30.0,
    value=10.0,
    step=1.0,
    help="주요 입력값에 랜덤 변동을 주는 범위입니다."
)
confidence_mape = st.sidebar.slider(
    "허용 변동폭(평균 오차, %)",
    min_value=5.0,
    max_value=30.0,
    value=15.0,
    step=1.0,
    help="기준 추이와 평균 오차가 이 값 이하인 시뮬레이션의 비율을 신뢰도로 계산합니다."
)

st.sidebar.markdown("---")
st.sidebar.header("🇰🇷 Upbit 평균 시나리오")
show_upbit_baseline = st.sidebar.checkbox(
    "Upbit 평균 그래프 표시",
    value=False,
    help="한국 주요 거래소의 평균 추정치를 기준으로 그래프를 비교 표시합니다."
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
steps_per_month = st.sidebar.selectbox(
    "거래 분할 단위",
    options=[30, 7],
    index=0,
    format_func=lambda x: f"{x}일 분할",
    help="월간 매수/매도를 일/주 단위로 분할해 변동성을 완화합니다.",
    key="steps_per_month"
)
turnover_ratio = st.sidebar.slider(
    "신규 유입 회전율(총합, %)",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=0.5,
    help="신규 유입 매수·매도 총 회전율입니다. 비대칭 비율로 매수/매도 분배합니다.",
    key="turnover_ratio"
)
turnover_buy_share = st.sidebar.slider(
    "회전율 매수 비중(%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    step=5.0,
    help="회전율 중 매수로 반영되는 비중입니다. 나머지는 매도로 반영됩니다.",
    key="turnover_buy_share"
)
lp_growth_rate = st.sidebar.slider(
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
max_buy_usdt_ratio = st.sidebar.slider(
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
max_sell_token_ratio = st.sidebar.slider(
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
st.sidebar.header("🧠 시장 심리/비선형")
panic_sensitivity = st.sidebar.slider(
    "패닉 민감도",
    min_value=1.0,
    max_value=3.0,
    value=1.5,
    step=0.1,
    help="가격 하락 시 매도 압력을 증폭시키는 강도입니다."
)
fomo_sensitivity = st.sidebar.slider(
    "FOMO 민감도",
    min_value=1.0,
    max_value=2.0,
    value=1.2,
    step=0.1,
    help="가격 상승 시 추격 매수를 증폭시키는 강도입니다."
)
private_sale_price = st.sidebar.number_input(
    "초기 투자자 평단가($)",
    value=0.05,
    step=0.01,
    help="초기 투자자의 평균 매입 단가입니다. 이 가격 이하에서는 매도가 둔화됩니다."
)
profit_taking_multiple = st.sidebar.slider(
    "이익실현 목표 배수",
    min_value=1.0,
    max_value=10.0,
    value=5.0,
    step=0.5,
    help="초기 투자자가 평단가 대비 몇 배 상승 시 이익실현 매도를 강화할지 설정합니다."
)
arbitrage_threshold = st.sidebar.slider(
    "차익거래 임계값(%)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.5,
    help="가격 변동률이 이 값을 넘으면 차익거래 유입을 가정합니다.",
    format="%.1f%%"
)
min_depth_ratio = st.sidebar.slider(
    "패닉 시 오더북 깊이 하한",
    min_value=0.1,
    max_value=1.0,
    value=0.3,
    step=0.05,
    help="패닉 국면에서 오더북 깊이가 줄어드는 최소 비율입니다."
)

market_sentiment_config = {
    "panic_sensitivity": panic_sensitivity,
    "fomo_sensitivity": fomo_sensitivity,
    "private_sale_price": private_sale_price,
    "profit_taking_multiple": profit_taking_multiple,
    "arbitrage_threshold": arbitrage_threshold / 100.0,
    "min_depth_ratio": min_depth_ratio
}

st.sidebar.markdown("---")
st.sidebar.header("🔥 소각/바이백 정책")
burn_fee_rate = st.sidebar.slider(
    "거래 수수료 소각률(%)",
    min_value=0.0,
    max_value=2.0,
    value=0.3,
    step=0.1,
    help="거래 수수료 중 일부를 토큰으로 소각합니다. 높을수록 유통량이 줄어 가격 상승 압력이 생깁니다.",
    key="burn_fee_rate"
)
monthly_buyback_usdt = st.sidebar.number_input(
    "월간 바이백 예산($)",
    value=0,
    step=100000,
    help="광고/NFT/수수료 등 사업 수익으로 토큰을 시장에서 매수해 소각하는 예산입니다.",
    key="monthly_buyback_usdt"
)

st.sidebar.markdown("---")
st.sidebar.header("🎯 $5.00 달성 목표 시나리오")
with st.sidebar.expander("시나리오 설명", expanded=False):
    st.markdown("""
- 공급 통제: 초기 유통량 3.0%, 언본딩 30일, 매도율 30%
- 수요 폭발: 1.6억명 × 0.5% 전환율 × $100 = 월 $6.6M 유입
- 리스크 제거: 마케팅 덤핑 시나리오 비활성화
""")
with st.sidebar.expander("KPI 체크리스트 & 예상 흐름", expanded=False):
    st.markdown("""
**2. 조건별 달성 목표 (KPI Checklist)**  
시뮬레이션 결과가 현실이 되기 위한 실제 KPI입니다.

| 구분 | 조건(Variable) | 목표치(Target) | 실행 전략(Action Item) |
|---|---|---|---|
| 법적(Legal) | 초기 유통량 | 3,000만 개 (3%) | 특약 제5조 발동. 나머지 7%는 예비비로 돌려 주소 공개 후 동결(Burn/Lock) 처리 |
| 영업(Sales) | 마케팅 물량 | 시장 유통 0개 | 마케팅 계약서 1억 개를 OTC(장외) 매도 금지 및 12개월 락업 특약에 서명 |
| 마케팅(Mkt) | 유저 전환율 | 0.5% (80만 명) | 1.6억 명 대상 앱 프로모션 진행 (예: 지갑 연동 시 $5 상당 토큰 에어드랍) |
| 운영(Ops) | 평균 매수액 | $100 (약 13만 원) | 소액 매수를 유도하는 스테이킹 이자(APR) 상품 출시 |
| 기술(Tech) | 언본딩 | 30일 강제 | 스마트 컨트랙트에 `undelegate period = 30 days` 검증 보고서 공개 |

**3. 시뮬레이션 예상 결과 (Output Preview)**  
이 값을 넣고 돌렸을 때 예상되는 차트 흐름입니다.

- **1개월 ~ 3개월**
  - 초기 유통량이 적어(3%) 작은 매수세에도 가격이 빠르게 상승 ($0.50 → $1.20)
  - 30일 언본딩으로 즉시 매물이 나오지 않아 상승세 유지
- **4개월 ~ 9개월**
  - 얼리어답터 유입이 본격화
  - 월 600만 달러 매수세가 지속되며 J-Curve 형성 ($1.20 → $3.50)
- **10개월 ~ 12개월**
  - FOMO(매수 공포)로 목표가 $5.00 돌파 가능
  - 12개월 차 대규모 락업 해제(Cliff)로 조정 가능성 주의

**결론**: 유통량 3% 고정(Supply Lock), 전환율 0.5%(Demand Push)일 때 $5.00 목표는 설계 가능 영역입니다.
""")

def apply_target_scenario():
    st.session_state["apply_target_scenario"] = True

st.sidebar.button("목표 시나리오 적용", on_click=apply_target_scenario)

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
adjusted_inputs, contract_notes = apply_contract_inputs(inputs, contract_mode)
result = run_sim_with_cache(adjusted_inputs)
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

if enable_confidence:
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

with st.expander("🎯 역산 목표 가격 시뮬레이션", expanded=(contract_mode == "역산목표가격")):
    target_price = st.number_input("목표 최종 가격 ($)", min_value=0.1, value=5.0, step=0.1)
    reverse_basis = st.selectbox(
        "역산 기준",
        options=["전환율 조정", "평균 매수액 조정", "전환율+매수액 균등"],
        index=0,
        help="목표가 달성을 위해 어떤 변수를 우선 조정할지 선택합니다."
    )
    volatility_mode = st.selectbox(
        "변동성 적용 방식",
        options=["완화", "중립", "공격"],
        index=0,
        help="목표가를 맞출 때 변동성을 줄이거나(완화), 유지(중립), 높이는(공격) 방향으로 설정합니다."
    )
    auto_price_model = st.checkbox(
        "가격 모델/오더북 자동 조정",
        value=True,
        help="역산 계산 시 가격 모델과 오더북 깊이도 함께 조정합니다."
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

        st.session_state["reverse_apply_payload"] = apply_payload
        st.session_state["apply_reverse_scenario"] = True

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

go = None
try:
    go = importlib.import_module("plotly.graph_objects")
except Exception:
    go = None

if go is not None:
    days = list(range(len(series)))
    fig = go.Figure()
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
    ))
    if upbit_baseline_result:
        up_series = upbit_baseline_result["daily_price_trend"]
        up_days = list(range(len(up_series)))
        fig.add_trace(go.Scatter(
            x=up_days,
            y=up_series,
            mode="lines",
            name="Upbit 평균 시나리오",
            line=dict(color="gray", dash="dash")
        ))
    fig.add_trace(go.Scatter(
        x=[0, len(series) - 1],
        y=[0.5, 0.5],
        mode="lines",
        name="Listing Price ($0.50)",
        line=dict(color="gray", dash="dot")
    ))

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
                customdata=list(zip(xai_reason, xai_action, xai_sentiment, xai_sell, xai_buy)),
                hovertemplate=(
                    "Day %{x}<br>"
                    "Price $%{y:.4f}<br>"
                    "원인 %{customdata[0]}<br>"
                    "대응 %{customdata[1]}<br>"
                    "심리 지수 %{customdata[2]:.2f}<br>"
                    "매도 압력 %{customdata[3]:,.0f}<br>"
                    "매수 지지력 %{customdata[4]:,.0f}"
                    "<extra></extra>"
                )
            ))

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
        ))
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
        ))

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
            ))

    fig.update_layout(
        xaxis_title="Day",
        yaxis_title="Price",
        yaxis=dict(dtick=0.25),
        hovermode="closest",
        height=420,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
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

    st.info(f"가장 큰 급락은 Day {drop_day}에 발생. 원인 추정: " + ", ".join(reasons))

# 로그 테이블
if result['risk_logs']:
    st.subheader("📜 리스크 발생 로그")
    st.table(pd.DataFrame(result['risk_logs']))
if result.get("action_logs"):
    st.subheader("📌 캠페인 액션 로그")
    st.table(pd.DataFrame(result["action_logs"]))
