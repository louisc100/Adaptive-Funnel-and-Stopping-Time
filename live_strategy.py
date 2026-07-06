"""
Live dry-run adapter for the trading algorithm.

This module keeps broker connectivity out of the strategy logic. It receives
completed mid-price bars, reuses the existing research strategy from main.py,
and emits terminal-friendly HOLD / WOULD BUY / WOULD SELL messages.

No orders are placed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np

from main import (
    estimate_sigma_seed,
    garch_volatility_estimates,
    kalman_drift_estimates,
    run_strategy_on_log_prices,
    summarize_simulation,
)


@dataclass
class MidBar:
    timestamp: str
    open_mid: float
    high_mid: float
    low_mid: float
    close_mid: float
    close_bid: float
    close_ask: float
    avg_spread_bps: float


@dataclass
class StrategySignal:
    timestamp: str
    symbol: str
    bars: int
    close_mid: float
    close_bid: float
    close_ask: float
    position: str
    signal: str
    z: float | None
    realized_w: float
    mtm_w: float
    trades: int
    note: str = ""
    blocked: bool = False
    block_reason: str = ""


class MidBarBuilder:
    """Aggregate quote mids into fixed-duration bars."""

    def __init__(self, bar_seconds=60):
        self.bar_seconds = max(float(bar_seconds), 1.0)
        self._bucket_start = None
        self._mids = []
        self._spreads_bps = []
        self._close_bid = None
        self._close_ask = None

    def update(self, quote_row):
        """Add a quote row and return a completed MidBar when one closes."""
        if quote_row.mid is None or quote_row.bid is None or quote_row.ask is None:
            return None

        now = datetime.fromisoformat(quote_row.timestamp)
        if self._bucket_start is None:
            self._bucket_start = now

        elapsed = (now - self._bucket_start).total_seconds()
        if elapsed >= self.bar_seconds and self._mids:
            bar = self._finish_bar(quote_row.timestamp)
            self._bucket_start = now
            self._mids = []
            self._spreads_bps = []
            self._close_bid = None
            self._close_ask = None
        else:
            bar = None

        self._mids.append(float(quote_row.mid))
        if quote_row.spread_bps is not None:
            self._spreads_bps.append(float(quote_row.spread_bps))
        self._close_bid = float(quote_row.bid)
        self._close_ask = float(quote_row.ask)
        return bar

    def _finish_bar(self, timestamp):
        mids = np.asarray(self._mids, dtype=float)
        avg_spread_bps = (
            float(np.mean(self._spreads_bps)) if self._spreads_bps else np.nan
        )
        return MidBar(
            timestamp=timestamp,
            open_mid=float(mids[0]),
            high_mid=float(np.max(mids)),
            low_mid=float(np.min(mids)),
            close_mid=float(mids[-1]),
            close_bid=float(self._close_bid),
            close_ask=float(self._close_ask),
            avg_spread_bps=avg_spread_bps,
        )


class LiveDryRunStrategy:
    """Re-evaluate the current bar path and report the latest strategy signal."""

    def __init__(
        self,
        symbol,
        k=1.2,
        delta=8,
        entry_delta=None,
        cost=0.0,
        drift_process_var=1e-7,
        max_funnel_lookback=60,
        trailing_stop=0.04,
        trend_entry_z=0.35,
        execution_spread=0.0,
        execution_slippage=0.0,
        initial_position=None,
        initial_avg_cost=None,
        regular_hours_only=False,
        max_spread_bps=None,
        max_quote_age=None,
        timezone="America/New_York",
        fixed_buy_fee=0.0,
        fixed_sell_fee=0.0,
        min_sell_profit=0.0,
        min_sell_profit_per_share=0.0,
        profit_target_mode="per_share",
        order_quantity=1,
        initial_buy_if_cash=False,
        capital_budget=1000.0,
    ):
        self.symbol = symbol.upper()
        self.k = float(k)
        self.delta = int(delta)
        self.entry_delta = int(self.delta if entry_delta is None else entry_delta)
        self.cost = float(cost)
        self.drift_process_var = float(drift_process_var)
        self.max_funnel_lookback = int(max_funnel_lookback)
        self.trailing_stop = float(trailing_stop)
        self.trend_entry_z = float(trend_entry_z)
        self.execution_spread = float(execution_spread)
        self.execution_slippage = float(execution_slippage)
        self.initial_position = None if initial_position is None else float(initial_position)
        self.initial_avg_cost = None if initial_avg_cost is None else float(initial_avg_cost)
        self.regular_hours_only = bool(regular_hours_only)
        self.max_spread_bps = None if max_spread_bps is None else float(max_spread_bps)
        self.max_quote_age = None if max_quote_age is None else float(max_quote_age)
        self.timezone = ZoneInfo(timezone)
        self.fixed_buy_fee = max(float(fixed_buy_fee), 0.0)
        self.fixed_sell_fee = max(float(fixed_sell_fee), 0.0)
        self.min_sell_profit = max(float(min_sell_profit), 0.0)
        self.min_sell_profit_per_share = max(float(min_sell_profit_per_share), 0.0)
        self.profit_target_mode = str(profit_target_mode)
        if self.profit_target_mode not in ("fixed", "per_share", "max"):
            self.profit_target_mode = "per_share"
        self.order_quantity = max(int(order_quantity), 1)
        self.initial_buy_if_cash = bool(initial_buy_if_cash)
        self.capital_budget = max(float(capital_budget), 0.0)
        self.live_cash = self._initial_live_cash()
        self._initial_buy_emitted = False
        self.bars = []
        self.display_bars = []
        self.display_funnel_mid = []
        self.display_funnel_up = []
        self.display_funnel_low = []
        self.display_z = []
        self.cycle_start_display_index = 0
        self.latest_data = None
        self.filled_markers = []
        self.cash_anchor_price = None
        self.cash_anchor_timestamp = None
        self.last_live_exit_reason = ""

    def _initial_live_cash(self):
        """Model cash remaining after the current known account position."""
        if (
            self.initial_position is not None
            and self.initial_position > 0
            and self.initial_avg_cost is not None
            and self.initial_avg_cost > 0
        ):
            buy_cost = (
                self.initial_position
                * self.initial_avg_cost
                * (1.0 + max(self.cost, 0.0))
                + self.fixed_buy_fee
            )
            return self.capital_budget - buy_cost
        return self.capital_budget

    def on_bar(self, bar, quote_age_seconds=None):
        blocked, block_reason = self._check_guards(bar, quote_age_seconds)
        self.display_bars.append(bar)
        self.display_funnel_mid.append(np.nan)
        self.display_funnel_up.append(np.nan)
        self.display_funnel_low.append(np.nan)
        self.display_z.append(np.nan)
        self.bars.append(bar)
        self._ensure_open_position_marker()
        if len(self.bars) < 2:
            return StrategySignal(
                timestamp=bar.timestamp,
                symbol=self.symbol,
                bars=len(self.bars),
                close_mid=bar.close_mid,
                close_bid=bar.close_bid,
                close_ask=bar.close_ask,
                position="WARMUP",
                signal="WAIT",
                z=None,
                realized_w=1000.0,
                mtm_w=1000.0,
                trades=0,
                note="Need at least two completed bars.",
                blocked=blocked,
                block_reason=block_reason,
            )

        strategy_bars = self._bars_for_strategy()
        mids = np.asarray([b.close_mid for b in strategy_bars], dtype=float)
        lp = np.log(mids)
        sigma_seed = estimate_sigma_seed(lp)
        effective_cost = self._effective_proportional_cost(bar.close_mid)
        if self.initial_position is not None and self.initial_position > 0:
            data = self._long_diagnostics(lp, effective_cost)
        else:
            data = run_strategy_on_log_prices(
                lp,
                k=self.k,
                delta=min(self.delta, max(1, len(lp) - 2)),
                sigma_seed=sigma_seed,
                c_buy=effective_cost["buy"],
                c_sell=effective_cost["sell"],
                drift_process_var=self.drift_process_var,
                max_funnel_lookback=self.max_funnel_lookback,
                trailing_stop=self.trailing_stop,
                trend_entry_z=self.trend_entry_z,
                execution_spread=self.execution_spread,
                execution_slippage=self.execution_slippage,
                mode="IBKR dry-run",
            )
        self.latest_data = data
        self._update_display_overlays(data)
        t = data["N"]
        buy_now = t in data["buy_times"]
        sell_now = t in data["sell_times"]
        cash_reentry = None
        if self.initial_position is not None and self.initial_position <= 0:
            cash_reentry = self._cash_reentry_signal(bar, blocked)
            if cash_reentry is not None:
                data = cash_reentry["data"]
                self.latest_data = data
                t = data["N"]
                buy_now = cash_reentry["buy_now"]
        research_position = "LONG" if bool(data["holding"][t]) else "CASH"
        if self.initial_position is None:
            position = research_position
            if buy_now and t == 1:
                signal = "INIT LONG"
                note = "Research strategy starts with an initial reference position."
            elif buy_now:
                signal = "WOULD BUY"
                note = "Dry run only; no order placed."
            elif sell_now:
                signal = "WOULD SELL"
                note = "Dry run only; no order placed."
            else:
                signal = "HOLD"
                note = ""
        elif self.initial_position > 0:
            position = "LONG"
            live_sell = self._live_long_sell_signal(bar, data)
            if blocked:
                signal = f"BLOCKED {block_reason}"
                note = (
                    f"Account-aware dry run; current account holds "
                    f"{self.initial_position:g} shares."
                )
            elif live_sell["sell_now"]:
                signal = "WOULD SELL"
                note = (
                    f"Account-aware dry run; current account holds "
                    f"{self.initial_position:g} shares. "
                    f"{live_sell['reason']}"
                )
            else:
                signal = "HOLD LONG"
                suffix = f" {live_sell['reason']}" if live_sell["reason"] else ""
                note = (
                    f"Account-aware dry run; current account holds "
                    f"{self.initial_position:g} shares."
                    f"{suffix}"
                )
        else:
            position = "CASH"
            if blocked:
                signal = f"BLOCKED {block_reason}"
                note = "Account-aware dry run; current account has no position."
            elif self.initial_buy_if_cash and not self._initial_buy_emitted:
                signal = "WOULD BUY"
                note = (
                    "GUI-style initial entry: account is cash, so the live "
                    "strategy starts by buying once to create tau_b."
                )
                self._initial_buy_emitted = True
            elif buy_now and t != 1:
                signal = "WOULD BUY"
                note = (
                    "Account-aware dry run; cash re-entry from last filled sell."
                    if cash_reentry is not None
                    else "Account-aware dry run; current account has no position."
                )
            else:
                signal = "HOLD CASH"
                note = "Account-aware dry run; current account has no position."

        metrics = summarize_simulation(data, t)
        z = data["Zsig"][t]
        return StrategySignal(
            timestamp=bar.timestamp,
            symbol=self.symbol,
            bars=len(self.bars),
            close_mid=bar.close_mid,
            close_bid=bar.close_bid,
            close_ask=bar.close_ask,
            position=position,
            signal=signal,
            z=None if np.isnan(z) else float(z),
            realized_w=float(data["realized_w"][t]),
            mtm_w=float(data["portfolio_w"][t]),
            trades=int(metrics["completed_trades"]),
            note=note,
            blocked=blocked,
            block_reason=block_reason,
        )

    def _long_diagnostics(self, lp, effective_cost):
        """Compute live long diagnostics without virtual buy/sell transitions."""
        n = len(lp)
        buy_t = 1 if n > 1 else 0
        buy_lp = lp[buy_t]
        log_returns = np.diff(lp)
        sigma_seed = estimate_sigma_seed(lp)
        mu_step, _ = kalman_drift_estimates(
            log_returns,
            init_mu=0.0,
            obs_var=sigma_seed * sigma_seed,
            process_var=self.drift_process_var,
        )
        var_step, _ = garch_volatility_estimates(
            log_returns,
            mu_prior=mu_step,
            init_var=sigma_seed * sigma_seed,
        )
        cumulative_drift = np.zeros(n)
        cumulative_var = np.zeros(n)
        cumulative_drift[1:] = np.cumsum(mu_step[1:])
        cumulative_var[1:] = np.cumsum(var_step[1:])

        zsig = np.full(n, np.nan)
        funnel_mid = np.full(n, np.nan)
        funnel_up = np.full(n, np.nan)
        funnel_low = np.full(n, np.nan)
        portfolio_w = np.full(n, 1000.0)
        liquidation_cost = (1.0 - effective_cost["sell"]) / (1.0 + effective_cost["buy"])

        for t in range(buy_t, n):
            el = t - buy_t
            ref_t = buy_t
            ref_lp = buy_lp
            if self.max_funnel_lookback is not None and el > self.max_funnel_lookback:
                ref_t = t - self.max_funnel_lookback
                ref_lp = lp[ref_t]
            drift = cumulative_drift[t] - cumulative_drift[ref_t]
            var = cumulative_var[t] - cumulative_var[ref_t]
            denom = np.sqrt(var)
            center = ref_lp + drift
            width = self.k * denom
            funnel_mid[t] = np.exp(center)
            funnel_up[t] = np.exp(center + width)
            funnel_low[t] = np.exp(center - width)
            zsig[t] = (lp[t] - ref_lp - drift) / denom if denom > 0 else np.nan
            portfolio_w[t] = 1000.0 * np.exp(lp[t] - buy_lp) * liquidation_cost

        return {
            "N": n - 1,
            "k": self.k,
            "trend_entry_z": self.trend_entry_z,
            "Zsig": zsig,
            "holding": np.ones(n, dtype=bool),
            "buy_times": set(),
            "sell_times": set(),
            "realized_w": np.full(n, 1000.0),
            "portfolio_w": portfolio_w,
            "buy_hold_w": portfolio_w.copy(),
            "funnel_mid": funnel_mid,
            "funnel_up": funnel_up,
            "funnel_low": funnel_low,
            "trade_log": [],
        }

    def _minimum_profitable_sell_price(self):
        entry_price = self.initial_avg_cost
        if entry_price is None or entry_price <= 0:
            return None
        quantity = max(int(self.order_quantity), 1)
        prop_cost = max(float(self.cost), 0.0)
        if prop_cost >= 1.0:
            return None
        target_profit = self._target_profit(quantity)
        required_proceeds = (
            entry_price * quantity * (1.0 + prop_cost)
            + self.fixed_buy_fee
            + self.fixed_sell_fee
            + target_profit
        )
        raw_limit = required_proceeds / (quantity * (1.0 - prop_cost))
        return np.ceil(raw_limit * 100.0) / 100.0

    def _live_long_sell_signal(self, bar, data):
        """Causal sell decision from actual filled anchor, not virtual trades."""
        result = {"sell_now": False, "reason": ""}
        if self.initial_avg_cost is None or self.initial_avg_cost <= 0:
            return result
        if not self.bars:
            return result

        mids = np.asarray(
            [self.initial_avg_cost] + [b.close_mid for b in self.bars],
            dtype=float,
        )
        peak_mid = float(np.max(mids))
        current_mid = float(bar.close_mid)
        current_bid = float(bar.close_bid)
        floor = self._minimum_profitable_sell_price()
        floor_ok = floor is None or current_bid >= floor

        trail_drawdown = np.log(peak_mid) - np.log(current_mid)
        trailing_exit = (
            self.trailing_stop is not None
            and trail_drawdown >= self.trailing_stop
        )

        t = int(data["N"])
        z = data["Zsig"][t]
        lp = np.log([b.close_mid for b in self._bars_for_strategy()])
        d = min(self.delta, max(1, len(lp) - 2))
        momentum = lp[t] - lp[t - d] if t >= d else np.nan
        funnel_exit = (
            t >= d + 1
            and not np.isnan(z)
            and z > self.k
            and not np.isnan(momentum)
            and momentum <= 0.0
        )

        if trailing_exit or funnel_exit:
            reason = "Trailing exit" if trailing_exit else "Funnel exit"
            threshold = peak_mid * np.exp(-float(self.trailing_stop or 0.0))
            if floor_ok:
                result["sell_now"] = True
                floor_text = f"{floor:.2f}" if floor is not None else "n/a"
                result["reason"] = (
                    f"{reason}: bid {current_bid:.2f}, "
                    f"floor {floor_text}, "
                    f"trail threshold {threshold:.2f}."
                )
            elif floor is not None:
                result["reason"] = (
                    f"{reason} active, but bid {current_bid:.2f} is below "
                    f"profit floor {floor:.2f}."
                )
        return result

    def _check_guards(self, bar, quote_age_seconds):
        if self.regular_hours_only and not self._is_regular_hours(bar.timestamp):
            return True, "MARKET_CLOSED"
        if (
            self.max_spread_bps is not None
            and not np.isnan(bar.avg_spread_bps)
            and bar.avg_spread_bps > self.max_spread_bps
        ):
            return True, "WIDE_SPREAD"
        if (
            self.max_quote_age is not None
            and quote_age_seconds is not None
            and quote_age_seconds > self.max_quote_age
        ):
            return True, "STALE_QUOTE"
        return False, ""

    def _effective_proportional_cost(self, reference_price):
        notional = max(float(reference_price) * self.order_quantity, 1e-12)
        target_profit = self._target_profit(self.order_quantity)
        return {
            "buy": self.cost + self.fixed_buy_fee / notional,
            "sell": self.cost + (self.fixed_sell_fee + target_profit) / notional,
        }

    def _target_profit(self, quantity=None):
        quantity = max(int(self.order_quantity if quantity is None else quantity), 1)
        fixed_profit = self.min_sell_profit
        per_share_profit = self.min_sell_profit_per_share * quantity
        if self.profit_target_mode == "fixed":
            return fixed_profit
        if self.profit_target_mode == "per_share":
            return per_share_profit
        return max(fixed_profit, per_share_profit)

    def apply_filled_order(self, action, quantity, price):
        """Synchronize live strategy state after the bridge receives a fill."""
        action = action.upper()
        quantity = max(float(quantity), 0.0)
        price = None if price is None else float(price)
        current_position = 0.0 if self.initial_position is None else self.initial_position

        if action == "BUY":
            new_position = current_position + quantity
            self.initial_position = new_position
            if price is not None and price > 0:
                self.live_cash -= (
                    quantity * price * (1.0 + max(self.cost, 0.0))
                    + self.fixed_buy_fee
                )
            if price is not None and price > 0:
                self.initial_avg_cost = price
            self.cash_anchor_price = None
            self.cash_anchor_timestamp = None
            self._initial_buy_emitted = True
            # A filled buy creates a fresh tau_b anchor for subsequent sell logic.
            fill_index = max(len(self.display_bars) - 1, 0)
            if price is not None and price > 0 and 0 <= fill_index < len(self.display_funnel_mid):
                self.display_funnel_mid[fill_index] = price
                self.display_funnel_up[fill_index] = price
                self.display_funnel_low[fill_index] = price
                if np.isnan(self.display_z[fill_index]):
                    self.display_z[fill_index] = 0.0
            self.bars = []
            self.cycle_start_display_index = len(self.display_bars)
            self.latest_data = None
            self.filled_markers.append({
                "action": "BUY",
                "index": fill_index,
                "price": price,
            })
        elif action == "SELL":
            marker_index = max(len(self.display_bars) - 1, 0)
            self.filled_markers.append({
                "action": "SELL",
                "index": marker_index,
                "price": price,
            })
            new_position = max(current_position - quantity, 0.0)
            self.initial_position = new_position
            if price is not None and price > 0:
                self.live_cash += (
                    quantity * price * (1.0 - max(self.cost, 0.0))
                    - self.fixed_sell_fee
                )
            if new_position <= 0:
                self.initial_avg_cost = None
                # A filled sell creates tau_s for cash re-entry logic.
                if price is not None and price > 0:
                    if 0 <= marker_index < len(self.display_funnel_mid):
                        self.display_funnel_mid[marker_index] = price
                        self.display_funnel_up[marker_index] = price
                        self.display_funnel_low[marker_index] = price
                        if np.isnan(self.display_z[marker_index]):
                            self.display_z[marker_index] = 0.0
                    self.cash_anchor_price = price
                    self.cash_anchor_timestamp = (
                        self.display_bars[-1].timestamp if self.display_bars else None
                    )
                self.bars = []
                self.cycle_start_display_index = len(self.display_bars)
                self.latest_data = None

    def _ensure_open_position_marker(self):
        """Recover a buy anchor marker when TWS says we already hold shares."""
        if (
            self.initial_position is None
            or self.initial_position <= 0
            or self.initial_avg_cost is None
            or self.initial_avg_cost <= 0
            or not self.display_bars
        ):
            return
        last_action = None
        if self.filled_markers:
            last_action = self.filled_markers[-1].get("action")
        if last_action == "BUY":
            return

        marker_index = max(len(self.display_bars) - 1, 0)
        price = float(self.initial_avg_cost)
        if 0 <= marker_index < len(self.display_funnel_mid):
            self.display_funnel_mid[marker_index] = price
            self.display_funnel_up[marker_index] = price
            self.display_funnel_low[marker_index] = price
            if np.isnan(self.display_z[marker_index]):
                self.display_z[marker_index] = 0.0
        self.filled_markers.append({
            "action": "BUY",
            "index": marker_index,
            "price": price,
            "recovered": True,
        })
        self._initial_buy_emitted = True

    def live_account_snapshot(self, bar=None):
        """Actual-dollar view using integer shares, fills, cash, and exit costs."""
        quantity = 0.0 if self.initial_position is None else float(self.initial_position)
        bid = None if bar is None else self._clean_float(bar.close_bid)
        mid = None if bar is None else self._clean_float(bar.close_mid)
        mark_price = bid if bid is not None else mid
        position_value = None
        liquidation_value = self.live_cash
        estimated_exit_fee = 0.0
        if quantity > 0 and mark_price is not None:
            position_value = quantity * mark_price
            estimated_exit_fee = self.fixed_sell_fee
            liquidation_value = (
                self.live_cash
                + quantity * mark_price * (1.0 - max(self.cost, 0.0))
                - estimated_exit_fee
            )
        pnl = liquidation_value - self.capital_budget
        ret = pnl / self.capital_budget if self.capital_budget > 0 else np.nan
        return {
            "capital_budget": self._clean_float(self.capital_budget),
            "cash": self._clean_float(self.live_cash),
            "shares": self._clean_float(quantity),
            "mark_price": self._clean_float(mark_price),
            "position_value": self._clean_float(position_value),
            "estimated_exit_fee": self._clean_float(estimated_exit_fee),
            "liquidation_value": self._clean_float(liquidation_value),
            "pnl": self._clean_float(pnl),
            "return": self._clean_float(ret),
        }

    def _bars_for_strategy(self):
        """Add a synthetic entry anchor when TWS reports an existing position."""
        if (
            self.initial_position is None
            or self.initial_position <= 0
            or self.initial_avg_cost is None
            or self.initial_avg_cost <= 0
        ):
            return self.bars

        entry = float(self.initial_avg_cost)
        first_ts = self.bars[0].timestamp
        anchor0 = MidBar(first_ts, entry, entry, entry, entry, entry, entry, 0.0)
        anchor1 = MidBar(first_ts, entry, entry, entry, entry, entry, entry, 0.0)
        return [anchor0, anchor1] + self.bars

    def _bars_for_cash_reentry(self):
        """Build a sell-anchored path for cash re-entry diagnostics."""
        if (
            self.cash_anchor_price is None
            or self.cash_anchor_price <= 0
            or not self.bars
        ):
            return None
        ts = self.cash_anchor_timestamp or self.bars[0].timestamp
        anchor = float(self.cash_anchor_price)
        anchor_bar = MidBar(ts, anchor, anchor, anchor, anchor, anchor, anchor, 0.0)
        return [anchor_bar] + self.bars

    def _cash_reentry_signal(self, bar, blocked):
        """Evaluate re-entry from the last filled sell anchor while in cash."""
        if blocked:
            return None
        strategy_bars = self._bars_for_cash_reentry()
        if strategy_bars is None or len(strategy_bars) < 2:
            return None

        mids = np.asarray([b.close_mid for b in strategy_bars], dtype=float)
        lp = np.log(mids)
        sigma_seed = estimate_sigma_seed(lp)
        log_returns = np.diff(lp)
        mu_step, _ = kalman_drift_estimates(
            log_returns,
            init_mu=0.0,
            obs_var=sigma_seed * sigma_seed,
            process_var=self.drift_process_var,
        )
        var_step, _ = garch_volatility_estimates(
            log_returns,
            mu_prior=mu_step,
            init_var=sigma_seed * sigma_seed,
        )
        cumulative_drift = np.zeros(len(lp))
        cumulative_var = np.zeros(len(lp))
        cumulative_drift[1:] = np.cumsum(mu_step[1:])
        cumulative_var[1:] = np.cumsum(var_step[1:])

        n = len(lp)
        t = n - 1
        d = min(max(1, self.entry_delta), t)
        ref_t = 0
        if self.max_funnel_lookback is not None and t > self.max_funnel_lookback:
            ref_t = t - self.max_funnel_lookback
        drift = cumulative_drift[t] - cumulative_drift[ref_t]
        var = cumulative_var[t] - cumulative_var[ref_t]
        denom = np.sqrt(var)
        z = (lp[t] - lp[ref_t] - drift) / denom if denom > 0 else np.nan
        momentum = lp[t] - lp[t - d] if t >= d else np.nan
        buy_now = (
            t >= d
            and not np.isnan(z)
            and not np.isnan(momentum)
            and (
                (z < -self.k and momentum >= 0.0)
                or (
                    self.trend_entry_z is not None
                    and z > self.trend_entry_z
                    and momentum > 0.0
                )
            )
        )

        data = {
            "N": t,
            "k": self.k,
            "trend_entry_z": self.trend_entry_z,
            "Zsig": np.full(n, np.nan),
            "holding": np.zeros(n, dtype=bool),
            "buy_times": {t} if buy_now else set(),
            "sell_times": set(),
            "realized_w": np.full(n, 1000.0),
            "portfolio_w": np.full(n, 1000.0),
            "buy_hold_w": np.full(n, 1000.0),
            "funnel_mid": np.full(n, np.nan),
            "funnel_up": np.full(n, np.nan),
            "funnel_low": np.full(n, np.nan),
            "trade_log": [],
        }
        data["funnel_mid"][0] = np.exp(lp[0])
        data["funnel_up"][0] = np.exp(lp[0])
        data["funnel_low"][0] = np.exp(lp[0])
        data["Zsig"][0] = 0.0
        for i in range(1, n):
            ref_i = 0
            if self.max_funnel_lookback is not None and i > self.max_funnel_lookback:
                ref_i = i - self.max_funnel_lookback
            drift_i = cumulative_drift[i] - cumulative_drift[ref_i]
            var_i = cumulative_var[i] - cumulative_var[ref_i]
            denom_i = np.sqrt(var_i)
            center_i = lp[ref_i] + drift_i
            width_i = self.k * denom_i
            data["funnel_mid"][i] = np.exp(center_i)
            data["funnel_up"][i] = np.exp(center_i + width_i)
            data["funnel_low"][i] = np.exp(center_i - width_i)
            data["Zsig"][i] = (
                (lp[i] - lp[ref_i] - drift_i) / denom_i
                if denom_i > 0
                else np.nan
            )
        self._update_display_overlays(data)
        return {"buy_now": buy_now, "data": data}

    def _update_display_overlays(self, data):
        strategy_end = int(data["N"]) + 1
        for strategy_index in range(strategy_end):
            if (
                self.initial_position is not None
                and self.initial_position > 0
                and self.initial_avg_cost is not None
                and self.initial_avg_cost > 0
                and strategy_index < 2
            ):
                continue
            display_index = self.strategy_x_to_display_x(strategy_index)
            if 0 <= display_index < len(self.display_bars):
                self.display_funnel_mid[display_index] = data["funnel_mid"][strategy_index]
                self.display_funnel_up[display_index] = data["funnel_up"][strategy_index]
                self.display_funnel_low[display_index] = data["funnel_low"][strategy_index]
                self.display_z[display_index] = data["Zsig"][strategy_index]

    def strategy_x_to_display_x(self, strategy_index):
        """Map current-cycle strategy index into persistent display index."""
        anchor_count = 0
        if (
            self.initial_position is not None
            and self.initial_position > 0
            and self.initial_avg_cost is not None
            and self.initial_avg_cost > 0
        ):
            anchor_count = 2
        elif (
            self.initial_position is not None
            and self.initial_position <= 0
            and self.cash_anchor_price is not None
            and self.cash_anchor_price > 0
        ):
            anchor_count = 1
        return self.cycle_start_display_index + int(strategy_index) - anchor_count

    @staticmethod
    def _clean_float(value):
        if value is None:
            return None
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return None
        return value

    @classmethod
    def _bar_to_dict(cls, bar):
        return {
            "timestamp": bar.timestamp,
            "open_mid": cls._clean_float(bar.open_mid),
            "high_mid": cls._clean_float(bar.high_mid),
            "low_mid": cls._clean_float(bar.low_mid),
            "close_mid": cls._clean_float(bar.close_mid),
            "close_bid": cls._clean_float(bar.close_bid),
            "close_ask": cls._clean_float(bar.close_ask),
            "avg_spread_bps": cls._clean_float(bar.avg_spread_bps),
        }

    @classmethod
    def _bar_from_dict(cls, data):
        return MidBar(
            timestamp=str(data["timestamp"]),
            open_mid=float(data["open_mid"]),
            high_mid=float(data["high_mid"]),
            low_mid=float(data["low_mid"]),
            close_mid=float(data["close_mid"]),
            close_bid=float(data["close_bid"]),
            close_ask=float(data["close_ask"]),
            avg_spread_bps=float(data.get("avg_spread_bps") or 0.0),
        )

    @classmethod
    def _series_to_json(cls, values):
        return [cls._clean_float(value) for value in values]

    @staticmethod
    def _series_from_json(values):
        return [np.nan if value is None else float(value) for value in values]

    def export_state(self):
        """Return JSON-safe live state needed to continue after restart."""
        return {
            "version": 1,
            "symbol": self.symbol,
            "params": {
                "k": self.k,
                "delta": self.delta,
                "entry_delta": self.entry_delta,
                "cost": self.cost,
                "drift_process_var": self.drift_process_var,
                "max_funnel_lookback": self.max_funnel_lookback,
                "trailing_stop": self.trailing_stop,
                "trend_entry_z": self.trend_entry_z,
                "fixed_buy_fee": self.fixed_buy_fee,
                "fixed_sell_fee": self.fixed_sell_fee,
                "min_sell_profit": self.min_sell_profit,
                "min_sell_profit_per_share": self.min_sell_profit_per_share,
                "profit_target_mode": self.profit_target_mode,
                "order_quantity": self.order_quantity,
                "capital_budget": self.capital_budget,
            },
            "capital_budget": self._clean_float(self.capital_budget),
            "live_cash": self._clean_float(self.live_cash),
            "initial_position": self._clean_float(self.initial_position),
            "initial_avg_cost": self._clean_float(self.initial_avg_cost),
            "initial_buy_emitted": bool(self._initial_buy_emitted),
            "bars": [self._bar_to_dict(bar) for bar in self.bars],
            "display_bars": [self._bar_to_dict(bar) for bar in self.display_bars],
            "display_funnel_mid": self._series_to_json(self.display_funnel_mid),
            "display_funnel_up": self._series_to_json(self.display_funnel_up),
            "display_funnel_low": self._series_to_json(self.display_funnel_low),
            "display_z": self._series_to_json(self.display_z),
            "cycle_start_display_index": int(self.cycle_start_display_index),
            "filled_markers": list(self.filled_markers),
            "cash_anchor_price": self._clean_float(self.cash_anchor_price),
            "cash_anchor_timestamp": self.cash_anchor_timestamp,
            "last_live_exit_reason": self.last_live_exit_reason,
        }

    def import_state(self, state, account_position=None, account_avg_cost=None):
        """Restore live state, then reconcile position with current broker state."""
        if not state:
            return
        saved_symbol = str(state.get("symbol", "")).upper()
        if saved_symbol and saved_symbol != self.symbol:
            raise ValueError(
                f"State symbol {saved_symbol} does not match requested symbol {self.symbol}."
            )
        params = state.get("params", {})
        if params:
            self.k = float(params.get("k", self.k))
            self.delta = int(params.get("delta", self.delta))
            self.entry_delta = int(params.get("entry_delta", self.entry_delta))
            self.cost = float(params.get("cost", self.cost))
            self.drift_process_var = float(
                params.get("drift_process_var", self.drift_process_var)
            )
            self.max_funnel_lookback = int(
                params.get("max_funnel_lookback", self.max_funnel_lookback)
            )
            self.trailing_stop = params.get("trailing_stop", self.trailing_stop)
            self.trend_entry_z = params.get("trend_entry_z", self.trend_entry_z)
            self.fixed_buy_fee = float(params.get("fixed_buy_fee", self.fixed_buy_fee))
            self.fixed_sell_fee = float(params.get("fixed_sell_fee", self.fixed_sell_fee))
            self.min_sell_profit = float(params.get("min_sell_profit", self.min_sell_profit))
            self.min_sell_profit_per_share = float(
                params.get("min_sell_profit_per_share", self.min_sell_profit_per_share)
            )
            self.profit_target_mode = str(
                params.get("profit_target_mode", self.profit_target_mode)
            )
            if self.profit_target_mode not in ("fixed", "per_share", "max"):
                self.profit_target_mode = "per_share"
            self.order_quantity = int(params.get("order_quantity", self.order_quantity))
            self.capital_budget = float(params.get("capital_budget", self.capital_budget))

        self.initial_position = state.get("initial_position")
        self.initial_position = (
            None if self.initial_position is None else float(self.initial_position)
        )
        self.initial_avg_cost = state.get("initial_avg_cost")
        self.initial_avg_cost = (
            None if self.initial_avg_cost is None else float(self.initial_avg_cost)
        )
        self.capital_budget = float(state.get("capital_budget", self.capital_budget))
        self.live_cash = float(state.get("live_cash", self._initial_live_cash()))
        self._initial_buy_emitted = bool(state.get("initial_buy_emitted", False))
        self.bars = [self._bar_from_dict(item) for item in state.get("bars", [])]
        self.display_bars = [
            self._bar_from_dict(item) for item in state.get("display_bars", [])
        ]
        self.display_funnel_mid = self._series_from_json(
            state.get("display_funnel_mid", [])
        )
        self.display_funnel_up = self._series_from_json(
            state.get("display_funnel_up", [])
        )
        self.display_funnel_low = self._series_from_json(
            state.get("display_funnel_low", [])
        )
        self.display_z = self._series_from_json(state.get("display_z", []))
        self.cycle_start_display_index = int(
            state.get("cycle_start_display_index", 0)
        )
        self.filled_markers = list(state.get("filled_markers", []))
        self.cash_anchor_price = state.get("cash_anchor_price")
        self.cash_anchor_price = (
            None if self.cash_anchor_price is None else float(self.cash_anchor_price)
        )
        self.cash_anchor_timestamp = state.get("cash_anchor_timestamp")
        self.last_live_exit_reason = state.get("last_live_exit_reason", "")
        self.latest_data = None

        n_display = len(self.display_bars)
        for attr in (
            "display_funnel_mid",
            "display_funnel_up",
            "display_funnel_low",
            "display_z",
        ):
            series = getattr(self, attr)
            if len(series) < n_display:
                series.extend([np.nan] * (n_display - len(series)))
            elif len(series) > n_display:
                del series[n_display:]

        if account_position is not None:
            prior_position = self.initial_position
            account_position = float(account_position)
            self.initial_position = account_position
            if account_position > 0:
                if self.initial_avg_cost is None and account_avg_cost is not None:
                    self.initial_avg_cost = float(account_avg_cost)
                if prior_position is None or prior_position <= 0:
                    self.live_cash = self._initial_live_cash()
                    self._initial_buy_emitted = True
                self.cash_anchor_price = None
                self.cash_anchor_timestamp = None
            else:
                self.initial_avg_cost = None

    def _is_regular_hours(self, timestamp):
        dt = datetime.fromisoformat(timestamp)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
        eastern = dt.astimezone(self.timezone)
        if eastern.weekday() >= 5:
            return False
        return time(9, 30) <= eastern.time() <= time(16, 0)


def print_strategy_signal(signal):
    z_text = "n/a" if signal.z is None else f"{signal.z:.3f}"
    note = f" | {signal.note}" if signal.note else ""
    print(
        f"{signal.timestamp} | {signal.symbol:<5} | "
        f"bars {signal.bars:>3} | "
        f"mid {signal.close_mid:>10.4f} | "
        f"bid {signal.close_bid:>10.4f} | "
        f"ask {signal.close_ask:>10.4f} | "
        f"state {signal.position:<6} | "
        f"signal {signal.signal:<10} | "
        f"Z {z_text:>7} | "
        f"MtM {signal.mtm_w:>9.2f} | "
        f"trades {signal.trades}{note}"
    )


def signal_to_dict(signal):
    return {
        "timestamp": signal.timestamp,
        "symbol": signal.symbol,
        "bars": signal.bars,
        "close_mid": signal.close_mid,
        "close_bid": signal.close_bid,
        "close_ask": signal.close_ask,
        "position": signal.position,
        "signal": signal.signal,
        "z": signal.z,
        "realized_w": signal.realized_w,
        "mtm_w": signal.mtm_w,
        "trades": signal.trades,
        "blocked": signal.blocked,
        "block_reason": signal.block_reason,
        "note": signal.note,
    }
