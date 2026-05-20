"""
Live dry-run adapter for the trading algorithm.

This module keeps broker connectivity out of the strategy logic. It receives
completed mid-price bars, reuses the existing research strategy from main.py,
and emits terminal-friendly HOLD / WOULD BUY / WOULD SELL messages.

No orders are placed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from main import estimate_sigma_seed, run_strategy_on_log_prices, summarize_simulation


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
        cost=0.0,
        drift_process_var=1e-7,
        max_funnel_lookback=60,
        trailing_stop=0.04,
        trend_entry_z=0.35,
        execution_spread=0.0,
        execution_slippage=0.0,
    ):
        self.symbol = symbol.upper()
        self.k = float(k)
        self.delta = int(delta)
        self.cost = float(cost)
        self.drift_process_var = float(drift_process_var)
        self.max_funnel_lookback = int(max_funnel_lookback)
        self.trailing_stop = float(trailing_stop)
        self.trend_entry_z = float(trend_entry_z)
        self.execution_spread = float(execution_spread)
        self.execution_slippage = float(execution_slippage)
        self.bars = []

    def on_bar(self, bar):
        self.bars.append(bar)
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
            )

        mids = np.asarray([b.close_mid for b in self.bars], dtype=float)
        lp = np.log(mids)
        sigma_seed = estimate_sigma_seed(lp)
        data = run_strategy_on_log_prices(
            lp,
            k=self.k,
            delta=min(self.delta, max(1, len(lp) - 2)),
            sigma_seed=sigma_seed,
            c_buy=self.cost,
            c_sell=self.cost,
            drift_process_var=self.drift_process_var,
            max_funnel_lookback=self.max_funnel_lookback,
            trailing_stop=self.trailing_stop,
            trend_entry_z=self.trend_entry_z,
            execution_spread=self.execution_spread,
            execution_slippage=self.execution_slippage,
            mode="IBKR dry-run",
        )
        t = data["N"]
        buy_now = t in data["buy_times"]
        sell_now = t in data["sell_times"]
        position = "LONG" if bool(data["holding"][t]) else "CASH"
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
        )


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
