"""
Paper-trading layer for the funnel trading project.

This module does not place live broker orders. It fetches recent Yahoo bars,
runs the existing strategy engine, processes every new bar since the last paper
check, fake-fills any paper BUY/SELL transitions, and persists the account
state to JSON.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import main


DEFAULT_STATE_PATH = Path("data/paper_trading_state.json")


@dataclass
class StrategyConfig:
    symbol: str = main.DEFAULT_REAL_SYMBOL
    lookback_days: float = 365.25
    interval: str = "1d"
    mode: str = "fixed"
    k: float = 1.2
    delta: int = 8
    drift_q: float = 1e-7
    lookback_L: int = 60
    trailing_stop: float = 0.04
    trend_entry_z: float = 0.35
    learning_rate: float = 0.25
    cost: float = 0.0
    generalized_momentum_c: float = 0.02
    optimize_params: list[str] = field(default_factory=lambda: ["k", "delta"])


@dataclass
class PaperTrade:
    time: str
    symbol: str
    action: str
    price: float
    shares: float
    cash_after: float
    position_after: str
    reason: str


@dataclass
class PaperState:
    symbol: str | None = None
    cash: float = 10000.0
    shares: float = 0.0
    position: str = "CASH"
    last_bar_time: str | None = None
    last_price: float | None = None
    updated_at: str | None = None
    trades: list[dict] = field(default_factory=list)

    @property
    def equity(self) -> float:
        if self.last_price is None:
            return float(self.cash)
        return float(self.cash + self.shares * self.last_price)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def epoch_to_iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), timezone.utc).isoformat(timespec="seconds")


def load_state(path: Path = DEFAULT_STATE_PATH) -> PaperState:
    if not path.exists():
        return PaperState()
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return PaperState(**raw)


def save_state(state: PaperState, path: Path = DEFAULT_STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(state), f, indent=2)


def fetch_yahoo_bars(symbol: str, lookback_days: float, interval: str) -> tuple[np.ndarray, np.ndarray]:
    """Fetch Yahoo chart bars and return UTC timestamps plus positive prices."""
    period2 = int(time.time())
    period1 = period2 - int(float(lookback_days) * 24 * 60 * 60)
    query_symbol = urllib.parse.quote(symbol.upper())
    query_interval = urllib.parse.quote(interval)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{query_symbol}"
        f"?period1={period1}&period2={period2}"
        f"&interval={query_interval}&events=history&includeAdjustedClose=true"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not fetch Yahoo data for {symbol}: {exc}") from exc

    error = payload.get("chart", {}).get("error")
    if error:
        raise ValueError(f"Yahoo returned an error for {symbol}: {error}")
    results = payload.get("chart", {}).get("result") or []
    if not results:
        raise ValueError(f"Yahoo returned no data for {symbol}.")

    result = results[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators", {})
    prices = indicators.get("adjclose", [{}])[0].get("adjclose")
    if not prices:
        prices = indicators.get("quote", [{}])[0].get("close")
    if not prices:
        raise ValueError(f"Yahoo returned no close prices for {symbol}.")

    cleaned = [
        (int(ts), float(price))
        for ts, price in zip(timestamps, prices)
        if price is not None and float(price) > 0
    ]
    if len(cleaned) < 2:
        raise ValueError(f"Yahoo returned insufficient positive bars for {symbol}.")
    ts_arr = np.asarray([item[0] for item in cleaned], dtype=int)
    price_arr = np.asarray([item[1] for item in cleaned], dtype=float)
    return ts_arr, price_arr


def run_configured_strategy(log_prices: np.ndarray, config: StrategyConfig) -> dict:
    """Run the selected project strategy on the supplied log-price history."""
    sigma_seed = main.estimate_sigma_seed(log_prices)
    if config.mode == "learn":
        return main.run_online_adaptive_strategy(
            log_prices,
            c_buy=config.cost,
            c_sell=config.cost,
            drift_process_var=config.drift_q,
            learning_rate=config.learning_rate,
            k_init=config.k,
            delta_init=config.delta,
            max_funnel_lookback=config.lookback_L,
            trailing_stop=config.trailing_stop,
            trend_entry_z=config.trend_entry_z,
            optimize_params=config.optimize_params,
            source=f"Paper Yahoo {config.symbol.upper()}",
        )

    return main.run_strategy_on_log_prices(
        log_prices,
        k=config.k,
        delta=config.delta,
        sigma_seed=sigma_seed,
        c_buy=config.cost,
        c_sell=config.cost,
        drift_process_var=config.drift_q,
        max_funnel_lookback=config.lookback_L,
        trailing_stop=config.trailing_stop,
        trend_entry_z=config.trend_entry_z,
        generalized_momentum_c=(
            config.generalized_momentum_c
            if config.mode == "generalized_momentum"
            else None
        ),
        use_hmm_regime=config.mode == "regime_hmm",
        hmm_states=3,
        randomized_stopping=config.mode == "randomized_stopping",
        mode=f"Paper {config.symbol.upper()} {config.mode}",
    )


def decision_at_bar(data: dict, index: int, current_position: str) -> tuple[str, str]:
    """
    Convert strategy exposure at one bar into a paper action.

    On the first run, this intentionally aligns the paper account with the
    strategy's current target exposure, even if the original backtest entry
    happened before the latest bar.
    """
    desired_long = bool(data["holding"][index])
    if desired_long and current_position != "LONG":
        return "BUY", "strategy target exposure is LONG"
    if not desired_long and current_position == "LONG":
        return "SELL", "strategy target exposure is CASH"
    return "HOLD", "paper account already matches strategy exposure"


def latest_decision(data: dict, current_position: str) -> tuple[str, str]:
    """Backward-compatible latest-bar decision helper."""
    return decision_at_bar(data, -1, current_position)


def apply_paper_action(
    state: PaperState,
    config: StrategyConfig,
    action: str,
    price: float,
    bar_time: str,
    reason: str,
) -> PaperTrade | None:
    """Fake-fill a paper action at the latest bar price."""
    if action == "BUY" and state.cash > 0:
        gross_cash = state.cash
        shares = gross_cash * (1.0 - config.cost) / price
        state.cash = 0.0
        state.shares = float(shares)
        state.position = "LONG"
    elif action == "SELL" and state.shares > 0:
        shares = state.shares
        state.cash = float(shares * price * (1.0 - config.cost))
        state.shares = 0.0
        state.position = "CASH"
    else:
        return None

    trade = PaperTrade(
        time=bar_time,
        symbol=config.symbol.upper(),
        action=action,
        price=float(price),
        shares=float(shares),
        cash_after=float(state.cash),
        position_after=state.position,
        reason=reason,
    )
    state.symbol = config.symbol.upper()
    state.trades.append(asdict(trade))
    return trade


def run_paper_check(
    config: StrategyConfig,
    state_path: Path = DEFAULT_STATE_PATH,
) -> dict:
    """Fetch latest bars, process all new bars in order, and save state."""
    state = load_state(state_path)
    symbol = config.symbol.upper()
    if state.symbol and state.symbol != symbol and state.position == "LONG":
        raise ValueError(
            "Paper account is currently LONG "
            f"{state.symbol}. Reset or close the paper account before switching to {symbol}."
        )
    if state.symbol is None or state.position == "CASH":
        state.symbol = symbol

    timestamps, prices = fetch_yahoo_bars(
        config.symbol,
        config.lookback_days,
        config.interval,
    )
    bar_time = epoch_to_iso(int(timestamps[-1]))
    latest_price = float(prices[-1])

    if state.last_bar_time == bar_time:
        state.last_price = latest_price
        state.updated_at = utc_now_iso()
        save_state(state, state_path)
        return {
            "action": "HOLD",
            "reason": "latest Yahoo bar was already processed",
            "bar_time": bar_time,
            "price": latest_price,
            "state": state,
            "trade": None,
            "trades": [],
            "trade_indices": [],
            "paper_buy_times": set(),
            "paper_sell_times": set(),
            "processed_bars": 0,
        }

    data = run_configured_strategy(np.log(prices), config)
    iso_times = [epoch_to_iso(int(ts)) for ts in timestamps]
    if state.last_bar_time is None:
        new_indices = [len(prices) - 1]
    else:
        new_indices = [
            idx for idx, iso_time in enumerate(iso_times)
            if iso_time > state.last_bar_time
        ]
        if not new_indices:
            new_indices = [len(prices) - 1]

    last_action = "HOLD"
    last_reason = "no new strategy transition"
    trades = []
    trade_indices = []
    for idx in new_indices:
        step_time = iso_times[idx]
        step_price = float(prices[idx])
        action, reason = decision_at_bar(data, idx, state.position)
        state.last_bar_time = step_time
        state.last_price = step_price
        if action in ("BUY", "SELL"):
            trade = apply_paper_action(state, config, action, step_price, step_time, reason)
            if trade is not None:
                trades.append(trade)
                trade_indices.append(idx)
        last_action = action
        last_reason = reason

    latest_price = float(prices[new_indices[-1]])
    bar_time = iso_times[new_indices[-1]]
    state.updated_at = utc_now_iso()
    save_state(state, state_path)
    paper_buy_times = set()
    paper_sell_times = set()
    time_to_index = {iso_time: idx for idx, iso_time in enumerate(iso_times)}
    for trade_record in state.trades:
        idx = time_to_index.get(trade_record.get("time"))
        if idx is None:
            continue
        if trade_record.get("action") == "BUY":
            paper_buy_times.add(idx)
        elif trade_record.get("action") == "SELL":
            paper_sell_times.add(idx)

    return {
        "action": last_action,
        "reason": last_reason,
        "bar_time": bar_time,
        "price": latest_price,
        "state": state,
        "trade": trades[-1] if trades else None,
        "trades": trades,
        "trade_indices": trade_indices,
        "paper_buy_times": paper_buy_times,
        "paper_sell_times": paper_sell_times,
        "processed_bars": len(new_indices),
        "strategy": data,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one paper-trading check.")
    parser.add_argument("--symbol", default=main.DEFAULT_REAL_SYMBOL)
    parser.add_argument("--lookback-days", type=float, default=365.25)
    parser.add_argument("--interval", default="1d")
    parser.add_argument(
        "--mode",
        choices=["fixed", "learn", "generalized_momentum", "regime_hmm", "randomized_stopping"],
        default="fixed",
    )
    parser.add_argument("--k", type=float, default=1.2)
    parser.add_argument("--delta", type=int, default=8)
    parser.add_argument("--drift-q", type=float, default=1e-7)
    parser.add_argument("--lookback-L", type=int, default=60)
    parser.add_argument("--trailing-stop", type=float, default=0.04)
    parser.add_argument("--trend-entry-z", type=float, default=0.35)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--cost", type=float, default=0.0)
    parser.add_argument("--generalized-momentum-c", type=float, default=0.02)
    parser.add_argument("--optimize", default="k,delta")
    parser.add_argument("--state", default=str(DEFAULT_STATE_PATH))
    return parser


def main_cli() -> None:
    args = build_arg_parser().parse_args()
    optimize_params = [part.strip() for part in args.optimize.split(",") if part.strip()]
    config = StrategyConfig(
        symbol=args.symbol,
        lookback_days=args.lookback_days,
        interval=args.interval,
        mode=args.mode,
        k=args.k,
        delta=args.delta,
        drift_q=args.drift_q,
        lookback_L=args.lookback_L,
        trailing_stop=args.trailing_stop,
        trend_entry_z=args.trend_entry_z,
        learning_rate=args.learning_rate,
        cost=args.cost,
        generalized_momentum_c=args.generalized_momentum_c,
        optimize_params=optimize_params,
    )
    result = run_paper_check(config, Path(args.state))
    state = result["state"]
    print(
        f"{result['bar_time']} {config.symbol.upper()} "
        f"{result['action']} @ {result['price']:.2f} | "
        f"{result['reason']} | equity={state.equity:.2f} "
        f"cash={state.cash:.2f} shares={state.shares:.6f}"
    )


if __name__ == "__main__":
    main_cli()
