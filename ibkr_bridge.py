"""
IBKR TWS bridge for the trading project.

Version 1 goal:
    Connect to a local TWS / IB Gateway session and print bid, ask, and mid.

By default this script is read-only. It places orders only when explicitly
started with --manual-orders or --auto-orders.

Setup:
    1. Open TWS Demo/Paper.
    2. Enable API socket clients in TWS:
       Global Configuration -> API -> Settings -> Enable ActiveX and Socket Clients
    3. Use the paper/demo port, usually 7497.
    4. Install dependency:
       pip install ib_insync

Example:
    python3 ibkr_bridge.py --symbol AAPL --port 7497
    python3 ibkr_bridge.py --preset apple
    python3 ibkr_bridge.py --preset spy --market-data-type delayed
    python3 ibkr_bridge.py --preset spy --watch --duration 30
    python3 ibkr_bridge.py --symbol NVDA --market-data-type live --dry-run-strategy
    python3 ibkr_bridge.py --positions
    python3 ibkr_bridge.py --symbol NVDA --market-data-type live --dry-run-strategy --manual-orders
    python3 ibkr_bridge.py --symbol NVDA --market-data-type live --dry-run-strategy --use-account-position --auto-orders --regular-hours-only --max-spread-bps 20 --max-quote-age 10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import select
import sys
import time
from dataclasses import dataclass
from datetime import datetime, time as day_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7497
DEFAULT_CLIENT_ID = 17
DEFAULT_SYMBOL = "AAPL"
DEFAULT_MARKET_DATA_TYPE = "delayed"
SYMBOL_PRESETS = {
    "apple": "AAPL",
    "aapl": "AAPL",
    "spy": "SPY",
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "microsoft": "MSFT",
    "msft": "MSFT",
    "tesla": "TSLA",
    "tsla": "TSLA",
}
MARKET_DATA_TYPES = {
    "live": 1,
    "frozen": 2,
    "delayed": 3,
    "delayed-frozen": 4,
}
MARKET_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = day_time(9, 30)
MARKET_CLOSE = day_time(16, 0)


@dataclass
class QuoteSnapshot:
    symbol: str
    bid: float | None
    ask: float | None
    last: float | None
    mid: float | None
    market_data_type: str


@dataclass
class QuoteRow:
    timestamp: str
    symbol: str
    bid: float | None
    ask: float | None
    last: float | None
    mid: float | None
    spread: float | None
    spread_bps: float | None
    market_data_type: str


def _safe_float(value):
    """Return None for IBKR's unset/NaN quote values."""
    try:
        if value is None or value != value:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_price(value):
    return "n/a" if value is None else f"{value:.4f}"


def _format_quantity(value):
    if value is None:
        return "n/a"
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def _format_bps(value):
    return "n/a" if value is None else f"{value:.2f}"


def _regular_market_window(now=None):
    """Return today's regular US equity market open/close in Eastern time."""
    now = datetime.now(MARKET_TZ) if now is None else now.astimezone(MARKET_TZ)
    open_dt = datetime.combine(now.date(), MARKET_OPEN, tzinfo=MARKET_TZ)
    close_dt = datetime.combine(now.date(), MARKET_CLOSE, tzinfo=MARKET_TZ)
    return open_dt, close_dt


def _is_regular_market_open(now=None):
    now = datetime.now(MARKET_TZ) if now is None else now.astimezone(MARKET_TZ)
    if now.weekday() >= 5:
        return False
    open_dt, close_dt = _regular_market_window(now)
    return open_dt <= now <= close_dt


def _next_regular_market_open(now=None):
    """Return the next regular US equity market open, ignoring holidays."""
    now = datetime.now(MARKET_TZ) if now is None else now.astimezone(MARKET_TZ)
    candidate = now
    while True:
        open_dt, _ = _regular_market_window(candidate)
        if candidate.weekday() < 5 and candidate < open_dt:
            return open_dt
        candidate = datetime.combine(
            candidate.date() + timedelta(days=1),
            day_time(0, 0),
            tzinfo=MARKET_TZ,
        )


def wait_for_regular_market_open(check_seconds=60):
    """Sleep until regular US stock-market hours begin."""
    check_seconds = max(float(check_seconds), 5.0)
    while not _is_regular_market_open():
        now = datetime.now(MARKET_TZ)
        next_open = _next_regular_market_open(now)
        wait_seconds = max((next_open - now).total_seconds(), 0.0)
        print(
            "Market is closed. Waiting until regular US market open: "
            f"{next_open.isoformat(timespec='minutes')} ET "
            f"(about {wait_seconds / 3600.0:.2f} hours)."
        )
        time.sleep(min(check_seconds, wait_seconds if wait_seconds > 0 else check_seconds))


def _first_price(*values):
    """Return the first usable price from normal or delayed ticker fields."""
    for value in values:
        price = _safe_float(value)
        if price is not None:
            return price
    return None


def _limit_price_for_signal(signal, limit_buffer_bps):
    buffer = max(float(limit_buffer_bps), 0.0) / 10000.0
    if signal.signal == "WOULD BUY":
        return signal.close_ask * (1.0 + buffer)
    if signal.signal == "WOULD SELL":
        return signal.close_bid * (1.0 - buffer)
    return None


def _quantity_for_signal(
    signal,
    strategy,
    fallback_quantity,
    sizing_mode="fixed",
    cash_reserve=0.0,
    limit_price=None,
):
    """Choose order size for a strategy signal."""
    action = "BUY" if signal.signal == "WOULD BUY" else "SELL"
    fallback_quantity = max(int(fallback_quantity), 1)
    if sizing_mode != "cash_reserve" or strategy is None:
        return fallback_quantity
    if action == "SELL":
        quantity_fn = getattr(strategy, "active_sell_quantity", None)
        if callable(quantity_fn):
            return int(quantity_fn())
        position = _safe_float(getattr(strategy, "initial_position", None))
        return max(int(math.floor(position or 0.0)), 0)
    reference_price = _safe_float(limit_price or signal.close_ask)
    quantity_fn = getattr(strategy, "active_buy_quantity", None)
    if callable(quantity_fn):
        return int(quantity_fn(reference_price))
    live_cash = _safe_float(getattr(strategy, "live_cash", None))
    fixed_buy_fee = max(float(getattr(strategy, "fixed_buy_fee", 0.0)), 0.0)
    prop_cost = max(float(getattr(strategy, "cost", 0.0)), 0.0)
    if live_cash is None or reference_price is None or reference_price <= 0:
        return 0
    spendable_cash = live_cash - max(float(cash_reserve), 0.0) - fixed_buy_fee
    unit_cost = reference_price * (1.0 + prop_cost)
    return max(int(math.floor(spendable_cash / unit_cost)), 0)


def _minimum_profitable_sell_limit(
    strategy,
    quantity,
    min_profit=0.01,
    min_profit_per_share=0.0,
    profit_target_mode="per_share",
):
    """Minimum sell limit that guarantees positive net profit if filled."""
    if strategy is None:
        return None
    entry_price = _safe_float(getattr(strategy, "initial_avg_cost", None))
    if entry_price is None or entry_price <= 0:
        return None
    quantity = int(quantity)
    if quantity <= 0:
        return None
    prop_cost = max(float(getattr(strategy, "cost", 0.0)), 0.0)
    if prop_cost >= 1.0:
        return None
    fixed_buy_fee = max(float(getattr(strategy, "fixed_buy_fee", 0.0)), 0.0)
    fixed_sell_fee = max(float(getattr(strategy, "fixed_sell_fee", 0.0)), 0.0)
    min_profit = max(float(min_profit), 0.0)
    min_profit_per_share = max(float(min_profit_per_share), 0.0)
    if profit_target_mode == "fixed":
        target_profit = min_profit
    elif profit_target_mode == "per_share":
        target_profit = min_profit_per_share * quantity
    else:
        target_profit = max(min_profit, min_profit_per_share * quantity)
    required_proceeds = (
        entry_price * quantity * (1.0 + prop_cost)
        + fixed_buy_fee
        + fixed_sell_fee
        + target_profit
    )
    raw_limit = required_proceeds / (quantity * (1.0 - prop_cost))
    # IBKR equity limit prices are cent-based. Round upward so the floor survives
    # price rounding and still leaves at least the requested profit buffer.
    return math.ceil(raw_limit * 100.0) / 100.0


def _apply_profitable_sell_floor(
    limit_price,
    strategy,
    quantity,
    min_profit=0.01,
    min_profit_per_share=0.0,
    profit_target_mode="per_share",
):
    floor = _minimum_profitable_sell_limit(
        strategy,
        quantity,
        min_profit=min_profit,
        min_profit_per_share=min_profit_per_share,
        profit_target_mode=profit_target_mode,
    )
    if floor is None:
        return limit_price, None
    if limit_price is None or limit_price < floor:
        return floor, floor
    return limit_price, floor


def _submit_limit_order(
    ib,
    contract,
    signal,
    quantity,
    limit_buffer_bps,
    strategy=None,
    min_sell_profit=0.01,
    min_sell_profit_per_share=0.0,
    profit_target_mode="per_share",
):
    """Submit a limit order for a WOULD BUY / WOULD SELL signal."""
    if signal.signal not in ("WOULD BUY", "WOULD SELL"):
        return None
    if signal.blocked:
        print(f"Order skipped: signal blocked by {signal.block_reason}.")
        return None
    limit_price = _limit_price_for_signal(signal, limit_buffer_bps)
    if limit_price is None:
        return None
    action = "BUY" if signal.signal == "WOULD BUY" else "SELL"
    quantity = int(quantity)
    if quantity <= 0:
        print("Order skipped: quantity must be positive.")
        return None
    floor = None
    if action == "SELL":
        limit_price, floor = _apply_profitable_sell_floor(
            limit_price,
            strategy,
            quantity,
            min_profit=min_sell_profit,
            min_profit_per_share=min_sell_profit_per_share,
            profit_target_mode=profit_target_mode,
        )

    from ib_insync import LimitOrder

    order = LimitOrder(action, quantity, round(limit_price, 2), tif="DAY")
    trade = ib.placeOrder(contract, order)
    ib.sleep(1.0)
    print(
        "Submitted order: "
        f"{action} {quantity} {signal.symbol} LMT {_format_price(order.lmtPrice)} "
        f"status={trade.orderStatus.status}"
    )
    if floor is not None:
        print(
            "Profit-safe sell floor applied: "
            f"minimum LMT {_format_price(floor)} includes modeled fees."
        )
    return trade


def _limit_price_for_action(row, action, limit_buffer_bps):
    buffer = max(float(limit_buffer_bps), 0.0) / 10000.0
    action = action.upper()
    if action == "BUY":
        if row.ask is None:
            return None
        return row.ask * (1.0 + buffer)
    if action == "SELL":
        if row.bid is None:
            return None
        return row.bid * (1.0 - buffer)
    return None


def _submit_action_limit_order(
    ib,
    contract,
    symbol,
    action,
    quantity,
    limit_price,
):
    """Submit a direct BUY/SELL limit order from an interactive override."""
    action = action.upper()
    quantity = int(quantity)
    if action not in ("BUY", "SELL"):
        print("Order skipped: action must be BUY or SELL.")
        return None
    if quantity <= 0:
        print("Order skipped: quantity must be positive.")
        return None
    if limit_price is None:
        print("Order skipped: limit price is unavailable.")
        return None

    from ib_insync import LimitOrder

    order = LimitOrder(action, quantity, round(float(limit_price), 2), tif="DAY")
    trade = ib.placeOrder(contract, order)
    ib.sleep(1.0)
    print(
        "Submitted interactive order: "
        f"{action} {quantity} {symbol} LMT {_format_price(order.lmtPrice)} "
        f"status={trade.orderStatus.status}"
    )
    return trade


def _handle_order_signal(
    ib,
    contract,
    signal,
    quantity,
    limit_buffer_bps,
    strategy=None,
    min_sell_profit=0.01,
    min_sell_profit_per_share=0.0,
    profit_target_mode="per_share",
    sizing_mode="fixed",
    cash_reserve=0.0,
    auto_orders=False,
):
    """Prompt or auto-submit when the strategy emits an actionable signal."""
    if signal.signal not in ("WOULD BUY", "WOULD SELL"):
        return None
    if signal.blocked:
        print(f"Order skipped: signal blocked by {signal.block_reason}.")
        return None
    limit_price = _limit_price_for_signal(signal, limit_buffer_bps)
    if limit_price is None:
        return None
    action = "BUY" if signal.signal == "WOULD BUY" else "SELL"
    quantity = _quantity_for_signal(
        signal,
        strategy,
        quantity,
        sizing_mode=sizing_mode,
        cash_reserve=cash_reserve,
        limit_price=limit_price,
    )
    if quantity <= 0:
        if action == "BUY" and sizing_mode == "cash_reserve":
            print(
                "Order skipped: cash-reserve sizing leaves no spendable cash "
                "for at least one share."
            )
        else:
            print("Order skipped: quantity must be positive.")
        return None
    floor = None
    if action == "SELL":
        limit_price, floor = _apply_profitable_sell_floor(
            limit_price,
            strategy,
            quantity,
            min_profit=min_sell_profit,
            min_profit_per_share=min_sell_profit_per_share,
            profit_target_mode=profit_target_mode,
        )

    print("")
    print("Order signal")
    print(f"Signal: {signal.signal}")
    print(f"Symbol: {signal.symbol}")
    print(f"Action: {action}")
    print(f"Quantity: {quantity}")
    print(f"Reference bid/ask: {_format_price(signal.close_bid)} / {_format_price(signal.close_ask)}")
    print(f"Suggested LMT price: {_format_price(limit_price)}")
    if floor is not None:
        print(
            "Profit-safe sell floor: "
            f"{_format_price(floor)} "
            "(includes anchored buy cost and modeled fees)."
        )
        if signal.close_bid is not None and floor > signal.close_bid:
            print(
                "Current bid is below the profit-safe floor; "
                "this sell limit may wait instead of filling immediately."
            )
    if auto_orders:
        print("Auto-order mode is ON: submitting without y/n confirmation.")
        return _submit_limit_order(
            ib,
            contract,
            signal,
            quantity,
            limit_buffer_bps,
            strategy=strategy,
            min_sell_profit=min_sell_profit,
            min_sell_profit_per_share=min_sell_profit_per_share,
            profit_target_mode=profit_target_mode,
        )

    answer = input("Submit this limit order to TWS Paper? Type y to submit: ").strip().lower()
    if answer == "y":
        return _submit_limit_order(
            ib,
            contract,
            signal,
            quantity,
            limit_buffer_bps,
            strategy=strategy,
            min_sell_profit=min_sell_profit,
            min_sell_profit_per_share=min_sell_profit_per_share,
            profit_target_mode=profit_target_mode,
        )
    print("Order skipped by user.")
    return None


def _trade_fill_price(trade):
    status = getattr(trade, "orderStatus", None)
    avg_fill = _safe_float(getattr(status, "avgFillPrice", None))
    if avg_fill is not None and avg_fill > 0:
        return avg_fill
    fills = getattr(trade, "fills", None) or []
    if fills:
        price = _safe_float(getattr(fills[-1].execution, "price", None))
        if price is not None and price > 0:
            return price
    order = getattr(trade, "order", None)
    return _safe_float(getattr(order, "lmtPrice", None))


def _sync_strategy_after_fill(strategy, trade):
    """Update account-aware strategy state only after TWS reports a fill."""
    if strategy is None or trade is None:
        return None
    order = getattr(trade, "order", None)
    status = getattr(trade, "orderStatus", None)
    if order is None:
        return None
    action = getattr(order, "action", None)
    filled = _safe_float(getattr(status, "filled", None))
    synced_filled = _safe_float(getattr(trade, "_strategy_synced_filled", 0.0)) or 0.0
    if action is None or filled is None or filled <= synced_filled:
        return None
    fill_delta = filled - synced_filled
    fill_price = _trade_fill_price(trade)
    if hasattr(strategy, "apply_filled_order"):
        strategy.apply_filled_order(action, fill_delta, fill_price)
        setattr(trade, "_strategy_synced_filled", filled)
        print(
            "Strategy state synced after fill: "
            f"{action} {_format_quantity(fill_delta)} at anchor {_format_price(fill_price)}."
        )
        return {
            "action": action,
            "quantity": fill_delta,
            "price": fill_price,
            "filled": filled,
        }
    return None


def _check_pending_fills(strategy, pending_trades):
    """Sync newly filled pending trades. Return (should_stop, fill_events)."""
    should_stop = False
    fill_events = []
    remaining = []
    for item in pending_trades:
        trade = item["trade"]
        stop_after_fill = item.get("stop_after_fill", False)
        fill_event = _sync_strategy_after_fill(strategy, trade)
        if fill_event is not None:
            fill_events.append(fill_event)
        status = getattr(getattr(trade, "orderStatus", None), "status", "")
        remaining_qty = _safe_float(getattr(getattr(trade, "orderStatus", None), "remaining", None))
        if fill_event is not None and stop_after_fill:
            should_stop = True
        if status not in ("Filled", "Cancelled", "Inactive") and remaining_qty != 0:
            remaining.append(item)
    pending_trades[:] = remaining
    return should_stop, fill_events


def _read_interactive_order_command():
    """Return one pending terminal command without blocking the quote loop."""
    if not sys.stdin or not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return None
    return sys.stdin.readline().strip()


def _handle_interactive_order_command(
    command,
    ib,
    contract,
    symbol,
    row,
    strategy,
    default_quantity,
    limit_buffer_bps,
    min_sell_profit=0.01,
    min_sell_profit_per_share=0.0,
    profit_target_mode="per_share",
    sizing_mode="fixed",
    cash_reserve=0.0,
):
    """Handle terminal commands: buy [qty] [limit], sell [qty] [limit]."""
    if not command:
        return False
    parts = command.split()
    action = parts[0].lower()
    if action in ("help", "?"):
        print("Interactive commands: buy [qty] [limit], sell [qty] [limit], status, quit")
        return False, None
    if action == "status":
        position = getattr(strategy, "initial_position", None)
        avg_cost = getattr(strategy, "initial_avg_cost", None)
        print(
            "Strategy state: "
            f"position={_format_quantity(position)}, "
            f"anchor={_format_price(avg_cost)}, "
            f"bid/ask={_format_price(row.bid)} / {_format_price(row.ask)}"
        )
        return False, None
    if action in ("quit", "exit", "stop"):
        print("Interactive stop requested.")
        return True, None
    if action not in ("buy", "sell"):
        print(f"Unknown interactive command: {command!r}. Type 'help' for options.")
        return False, None

    quantity = None
    if len(parts) >= 2:
        try:
            quantity = int(parts[1])
        except ValueError:
            print("Interactive order skipped: quantity must be an integer.")
            return False, None

    limit_price = None
    if len(parts) >= 3:
        try:
            limit_price = float(parts[2])
        except ValueError:
            print("Interactive order skipped: limit price must be numeric.")
            return False, None
    else:
        limit_price = _limit_price_for_action(row, action.upper(), limit_buffer_bps)
    if quantity is None:
        if action == "buy":
            synthetic_signal = type("InteractiveSignal", (), {
                "signal": "WOULD BUY",
                "close_ask": row.ask,
            })()
        else:
            synthetic_signal = type("InteractiveSignal", (), {
                "signal": "WOULD SELL",
                "close_bid": row.bid,
            })()
        quantity = _quantity_for_signal(
            synthetic_signal,
            strategy,
            default_quantity,
            sizing_mode=sizing_mode,
            cash_reserve=cash_reserve,
            limit_price=limit_price,
        )
    floor = None
    if action == "sell":
        limit_price, floor = _apply_profitable_sell_floor(
            limit_price,
            strategy,
            quantity,
            min_profit=min_sell_profit,
            min_profit_per_share=min_sell_profit_per_share,
            profit_target_mode=profit_target_mode,
        )

    print("")
    print("Interactive override")
    print(f"Command: {command}")
    print(f"Reference bid/ask: {_format_price(row.bid)} / {_format_price(row.ask)}")
    if floor is not None:
        print(
            "Profit-safe sell floor: "
            f"{_format_price(floor)} "
            "(includes anchored buy cost and modeled fees)."
        )
        if row.bid is not None and floor > row.bid:
            print(
                "Current bid is below the profit-safe floor; "
                "this sell limit may wait instead of filling immediately."
            )
    print(f"Submitting: {action.upper()} {quantity} {symbol} LMT {_format_price(limit_price)}")
    trade = _submit_action_limit_order(
        ib,
        contract,
        symbol,
        action.upper(),
        quantity,
        limit_price,
    )
    return False, trade


def place_direct_limit_order(
    symbol=DEFAULT_SYMBOL,
    action="BUY",
    quantity=1,
    limit_price=None,
    limit_buffer_bps=5.0,
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    client_id=DEFAULT_CLIENT_ID,
    timeout=8.0,
    market_data_type=DEFAULT_MARKET_DATA_TYPE,
    confirm_order=False,
):
    """Place one explicit limit order, or preview it without --confirm-order."""
    try:
        from ib_insync import IB, LimitOrder, Stock
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: ib_insync.\n"
            "Install it inside your project environment with:\n"
            "    pip install ib_insync\n"
            "or, if using your venv explicitly:\n"
            "    ./venv/bin/python -m pip install ib_insync"
        ) from exc

    symbol = symbol.upper().strip()
    action = action.upper().strip()
    quantity = int(quantity)
    if action not in ("BUY", "SELL"):
        raise ValueError("--place-order must be BUY or SELL.")
    if quantity <= 0:
        raise ValueError("--order-quantity must be positive.")
    if market_data_type not in MARKET_DATA_TYPES:
        raise ValueError(
            "market_data_type must be one of: "
            + ", ".join(MARKET_DATA_TYPES)
        )

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, readonly=False, timeout=timeout)
        ib.reqMarketDataType(MARKET_DATA_TYPES[market_data_type])
        ib.sleep(0.5)
        contract = Stock(symbol, "SMART", "USD")
        ib.qualifyContracts(contract)

        row = None
        if limit_price is None:
            ticker = ib.reqMktData(contract, "", snapshot=False, regulatorySnapshot=False)
            ib.sleep(3.0)
            row = _quote_from_ticker(ticker, symbol, market_data_type)
            try:
                ib.cancelMktData(contract)
            except Exception:
                pass
            buffer = max(float(limit_buffer_bps), 0.0) / 10000.0
            if action == "BUY":
                if row.ask is None:
                    raise RuntimeError("Cannot infer BUY limit price because ask is unavailable.")
                limit_price = row.ask * (1.0 + buffer)
            else:
                if row.bid is None:
                    raise RuntimeError("Cannot infer SELL limit price because bid is unavailable.")
                limit_price = row.bid * (1.0 - buffer)

        limit_price = round(float(limit_price), 2)
        print("")
        print("Direct order preview")
        print(f"Symbol: {symbol}")
        print(f"Action: {action}")
        print(f"Quantity: {quantity}")
        if row is not None:
            print(f"Reference bid/ask: {_format_price(row.bid)} / {_format_price(row.ask)}")
        print(f"Limit price: {_format_price(limit_price)}")
        print(f"Market data type: {market_data_type}")
        if not confirm_order:
            print("")
            print("No order placed. Add --confirm-order to submit this order.")
            return None

        order = LimitOrder(action, quantity, limit_price, tif="DAY")
        trade = ib.placeOrder(contract, order)
        ib.sleep(1.0)
        print("")
        print(
            "Submitted direct order: "
            f"{action} {quantity} {symbol} LMT {_format_price(order.lmtPrice)} "
            f"status={trade.orderStatus.status}"
        )
        return trade
    finally:
        if ib.isConnected():
            ib.disconnect()


def _quote_from_ticker(ticker, symbol, market_data_type):
    bid = _first_price(ticker.bid, getattr(ticker, "delayedBid", None))
    ask = _first_price(ticker.ask, getattr(ticker, "delayedAsk", None))
    last = _first_price(ticker.last, getattr(ticker, "delayedLast", None), ticker.close)
    mid = (bid + ask) / 2.0 if bid is not None and ask is not None else None
    spread = ask - bid if bid is not None and ask is not None else None
    spread_bps = 10000.0 * spread / mid if spread is not None and mid else None
    return QuoteRow(
        timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
        symbol=symbol,
        bid=bid,
        ask=ask,
        last=last,
        mid=mid,
        spread=spread,
        spread_bps=spread_bps,
        market_data_type=market_data_type,
    )


def _print_quote_row(row):
    print(
        f"{row.timestamp} | {row.symbol:<5} | "
        f"bid {_format_price(row.bid):>10} | "
        f"ask {_format_price(row.ask):>10} | "
        f"mid {_format_price(row.mid):>10} | "
        f"last {_format_price(row.last):>10} | "
        f"spread {_format_price(row.spread):>9} | "
        f"{_format_bps(row.spread_bps):>7} bp"
    )


def _open_quote_log(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    file_obj = path.open("a", newline="")
    writer = csv.DictWriter(
        file_obj,
        fieldnames=[
            "timestamp",
            "symbol",
            "bid",
            "ask",
            "last",
            "mid",
            "spread",
            "spread_bps",
            "market_data_type",
        ],
    )
    if not exists:
        writer.writeheader()
    return file_obj, writer


def _open_signal_log(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    file_obj = path.open("a", newline="")
    writer = csv.DictWriter(
        file_obj,
        fieldnames=[
            "timestamp",
            "symbol",
            "bars",
            "close_mid",
            "close_bid",
            "close_ask",
            "position",
            "signal",
            "z",
            "realized_w",
            "mtm_w",
            "trades",
            "blocked",
            "block_reason",
            "note",
        ],
    )
    if not exists:
        writer.writeheader()
    return file_obj, writer


def _fill_event_to_signal_row(fill_event, row, strategy):
    """Represent an IBKR fill as a signal-log row for auditability."""
    action = fill_event["action"].upper()
    position = (
        "LONG"
        if getattr(strategy, "initial_position", 0.0)
        and strategy.initial_position > 0
        else "CASH"
    )
    return {
        "timestamp": row.timestamp,
        "symbol": row.symbol,
        "bars": len(getattr(strategy, "display_bars", [])),
        "close_mid": row.mid,
        "close_bid": row.bid,
        "close_ask": row.ask,
        "position": position,
        "signal": f"FILLED {action}",
        "z": None,
        "realized_w": "",
        "mtm_w": "",
        "trades": "",
        "blocked": False,
        "block_reason": "",
        "note": (
            f"IBKR fill confirmed: {action} "
            f"{_format_quantity(fill_event['quantity'])} at "
            f"{_format_price(fill_event['price'])}. Anchor updated now."
        ),
    }


def _load_strategy_state(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _save_strategy_state(strategy, path):
    if strategy is None or path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file_obj:
        json.dump(strategy.export_state(), file_obj, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _save_live_strategy_plot(strategy, signal, path, window_bars=120):
    """Save a refreshed live plot of mid-price, funnel, and Z statistic."""
    data = getattr(strategy, "latest_data", None)
    display_bars = getattr(strategy, "display_bars", [])
    if not display_bars:
        return

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    display_prices = [bar.close_mid for bar in display_bars]
    display_funnel_mid = getattr(strategy, "display_funnel_mid", [])
    display_funnel_up = getattr(strategy, "display_funnel_up", [])
    display_funnel_low = getattr(strategy, "display_funnel_low", [])
    display_z = getattr(strategy, "display_z", [])
    t_end = len(display_prices)
    start = max(0, t_end - int(window_bars))
    xs = list(range(start, t_end))

    fig = Figure(figsize=(11, 7), dpi=120, facecolor="#10151f")
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.24)
    ax_p = fig.add_subplot(gs[0])
    ax_z = fig.add_subplot(gs[1], sharex=ax_p)
    for ax in (ax_p, ax_z):
        ax.set_facecolor("#151b26")
        ax.grid(True, color="#2a3342", alpha=0.65, linewidth=0.7)
        ax.tick_params(colors="#d6deeb")
        for spine in ax.spines.values():
            spine.set_color("#3b4658")

    ax_p.plot(xs, display_prices[start:t_end], color="#5dade2", lw=1.6, label="Mid price")
    if len(display_funnel_up) == t_end:
        ax_p.plot(xs, display_funnel_up[start:t_end], color="#f59e0b", lw=1.0, ls="--", label="Upper funnel")
        ax_p.plot(xs, display_funnel_low[start:t_end], color="#7dd3fc", lw=1.0, ls="--", label="Lower funnel")
        ax_p.plot(xs, display_funnel_mid[start:t_end], color="#cbd5e1", lw=0.9, ls=":", label="Funnel center")

    z_x = []
    z_y = []
    if data is not None:
        strategy_end = int(data["N"]) + 1
        mapped = [
            (strategy.strategy_x_to_display_x(i), i)
            for i in range(strategy_end)
        ]
        mapped = [
            (display_x, strategy_x)
            for display_x, strategy_x in mapped
            if start <= display_x < t_end
        ]
        if mapped:
            mx = [display_x for display_x, _ in mapped]
            mi = [strategy_x for _, strategy_x in mapped]
            z_x = mx
            z_y = [data["Zsig"][i] for i in mi]

        # Live plots show only broker-confirmed fills as trade markers. The
        # research engine can emit virtual buy/sell times while recomputing
        # diagnostics, but those are not orders and should not appear here.

    filled_markers = getattr(strategy, "filled_markers", [])
    filled_buys = [
        m for m in filled_markers
        if (
            m.get("action") == "BUY"
            and m.get("price") is not None
            and start <= int(m.get("index", -1)) < t_end
        )
    ]
    filled_sells = [
        m for m in filled_markers
        if (
            m.get("action") == "SELL"
            and m.get("price") is not None
            and start <= int(m.get("index", -1)) < t_end
        )
    ]
    if filled_buys:
        ax_p.scatter(
            [int(m["index"]) for m in filled_buys],
            [float(m["price"]) for m in filled_buys],
            marker="^",
            s=95,
            color="#16a34a",
            edgecolors="#f8fafc",
            linewidths=0.8,
            label="Filled buy",
            zorder=6,
        )
    if filled_sells:
        ax_p.scatter(
            [int(m["index"]) for m in filled_sells],
            [float(m["price"]) for m in filled_sells],
            marker="v",
            s=95,
            color="#dc2626",
            edgecolors="#f8fafc",
            linewidths=0.8,
            label="Filled sell",
            zorder=6,
        )

    current_color = "#22c55e" if getattr(strategy, "initial_position", 0.0) and strategy.initial_position > 0 else "#f97316"
    ax_p.scatter([t_end - 1], [display_prices[-1]], s=65, color=current_color, zorder=5)
    ax_p.set_ylabel("Price", color="#d6deeb")
    signal_text = "n/a" if signal is None else signal.signal
    mid_text = display_prices[-1]
    mtm_text = "n/a" if signal is None else f"{signal.mtm_w:.2f}"
    ax_p.set_title(
        f"{strategy.symbol} live strategy | {signal_text} | "
        f"mid {mid_text:.2f} | W {mtm_text}",
        color="#f8fafc",
    )
    ax_p.legend(loc="upper left", ncol=4, fontsize=8, facecolor="#10151f", edgecolor="#3b4658", labelcolor="#d6deeb")

    if len(display_z) == t_end:
        ax_z.plot(xs, display_z[start:t_end], color="#14b8a6", lw=1.4, label="Z statistic")
    if data is not None:
        ax_z.axhline(float(data["k"]), color="#ef4444", lw=0.9, ls="--", label="+k")
        ax_z.axhline(-float(data["k"]), color="#22c55e", lw=0.9, ls="--", label="-k")
        trend_z = data.get("trend_entry_z")
        if trend_z is not None:
            ax_z.axhline(float(trend_z), color="#f59e0b", lw=0.8, ls=":", label="z_trend")
    ax_z.set_ylabel("Z", color="#d6deeb")
    ax_z.set_xlabel("Completed live bar index", color="#d6deeb")
    ax_z.legend(loc="upper left", ncol=4, fontsize=8, facecolor="#10151f", edgecolor="#3b4658", labelcolor="#d6deeb")

    fig.savefig(path, bbox_inches="tight")
    canvas.draw()


def _json_float(value):
    """Return JSON-safe float values, preserving missing values as null."""
    value = _safe_float(value)
    return None if value is None else value


def _save_live_strategy_html_plot(strategy, signal, path):
    """Save a self-contained browser plot with pan/zoom over all live bars."""
    display_bars = getattr(strategy, "display_bars", [])
    if not display_bars:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = []
    display_funnel_mid = getattr(strategy, "display_funnel_mid", [])
    display_funnel_up = getattr(strategy, "display_funnel_up", [])
    display_funnel_low = getattr(strategy, "display_funnel_low", [])
    display_z = getattr(strategy, "display_z", [])
    for i, bar in enumerate(display_bars):
        points.append({
            "x": i,
            "timestamp": bar.timestamp,
            "mid": _json_float(bar.close_mid),
            "bid": _json_float(bar.close_bid),
            "ask": _json_float(bar.close_ask),
            "upper": _json_float(display_funnel_up[i]) if i < len(display_funnel_up) else None,
            "lower": _json_float(display_funnel_low[i]) if i < len(display_funnel_low) else None,
            "center": _json_float(display_funnel_mid[i]) if i < len(display_funnel_mid) else None,
            "z": _json_float(display_z[i]) if i < len(display_z) else None,
        })

    filled_markers = [
        {
            "x": int(marker.get("index", 0)),
            "action": str(marker.get("action", "")).upper(),
            "price": _json_float(marker.get("price")),
            "quantity": _json_float(marker.get("quantity")),
            "timestamp": marker.get("timestamp", ""),
        }
        for marker in getattr(strategy, "filled_markers", [])
        if marker.get("price") is not None
    ]

    latest_data = getattr(strategy, "latest_data", None) or {}
    sell_floor = None
    if getattr(strategy, "initial_position", 0.0) and strategy.initial_position > 0:
        floor_fn = getattr(strategy, "_minimum_profitable_sell_price", None)
        if callable(floor_fn):
            sell_floor = floor_fn()
    latest_bar = display_bars[-1]
    latest_spread = None
    if latest_bar.close_bid is not None and latest_bar.close_ask is not None:
        latest_spread = latest_bar.close_ask - latest_bar.close_bid
    buy_fills = [marker for marker in filled_markers if marker.get("action") == "BUY"]
    sell_fills = [marker for marker in filled_markers if marker.get("action") == "SELL"]
    live_account = {}
    live_account_fn = getattr(strategy, "live_account_snapshot", None)
    if callable(live_account_fn):
        live_account = live_account_fn(latest_bar)
    stats = {
        "bars": len(display_bars),
        "cycleBars": len(getattr(strategy, "bars", [])),
        "latestMid": _json_float(latest_bar.close_mid),
        "latestBid": _json_float(latest_bar.close_bid),
        "latestAsk": _json_float(latest_bar.close_ask),
        "latestSpread": _json_float(latest_spread),
        "latestSpreadBps": _json_float(latest_bar.avg_spread_bps),
        "positionQty": _json_float(getattr(strategy, "initial_position", None)),
        "anchorPrice": _json_float(getattr(strategy, "initial_avg_cost", None)),
        "cashAnchorPrice": _json_float(getattr(strategy, "cash_anchor_price", None)),
        "realizedW": None if signal is None else _json_float(signal.realized_w),
        "mtmW": None if signal is None else _json_float(signal.mtm_w),
        "trades": None if signal is None else int(signal.trades),
        "filledBuys": len(buy_fills),
        "filledSells": len(sell_fills),
        "lastBuy": buy_fills[-1] if buy_fills else None,
        "lastSell": sell_fills[-1] if sell_fills else None,
        "sellFloor": _json_float(sell_floor),
        "liveAccount": live_account,
        "blockReason": None if signal is None else signal.block_reason,
        "note": None if signal is None else signal.note,
    }
    params = {
        "k": _json_float(getattr(strategy, "k", None)),
        "delta": getattr(strategy, "delta", None),
        "entryDelta": getattr(strategy, "entry_delta", None),
        "lookbackL": getattr(strategy, "max_funnel_lookback", None),
        "trailA": _json_float(getattr(strategy, "trailing_stop", None)),
        "zTrend": _json_float(getattr(strategy, "trend_entry_z", None)),
        "driftQ": _json_float(getattr(strategy, "drift_process_var", None)),
        "cost": _json_float(getattr(strategy, "cost", None)),
        "fixedBuyFee": _json_float(getattr(strategy, "fixed_buy_fee", None)),
        "fixedSellFee": _json_float(getattr(strategy, "fixed_sell_fee", None)),
        "minSellProfit": _json_float(getattr(strategy, "min_sell_profit", None)),
        "minSellProfitPerShare": _json_float(getattr(strategy, "min_sell_profit_per_share", None)),
        "profitTargetMode": getattr(strategy, "profit_target_mode", None),
        "orderQuantity": getattr(strategy, "order_quantity", None),
        "capitalBudget": _json_float(getattr(strategy, "capital_budget", None)),
        "sizingMode": getattr(strategy, "sizing_mode", None),
        "cashReserve": _json_float(getattr(strategy, "cash_reserve", None)),
        "regularHoursOnly": bool(getattr(strategy, "regular_hours_only", False)),
        "maxSpreadBps": _json_float(getattr(strategy, "max_spread_bps", None)),
        "maxQuoteAge": _json_float(getattr(strategy, "max_quote_age", None)),
        "cashReentryCooldownBars": getattr(
            strategy, "cash_reentry_cooldown_bars", None
        ),
    }
    payload = {
        "symbol": getattr(strategy, "symbol", ""),
        "signal": None if signal is None else signal.signal,
        "position": None if signal is None else signal.position,
        "k": _json_float(latest_data.get("k")),
        "zTrend": _json_float(latest_data.get("trend_entry_z")),
        "sellFloor": _json_float(sell_floor),
        "updatedAt": datetime.now().isoformat(timespec="seconds"),
        "stats": stats,
        "params": params,
        "points": points,
        "markers": filled_markers,
    }

    payload_json = json.dumps(payload, allow_nan=False)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{payload['symbol']} Live Strategy Plot</title>
<style>
  :root {{
    --bg: #0b1018;
    --panel: #111827;
    --grid: #283244;
    --text: #e5eefc;
    --muted: #94a3b8;
    --blue: #60a5fa;
    --orange: #f59e0b;
    --cyan: #7dd3fc;
    --green: #22c55e;
    --red: #ef4444;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: radial-gradient(circle at top left, #172033, var(--bg) 48%);
    color: var(--text);
    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  main {{
    width: min(1500px, 100vw);
    margin: 0 auto;
    padding: 18px;
  }}
  .topbar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 12px;
  }}
  h1 {{
    margin: 0;
    font-size: 22px;
    letter-spacing: 0.02em;
  }}
  .meta {{
    color: var(--muted);
    font-size: 13px;
  }}
  .chart-card {{
    background: rgba(17, 24, 39, 0.92);
    border: 1px solid #263244;
    border-radius: 18px;
    box-shadow: 0 18px 55px rgba(0, 0, 0, 0.35);
    padding: 14px;
  }}
  .info-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
    margin-top: 14px;
  }}
  .info-card {{
    background: rgba(17, 24, 39, 0.92);
    border: 1px solid #263244;
    border-radius: 18px;
    padding: 14px;
  }}
  .info-card h2 {{
    margin: 0 0 10px;
    font-size: 16px;
    color: #f8fafc;
  }}
  .kv {{
    display: grid;
    grid-template-columns: minmax(130px, 1fr) minmax(90px, auto);
    gap: 7px 14px;
    color: var(--muted);
    font-size: 13px;
  }}
  .kv strong {{
    color: var(--text);
    font-weight: 650;
    text-align: right;
  }}
  .note {{
    margin-top: 10px;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.45;
  }}
  canvas {{
    width: 100%;
    height: 680px;
    display: block;
    background: #0f1724;
    border-radius: 12px;
    cursor: grab;
  }}
  canvas:active {{ cursor: grabbing; }}
  .controls {{
    display: grid;
    grid-template-columns: 1fr auto auto auto auto;
    gap: 10px;
    align-items: center;
    margin-top: 12px;
  }}
  input[type="range"] {{ width: 100%; }}
  button {{
    color: var(--text);
    background: #1f2937;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 8px 11px;
    cursor: pointer;
  }}
  button:hover {{ background: #273449; }}
  label {{
    color: var(--muted);
    font-size: 13px;
    white-space: nowrap;
  }}
  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 13px;
    margin-top: 10px;
    color: var(--muted);
    font-size: 13px;
  }}
  .swatch {{
    display: inline-block;
    width: 12px;
    height: 3px;
    margin-right: 5px;
    vertical-align: middle;
  }}
  @media (max-width: 800px) {{
    canvas {{ height: 520px; }}
    .controls {{ grid-template-columns: 1fr 1fr; }}
    .info-grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<main>
  <div class="topbar">
    <div>
      <h1 id="title"></h1>
      <div class="meta" id="meta"></div>
    </div>
    <div class="meta">Drag to pan. Mouse wheel or trackpad scroll to zoom. Slider jumps through history.</div>
  </div>
  <section class="chart-card">
    <canvas id="chart"></canvas>
    <div class="controls">
      <input id="range" type="range" min="0" value="0" step="1">
      <button id="zoomIn">Zoom in</button>
      <button id="zoomOut">Zoom out</button>
      <button id="latest">Latest</button>
      <label><input id="autoRefresh" type="checkbox" checked> Auto-refresh</label>
    </div>
    <div class="legend">
      <span><i class="swatch" style="background:#60a5fa"></i>Mid</span>
      <span><i class="swatch" style="background:#f59e0b"></i>Upper funnel</span>
      <span><i class="swatch" style="background:#7dd3fc"></i>Lower funnel</span>
      <span><i class="swatch" style="background:#cbd5e1"></i>Center</span>
      <span><i class="swatch" style="background:#fb7185"></i>Profit floor</span>
      <span><i class="swatch" style="background:#14b8a6"></i>Z statistic</span>
      <span style="color:#22c55e">● Latest price</span>
      <span style="color:#22c55e">▲ Filled buy</span>
      <span style="color:#ef4444">▼ Filled sell</span>
    </div>
  </section>
  <section class="info-grid">
    <div class="info-card">
      <h2>Live Stats</h2>
      <div class="kv" id="statsGrid"></div>
      <div class="note" id="statsNote"></div>
    </div>
    <div class="info-card">
      <h2>Parameters</h2>
      <div class="kv" id="paramsGrid"></div>
    </div>
  </section>
</main>
<script>
const payload = {payload_json};
const canvas = document.getElementById("chart");
const ctx = canvas.getContext("2d");
const range = document.getElementById("range");
const autoRefresh = document.getElementById("autoRefresh");
const points = payload.points || [];
const markers = payload.markers || [];
const storageKey = `ibkrLivePlot:${{payload.symbol}}`;
const storedView = (() => {{
  try {{
    return JSON.parse(localStorage.getItem(storageKey) || "{{}}");
  }} catch (_) {{
    return {{}};
  }}
}})();
let windowSize = Math.min(Math.max(120, Math.floor(points.length * 0.35)), Math.max(points.length, 1));
if (Number.isFinite(storedView.windowSize)) {{
  windowSize = Math.max(20, Math.min(points.length || 1, Math.round(storedView.windowSize)));
}}
let followLatest = storedView.followLatest !== false;
let start = followLatest
  ? Math.max(0, points.length - windowSize)
  : Math.max(0, Math.min(Number(storedView.start || 0), Math.max(0, points.length - windowSize)));
autoRefresh.checked = storedView.autoRefresh !== false;
let dragging = false;
let dragStartX = 0;
let dragStartStart = 0;

document.getElementById("title").textContent =
  `${{payload.symbol}} live strategy | ${{payload.signal || "n/a"}} | ${{payload.position || "n/a"}}`;
document.getElementById("meta").textContent =
  `Updated ${{payload.updatedAt}} | Bars ${{points.length}} | k=${{fmt(payload.k)}} | z_trend=${{fmt(payload.zTrend)}} | sell floor=${{fmt(payload.sellFloor)}}`;

function renderGrid(id, rows) {{
  const el = document.getElementById(id);
  el.innerHTML = rows.map(([label, value]) =>
    `<span>${{label}}</span><strong>${{value}}</strong>`
  ).join("");
}}

function money(value) {{
  return value === null || value === undefined ? "n/a" : `$${{Number(value).toFixed(2)}}`;
}}

function pct(value) {{
  return value === null || value === undefined ? "n/a" : `${{Number(value).toFixed(2)}}%`;
}}

function markerText(marker) {{
  if (!marker) return "n/a";
  return `${{marker.action}} @ ${{fmt(marker.price)}}`;
}}

const stats = payload.stats || {{}};
const params = payload.params || {{}};
const liveAccount = stats.liveAccount || {{}};
renderGrid("statsGrid", [
  ["Signal", payload.signal || "n/a"],
  ["Position", `${{payload.position || "n/a"}} (${{fmt(stats.positionQty)}} sh)`],
  ["Bars / cycle bars", `${{fmt(stats.bars)}} / ${{fmt(stats.cycleBars)}}`],
  ["Mid / bid / ask", `${{money(stats.latestMid)}} / ${{money(stats.latestBid)}} / ${{money(stats.latestAsk)}}`],
  ["Spread", `${{money(stats.latestSpread)}} (${{fmt(stats.latestSpreadBps)}} bps)`],
  ["Capital budget", money(liveAccount.capital_budget)],
  ["Live cash", money(liveAccount.cash)],
  ["Live shares", fmt(liveAccount.shares)],
  ["Live stock value", money(liveAccount.position_value)],
  ["Live liquidation W", money(liveAccount.liquidation_value)],
  ["Live net P&L", money(liveAccount.pnl)],
  ["Live return", pct((liveAccount.return ?? null) === null ? null : liveAccount.return * 100)],
  ["Anchor price", money(stats.anchorPrice)],
  ["Cash anchor", money(stats.cashAnchorPrice)],
  ["Profit floor", money(stats.sellFloor)],
  ["Research realized W", money(stats.realizedW)],
  ["Research MtM W", money(stats.mtmW)],
  ["Completed trades", fmt(stats.trades)],
  ["Filled buys / sells", `${{fmt(stats.filledBuys)}} / ${{fmt(stats.filledSells)}}`],
  ["Last buy", markerText(stats.lastBuy)],
  ["Last sell", markerText(stats.lastSell)],
]);
document.getElementById("statsNote").textContent =
  stats.note ? `Note: ${{stats.note}}` : (stats.blockReason ? `Blocked: ${{stats.blockReason}}` : "");
renderGrid("paramsGrid", [
  ["k", fmt(params.k)],
  ["delta / entry_delta", `${{fmt(params.delta)}} / ${{fmt(params.entryDelta)}}`],
  ["L", fmt(params.lookbackL)],
  ["trailing a", fmt(params.trailA)],
  ["z_trend", fmt(params.zTrend)],
  ["drift q", fmt(params.driftQ)],
  ["proportional cost", fmt(params.cost)],
  ["fixed buy / sell fee", `${{money(params.fixedBuyFee)}} / ${{money(params.fixedSellFee)}}`],
  ["min sell profit", money(params.minSellProfit)],
  ["min profit/share", money(params.minSellProfitPerShare)],
  ["profit target mode", params.profitTargetMode || "n/a"],
  ["order quantity", fmt(params.orderQuantity)],
  ["capital budget", money(params.capitalBudget)],
  ["sizing mode", params.sizingMode || "n/a"],
  ["cash reserve", money(params.cashReserve)],
  ["regular hours only", params.regularHoursOnly ? "yes" : "no"],
  ["max spread", `${{fmt(params.maxSpreadBps)}} bps`],
  ["max quote age", `${{fmt(params.maxQuoteAge)}} sec`],
  ["cash cooldown bars", fmt(params.cashReentryCooldownBars)],
]);

function fmt(value) {{
  return value === null || value === undefined ? "n/a" : Number(value).toFixed(4).replace(/0+$/, "").replace(/\\.$/, "");
}}

function resizeCanvas() {{
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.floor(rect.width * ratio);
  canvas.height = Math.floor(rect.height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}}

function visiblePoints() {{
  const end = Math.min(points.length, start + windowSize);
  return points.slice(start, end);
}}

function finite(values) {{
  return values.filter(v => v !== null && v !== undefined && Number.isFinite(v));
}}

function yScale(values, top, bottom) {{
  const vals = finite(values);
  let min = vals.length ? Math.min(...vals) : 0;
  let max = vals.length ? Math.max(...vals) : 1;
  if (Math.abs(max - min) < 1e-9) {{
    max += 1;
    min -= 1;
  }}
  const pad = (max - min) * 0.08;
  min -= pad;
  max += pad;
  const scale = value => bottom - ((value - min) / (max - min)) * (bottom - top);
  scale.min = min;
  scale.max = max;
  return scale;
}}

function drawGrid(left, top, right, bottom, yFor=null, xStart=null, xEnd=null) {{
  ctx.strokeStyle = "#283244";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#94a3b8";
  ctx.font = "12px system-ui";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 5; i++) {{
    const y = top + (bottom - top) * i / 5;
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
    if (yFor) {{
      const value = yFor.max - (yFor.max - yFor.min) * i / 5;
      ctx.fillText(fmtAxis(value), left - 8, y);
    }}
  }}
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let i = 0; i <= 8; i++) {{
    const x = left + (right - left) * i / 8;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
    if (xStart !== null && xEnd !== null) {{
      const label = Math.round(xStart + (xEnd - xStart) * i / 8);
      ctx.fillText(String(label), x, bottom + 7);
    }}
  }}
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
}}

function fmtAxis(value) {{
  const absValue = Math.abs(value);
  if (absValue >= 100) return value.toFixed(2);
  if (absValue >= 10) return value.toFixed(3).replace(/0+$/, "").replace(/\\.$/, "");
  return value.toFixed(4).replace(/0+$/, "").replace(/\\.$/, "");
}}

function line(series, xFor, yFor, color, dash=[]) {{
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash(dash);
  ctx.beginPath();
  let active = false;
  series.forEach(p => {{
    const yv = p.value;
    if (yv === null || yv === undefined || !Number.isFinite(yv)) {{
      active = false;
      return;
    }}
    const x = xFor(p.x);
    const y = yFor(yv);
    if (!active) {{
      ctx.moveTo(x, y);
      active = true;
    }} else {{
      ctx.lineTo(x, y);
    }}
  }});
  ctx.stroke();
  ctx.setLineDash([]);
}}

function drawTriangle(x, y, up, color) {{
  ctx.fillStyle = color;
  ctx.strokeStyle = "#f8fafc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  if (up) {{
    ctx.moveTo(x, y - 9);
    ctx.lineTo(x - 8, y + 7);
    ctx.lineTo(x + 8, y + 7);
  }} else {{
    ctx.moveTo(x, y + 9);
    ctx.lineTo(x - 8, y - 7);
    ctx.lineTo(x + 8, y - 7);
  }}
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}}

function drawCircle(x, y, radius, color) {{
  ctx.fillStyle = color;
  ctx.strokeStyle = "#f8fafc";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}}

function draw() {{
  resizeCanvas();
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!points.length) return;
  range.max = Math.max(0, points.length - windowSize);
  range.value = start;

  const left = 76, right = rect.width - 24;
  const priceTop = 34, priceBottom = Math.floor(rect.height * 0.63);
  const zTop = priceBottom + 62, zBottom = rect.height - 50;
  const vis = visiblePoints();
  const xEnd = Math.min(points.length - 1, start + windowSize - 1);
  const denom = Math.max(vis.length - 1, 1);
  const xFor = x => left + ((x - start) / denom) * (right - left);
  const priceVals = [];
  vis.forEach(p => priceVals.push(p.mid, p.upper, p.lower, p.center));
  if (payload.sellFloor !== null && payload.sellFloor !== undefined) priceVals.push(payload.sellFloor);
  const zVals = [];
  vis.forEach(p => zVals.push(p.z));
  if (payload.k !== null && payload.k !== undefined) zVals.push(payload.k, -payload.k);
  if (payload.zTrend !== null && payload.zTrend !== undefined) zVals.push(payload.zTrend);
  const yPrice = yScale(priceVals, priceTop, priceBottom);
  const yZ = yScale(zVals, zTop, zBottom);

  drawGrid(left, priceTop, right, priceBottom, yPrice, start, xEnd);
  drawGrid(left, zTop, right, zBottom, yZ, start, xEnd);

  line(vis.map(p => ({{x: p.x, value: p.mid}})), xFor, yPrice, "#60a5fa");
  line(vis.map(p => ({{x: p.x, value: p.upper}})), xFor, yPrice, "#f59e0b", [7, 5]);
  line(vis.map(p => ({{x: p.x, value: p.lower}})), xFor, yPrice, "#7dd3fc", [7, 5]);
  line(vis.map(p => ({{x: p.x, value: p.center}})), xFor, yPrice, "#cbd5e1", [2, 5]);
  line(vis.map(p => ({{x: p.x, value: p.z}})), xFor, yZ, "#14b8a6");

  function priceHline(value, color, label, dash=[]) {{
    if (value === null || value === undefined) return;
    const y = yPrice(value);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = color;
    ctx.font = "12px system-ui";
    ctx.fillText(`${{label}} ${{fmt(value)}}`, right - 112, y - 5);
  }}
  priceHline(payload.sellFloor, "#fb7185", "floor", [2, 5]);

  function hline(value, color, label, dash=[]) {{
    if (value === null || value === undefined) return;
    const y = yZ(value);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = color;
    ctx.font = "12px system-ui";
    ctx.fillText(label, right - 56, y - 5);
  }}
  hline(payload.k, "#ef4444", "+k", [7, 5]);
  hline(payload.k === null || payload.k === undefined ? null : -payload.k, "#22c55e", "-k", [7, 5]);
  hline(payload.zTrend, "#f59e0b", "z_trend", [2, 5]);

  markers
    .filter(m => m.price !== null && m.x >= start && m.x < start + windowSize)
    .forEach(m => drawTriangle(xFor(m.x), yPrice(m.price), m.action === "BUY", m.action === "BUY" ? "#22c55e" : "#ef4444"));

  const latest = points[points.length - 1];
  if (
    latest
    && latest.mid !== null
    && latest.mid !== undefined
    && latest.x >= start
    && latest.x < start + windowSize
  ) {{
    drawCircle(xFor(latest.x), yPrice(latest.mid), 6.5, "#22c55e");
  }}

  ctx.fillStyle = "#e5eefc";
  ctx.font = "13px system-ui";
  ctx.fillText(`Bars ${{start}}-${{Math.min(points.length - 1, start + windowSize - 1)}} of ${{points.length - 1}}`, left, 22);
  ctx.fillStyle = "#94a3b8";
  ctx.fillText(`Latest mid ${{fmt(latest.mid)}} | bid ${{fmt(latest.bid)}} | ask ${{fmt(latest.ask)}}`, right - 280, 22);
  ctx.textAlign = "center";
  ctx.fillText("Completed live bar index", (left + right) / 2, rect.height - 14);
  ctx.textAlign = "left";
  ctx.fillStyle = "#94a3b8";
  ctx.fillText("Price", 12, priceTop + 18);
  ctx.fillText("Z", 26, zTop + 18);
}}

function setStart(value) {{
  const maxStart = Math.max(0, points.length - windowSize);
  start = Math.max(0, Math.min(Number(value), maxStart));
  followLatest = start >= maxStart - 1;
  saveViewState();
  draw();
}}

function zoom(factor) {{
  const center = start + windowSize / 2;
  windowSize = Math.max(20, Math.min(points.length || 1, Math.round(windowSize * factor)));
  setStart(Math.round(center - windowSize / 2));
}}

function saveViewState() {{
  localStorage.setItem(storageKey, JSON.stringify({{
    start,
    windowSize,
    followLatest,
    autoRefresh: autoRefresh.checked
  }}));
}}

range.addEventListener("input", e => setStart(e.target.value));
document.getElementById("zoomIn").addEventListener("click", () => zoom(0.7));
document.getElementById("zoomOut").addEventListener("click", () => zoom(1.35));
document.getElementById("latest").addEventListener("click", () => {{
  followLatest = true;
  setStart(Math.max(0, points.length - windowSize));
}});
autoRefresh.addEventListener("change", saveViewState);
canvas.addEventListener("wheel", e => {{
  e.preventDefault();
  zoom(e.deltaY < 0 ? 0.82 : 1.18);
}}, {{passive: false}});
canvas.addEventListener("mousedown", e => {{
  dragging = true;
  dragStartX = e.clientX;
  dragStartStart = start;
}});
window.addEventListener("mouseup", () => dragging = false);
window.addEventListener("mousemove", e => {{
  if (!dragging) return;
  const rect = canvas.getBoundingClientRect();
  const dx = e.clientX - dragStartX;
  const barsMoved = Math.round(-dx / Math.max(rect.width - 82, 1) * windowSize);
  setStart(dragStartStart + barsMoved);
}});
window.addEventListener("resize", draw);
setInterval(() => {{
  saveViewState();
  if (autoRefresh.checked) {{
    window.location.reload();
  }}
}}, 20000);
draw();
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def fetch_stock_quote(
    symbol=DEFAULT_SYMBOL,
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    client_id=DEFAULT_CLIENT_ID,
    timeout=8.0,
    market_data_type=DEFAULT_MARKET_DATA_TYPE,
):
    """Connect to TWS and request a read-only stock quote snapshot."""
    try:
        from ib_insync import IB, Stock
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: ib_insync.\n"
            "Install it inside your project environment with:\n"
            "    pip install ib_insync\n"
            "or, if using your venv explicitly:\n"
            "    ./venv/bin/python -m pip install ib_insync"
        ) from exc

    symbol = symbol.upper().strip()
    if not symbol:
        raise ValueError("Symbol cannot be empty.")
    if market_data_type not in MARKET_DATA_TYPES:
        raise ValueError(
            "market_data_type must be one of: "
            + ", ".join(MARKET_DATA_TYPES)
        )

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, readonly=True, timeout=timeout)
        ib.reqMarketDataType(MARKET_DATA_TYPES[market_data_type])
        # Give TWS a moment to apply the requested market-data mode before
        # creating the subscription. This matters most when forcing delayed.
        ib.sleep(0.5)
        contract = Stock(symbol, "SMART", "USD")
        ib.qualifyContracts(contract)

        ticker = ib.reqMktData(contract, "", snapshot=False, regulatorySnapshot=False)
        ib.sleep(4.0)

        row = _quote_from_ticker(ticker, symbol, market_data_type)

        return QuoteSnapshot(
            symbol=symbol,
            bid=row.bid,
            ask=row.ask,
            last=row.last,
            mid=row.mid,
            market_data_type=market_data_type,
        )
    finally:
        if ib.isConnected():
            ib.disconnect()


def watch_stock_quote(
    symbol=DEFAULT_SYMBOL,
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    client_id=DEFAULT_CLIENT_ID,
    timeout=8.0,
    market_data_type=DEFAULT_MARKET_DATA_TYPE,
    interval=5.0,
    duration=None,
    log_path=None,
    dry_run_strategy=False,
    bar_seconds=60,
    strategy_kwargs=None,
    signal_log_path=None,
    live_plot_path=None,
    live_plot_html_path=None,
    strategy_state_path=None,
    resume_strategy_state=False,
    plot_window_bars=120,
    manual_orders=False,
    auto_orders=False,
    interactive_orders=False,
    order_quantity=4,
    limit_buffer_bps=5.0,
    min_sell_profit=0.01,
    min_sell_profit_per_share=0.0,
    profit_target_mode="per_share",
    sizing_mode="fixed",
    cash_reserve=0.0,
    stop_after_order=True,
):
    """Continuously print Level 1 quotes from TWS and optionally act on signals."""
    try:
        from ib_insync import IB, Stock
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: ib_insync.\n"
            "Install it inside your project environment with:\n"
            "    pip install ib_insync\n"
            "or, if using your venv explicitly:\n"
            "    ./venv/bin/python -m pip install ib_insync"
        ) from exc

    symbol = symbol.upper().strip()
    if not symbol:
        raise ValueError("Symbol cannot be empty.")
    if market_data_type not in MARKET_DATA_TYPES:
        raise ValueError(
            "market_data_type must be one of: "
            + ", ".join(MARKET_DATA_TYPES)
        )
    interval = max(float(interval), 0.5)
    duration = None if duration is None else max(float(duration), 0.0)

    log_file = None
    log_writer = None
    signal_log_file = None
    signal_log_writer = None
    bar_builder = None
    strategy = None
    if dry_run_strategy:
        from live_strategy import (
            LiveDryRunStrategy,
            MidBarBuilder,
            print_strategy_signal,
            signal_to_dict,
        )
        bar_builder = MidBarBuilder(bar_seconds=bar_seconds)
        strategy = LiveDryRunStrategy(symbol=symbol, **(strategy_kwargs or {}))
        if resume_strategy_state and strategy_state_path:
            state = _load_strategy_state(strategy_state_path)
            if state is None:
                print(f"No previous strategy state found at {strategy_state_path}; starting fresh.")
            else:
                strategy.import_state(
                    state,
                    account_position=(strategy_kwargs or {}).get("initial_position"),
                    account_avg_cost=(strategy_kwargs or {}).get("initial_avg_cost"),
                )
                floor_fn = getattr(strategy, "_minimum_profitable_sell_price", None)
                floor = floor_fn() if callable(floor_fn) else None
                print(
                    "Resumed strategy state: "
                    f"display_bars={len(strategy.display_bars)}, "
                    f"cycle_bars={len(strategy.bars)}, "
                    f"position={_format_quantity(strategy.initial_position)}, "
                    f"anchor={_format_price(strategy.initial_avg_cost)}, "
                    f"sell_floor={_format_price(floor)}"
                )
        if signal_log_path:
            signal_log_file, signal_log_writer = _open_signal_log(signal_log_path)

    ib = IB()
    try:
        ib.connect(
            host,
            port,
            clientId=client_id,
            readonly=not (manual_orders or auto_orders or interactive_orders),
            timeout=timeout,
        )
        ib.reqMarketDataType(MARKET_DATA_TYPES[market_data_type])
        ib.sleep(0.5)
        contract = Stock(symbol, "SMART", "USD")
        ib.qualifyContracts(contract)
        ticker = ib.reqMktData(contract, "", snapshot=False, regulatorySnapshot=False)
        if log_path:
            log_file, log_writer = _open_quote_log(log_path)

        print(
            f"Watching {symbol} {market_data_type} quotes "
            f"every {interval:g}s. Press Ctrl+C to stop."
        )
        if log_path:
            print(f"Logging to {log_path}")
        if signal_log_path:
            print(f"Logging strategy signals to {signal_log_path}")
        if live_plot_path:
            print(f"Refreshing live strategy plot at {live_plot_path}")
        if live_plot_html_path:
            print(f"Refreshing interactive live strategy plot at {live_plot_html_path}")
        if strategy_state_path:
            print(f"Saving live strategy state to {strategy_state_path}")
        if interactive_orders:
            print(
                "Interactive overrides enabled. Type commands like "
                "'buy 1', 'sell 1', 'buy 1 123.45', 'status', or 'quit'."
            )
        start = time.monotonic()
        last_good_quote_monotonic = None
        should_stop = False
        pending_trades = []
        while True:
            ib.sleep(interval)
            row = _quote_from_ticker(ticker, symbol, market_data_type)
            if row.mid is not None:
                last_good_quote_monotonic = time.monotonic()
            quote_age_seconds = (
                None if last_good_quote_monotonic is None
                else time.monotonic() - last_good_quote_monotonic
            )
            _print_quote_row(row)
            if pending_trades:
                stop_for_fill, fill_events = _check_pending_fills(strategy, pending_trades)
                if signal_log_writer and fill_events:
                    for fill_event in fill_events:
                        signal_log_writer.writerow(
                            _fill_event_to_signal_row(fill_event, row, strategy)
                        )
                    signal_log_file.flush()
                if live_plot_path and fill_events:
                    _save_live_strategy_plot(
                        strategy,
                        signal if "signal" in locals() else None,
                        live_plot_path,
                        window_bars=plot_window_bars,
                    )
                if live_plot_html_path and fill_events:
                    _save_live_strategy_html_plot(
                        strategy,
                        signal if "signal" in locals() else None,
                        live_plot_html_path,
                    )
                if strategy_state_path and fill_events:
                    _save_strategy_state(strategy, strategy_state_path)
                if stop_for_fill:
                    should_stop = True
                if (
                    live_plot_path
                    and not fill_events
                    and "signal" in locals()
                    and getattr(strategy, "latest_data", None) is not None
                ):
                    _save_live_strategy_plot(
                        strategy,
                        signal,
                        live_plot_path,
                        window_bars=plot_window_bars,
                    )
                if (
                    live_plot_html_path
                    and not fill_events
                    and "signal" in locals()
                    and getattr(strategy, "latest_data", None) is not None
                ):
                    _save_live_strategy_html_plot(
                        strategy,
                        signal,
                        live_plot_html_path,
                    )
                if (
                    strategy_state_path
                    and not fill_events
                    and getattr(strategy, "latest_data", None) is not None
                ):
                    _save_strategy_state(strategy, strategy_state_path)
            if interactive_orders:
                command = _read_interactive_order_command()
                if command:
                    should_stop, trade = _handle_interactive_order_command(
                        command,
                        ib,
                        contract,
                        symbol,
                        row,
                        strategy,
                        order_quantity,
                        limit_buffer_bps,
                        min_sell_profit=min_sell_profit,
                        min_sell_profit_per_share=min_sell_profit_per_share,
                        profit_target_mode=profit_target_mode,
                        sizing_mode=sizing_mode,
                        cash_reserve=cash_reserve,
                    )
                    if trade is not None:
                        pending_trades.append({
                            "trade": trade,
                            "stop_after_fill": False,
                        })
            if should_stop:
                break
            if dry_run_strategy:
                bar = bar_builder.update(row)
                if bar is not None:
                    signal = strategy.on_bar(
                        bar,
                        quote_age_seconds=quote_age_seconds,
                    )
                    print_strategy_signal(signal)
                    if signal_log_writer:
                        signal_log_writer.writerow(signal_to_dict(signal))
                        signal_log_file.flush()
                    if live_plot_path:
                        _save_live_strategy_plot(
                            strategy,
                            signal,
                            live_plot_path,
                            window_bars=plot_window_bars,
                        )
                    if live_plot_html_path:
                        _save_live_strategy_html_plot(
                            strategy,
                            signal,
                            live_plot_html_path,
                        )
                    if strategy_state_path:
                        _save_strategy_state(strategy, strategy_state_path)
                    if (manual_orders or auto_orders) and not pending_trades:
                        trade = _handle_order_signal(
                            ib,
                            contract,
                            signal,
                            quantity=order_quantity,
                            limit_buffer_bps=limit_buffer_bps,
                            strategy=strategy,
                            min_sell_profit=min_sell_profit,
                            min_sell_profit_per_share=min_sell_profit_per_share,
                            profit_target_mode=profit_target_mode,
                            sizing_mode=sizing_mode,
                            cash_reserve=cash_reserve,
                            auto_orders=auto_orders,
                        )
                        if trade is not None:
                            pending_trades.append({
                                "trade": trade,
                                "stop_after_fill": bool(auto_orders and stop_after_order),
                            })
                    elif auto_orders and pending_trades and signal.signal in ("WOULD BUY", "WOULD SELL"):
                        print("Auto-order signal ignored while an earlier order is still pending fill.")
            if log_writer:
                log_writer.writerow(row.__dict__)
                log_file.flush()
            if duration is not None and time.monotonic() - start >= duration:
                break
    finally:
        if log_file:
            log_file.close()
        if signal_log_file:
            signal_log_file.close()
        if ib.isConnected():
            try:
                ib.cancelMktData(contract)
            except Exception:
                pass
            ib.disconnect()


def fetch_positions(
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    client_id=DEFAULT_CLIENT_ID,
    timeout=8.0,
):
    """Fetch current account positions from TWS without placing orders."""
    try:
        from ib_insync import IB
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: ib_insync.\n"
            "Install it inside your project environment with:\n"
            "    pip install ib_insync\n"
            "or, if using your venv explicitly:\n"
            "    ./venv/bin/python -m pip install ib_insync"
        ) from exc

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, readonly=True, timeout=timeout)
        positions = ib.positions()
        return [
            {
                "account": p.account,
                "symbol": getattr(p.contract, "symbol", ""),
                "sec_type": getattr(p.contract, "secType", ""),
                "exchange": getattr(p.contract, "exchange", ""),
                "currency": getattr(p.contract, "currency", ""),
                "position": float(p.position),
                "avg_cost": float(p.avgCost),
            }
            for p in positions
        ]
    finally:
        if ib.isConnected():
            ib.disconnect()


def print_positions(positions):
    if not positions:
        print("No open positions.")
        return
    print(
        f"{'Account':<14} {'Symbol':<8} {'Type':<6} {'Exchange':<10} "
        f"{'Currency':<8} {'Position':>12} {'Avg Cost':>12}"
    )
    print("-" * 78)
    for p in positions:
        print(
            f"{p['account']:<14} "
            f"{p['symbol']:<8} "
            f"{p['sec_type']:<6} "
            f"{p['exchange']:<10} "
            f"{p['currency']:<8} "
            f"{_format_quantity(p['position']):>12} "
            f"{_format_price(p['avg_cost']):>12}"
        )


def find_position(positions, symbol):
    """Return the first matching stock position for a symbol, or None."""
    symbol = symbol.upper()
    for position in positions:
        if (
            position["symbol"].upper() == symbol
            and position["sec_type"].upper() == "STK"
        ):
            return position
    return None


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Read bid/ask/mid from a local TWS demo/paper session. "
            "Orders are placed only with --manual-orders or --auto-orders."
        ),
        epilog=(
            "Examples:\n"
            "  ./venv/bin/python ibkr_bridge.py --preset apple\n"
            "  ./venv/bin/python ibkr_bridge.py --symbol AAPL --port 7497 --market-data-type delayed\n"
            "  ./venv/bin/python ibkr_bridge.py --preset spy --market-data-type delayed\n"
            "  ./venv/bin/python ibkr_bridge.py --preset spy --watch --duration 30\n"
            "  ./venv/bin/python ibkr_bridge.py --symbol NVDA --market-data-type live --dry-run-strategy\n"
            "  ./venv/bin/python ibkr_bridge.py --symbol NVDA --market-data-type live --dry-run-strategy --use-account-position\n"
            "  ./venv/bin/python ibkr_bridge.py --symbol NVDA --market-data-type live --dry-run-strategy --manual-orders\n"
            "  ./venv/bin/python ibkr_bridge.py --symbol NVDA --market-data-type live --dry-run-strategy --use-account-position --auto-orders --regular-hours-only --max-spread-bps 20 --max-quote-age 10\n"
            "  ./venv/bin/python ibkr_bridge.py --symbol NVDA --market-data-type live --dry-run-strategy --use-account-position --initial-buy-if-cash --auto-orders --regular-hours-only --max-spread-bps 20 --max-quote-age 10\n"
            "  ./venv/bin/python ibkr_bridge.py --symbol NVDA --market-data-type live --place-order BUY --order-quantity 1 --confirm-order\n"
            "  ./venv/bin/python ibkr_bridge.py --positions"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--symbol", default=None, help="Stock symbol, e.g. AAPL.")
    parser.add_argument(
        "--preset",
        choices=tuple(SYMBOL_PRESETS),
        default="apple",
        help="Friendly symbol preset. Ignored if --symbol is provided.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="TWS host, usually 127.0.0.1.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TWS API port, usually 7497 for paper/demo.")
    parser.add_argument("--client-id", type=int, default=DEFAULT_CLIENT_ID, help="Unique API client id.")
    parser.add_argument("--timeout", type=float, default=8.0, help="Connection timeout in seconds.")
    parser.add_argument("--positions", action="store_true", help="Print current account positions and exit.")
    parser.add_argument(
        "--place-order",
        choices=("BUY", "SELL"),
        default=None,
        help="Place one explicit limit order. Requires --confirm-order to actually submit.",
    )
    parser.add_argument(
        "--limit-price",
        type=float,
        default=None,
        help="Explicit limit price for --place-order. If omitted, uses ask for BUY or bid for SELL plus --limit-buffer-bps.",
    )
    parser.add_argument(
        "--confirm-order",
        action="store_true",
        help="Required to submit --place-order. Without this, the command only previews.",
    )
    parser.add_argument("--watch", action="store_true", help="Continuously print quotes until stopped.")
    parser.add_argument("--interval", type=float, default=5.0, help="Watch-mode print interval in seconds.")
    parser.add_argument("--duration", type=float, default=None, help="Optional watch-mode duration in seconds.")
    parser.add_argument("--log-csv", default=None, help="Optional CSV path for watch-mode quote logging.")
    parser.add_argument(
        "--wait-for-open",
        action="store_true",
        help="Sleep until regular US stock-market hours before connecting/running.",
    )
    parser.add_argument(
        "--wait-check-seconds",
        type=float,
        default=60.0,
        help="How often --wait-for-open checks the clock.",
    )
    parser.add_argument(
        "--dry-run-strategy",
        action="store_true",
        help="Build mid-price bars and print HOLD/WOULD BUY/WOULD SELL. No orders.",
    )
    parser.add_argument(
        "--use-account-position",
        action="store_true",
        help="For dry-run strategy, initialize CASH/LONG from the current TWS account position.",
    )
    parser.add_argument(
        "--initial-buy-if-cash",
        action="store_true",
        help="If account-aware mode starts with zero shares, emit one GUI-style initial WOULD BUY.",
    )
    parser.add_argument("--bar-seconds", type=float, default=60.0, help="Dry-run strategy bar size in seconds.")
    parser.add_argument(
        "--regular-hours-only",
        action="store_true",
        help="Block dry-run trade signals outside regular US stock market hours.",
    )
    parser.add_argument(
        "--max-spread-bps",
        type=float,
        default=None,
        help="Block dry-run trade signals when average bar spread exceeds this many basis points.",
    )
    parser.add_argument(
        "--max-quote-age",
        type=float,
        default=None,
        help="Block dry-run trade signals when the latest usable quote is older than this many seconds.",
    )
    parser.add_argument(
        "--signal-log",
        default=None,
        help="Optional CSV path for dry-run strategy signals.",
    )
    parser.add_argument(
        "--live-plot",
        default=None,
        help="Optional PNG path refreshed after each completed strategy bar.",
    )
    parser.add_argument(
        "--live-plot-html",
        default=None,
        help="Optional self-contained HTML path with an interactive all-history live plot.",
    )
    parser.add_argument(
        "--strategy-state",
        default=None,
        help="Optional JSON path for saving restartable live strategy state.",
    )
    parser.add_argument(
        "--resume-strategy-state",
        action="store_true",
        help="Load --strategy-state at startup and continue the prior live cycle.",
    )
    parser.add_argument(
        "--plot-window-bars",
        type=int,
        default=120,
        help="Number of recent completed bars shown in --live-plot.",
    )
    parser.add_argument(
        "--manual-orders",
        action="store_true",
        help="Prompt before submitting a limit order when dry-run emits WOULD BUY/WOULD SELL.",
    )
    parser.add_argument(
        "--auto-orders",
        action="store_true",
        help="Submit a limit order automatically when dry-run emits WOULD BUY/WOULD SELL.",
    )
    parser.add_argument(
        "--interactive-orders",
        action="store_true",
        help="Allow terminal overrides while running: buy [qty] [limit], sell [qty] [limit].",
    )
    parser.add_argument(
        "--keep-running-after-order",
        action="store_true",
        help="Keep watching after an auto order is filled. Default stops after one auto fill.",
    )
    parser.add_argument("--order-quantity", type=int, default=4, help="Order quantity.")
    parser.add_argument(
        "--capital-budget",
        type=float,
        default=1000.0,
        help="Dollar budget used for live discrete-share wealth accounting.",
    )
    parser.add_argument(
        "--sizing-mode",
        choices=("fixed", "cash_reserve"),
        default="fixed",
        help=(
            "Order sizing rule. fixed uses --order-quantity. cash_reserve buys "
            "as many whole shares as possible while leaving --cash-reserve dollars, "
            "and sells the full current strategy position."
        ),
    )
    parser.add_argument(
        "--cash-reserve",
        type=float,
        default=100.0,
        help="Cash to leave unused when --sizing-mode cash_reserve is active.",
    )
    parser.add_argument(
        "--limit-buffer-bps",
        type=float,
        default=5.0,
        help="Order buffer in basis points: buy above ask, sell below bid.",
    )
    parser.add_argument("--k", type=float, default=0.8, help="Strategy funnel threshold.")
    parser.add_argument("--delta", type=int, default=3, help="Strategy momentum window in bars.")
    parser.add_argument("--entry-delta", type=int, default=2, help="Cash re-entry momentum window in bars.")
    parser.add_argument(
        "--cash-reentry-cooldown-bars",
        type=int,
        default=10,
        help="Minimum completed bars after a filled sell before cash re-entry can buy again.",
    )
    parser.add_argument("--cost", type=float, default=0.0, help="Per-side proportional transaction cost.")
    parser.add_argument(
        "--fixed-buy-fee",
        type=float,
        default=1.0,
        help="Fixed dollar buy commission folded into the strategy cost h.",
    )
    parser.add_argument(
        "--fixed-sell-fee",
        type=float,
        default=1.03,
        help="Fixed dollar sell commission folded into the strategy cost h.",
    )
    parser.add_argument(
        "--min-sell-profit",
        type=float,
        default=0.01,
        help="Minimum net dollar profit required before a live sell limit can fill.",
    )
    parser.add_argument(
        "--min-sell-profit-per-share",
        type=float,
        default=1.0,
        help="Minimum net profit per share required before a live sell limit can fill.",
    )
    parser.add_argument(
        "--profit-target-mode",
        choices=("fixed", "per_share", "max"),
        default="per_share",
        help=(
            "How to interpret sell profit target: fixed uses --min-sell-profit "
            "as total dollars; per_share uses --min-sell-profit-per-share times "
            "quantity; max requires the larger of both."
        ),
    )
    parser.add_argument("--lookback-L", type=int, default=20, help="Bounded funnel lookback.")
    parser.add_argument("--trail-a", type=float, default=0.01, help="Trailing-profit log drawdown threshold.")
    parser.add_argument("--z-trend", type=float, default=0.35, help="Trend re-entry Z threshold.")
    parser.add_argument("--drift-q", type=float, default=1e-7, help="Kalman drift process variance.")
    parser.add_argument(
        "--market-data-type",
        choices=tuple(MARKET_DATA_TYPES),
        default=DEFAULT_MARKET_DATA_TYPE,
        help=(
            "Requested IBKR market data mode. Default is delayed "
            "(IBKR API marketDataType=3), not live."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    symbol = args.symbol or SYMBOL_PRESETS[args.preset]
    if args.wait_for_open:
        try:
            wait_for_regular_market_open(check_seconds=args.wait_check_seconds)
        except KeyboardInterrupt:
            print("\nStopped while waiting for market open.")
            return 0
    if args.place_order:
        try:
            place_direct_limit_order(
                symbol=symbol,
                action=args.place_order,
                quantity=args.order_quantity,
                limit_price=args.limit_price,
                limit_buffer_bps=args.limit_buffer_bps,
                host=args.host,
                port=args.port,
                client_id=args.client_id,
                timeout=args.timeout,
                market_data_type=args.market_data_type,
                confirm_order=args.confirm_order,
            )
        except Exception as exc:
            print("IBKR direct order failed.")
            print(str(exc))
            return 1
        return 0
    if args.positions:
        try:
            positions = fetch_positions(
                host=args.host,
                port=args.port,
                client_id=args.client_id,
                timeout=args.timeout,
            )
        except Exception as exc:
            print("IBKR positions check failed.")
            print(str(exc))
            return 1
        print_positions(positions)
        print("")
        print("No orders were placed.")
        return 0

    if args.watch or args.dry_run_strategy:
        if args.resume_strategy_state and not args.strategy_state:
            print("--resume-strategy-state requires --strategy-state PATH.")
            return 1
        if args.manual_orders and not args.dry_run_strategy:
            print("--manual-orders requires --dry-run-strategy.")
            return 1
        if args.auto_orders and not args.dry_run_strategy:
            print("--auto-orders requires --dry-run-strategy.")
            return 1
        if args.interactive_orders and not args.dry_run_strategy:
            print("--interactive-orders requires --dry-run-strategy.")
            return 1
        if args.manual_orders and args.auto_orders:
            print("Choose only one: --manual-orders or --auto-orders.")
            return 1
        if args.auto_orders:
            missing = []
            if not args.use_account_position:
                missing.append("--use-account-position")
            if not args.regular_hours_only:
                missing.append("--regular-hours-only")
            if args.max_spread_bps is None:
                missing.append("--max-spread-bps")
            if args.max_quote_age is None:
                missing.append("--max-quote-age")
            if missing:
                print("--auto-orders requires these safety options:")
                print("  " + " ".join(missing))
                return 1
        try:
            initial_position = None
            initial_avg_cost = None
            if args.dry_run_strategy and args.use_account_position:
                positions = fetch_positions(
                    host=args.host,
                    port=args.port,
                    client_id=args.client_id + 1000,
                    timeout=args.timeout,
                )
                matched_position = find_position(positions, symbol)
                if matched_position:
                    initial_position = matched_position["position"]
                    initial_avg_cost = matched_position["avg_cost"]
                else:
                    initial_position = 0.0
                    initial_avg_cost = None
                print(
                    "Account-aware dry run: "
                    f"{symbol} position={_format_quantity(initial_position)}, "
                    f"avg_cost={_format_price(initial_avg_cost)}"
                )
            print(
                "Requesting "
                f"{args.market_data_type} market data "
                f"(IBKR marketDataType={MARKET_DATA_TYPES[args.market_data_type]})."
            )
            watch_stock_quote(
                symbol=symbol,
                host=args.host,
                port=args.port,
                client_id=args.client_id,
                timeout=args.timeout,
                market_data_type=args.market_data_type,
                interval=args.interval,
                duration=args.duration,
                log_path=args.log_csv,
                dry_run_strategy=args.dry_run_strategy,
                bar_seconds=args.bar_seconds,
                strategy_kwargs={
                    "k": args.k,
                    "delta": args.delta,
                    "entry_delta": args.entry_delta,
                    "cost": args.cost,
                    "drift_process_var": args.drift_q,
                    "max_funnel_lookback": args.lookback_L,
                    "trailing_stop": args.trail_a,
                    "trend_entry_z": args.z_trend,
                    "initial_position": initial_position,
                    "initial_avg_cost": initial_avg_cost,
                    "regular_hours_only": args.regular_hours_only,
                    "max_spread_bps": args.max_spread_bps,
                    "max_quote_age": args.max_quote_age,
                    "fixed_buy_fee": args.fixed_buy_fee,
                    "fixed_sell_fee": args.fixed_sell_fee,
                    "min_sell_profit": args.min_sell_profit,
                    "min_sell_profit_per_share": args.min_sell_profit_per_share,
                    "profit_target_mode": args.profit_target_mode,
                    "order_quantity": args.order_quantity,
                    "initial_buy_if_cash": args.initial_buy_if_cash,
                    "capital_budget": args.capital_budget,
                    "sizing_mode": args.sizing_mode,
                    "cash_reserve": args.cash_reserve,
                    "cash_reentry_cooldown_bars": args.cash_reentry_cooldown_bars,
                },
                signal_log_path=args.signal_log,
                live_plot_path=args.live_plot,
                live_plot_html_path=args.live_plot_html,
                strategy_state_path=args.strategy_state,
                resume_strategy_state=args.resume_strategy_state,
                plot_window_bars=args.plot_window_bars,
                manual_orders=args.manual_orders,
                auto_orders=args.auto_orders,
                interactive_orders=args.interactive_orders,
                order_quantity=args.order_quantity,
                limit_buffer_bps=args.limit_buffer_bps,
                min_sell_profit=args.min_sell_profit,
                min_sell_profit_per_share=args.min_sell_profit_per_share,
                profit_target_mode=args.profit_target_mode,
                sizing_mode=args.sizing_mode,
                cash_reserve=args.cash_reserve,
                stop_after_order=not args.keep_running_after_order,
            )
        except KeyboardInterrupt:
            print("\nStopped quote watch.")
            return 0
        except Exception as exc:
            print("IBKR bridge watch failed.")
            print(str(exc))
            return 1
        if args.manual_orders or args.auto_orders:
            print("Watch ended.")
        else:
            print("No orders were placed.")
        return 0

    try:
        print(
            "Requesting "
            f"{args.market_data_type} market data "
            f"(IBKR marketDataType={MARKET_DATA_TYPES[args.market_data_type]})."
        )
        quote = fetch_stock_quote(
            symbol=symbol,
            host=args.host,
            port=args.port,
            client_id=args.client_id,
            timeout=args.timeout,
            market_data_type=args.market_data_type,
        )
    except Exception as exc:
        print("IBKR bridge check failed.")
        print(str(exc))
        return 1

    quote_available = any(
        value is not None for value in (quote.bid, quote.ask, quote.last, quote.mid)
    )
    if quote_available:
        print("IBKR bridge check succeeded.")
    else:
        print("IBKR bridge connected, but no market quote was returned.")
    print(f"Symbol: {quote.symbol}")
    print(f"Data:   {quote.market_data_type}")
    print(f"Bid:    {_format_price(quote.bid)}")
    print(f"Ask:    {_format_price(quote.ask)}")
    print(f"Last:   {_format_price(quote.last)}")
    print(f"Mid:    {_format_price(quote.mid)}")
    print("")
    print("No orders were placed.")
    if not quote_available:
        print("")
        print("Most likely cause:")
        print("    TWS API market data permission is not available for this symbol/feed.")
        print("Things to try:")
        print("    1. Try the Apple preset: ./venv/bin/python ibkr_bridge.py --preset apple")
        print("    2. Retry during regular US market hours.")
        print("    3. In TWS, open Market Data Connections and check the subscription message.")
        print("    4. Try another market-data type, e.g. delayed-frozen.")
        print("    5. If delayed still returns 10089, IBKR is blocking even delayed API data for this symbol/feed.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
