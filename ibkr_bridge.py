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


def _submit_limit_order(ib, contract, signal, quantity, limit_buffer_bps):
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

    from ib_insync import LimitOrder

    order = LimitOrder(action, quantity, round(limit_price, 2), tif="DAY")
    trade = ib.placeOrder(contract, order)
    ib.sleep(1.0)
    print(
        "Submitted order: "
        f"{action} {quantity} {signal.symbol} LMT {_format_price(order.lmtPrice)} "
        f"status={trade.orderStatus.status}"
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
    quantity = int(quantity)
    if quantity <= 0:
        print("Order skipped: quantity must be positive.")
        return None

    print("")
    print("Order signal")
    print(f"Signal: {signal.signal}")
    print(f"Symbol: {signal.symbol}")
    print(f"Action: {action}")
    print(f"Quantity: {quantity}")
    print(f"Reference bid/ask: {_format_price(signal.close_bid)} / {_format_price(signal.close_ask)}")
    print(f"Suggested LMT price: {_format_price(limit_price)}")
    if auto_orders:
        print("Auto-order mode is ON: submitting without y/n confirmation.")
        return _submit_limit_order(ib, contract, signal, quantity, limit_buffer_bps)

    answer = input("Submit this limit order to TWS Paper? Type y to submit: ").strip().lower()
    if answer == "y":
        return _submit_limit_order(ib, contract, signal, quantity, limit_buffer_bps)
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

    quantity = default_quantity
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

    print("")
    print("Interactive override")
    print(f"Command: {command}")
    print(f"Reference bid/ask: {_format_price(row.bid)} / {_format_price(row.ask)}")
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

        buy_x = [
            strategy.strategy_x_to_display_x(x)
            for x in data["buy_times"]
        ]
        # The research engine starts from an internal reference buy at t=1.
        # In account-aware cash mode that is not a live buy signal, so hide it.
        if getattr(strategy, "initial_position", None) is not None:
            buy_x = [x for x in buy_x if x != strategy.strategy_x_to_display_x(1)]
        buy_x = [x for x in buy_x if start <= x < t_end]
        sell_x = [
            strategy.strategy_x_to_display_x(x)
            for x in data["sell_times"]
            if start <= strategy.strategy_x_to_display_x(x) < t_end
        ]
        if buy_x:
            ax_p.scatter(buy_x, [display_prices[x] for x in buy_x], marker="^", s=55, color="#22c55e", label="Strategy buy")
        if sell_x:
            ax_p.scatter(sell_x, [display_prices[x] for x in sell_x], marker="v", s=55, color="#ef4444", label="Strategy sell")

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
    plot_window_bars=120,
    manual_orders=False,
    auto_orders=False,
    interactive_orders=False,
    order_quantity=4,
    limit_buffer_bps=5.0,
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
                    if (manual_orders or auto_orders) and not pending_trades:
                        trade = _handle_order_signal(
                            ib,
                            contract,
                            signal,
                            quantity=order_quantity,
                            limit_buffer_bps=limit_buffer_bps,
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
        "--limit-buffer-bps",
        type=float,
        default=5.0,
        help="Order buffer in basis points: buy above ask, sell below bid.",
    )
    parser.add_argument("--k", type=float, default=0.8, help="Strategy funnel threshold.")
    parser.add_argument("--delta", type=int, default=3, help="Strategy momentum window in bars.")
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
    parser.add_argument("--lookback-L", type=int, default=20, help="Bounded funnel lookback.")
    parser.add_argument("--trail-a", type=float, default=0.02, help="Trailing-profit log drawdown threshold.")
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
                    "order_quantity": args.order_quantity,
                    "initial_buy_if_cash": args.initial_buy_if_cash,
                },
                signal_log_path=args.signal_log,
                live_plot_path=args.live_plot,
                plot_window_bars=args.plot_window_bars,
                manual_orders=args.manual_orders,
                auto_orders=args.auto_orders,
                interactive_orders=args.interactive_orders,
                order_quantity=args.order_quantity,
                limit_buffer_bps=args.limit_buffer_bps,
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
