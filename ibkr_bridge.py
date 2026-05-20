"""
Read-only IBKR TWS bridge for the trading project.

Version 1 goal:
    Connect to a local TWS / IB Gateway session and print bid, ask, and mid.

This script does not place orders. It is intentionally conservative so we can
verify the local TWS API connection before wiring in strategy execution.

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
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


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


def _format_bps(value):
    return "n/a" if value is None else f"{value:.2f}"


def _first_price(*values):
    """Return the first usable price from normal or delayed ticker fields."""
    for value in values:
        price = _safe_float(value)
        if price is not None:
            return price
    return None


def _quote_from_ticker(ticker, symbol, market_data_type):
    bid = _first_price(ticker.bid, getattr(ticker, "delayedBid", None))
    ask = _first_price(ticker.ask, getattr(ticker, "delayedAsk", None))
    last = _first_price(ticker.last, getattr(ticker, "delayedLast", None), ticker.close)
    mid = (bid + ask) / 2.0 if bid is not None and ask is not None else None
    spread = ask - bid if bid is not None and ask is not None else None
    spread_bps = 10000.0 * spread / mid if spread is not None and mid else None
    return QuoteRow(
        timestamp=datetime.now().isoformat(timespec="seconds"),
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
):
    """Continuously print read-only Level 1 quotes from TWS."""
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
    bar_builder = None
    strategy = None
    if dry_run_strategy:
        from live_strategy import (
            LiveDryRunStrategy,
            MidBarBuilder,
            print_strategy_signal,
        )
        bar_builder = MidBarBuilder(bar_seconds=bar_seconds)
        strategy = LiveDryRunStrategy(symbol=symbol, **(strategy_kwargs or {}))

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, readonly=True, timeout=timeout)
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
        start = time.monotonic()
        while True:
            ib.sleep(interval)
            row = _quote_from_ticker(ticker, symbol, market_data_type)
            _print_quote_row(row)
            if dry_run_strategy:
                bar = bar_builder.update(row)
                if bar is not None:
                    signal = strategy.on_bar(bar)
                    print_strategy_signal(signal)
            if log_writer:
                log_writer.writerow(row.__dict__)
                log_file.flush()
            if duration is not None and time.monotonic() - start >= duration:
                break
    finally:
        if log_file:
            log_file.close()
        if ib.isConnected():
            try:
                ib.cancelMktData(contract)
            except Exception:
                pass
            ib.disconnect()


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Read bid/ask/mid from a local TWS demo/paper session. "
            "This script is read-only and never places orders."
        ),
        epilog=(
            "Examples:\n"
            "  ./venv/bin/python ibkr_bridge.py --preset apple\n"
            "  ./venv/bin/python ibkr_bridge.py --symbol AAPL --port 7497 --market-data-type delayed\n"
            "  ./venv/bin/python ibkr_bridge.py --preset spy --market-data-type delayed\n"
            "  ./venv/bin/python ibkr_bridge.py --preset spy --watch --duration 30\n"
            "  ./venv/bin/python ibkr_bridge.py --symbol NVDA --market-data-type live --dry-run-strategy"
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
    parser.add_argument("--watch", action="store_true", help="Continuously print quotes until stopped.")
    parser.add_argument("--interval", type=float, default=5.0, help="Watch-mode print interval in seconds.")
    parser.add_argument("--duration", type=float, default=None, help="Optional watch-mode duration in seconds.")
    parser.add_argument("--log-csv", default=None, help="Optional CSV path for watch-mode quote logging.")
    parser.add_argument(
        "--dry-run-strategy",
        action="store_true",
        help="Build mid-price bars and print HOLD/WOULD BUY/WOULD SELL. No orders.",
    )
    parser.add_argument("--bar-seconds", type=float, default=60.0, help="Dry-run strategy bar size in seconds.")
    parser.add_argument("--k", type=float, default=1.2, help="Strategy funnel threshold.")
    parser.add_argument("--delta", type=int, default=8, help="Strategy momentum window in bars.")
    parser.add_argument("--cost", type=float, default=0.0, help="Per-side proportional transaction cost.")
    parser.add_argument("--lookback-L", type=int, default=60, help="Bounded funnel lookback.")
    parser.add_argument("--trail-a", type=float, default=0.04, help="Trailing-profit log drawdown threshold.")
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
    if args.watch or args.dry_run_strategy:
        try:
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
                },
            )
        except KeyboardInterrupt:
            print("\nStopped quote watch.")
            return 0
        except Exception as exc:
            print("IBKR bridge watch failed.")
            print(str(exc))
            return 1
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
