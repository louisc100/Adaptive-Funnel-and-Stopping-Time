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
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass


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


def _first_price(*values):
    """Return the first usable price from normal or delayed ticker fields."""
    for value in values:
        price = _safe_float(value)
        if price is not None:
            return price
    return None


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

        bid = _first_price(ticker.bid, getattr(ticker, "delayedBid", None))
        ask = _first_price(ticker.ask, getattr(ticker, "delayedAsk", None))
        last = _first_price(ticker.last, getattr(ticker, "delayedLast", None), ticker.close)
        mid = (bid + ask) / 2.0 if bid is not None and ask is not None else None

        return QuoteSnapshot(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=last,
            mid=mid,
            market_data_type=market_data_type,
        )
    finally:
        if ib.isConnected():
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
            "  ./venv/bin/python ibkr_bridge.py --preset spy --market-data-type delayed"
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
