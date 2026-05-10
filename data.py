"""
Fetch and cache OHLCV data from Bybit via ccxt.
Cache lives in ./cache/ as Parquet files so re-runs are instant.
"""

import os
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt
import pandas as pd

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

_exchange = None


def _ex() -> ccxt.bybit:
    global _exchange
    if _exchange is None:
        _exchange = ccxt.bybit({"enableRateLimit": True})
    return _exchange


def _cache_path(symbol: str, timeframe: str) -> Path:
    safe = symbol.replace("/", "-")
    return CACHE_DIR / f"{safe}_{timeframe}.parquet"


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    since_days: int = 730,
) -> pd.DataFrame:
    """Return OHLCV DataFrame. Refreshes cache if last candle is stale."""
    path = _cache_path(symbol, timeframe)
    since_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=since_days)).timestamp() * 1000
    )

    # load cache if fresh enough
    if path.exists():
        df = pd.read_parquet(path)
        if not df.empty:
            last_ts = df.index[-1]
            stale_after = _stale_threshold(timeframe)
            if last_ts > datetime.now(timezone.utc) - stale_after:
                return df

    log.info(f"Fetching {symbol} {timeframe} from Bybit…")
    rows = []
    fetch_since = since_ms
    while True:
        batch = _ex().fetch_ohlcv(symbol, timeframe, since=fetch_since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        fetch_since = batch[-1][0] + 1

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.to_parquet(path)
    return df


def _stale_threshold(timeframe: str) -> timedelta:
    mapping = {
        "1m": timedelta(minutes=2),
        "5m": timedelta(minutes=10),
        "15m": timedelta(minutes=30),
        "1h": timedelta(hours=2),
        "4h": timedelta(hours=8),
        "1d": timedelta(hours=25),
    }
    return mapping.get(timeframe, timedelta(hours=25))


def available_symbols() -> list[str]:
    """Return top perpetual futures symbols by volume."""
    try:
        markets = _ex().load_markets()
        perps = [
            m for m in markets.values()
            if m.get("type") == "swap" and m.get("quote") == "USDT" and m.get("active")
        ]
        return sorted(m["symbol"] for m in perps[:200])
    except Exception as e:
        log.warning(f"Could not load markets: {e}")
        return ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
