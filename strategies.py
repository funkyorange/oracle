"""
Signal generators.  Each function takes an OHLCV DataFrame and returns
a Series of {1 = long entry, -1 = short entry, 0 = no signal}.

Add new strategies here; they are auto-discovered by the backtester.
"""

import numpy as np
import pandas as pd

REGISTRY: dict[str, callable] = {}


def strategy(name: str):
    def decorator(fn):
        REGISTRY[name] = fn
        return fn
    return decorator


# ── helpers ────────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hi, lo, cl = df["high"], df["low"], df["close"]
    tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# ── strategies ─────────────────────────────────────────────────────────────────

@strategy("ema_cross")
def ema_cross(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> pd.Series:
    """Classic EMA crossover — long when fast crosses above slow."""
    fast_e = _ema(df["close"], fast)
    slow_e = _ema(df["close"], slow)
    cross_up   = (fast_e > slow_e) & (fast_e.shift() <= slow_e.shift())
    cross_down = (fast_e < slow_e) & (fast_e.shift() >= slow_e.shift())
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   = 1
    sig[cross_down] = -1
    return sig


@strategy("ema_trend_rsi")
def ema_trend_rsi(
    df: pd.DataFrame,
    trend_ema: int = 50,
    rsi_period: int = 14,
    rsi_oversold: int = 40,
    rsi_overbought: int = 60,
) -> pd.Series:
    """Trade in direction of EMA trend, enter on RSI pullback."""
    trend = _ema(df["close"], trend_ema)
    rsi   = _rsi(df["close"], rsi_period)
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > trend) & (rsi < rsi_oversold)]  = 1
    sig[(df["close"] < trend) & (rsi > rsi_overbought)] = -1
    return sig


@strategy("rsi_mean_revert")
def rsi_mean_revert(
    df: pd.DataFrame,
    period: int = 14,
    oversold: int = 30,
    overbought: int = 70,
) -> pd.Series:
    """Mean-reversion: buy oversold, sell overbought."""
    rsi = _rsi(df["close"], period)
    sig = pd.Series(0, index=df.index)
    sig[rsi < oversold]  = 1
    sig[rsi > overbought] = -1
    return sig


@strategy("breakout")
def breakout(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Donchian breakout — long on N-bar high, short on N-bar low."""
    hi = df["high"].rolling(window).max().shift(1)
    lo = df["low"].rolling(window).min().shift(1)
    sig = pd.Series(0, index=df.index)
    sig[df["close"] > hi] = 1
    sig[df["close"] < lo] = -1
    return sig


@strategy("macd")
def macd_signal(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    """MACD histogram cross."""
    macd_line = _ema(df["close"], fast) - _ema(df["close"], slow)
    sig_line  = _ema(macd_line, signal)
    hist = macd_line - sig_line
    cross_up   = (hist > 0) & (hist.shift() <= 0)
    cross_down = (hist < 0) & (hist.shift() >= 0)
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   = 1
    sig[cross_down] = -1
    return sig


def list_strategies() -> list[str]:
    return sorted(REGISTRY.keys())


def generate_signals(name: str, df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    fn = REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"Unknown strategy: {name}")
    return fn(df, **(params or {}))
