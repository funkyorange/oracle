"""
Regime classification based on BTC trend and volatility.

Regime labels (matches TradeManager account regime field):
  BULLISH  — uptrend, low/normal vol
  NEUTRAL  — sideways, normal vol
  CAUTION  — weakening trend or elevated vol
  STRESS   — downtrend or high vol / drawdown
"""

import numpy as np
import pandas as pd


def classify(btc: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Classify each bar into a regime.
    btc: OHLCV DataFrame with DatetimeIndex (UTC).
    Returns Series of regime strings, same index as btc.
    """
    close = btc["close"]

    ema_fast = close.ewm(span=window, adjust=False).mean()
    ema_slow = close.ewm(span=window * 3, adjust=False).mean()

    # rolling 30-day volatility (annualised)
    returns = close.pct_change()
    vol = returns.rolling(30).std() * np.sqrt(365)

    vol_median = vol.median()
    vol_high = vol_median * 1.5

    trend = ema_fast - ema_slow  # positive = uptrend

    # rolling 60-day drawdown from peak
    rolling_max = close.rolling(60, min_periods=1).max()
    drawdown = (close - rolling_max) / rolling_max  # ≤ 0

    regime = pd.Series("NEUTRAL", index=btc.index)

    regime[
        (trend > 0) & (vol <= vol_median)
    ] = "BULLISH"

    regime[
        (trend < 0) & (vol > vol_high)
    ] = "STRESS"

    regime[
        (trend < 0) | (vol > vol_median) | (drawdown < -0.15)
    ] = "CAUTION"

    # BULLISH overrides (apply last so it wins when both conditions met)
    regime[
        (trend > 0) & (vol <= vol_median) & (drawdown > -0.10)
    ] = "BULLISH"

    regime[
        (drawdown < -0.25) | (vol > vol_high * 1.3)
    ] = "STRESS"

    return regime


def current_regime(btc: pd.DataFrame) -> str:
    """Return the latest regime label."""
    series = classify(btc)
    return series.iloc[-1]


def regime_history(btc: pd.DataFrame) -> list[dict]:
    """Return regime series as a list of {date, regime} records."""
    series = classify(btc)
    return [
        {"date": ts.date().isoformat(), "regime": r}
        for ts, r in series.items()
    ]
