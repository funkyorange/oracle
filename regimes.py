"""
Regime classifier — multi-indicator composite scoring.

Four signals weighted into a 0-100 score:
  Trend     40%  EMA 20/50/200 alignment, price position
  Volatility 30%  30-day realised vol vs 1-year percentile rank
  Momentum  20%  rolling 7d / 30d / 90d returns
  Drawdown  10%  current drawdown from 60-day peak

Score bands:  ≥70 BULLISH | 50-69 NEUTRAL | 35-49 CAUTION | <35 STRESS

Why these indicators:
  - EMA 200 is the most reliable long-term crypto trend filter (golden/death cross)
  - Realised vol percentile rank is more stable than absolute vol (adapts to crypto's
    structurally higher vol)
  - Multi-period momentum catches regime transitions faster than trend alone
  - Drawdown component adds a capital-preservation floor that triggers CAUTION before
    the full trend signal would
"""

import numpy as np
import pandas as pd


# ── per-indicator scoring (each returns 0-100) ────────────────────────────────

def _trend_score(close: pd.Series) -> pd.Series:
    """
    EMA alignment score.
    Components:
      - price above EMA200: strong baseline (+35)
      - EMA20 > EMA50: short-term momentum (+25)
      - EMA50 > EMA200: medium-term trend (+25)
      - EMA20 above EMA200 by >2%: conviction (+15)
    """
    e20  = close.ewm(span=20,  adjust=False).mean()
    e50  = close.ewm(span=50,  adjust=False).mean()
    e200 = close.ewm(span=200, adjust=False).mean()

    score = pd.Series(0.0, index=close.index)
    score += (close > e200).astype(float) * 35
    score += (e20 > e50).astype(float) * 25
    score += (e50 > e200).astype(float) * 25
    score += ((close - e200) / e200.replace(0, np.nan) > 0.02).astype(float) * 15
    return score.clip(0, 100)


def _vol_score(close: pd.Series, window: int = 30, rank_window: int = 252) -> pd.Series:
    """
    Inverse volatility percentile rank.
    Low vol → high score (calm markets favour systematic strategies).
    Uses percentile rank to normalise across different market eras.
    """
    rets = close.pct_change()
    rv   = rets.rolling(window).std() * np.sqrt(365)
    rank = rv.rolling(rank_window, min_periods=60).rank(pct=True)
    # invert: low-vol percentile → high score
    return ((1 - rank) * 100).clip(0, 100)


def _momentum_score(close: pd.Series) -> pd.Series:
    """
    Weighted blend of 7d, 30d, 90d returns normalised to 0-100.
    Uses tanh to bound outliers; calibrated so +20% monthly ≈ 90 score.
    """
    r7  = close.pct_change(7)
    r30 = close.pct_change(30)
    r90 = close.pct_change(90)

    def norm(r, scale):
        return (np.tanh(r / scale) + 1) / 2 * 100

    return (norm(r7, 0.10) * 0.25 + norm(r30, 0.20) * 0.45 + norm(r90, 0.35) * 0.30).clip(0, 100)


def _drawdown_score(close: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling drawdown from N-day peak.
    0% drawdown → 100 score | -25%+ → 0 score.
    """
    rolling_max = close.rolling(window, min_periods=1).max()
    dd = (close - rolling_max) / rolling_max.replace(0, np.nan)
    return ((1 + dd.clip(-0.25, 0) / 0.25) * 100).clip(0, 100)


# ── weights ───────────────────────────────────────────────────────────────────

WEIGHTS = {"trend": 0.40, "vol": 0.30, "momentum": 0.20, "drawdown": 0.10}


def _composite_score(close: pd.Series) -> pd.Series:
    t = _trend_score(close)
    v = _vol_score(close)
    m = _momentum_score(close)
    d = _drawdown_score(close)
    return (
        t * WEIGHTS["trend"] +
        v * WEIGHTS["vol"]   +
        m * WEIGHTS["momentum"] +
        d * WEIGHTS["drawdown"]
    ).clip(0, 100)


def _score_to_label(score: float) -> str:
    if score >= 70:
        return "BULLISH"
    if score >= 50:
        return "NEUTRAL"
    if score >= 35:
        return "CAUTION"
    return "STRESS"


# ── public API ────────────────────────────────────────────────────────────────

def classify(btc: pd.DataFrame, _window: int = 20) -> pd.Series:
    """
    Classify each bar into a regime label.
    _window is kept for backward-compat but not used (composite approach instead).
    """
    score = _composite_score(btc["close"])
    return score.map(_score_to_label)


def classify_with_scores(btc: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with per-indicator scores and the composite label.
    Useful for the /regimes/breakdown endpoint.
    """
    close = btc["close"]
    t = _trend_score(close).rename("trend")
    v = _vol_score(close).rename("vol")
    m = _momentum_score(close).rename("momentum")
    d = _drawdown_score(close).rename("drawdown")
    composite = _composite_score(close).rename("composite")
    label = composite.map(_score_to_label).rename("regime")
    return pd.concat([t, v, m, d, composite, label], axis=1).dropna(subset=["composite"])


def current_regime(btc: pd.DataFrame) -> str:
    return classify(btc).iloc[-1]


def current_regime_detail(btc: pd.DataFrame) -> dict:
    """Return the latest composite score with per-component breakdown."""
    close = btc["close"]
    t = float(_trend_score(close).iloc[-1])
    v = float(_vol_score(close).iloc[-1])
    m = float(_momentum_score(close).iloc[-1])
    d = float(_drawdown_score(close).iloc[-1])
    composite = t * WEIGHTS["trend"] + v * WEIGHTS["vol"] + m * WEIGHTS["momentum"] + d * WEIGHTS["drawdown"]

    return {
        "label":     _score_to_label(composite),
        "score":     round(composite, 1),
        "components": {
            "trend":     {"score": round(t, 1), "weight": WEIGHTS["trend"],
                          "desc": "EMA 20/50/200 alignment"},
            "volatility":{"score": round(v, 1), "weight": WEIGHTS["vol"],
                          "desc": "Realised vol percentile rank (inverted)"},
            "momentum":  {"score": round(m, 1), "weight": WEIGHTS["momentum"],
                          "desc": "7d/30d/90d return blend"},
            "drawdown":  {"score": round(d, 1), "weight": WEIGHTS["drawdown"],
                          "desc": "60-day rolling drawdown"},
        },
        "thresholds": {"BULLISH": "≥70", "NEUTRAL": "50–69", "CAUTION": "35–49", "STRESS": "<35"},
    }


def regime_history(btc: pd.DataFrame) -> list[dict]:
    return [
        {"date": ts.date().isoformat(), "regime": r}
        for ts, r in classify(btc).items()
    ]


def regime_history_with_scores(btc: pd.DataFrame, tail: int = 0) -> list[dict]:
    """Full per-day breakdown including component scores."""
    df = classify_with_scores(btc)
    if tail:
        df = df.tail(tail)
    return [
        {
            "date":     ts.date().isoformat(),
            "regime":   row["regime"],
            "score":    round(row["composite"], 1),
            "trend":    round(row["trend"], 1),
            "vol":      round(row["vol"], 1),
            "momentum": round(row["momentum"], 1),
            "drawdown": round(row["drawdown"], 1),
        }
        for ts, row in df.iterrows()
    ]


def regime_transitions(btc: pd.DataFrame) -> list[dict]:
    """Return only the bars where the regime label changed."""
    series = classify(btc)
    prev   = series.shift(1)
    mask   = series != prev
    transitions = series[mask].dropna()
    return [
        {"date": ts.date().isoformat(), "to": r, "from": str(prev.loc[ts])}
        for ts, r in transitions.items()
        if str(prev.loc[ts]) != "nan"
    ]
