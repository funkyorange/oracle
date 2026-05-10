"""
Portfolio optimizer.
Takes a list of backtest results and computes optimal capital allocations
using mean-variance (max Sharpe) or risk-parity weighting.
"""

import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize

from backtester import BacktestResult


@dataclass
class Allocation:
    strategy: str
    symbol: str
    weight_pct: float
    sharpe: float
    cagr_pct: float
    max_drawdown_pct: float


def _equity_returns(result: BacktestResult) -> np.ndarray:
    values = np.array([e["equity"] for e in result.equity_curve], dtype=float)
    if len(values) < 2:
        return np.array([0.0])
    return np.diff(values) / values[:-1]


def max_sharpe(results: list[BacktestResult]) -> list[Allocation]:
    """Markowitz max-Sharpe allocation (long-only, no leverage)."""
    if not results:
        return []

    ret_series = [_equity_returns(r) for r in results]
    min_len = min(len(s) for s in ret_series)
    if min_len < 30:
        return _equal_weight(results)

    mat = np.column_stack([s[-min_len:] for s in ret_series])
    mu  = mat.mean(axis=0) * 365
    cov = np.cov(mat.T) * 365

    n = len(results)

    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        return -port_ret / (port_vol + 1e-9)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n

    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = res.x if res.success else w0

    return [
        Allocation(
            strategy=r.strategy,
            symbol=r.symbol,
            weight_pct=round(float(w) * 100, 1),
            sharpe=r.sharpe,
            cagr_pct=r.cagr_pct,
            max_drawdown_pct=r.max_drawdown_pct,
        )
        for r, w in zip(results, weights)
    ]


def risk_parity(results: list[BacktestResult]) -> list[Allocation]:
    """Equal-risk-contribution weights."""
    if not results:
        return []

    ret_series = [_equity_returns(r) for r in results]
    min_len = min(len(s) for s in ret_series)
    if min_len < 30:
        return _equal_weight(results)

    mat = np.column_stack([s[-min_len:] for s in ret_series])
    cov = np.cov(mat.T) * 365
    n   = len(results)

    def risk_budget_obj(w):
        w = np.array(w)
        sigma = np.sqrt(w @ cov @ w)
        mrc = cov @ w / sigma
        rc  = w * mrc
        target = sigma / n
        return np.sum((rc - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 1.0)] * n
    w0 = np.ones(n) / n
    res = minimize(risk_budget_obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = res.x / res.x.sum() if res.success else w0

    return [
        Allocation(
            strategy=r.strategy,
            symbol=r.symbol,
            weight_pct=round(float(w) * 100, 1),
            sharpe=r.sharpe,
            cagr_pct=r.cagr_pct,
            max_drawdown_pct=r.max_drawdown_pct,
        )
        for r, w in zip(results, weights)
    ]


def _equal_weight(results: list[BacktestResult]) -> list[Allocation]:
    w = 100.0 / len(results)
    return [
        Allocation(
            strategy=r.strategy,
            symbol=r.symbol,
            weight_pct=round(w, 1),
            sharpe=r.sharpe,
            cagr_pct=r.cagr_pct,
            max_drawdown_pct=r.max_drawdown_pct,
        )
        for r in results
    ]
