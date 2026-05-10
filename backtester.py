"""
Event-driven backtester.
Simulates long-only or long/short trades from a signal series.
Computes per-trade PnL and aggregate metrics.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    timeframe: str
    regime_filter: str | None      # None = run on all regimes
    params: dict

    # aggregate metrics
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar: float = 0.0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pct: float = 0.0
    total_trades: int = 0

    # time series (date → value)
    equity_curve: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def run(
    df: pd.DataFrame,
    signals: pd.Series,
    regime_series: pd.Series | None = None,
    regime_filter: str | None = None,
    symbol: str = "",
    strategy: str = "",
    timeframe: str = "1d",
    params: dict | None = None,
    fee_pct: float = 0.055,   # Bybit linear taker fee 0.055%
    slippage_pct: float = 0.02,
    initial_capital: float = 10_000,
    position_size_pct: float = 100.0,  # % of capital per trade
    long_only: bool = False,
) -> BacktestResult:
    result = BacktestResult(
        symbol=symbol,
        strategy=strategy,
        timeframe=timeframe,
        regime_filter=regime_filter,
        params=params or {},
    )

    close = df["close"]
    capital = initial_capital
    equity = []
    trades_log = []

    position = 0       # 0 = flat, 1 = long, -1 = short
    entry_price = 0.0
    entry_date = None

    cost = (fee_pct + slippage_pct) / 100  # one-way cost fraction

    for i in range(1, len(df)):
        ts   = df.index[i]
        price = close.iloc[i]

        # regime gate — skip signals outside allowed regime
        in_regime = True
        if regime_filter and regime_series is not None:
            in_regime = regime_series.iloc[i] == regime_filter

        sig = signals.iloc[i] if in_regime else 0

        # ── close open position on opposing signal ──────────────────────────
        if position != 0 and sig != 0 and sig != position:
            pnl_pct = _trade_pnl(position, entry_price, price, cost)
            capital *= (1 + pnl_pct / 100)
            trades_log.append({
                "entry_date": entry_date.isoformat() if entry_date else "",
                "exit_date":  ts.date().isoformat(),
                "direction":  "long" if position == 1 else "short",
                "entry_price": round(entry_price, 4),
                "exit_price":  round(price, 4),
                "pnl_pct":     round(pnl_pct, 4),
            })
            position = 0

        # ── open new position ───────────────────────────────────────────────
        if position == 0 and sig != 0:
            if long_only and sig == -1:
                pass
            else:
                position    = sig
                entry_price = price * (1 + cost * sig)  # adjust for cost
                entry_date  = ts

        equity.append({"date": ts.date().isoformat(), "equity": round(capital, 2)})

    # close any open position at end
    if position != 0 and entry_price:
        price = close.iloc[-1]
        pnl_pct = _trade_pnl(position, entry_price, price, cost)
        capital *= (1 + pnl_pct / 100)
        trades_log.append({
            "entry_date": entry_date.isoformat() if entry_date else "",
            "exit_date":  df.index[-1].date().isoformat(),
            "direction":  "long" if position == 1 else "short",
            "entry_price": round(entry_price, 4),
            "exit_price":  round(price, 4),
            "pnl_pct":     round(pnl_pct, 4),
        })

    result.equity_curve = equity
    result.trades       = trades_log
    result.total_trades = len(trades_log)
    result.total_return_pct = round((capital / initial_capital - 1) * 100, 2)

    _fill_metrics(result, initial_capital, equity, trades_log)
    return result


def _trade_pnl(direction: int, entry: float, exit_: float, cost: float) -> float:
    raw = (exit_ - entry) / entry * direction * 100
    return raw - cost * 100


def _fill_metrics(result: BacktestResult, initial: float, equity: list, trades: list):
    if not equity:
        return

    values = [e["equity"] for e in equity]
    arr    = np.array(values, dtype=float)
    rets   = np.diff(arr) / arr[:-1]

    # CAGR
    years = len(equity) / 365.0
    final = arr[-1]
    result.cagr_pct = round(((final / initial) ** (1 / max(years, 0.01)) - 1) * 100, 2)

    # Sharpe (daily, annualised, 0% risk-free)
    if rets.std() > 0:
        result.sharpe = round(rets.mean() / rets.std() * np.sqrt(365), 2)

    # Sortino
    downside = rets[rets < 0]
    if len(downside) > 0 and downside.std() > 0:
        result.sortino = round(rets.mean() / downside.std() * np.sqrt(365), 2)

    # Max drawdown
    peaks = np.maximum.accumulate(arr)
    dd    = (arr - peaks) / peaks
    result.max_drawdown_pct = round(dd.min() * 100, 2)

    # Calmar
    if result.max_drawdown_pct < 0:
        result.calmar = round(result.cagr_pct / abs(result.max_drawdown_pct), 2)

    # Trade metrics
    if trades:
        pnls = [t["pnl_pct"] for t in trades]
        wins  = [p for p in pnls if p > 0]
        loses = [p for p in pnls if p <= 0]
        result.win_rate_pct = round(len(wins) / len(pnls) * 100, 1)
        result.avg_trade_pct = round(np.mean(pnls), 3)
        gross_profit = sum(wins)
        gross_loss   = abs(sum(loses))
        result.profit_factor = round(gross_profit / gross_loss, 2) if gross_loss else 999.0
