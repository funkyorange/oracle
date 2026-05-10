"""
Oracle — backtesting & portfolio optimisation API.
Called by TradeManager; results surface in the TradeManager dashboard.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import data as datamod
import regimes as regmod
import strategies as strats
import backtester
import optimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="oracle")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory job store (replace with DB if needed)
_jobs: dict[str, dict] = {}


# ── health ──────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True, "strategies": strats.list_strategies()}


# ── data / universe ─────────────────────────────────────────────────────────────

@app.get("/symbols")
def get_symbols():
    return datamod.available_symbols()


TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

@app.get("/timeframes")
def get_timeframes():
    return TIMEFRAMES


# ── strategies ──────────────────────────────────────────────────────────────────

@app.get("/strategies")
def get_strategies():
    return strats.list_strategies()


# ── regimes ─────────────────────────────────────────────────────────────────────

@app.get("/regimes/current")
def get_current_regime(symbol: str = "BTC/USDT:USDT"):
    df = datamod.fetch_ohlcv(symbol, "1d", since_days=120)
    if df.empty:
        raise HTTPException(status_code=503, detail="Could not fetch data")
    return {"symbol": symbol, "regime": regmod.current_regime(df)}


@app.get("/regimes/history")
def get_regime_history(symbol: str = "BTC/USDT:USDT", days: int = 365):
    df = datamod.fetch_ohlcv(symbol, "1d", since_days=days)
    if df.empty:
        raise HTTPException(status_code=503, detail="Could not fetch data")
    history = regmod.regime_history(df)
    return {"symbol": symbol, "history": history}


# ── backtests ────────────────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    symbol: str = "BTC/USDT:USDT"
    timeframe: str = "1d"
    strategy: str = "ema_cross"
    params: Optional[dict] = None
    regime_filter: Optional[str] = None   # BULLISH / NEUTRAL / CAUTION / STRESS / None
    since_days: int = 730
    long_only: bool = False
    fee_pct: float = 0.055
    slippage_pct: float = 0.02


def _run_backtest_job(job_id: str, req: BacktestRequest):
    try:
        _jobs[job_id]["status"] = "running"
        df = datamod.fetch_ohlcv(req.symbol, req.timeframe, req.since_days)
        if df.empty:
            _jobs[job_id] = {"status": "error", "error": "No data returned"}
            return

        signals = strats.generate_signals(req.strategy, df, req.params)

        regime_series = None
        if req.regime_filter:
            btc = datamod.fetch_ohlcv("BTC/USDT:USDT", "1d", req.since_days)
            # resample regime to match df timeframe if needed
            regime_series = regmod.classify(btc)
            if req.timeframe != "1d":
                regime_series = regime_series.reindex(df.index, method="ffill")
            else:
                regime_series = regime_series.reindex(df.index, method="ffill")

        result = backtester.run(
            df=df,
            signals=signals,
            regime_series=regime_series,
            regime_filter=req.regime_filter,
            symbol=req.symbol,
            strategy=req.strategy,
            timeframe=req.timeframe,
            params=req.params or {},
            fee_pct=req.fee_pct,
            slippage_pct=req.slippage_pct,
            long_only=req.long_only,
        )
        _jobs[job_id] = {"status": "done", "result": result.to_dict()}
    except Exception as e:
        log.error(f"Backtest job {job_id} failed: {e}")
        _jobs[job_id] = {"status": "error", "error": str(e)}


@app.post("/backtest")
def start_backtest(req: BacktestRequest, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued"}
    bg.add_task(_run_backtest_job, job_id, req)
    return {"job_id": job_id}


@app.get("/backtest/{job_id}")
def get_backtest(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/backtests")
def list_backtests():
    return [
        {"job_id": jid, "status": j["status"],
         "strategy": j.get("result", {}).get("strategy", ""),
         "symbol":   j.get("result", {}).get("symbol", ""),
         "total_return_pct": j.get("result", {}).get("total_return_pct"),
         "sharpe":           j.get("result", {}).get("sharpe"),
         }
        for jid, j in _jobs.items()
    ]


# ── optimiser ────────────────────────────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    job_ids: list[str]           # completed backtest job IDs to optimise across
    method: str = "max_sharpe"   # max_sharpe | risk_parity


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    results = []
    for jid in req.job_ids:
        job = _jobs.get(jid)
        if not job or job["status"] != "done":
            raise HTTPException(status_code=400, detail=f"Job {jid} not ready")
        r = job["result"]
        br = backtester.BacktestResult(**{k: r[k] for k in backtester.BacktestResult.__dataclass_fields__})
        results.append(br)

    if req.method == "risk_parity":
        allocs = optimizer.risk_parity(results)
    else:
        allocs = optimizer.max_sharpe(results)

    return [
        {"strategy": a.strategy, "symbol": a.symbol,
         "weight_pct": a.weight_pct, "sharpe": a.sharpe,
         "cagr_pct": a.cagr_pct, "max_drawdown_pct": a.max_drawdown_pct}
        for a in allocs
    ]


# ── scenario ─────────────────────────────────────────────────────────────────────

class ScenarioRequest(BaseModel):
    job_ids: list[str]       # completed backtests to project
    regime: str              # force this regime label
    days: int = 30           # projection horizon


@app.post("/scenario")
def scenario(req: ScenarioRequest):
    """
    Filter each backtest's trade log to only trades taken in `regime`,
    compute average daily return for those periods, project `days` forward.
    """
    output = []
    for jid in req.job_ids:
        job = _jobs.get(jid)
        if not job or job["status"] != "done":
            continue
        r = job["result"]
        trades = r.get("trades", [])
        # use avg_trade_pct / avg days held as a proxy daily rate
        if not trades:
            continue
        avg_pct = r.get("avg_trade_pct", 0)
        # rough projection: compound avg trade return over `days`
        projected = round((1 + avg_pct / 100) ** req.days * 100 - 100, 2)
        output.append({
            "strategy": r["strategy"],
            "symbol":   r["symbol"],
            "regime":   req.regime,
            "avg_trade_pct": avg_pct,
            "projected_return_pct": projected,
        })
    return output
