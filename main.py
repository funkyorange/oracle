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
import sf_client as sf

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
    """Current regime label with full per-indicator breakdown."""
    df = datamod.fetch_ohlcv(symbol, "1d", since_days=400)
    if df.empty:
        raise HTTPException(status_code=503, detail="Could not fetch data")
    return {"symbol": symbol, **regmod.current_regime_detail(df)}


@app.get("/regimes/history")
def get_regime_history(symbol: str = "BTC/USDT:USDT", days: int = 365):
    """Daily regime label for the last N days."""
    df = datamod.fetch_ohlcv(symbol, "1d", since_days=days + 250)
    if df.empty:
        raise HTTPException(status_code=503, detail="Could not fetch data")
    history = regmod.regime_history(df)
    return {"symbol": symbol, "history": history[-days:]}


@app.get("/regimes/breakdown")
def get_regime_breakdown(symbol: str = "BTC/USDT:USDT", days: int = 180):
    """Daily regime with per-indicator scores — for charting component contributions."""
    df = datamod.fetch_ohlcv(symbol, "1d", since_days=days + 250)
    if df.empty:
        raise HTTPException(status_code=503, detail="Could not fetch data")
    rows = regmod.regime_history_with_scores(df, tail=days)
    return {"symbol": symbol, "rows": rows}


@app.get("/regimes/transitions")
def get_regime_transitions(symbol: str = "BTC/USDT:USDT", days: int = 730):
    """Only the dates where regime changed label — useful for annotating charts."""
    df = datamod.fetch_ohlcv(symbol, "1d", since_days=days + 250)
    if df.empty:
        raise HTTPException(status_code=503, detail="Could not fetch data")
    return {"symbol": symbol, "transitions": regmod.regime_transitions(df)}


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
        if not trades:
            continue
        avg_pct = r.get("avg_trade_pct", 0)
        projected = round((1 + avg_pct / 100) ** req.days * 100 - 100, 2)
        output.append({
            "strategy": r["strategy"],
            "symbol":   r["symbol"],
            "regime":   req.regime,
            "avg_trade_pct": avg_pct,
            "projected_return_pct": projected,
        })
    return output


# ── StrategyFactory analysis ──────────────────────────────────────────────────

_sf_client = sf.SFClient()

# simple in-process cache (TTL 5 min)
import time as _time
_sf_cache: dict = {}
_SF_TTL = 300


def _sf_cached(key: str, fn):
    entry = _sf_cache.get(key)
    if entry and (_time.time() - entry["ts"]) < _SF_TTL:
        return entry["data"]
    data = fn()
    _sf_cache[key] = {"ts": _time.time(), "data": data}
    return data


def _fetch_pool() -> list:
    raw = _sf_client.get("/api/user-api/strategies")
    return raw.get("items", []) if isinstance(raw, dict) else raw


def _fetch_bots() -> list:
    raw = _sf_client.get("/api/user-api/bots")
    return raw.get("items", []) if isinstance(raw, dict) else raw


def _fetch_strategy_detail(sid: str) -> dict:
    try:
        d = _sf_client.get(f"/api/user-api/strategies/{sid}")
        s = d.get("strategy", d)
        return {
            "raw": s,
            "m":   sf.extract_metrics(s),
            "dm":  sf.extract_detail_metrics(s),
            "name": s.get("name", sid),
        }
    except Exception:
        return {}


@app.get("/sf/regime")
def sf_regime():
    """Current market regime from the full SF strategy pool."""
    if not sf.SF_KEY:
        raise HTTPException(status_code=503, detail="SF_KEY not configured")
    try:
        pool = _sf_cached("pool", _fetch_pool)
        return sf.detect_regime(pool)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sf/strategies")
def sf_strategies(limit: int = 50):
    """Full SF strategy pool, scored and sorted."""
    if not sf.SF_KEY:
        raise HTTPException(status_code=503, detail="SF_KEY not configured")
    try:
        pool = _sf_cached("pool", _fetch_pool)
        result = []
        for s in pool:
            m = sf.extract_metrics(s)
            result.append({
                "id":          s.get("id") or s.get("slug"),
                "name":        s.get("name") or s.get("slug", ""),
                "slug":        s.get("slug", ""),
                "type":        sf.detect_type(s),
                "pair":        s.get("pair", ""),
                "timeframe":   s.get("timeframe", ""),
                "score":       sf.compute_score(m),
                "metrics": {
                    "sharpe_live":  m["sharpe_live"],
                    "sharpe_bt":    m["sharpe_bt"],
                    "dd":           m["dd"],
                    "wr_live":      m["wr_live"],
                    "pf":           m["pf"],
                    "trades_live":  m["trades_live"],
                    "pnl":          m["pnl"],
                    "sortino":      m["sortino"],
                    "lbt_ratio":    round(m["sharpe_live"] / m["sharpe_bt"], 3) if m["sharpe_bt"] > 0 else 0,
                },
            })
        result.sort(key=lambda x: x["score"], reverse=True)
        return result[:limit]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sf/bots")
def sf_bots():
    """Active bots with lifecycle verdicts (hold / warn / pause)."""
    if not sf.SF_KEY:
        raise HTTPException(status_code=503, detail="SF_KEY not configured")
    try:
        pool  = _sf_cached("pool", _fetch_pool)
        bots  = _sf_cached("bots", _fetch_bots)
        regime = sf.detect_regime(pool)

        result = []
        for bot in bots:
            sid    = bot.get("strategyId") or bot.get("strategySlug")
            detail = _sf_cached(f"detail:{sid}", lambda s=sid: _fetch_strategy_detail(s))
            m  = detail.get("m", {})
            dm = detail.get("dm", {})

            if not m:
                result.append({
                    "bot_id":   bot.get("id"),
                    "name":     bot.get("name", sid),
                    "status":   bot.get("status", "UNKNOWN"),
                    "strategy_id": sid,
                    "strategy_slug": bot.get("strategySlug", ""),
                    "verdict":  "UNKNOWN",
                    "flags":    [],
                    "action":   "No strategy data",
                    "metrics":  {},
                })
                continue

            lc  = sf.lifecycle_verdict(m, dm, regime)
            lbt = (m["sharpe_live"] / m["sharpe_bt"]) if m.get("sharpe_bt", 0) > 0 else 0

            result.append({
                "bot_id":        bot.get("id"),
                "name":          bot.get("name", sid),
                "status":        bot.get("status", "ACTIVE"),
                "strategy_id":   sid,
                "strategy_slug": bot.get("strategySlug", ""),
                "amount":        bot.get("amount", 0),
                "verdict":       lc["verdict"],
                "flags":         lc["flags"],
                "action":        lc["action"],
                "lbt_ratio":     round(lbt, 3),
                "metrics": {
                    "pct7":        dm.get("pct7", 0),
                    "pct30":       dm.get("pct30", 0),
                    "pct90":       dm.get("pct90", 0),
                    "sharpe_live": m.get("sharpe_live", 0),
                    "dd":          m.get("dd", 0),
                    "wr_live":     m.get("wr_live", 0),
                    "pf":          m.get("pf", 0),
                    "trades_live": m.get("trades_live", 0),
                },
            })

        return {"regime": regime, "bots": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sf/recommendations")
def sf_recommendations(top_candidates: int = 5):
    """
    The main decision endpoint.
    Returns:
      - Current regime
      - For each active bot: HOLD / WARN / PAUSE + reason
      - Top strategies to ADD from the pool (passed all gates)
      - A priority-ordered action list for the user
    """
    if not sf.SF_KEY:
        raise HTTPException(status_code=503, detail="SF_KEY not configured")
    try:
        pool   = _sf_cached("pool", _fetch_pool)
        bots   = _sf_cached("bots", _fetch_bots)
        regime = sf.detect_regime(pool)

        # ── lifecycle per active bot ──────────────────────────────────────────
        existing_ids   = set()
        existing_types = []
        bot_verdicts   = []
        actions        = []

        for bot in bots:
            sid    = bot.get("strategyId") or bot.get("strategySlug")
            existing_ids.add(sid)
            detail = _sf_cached(f"detail:{sid}", lambda s=sid: _fetch_strategy_detail(s))
            m      = detail.get("m", {})
            dm     = detail.get("dm", {})

            if not m:
                continue

            existing_types.append(sf.detect_type(detail.get("raw", {})))
            lc   = sf.lifecycle_verdict(m, dm, regime)
            lbt  = (m["sharpe_live"] / m["sharpe_bt"]) if m.get("sharpe_bt", 0) > 0 else 0
            entry = {
                "bot_id":        bot.get("id"),
                "name":          bot.get("name", sid),
                "strategy_slug": bot.get("strategySlug", ""),
                "verdict":       lc["verdict"],
                "action":        lc["action"],
                "flags":         lc["flags"],
                "lbt_ratio":     round(lbt, 3),
                "metrics": {
                    "pct7":        dm.get("pct7", 0),
                    "pct30":       dm.get("pct30", 0),
                    "sharpe_live": m.get("sharpe_live", 0),
                    "dd":          m.get("dd", 0),
                },
            }
            bot_verdicts.append(entry)

            if lc["verdict"] in ("PAUSE", "WARN"):
                priority = 1 if lc["verdict"] == "PAUSE" else 2
                actions.append({
                    "priority":  priority,
                    "type":      lc["verdict"],
                    "target":    bot.get("name", sid),
                    "bot_id":    bot.get("id"),
                    "action":    lc["action"],
                    "reason":    "; ".join(lc["flags"][:2]),
                })

        # ── add candidates ───────────────────────────────────────────────────
        candidates = sf.find_add_candidates(
            pool, existing_ids, existing_types, regime, top_n=top_candidates
        )
        for c in candidates:
            actions.append({
                "priority": 3,
                "type":     "ADD",
                "target":   c["name"],
                "bot_id":   None,
                "action":   f"Add strategy — score {c['score']}/100",
                "reason":   f"Sharpe {c['metrics']['sharpe_live']:.2f}, DD {c['metrics']['dd']:.1f}%, WR {c['metrics']['wr_live']:.0f}%",
            })

        # ── regime-level circuit breaker ─────────────────────────────────────
        if regime["label"] == "STRESS":
            actions.insert(0, {
                "priority": 0,
                "type":     "CIRCUIT_BREAKER",
                "target":   "ALL",
                "bot_id":   None,
                "action":   "Consider pausing all bots — market in STRESS",
                "reason":   regime["desc"],
            })

        actions.sort(key=lambda a: a["priority"])

        return {
            "as_of":       datetime.now(timezone.utc).isoformat(),
            "regime":      regime,
            "bots":        bot_verdicts,
            "candidates":  candidates,
            "actions":     actions,
            "summary": {
                "total_bots":    len(bot_verdicts),
                "hold":          sum(1 for b in bot_verdicts if b["verdict"] == "HOLD"),
                "warn":          sum(1 for b in bot_verdicts if b["verdict"] == "WARN"),
                "pause":         sum(1 for b in bot_verdicts if b["verdict"] == "PAUSE"),
                "add_candidates": len(candidates),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sf/rotation")
def sf_rotation():
    """
    Portfolio rotation analysis — the decay/diversification endpoint.

    Addresses two recurring problems:
      1. Good strategies decay — this endpoint detects decay early via a composite
         health score (Sharpe live, LBT ratio, 30d/7d returns) and zones each bot
         CORE / WATCH / ROTATE.
      2. Portfolio too small/concentrated — checks against the 8-12 target range
         and flags type over-concentration.

    Returns specific swap recommendations: which bot to drop and what to add instead.
    """
    if not sf.SF_KEY:
        raise HTTPException(status_code=503, detail="SF_KEY not configured")
    try:
        from rotation import (
            _decay_score, _classify_zone, TARGET_MIN, TARGET_MAX, MAX_PER_TYPE,
            _diversity_recommendation,
        )

        pool   = _sf_cached("pool", _fetch_pool)
        bots   = _sf_cached("bots", _fetch_bots)
        regime = sf.detect_regime(pool)

        # fetch detail for every bot
        existing_ids   = set()
        existing_types = []
        bot_health     = []

        for bot in bots:
            sid    = bot.get("strategyId") or bot.get("strategySlug")
            existing_ids.add(sid)
            detail = _sf_cached(f"detail:{sid}", lambda s=sid: _fetch_strategy_detail(s))
            m      = detail.get("m", {})
            dm     = detail.get("dm", {})
            raw_s  = detail.get("raw", {})

            if not m:
                continue

            existing_types.append(sf.detect_type(raw_s))
            lc     = sf.lifecycle_verdict(m, dm, regime)
            lbt    = m["sharpe_live"] / m["sharpe_bt"] if m.get("sharpe_bt", 0) > 0 else 0
            decay  = _decay_score(m, dm)
            zone   = _classify_zone(lc["verdict"], decay)

            bot_health.append({
                "bot_id":        bot.get("id"),
                "name":          bot.get("name", sid),
                "strategy_slug": bot.get("strategySlug", ""),
                "verdict":       lc["verdict"],
                "zone":          zone,
                "decay_score":   decay,
                "lbt_ratio":     round(lbt, 3),
                "metrics": {
                    "pct7":        dm.get("pct7", 0),
                    "pct30":       dm.get("pct30", 0),
                    "sharpe_live": m.get("sharpe_live", 0),
                    "dd":          m.get("dd", 0),
                    "wr_live":     m.get("wr_live", 0),
                    "trades_live": m.get("trades_live", 0),
                },
                "flags":  lc["flags"],
                "action": lc["action"],
            })

        # find replacements for ROTATE bots
        candidates = sf.find_add_candidates(pool, existing_ids, existing_types, regime, top_n=20)

        rotate_out = sorted(
            [b for b in bot_health if b["zone"] == "ROTATE"],
            key=lambda b: b["decay_score"]
        )
        swaps = []
        for i, drop in enumerate(rotate_out):
            replacement = candidates[i] if i < len(candidates) else None
            swaps.append({
                "drop": {
                    "name":        drop["name"],
                    "bot_id":      drop["bot_id"],
                    "decay_score": drop["decay_score"],
                    "zone":        drop["zone"],
                    "reason":      "; ".join(drop["flags"][:2]) or drop["action"],
                },
                "add": {
                    "name":    replacement["name"],
                    "id":      replacement["id"],
                    "score":   replacement["score"],
                    "metrics": replacement["metrics"],
                } if replacement else None,
            })

        # diversity report
        n = len(bot_health)
        type_counts: dict[str, int] = {}
        for b in bot_health:
            t = sf.detect_type({"slug": b["strategy_slug"]})
            type_counts[t] = type_counts.get(t, 0) + 1

        size_verdict = "OK" if TARGET_MIN <= n <= TARGET_MAX else ("TOO_FEW" if n < TARGET_MIN else "TOO_MANY")
        concentrated = {t: c for t, c in type_counts.items() if c > MAX_PER_TYPE}

        return {
            "as_of":   datetime.now(timezone.utc).isoformat(),
            "regime":  regime,
            "bots":    sorted(bot_health, key=lambda b: b["decay_score"]),
            "swaps":   swaps,
            "diversity": {
                "active_count":    n,
                "target_range":    f"{TARGET_MIN}-{TARGET_MAX}",
                "size_verdict":    size_verdict,
                "type_distribution": type_counts,
                "concentrated_types": concentrated,
                "recommendation":  _diversity_recommendation(n, size_verdict, concentrated, candidates),
            },
            "candidates_pool": candidates[:10],
            "summary": {
                "core":   sum(1 for b in bot_health if b["zone"] == "CORE"),
                "watch":  sum(1 for b in bot_health if b["zone"] == "WATCH"),
                "rotate": sum(1 for b in bot_health if b["zone"] == "ROTATE"),
                "swaps_needed": len(swaps),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
