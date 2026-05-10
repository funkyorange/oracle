"""
StrategyFactory API client + all scoring logic.
Ported from trademanager/portfolio_manager.py — same thresholds, no UI dependencies.
"""

import os
import statistics
from datetime import datetime

import requests

BASE_URL = "https://app.strategyfactory.ai"

SF_KEY    = os.getenv("SF_KEY", "")
SF_SECRET = os.getenv("SF_SECRET", "")

# ── Thresholds (keep in sync with trademanager/portfolio_manager.py) ──────────

SELECTION = {
    "min_trades_live":       100,
    "min_live_days":          60,
    "min_sharpe_live":        0.70,
    "max_dd_live":           22.0,
    "min_win_rate_live":     48.0,
    "min_pf_live":            1.25,
    "live_bt_sharpe_ratio":   0.75,
    "live_bt_wr_ratio":       0.88,
    "min_pct30":              0.0,
    "pct7_floor":            -5.0,
}

TRIM = {
    "pct30_warn":   0.0,
    "pct7_warn":   -3.0,
    "pct30_pause": -8.0,
    "pct7_pause":  -8.0,
    "dd_warn":      18.0,
    "dd_pause":     25.0,
    "sharpe_warn":   0.40,
    "sharpe_pause":  0.0,
    "live_bt_decay": 0.65,
    "no_trades_days": 14,
}

REGIME_THRESHOLDS = {
    "bullish_pct_pos":   60,
    "bearish_pct_pos":   38,
    "live_bt_ratio_ok":  0.80,
    "stress_threshold":  0.65,
    "circuit_breaker":   25.0,
}

MAX_PER_TYPE = 2


# ── HTTP client ───────────────────────────────────────────────────────────────

class SFClient:
    def __init__(self, key: str = SF_KEY, secret: str = SF_SECRET):
        self.headers = {
            "x-user-api-key":    key,
            "x-user-api-secret": secret,
            "Content-Type":      "application/json",
        }

    def get(self, path: str, params: dict = None) -> dict:
        r = requests.get(f"{BASE_URL}{path}", headers=self.headers, params=params, timeout=30)
        if r.status_code == 403:
            raise ConnectionError("SF authentication failed — check SF_KEY / SF_SECRET")
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            raise ValueError(data.get("error", "SF API returned ok=false"))
        return data["data"]


# ── Metric extraction ─────────────────────────────────────────────────────────

def _f(s, live_key, bt_key=None, default=0.0):
    val = s.get(live_key)
    if val is None and bt_key:
        val = s.get(bt_key)
    try:
        return float(val) if val is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def _i(s, live_key, bt_key=None, default=0):
    val = s.get(live_key)
    if val is None and bt_key:
        val = s.get(bt_key)
    try:
        return int(val) if val is not None else int(default)
    except (TypeError, ValueError):
        return int(default)


def extract_metrics(s: dict) -> dict:
    return {
        "sharpe":       _f(s, "sharpeLive",       "sharpeBacktest"),
        "sharpe_bt":    _f(s, "sharpeBacktest"),
        "sharpe_live":  _f(s, "sharpeLive"),
        "dd":           abs(_f(s, "maxDrawdownLive", "maxDrawdownBacktest")),
        "wr":           _f(s, "winRateLive",        "winRateBacktest"),
        "wr_bt":        _f(s, "winRateBacktest"),
        "wr_live":      _f(s, "winRateLive"),
        "pf":           _f(s, "profitFactorLive",   "profitFactorBacktest"),
        "trades":       _i(s, "tradesLive",         "tradesBacktest"),
        "trades_live":  _i(s, "tradesLive"),
        "pnl":          _f(s, "netProfitPctLive",   "netProfitPctBacktest"),
        "sortino":      _f(s, "sortinoLive",        "sortinoBacktest"),
    }


def extract_detail_metrics(s: dict) -> dict:
    return {
        "pct7":             _f(s, "pct7Days"),
        "pct30":            _f(s, "pct30Days"),
        "pct90":            _f(s, "pct90Days"),
        "kelly":            _f(s, "kellyCriterion"),
        "expected_monthly": _f(s, "expectedMonthly"),
        "calmar":           _f(s, "calmar"),
        "volatility":       _f(s, "volatility"),
        "first_date":       s.get("firstTradedDate"),
        "last_date":        s.get("lastTradedDate"),
    }


def detect_type(s: dict) -> str:
    api_type = (s.get("type") or "").lower()
    if api_type in ("trend", "trend_following"):
        return "Trend Following"
    if api_type in ("mean_reversion", "mr"):
        return "Mean Reversion"
    if api_type == "momentum":
        return "Momentum"
    nm = (s.get("name") or s.get("slug") or "").lower()
    if any(w in nm for w in ["trend", "follow", "breakout", "turtle", "dead zone", "bos", "stiff"]):
        return "Trend Following"
    if any(w in nm for w in ["mean", "reversion", "mr", "scalp", "bounce"]):
        return "Mean Reversion"
    if any(w in nm for w in ["momentum", "mom", "burst", "rsi"]):
        return "Momentum"
    if any(w in nm for w in ["vol", "volatility", "vix"]):
        return "Volatility"
    return "Systematic"


def compute_score(m: dict) -> int:
    if m["sharpe"] == 0 and m["pf"] == 0:
        return 0
    sh = min(m["sharpe"] / 2.0, 1.0) * 30
    dd = (1 - min(m["dd"] / 50.0, 1.0)) * 20
    wr = min(m["wr"] / 70.0, 1.0) * 15
    pf = min((m["pf"] - 1) / 2.0, 1.0) * 20 if m["pf"] > 1 else 0
    tr = min(m["trades"] / 200.0, 1.0) * 15
    return round(sh + dd + wr + pf + tr)


def live_days(s: dict) -> int:
    fd = s.get("firstTradedDate") or s.get("incubationDate")
    if not fd:
        return 0
    try:
        dt = datetime.fromisoformat(fd.replace("Z", "+00:00"))
        return (datetime.now(dt.tzinfo) - dt).days
    except Exception:
        return 0


# ── Regime detection ──────────────────────────────────────────────────────────

def detect_regime(pool: list) -> dict:
    ratios, live_sharpes, beat_bt = [], [], []
    for s in pool:
        sl = _f(s, "sharpeLive")
        sb = _f(s, "sharpeBacktest")
        tl = _i(s, "tradesLive")
        if sl > 0 and sb > 0 and tl >= 30:
            ratios.append(sl / sb)
            live_sharpes.append(sl)
            beat_bt.append(1 if sl >= sb * 0.9 else 0)

    if not ratios:
        return {"label": "UNKNOWN", "score": 50,
                "desc": "Insufficient data to determine regime.",
                "n_strats": 0, "avg_ratio": 0, "avg_live_sh": 0, "pct_beat_bt": 0}

    avg_ratio    = statistics.mean(ratios)
    avg_live_sh  = statistics.mean(live_sharpes)
    pct_beat_bt  = sum(beat_bt) / len(beat_bt) * 100
    n            = len(ratios)

    score = round(
        min(avg_ratio / 1.2, 1.0) * 40 +
        min(avg_live_sh / 1.0, 1.0) * 30 +
        min(pct_beat_bt / 70, 1.0) * 30
    )

    if score >= 65:
        label = "BULLISH"
        desc  = "Edge is being rewarded — conditions favour adding or holding exposure"
    elif score >= 45:
        label = "NEUTRAL"
        desc  = "Mixed conditions — hold current portfolio, be selective on new adds"
    elif score >= 30:
        label = "CAUTION"
        desc  = "Live underperforming backtest broadly — trim weak performers, raise bar for adds"
    else:
        label = "STRESS"
        desc  = "Market is punishing systematic strategies — reduce exposure, protect capital"

    return {
        "label":        label,
        "score":        score,
        "desc":         desc,
        "n_strats":     n,
        "avg_ratio":    round(avg_ratio, 3),
        "avg_live_sh":  round(avg_live_sh, 3),
        "pct_beat_bt":  round(pct_beat_bt, 1),
    }


# ── Lifecycle verdict ─────────────────────────────────────────────────────────

def lifecycle_verdict(m: dict, dm: dict, regime: dict) -> dict:
    flags   = []
    verdict = "HOLD"

    if dm["pct7"] < TRIM["pct7_pause"]:
        verdict = "PAUSE"
        flags.append(f"7d return {dm['pct7']:+.1f}% below pause floor {TRIM['pct7_pause']}%")

    if dm["pct30"] < TRIM["pct30_pause"]:
        verdict = "PAUSE"
        flags.append(f"30d return {dm['pct30']:+.1f}% below pause floor {TRIM['pct30_pause']}%")

    if m["dd"] > TRIM["dd_pause"] and m["dd"] > 0:
        verdict = "PAUSE"
        flags.append(f"Drawdown {m['dd']:.1f}% exceeds pause threshold {TRIM['dd_pause']}%")

    if m["sharpe_live"] > 0 and m["sharpe_live"] < TRIM["sharpe_pause"]:
        verdict = "PAUSE"
        flags.append(f"Live Sharpe {m['sharpe_live']:.2f} is negative")

    if verdict == "HOLD":
        if dm["pct30"] < TRIM["pct30_warn"]:
            verdict = "WARN"
            flags.append(f"30d return {dm['pct30']:+.1f}% (negative)")
        if dm["pct7"] < TRIM["pct7_warn"]:
            verdict = verdict if verdict != "HOLD" else "WARN"
            flags.append(f"7d return {dm['pct7']:+.1f}% below warning floor {TRIM['pct7_warn']}%")
        if m["dd"] > TRIM["dd_warn"] and m["dd"] > 0:
            verdict = verdict if verdict != "HOLD" else "WARN"
            flags.append(f"Drawdown {m['dd']:.1f}% approaching limit")
        if m["sharpe_live"] > 0 and m["sharpe_bt"] > 0:
            ratio = m["sharpe_live"] / m["sharpe_bt"]
            if ratio < TRIM["live_bt_decay"]:
                verdict = verdict if verdict != "HOLD" else "WARN"
                flags.append(f"Live Sharpe is {ratio:.0%} of backtest (edge decaying)")
        if regime["label"] == "STRESS":
            verdict = verdict if verdict != "HOLD" else "WARN"
            flags.append("Market in STRESS regime — reduce risk")

    if not flags:
        if dm["pct7"] > 5 and dm["pct30"] > 0 and regime["label"] in ("BULLISH", "NEUTRAL"):
            flags.append("Performing well in favourable regime")
        elif dm["pct30"] > 0:
            flags.append("Profitable over 30 days")

    action = {
        "HOLD":  "Maintain allocation",
        "WARN":  "Reduce allocation by 50% — watch closely",
        "PAUSE": "Pause bot — suspend trading",
    }.get(verdict, "Review")

    return {"verdict": verdict, "flags": flags, "action": action}


# ── Candidate finder ──────────────────────────────────────────────────────────

def find_add_candidates(pool: list, existing_ids: set, existing_types: list,
                        regime: dict, top_n: int = 10) -> list:
    candidates = []
    for s in pool:
        sid = s.get("id") or s.get("slug")
        if sid in existing_ids:
            continue
        m     = extract_metrics(s)
        gates = _check_gates(s, m, regime, existing_types)
        if gates["pass"]:
            candidates.append({
                "id":    sid,
                "name":  s.get("name") or s.get("slug", ""),
                "slug":  s.get("slug", ""),
                "type":  gates["type"],
                "pair":  s.get("pair", ""),
                "tf":    s.get("timeframe", ""),
                "score": compute_score(m),
                "metrics": {
                    "sharpe_live": m["sharpe_live"],
                    "dd":          m["dd"],
                    "wr_live":     m["wr_live"],
                    "pf":          m["pf"],
                    "trades_live": m["trades_live"],
                    "pnl":         m["pnl"],
                    "sortino":     m["sortino"],
                },
                "gates": gates["gates"],
            })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top_n]


def _check_gates(s: dict, m: dict, regime: dict, existing_types: list) -> dict:
    gates = []

    def gate(name, passed, reason):
        gates.append({"name": name, "pass": passed, "reason": reason})

    gate("Live trades ≥ 100",    m["trades_live"] >= SELECTION["min_trades_live"],  f"{m['trades_live']} live trades")
    gate("Live Sharpe ≥ 0.70",   m["sharpe_live"]  >= SELECTION["min_sharpe_live"], f"Sharpe {m['sharpe_live']:.2f}")
    gate("Max DD ≤ 22%",         m["dd"] <= SELECTION["max_dd_live"] or m["dd"] == 0, f"DD {m['dd']:.1f}%")
    gate("Win rate ≥ 48%",       m["wr_live"] >= SELECTION["min_win_rate_live"] or m["wr_live"] == 0, f"WR {m['wr_live']:.1f}%")
    gate("Profit factor ≥ 1.25", m["pf"] >= SELECTION["min_pf_live"],               f"PF {m['pf']:.2f}")

    if m["sharpe_bt"] > 0:
        ratio = m["sharpe_live"] / m["sharpe_bt"]
        gate("Live/BT Sharpe ≥ 75%", ratio >= SELECTION["live_bt_sharpe_ratio"], f"{ratio:.0%} of backtest Sharpe")
    if m["wr_bt"] > 0:
        ratio_wr = m["wr_live"] / m["wr_bt"]
        gate("Live/BT WR ≥ 88%", ratio_wr >= SELECTION["live_bt_wr_ratio"], f"{ratio_wr:.0%} of backtest WR")

    gate("Regime allows adds", regime["label"] in ("BULLISH", "NEUTRAL"), f"Regime is {regime['label']}")

    strat_type = detect_type(s)
    type_count = existing_types.count(strat_type)
    gate("Not over-concentrated", type_count < MAX_PER_TYPE, f"{strat_type} ({type_count}/{MAX_PER_TYPE} slots)")

    return {"pass": all(g["pass"] for g in gates), "gates": gates, "type": strat_type}
