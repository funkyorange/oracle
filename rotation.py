"""
Strategy rotation and decay detection.

Problem: a static portfolio of "good" strategies decays over time as edges erode.
Solution: a continuously-scored watchlist with three zones:

  CORE     — proven, stable, keep running (score ≥60, verdict HOLD, LBT ≥0.75)
  WATCH    — slipping but not failing — monitor, reduce size (score 40-59 or WARN)
  ROTATE   — failing — pause and replace (score <40 or PAUSE verdict)

Rotation logic:
  1. Identify ROTATE candidates from active bots
  2. Find best ADD candidates from full pool (not already active)
  3. Recommend specific swaps: drop X, add Y
  4. Flag portfolio concentration (too few = fragile, too many = diluted)

Target portfolio characteristics (for crypto systematic):
  - Size: 8-12 strategies (≥8 for correlation benefit, ≤12 to avoid dilution)
  - Type spread: no more than 3 of same type
  - Regime spread: include some mean-reversion for CAUTION/STRESS hedging
"""

from dataclasses import dataclass, asdict

import sf_client as sf


# ── Target portfolio parameters ───────────────────────────────────────────────

TARGET_MIN   = 8    # below this: under-diversified, single failure too costly
TARGET_MAX   = 12   # above this: each strategy has too little impact
MAX_PER_TYPE = 3    # allow slightly more types than the selection gate (which uses 2)
                    # because at portfolio scale, 3 trend followers is fine


# ── Decay scoring ─────────────────────────────────────────────────────────────

def _decay_score(m: dict, dm: dict) -> int:
    """
    0-100 health score for an active strategy (higher = healthier).
    Mirrors compute_score() but weights recent performance more heavily.
    """
    if not m:
        return 0

    sharpe   = min(m.get("sharpe_live", 0) / 2.0, 1.0) * 25
    lbt      = min(m.get("sharpe_live", 0) / max(m.get("sharpe_bt", 0.001), 0.001), 1.0) * 20
    dd       = (1 - min(m.get("dd", 0) / 50.0, 1.0)) * 15
    pct30    = _pct_score(dm.get("pct30", 0), lo=-8, hi=5) * 25
    pct7     = _pct_score(dm.get("pct7", 0), lo=-5, hi=3) * 15

    return round(sharpe + lbt + dd + pct30 + pct7)


def _pct_score(val: float, lo: float, hi: float) -> float:
    """Map val in [lo..hi] to [0..1]. Below lo → 0, above hi → 1."""
    if val >= hi:
        return 1.0
    if val <= lo:
        return 0.0
    return (val - lo) / (hi - lo)


def _classify_zone(verdict: str, decay: int) -> str:
    if verdict == "PAUSE" or decay < 40:
        return "ROTATE"
    if verdict == "WARN" or decay < 60:
        return "WATCH"
    return "CORE"


# ── Main analysis ─────────────────────────────────────────────────────────────

@dataclass
class BotHealth:
    bot_id:        str
    name:          str
    strategy_slug: str
    verdict:       str
    zone:          str          # CORE / WATCH / ROTATE
    decay_score:   int          # 0-100
    lbt_ratio:     float        # live/backtest Sharpe (1.0 = perfect)
    pct7:          float
    pct30:         float
    sharpe_live:   float
    dd:            float
    flags:         list[str]
    action:        str


def analyse_bots(bots: list, pool: list, regime: dict) -> dict:
    """
    Full rotation analysis.

    bots:   list of SF bot objects
    pool:   full SF strategy pool (list)
    regime: output of sf.detect_regime()

    Returns a rich dict with health classification, swap recommendations,
    and diversity report.
    """
    existing_ids   = set()
    existing_types = []
    bot_health     = []

    for bot in bots:
        sid    = bot["_detail"].get("raw", {})  # pre-fetched detail
        m      = bot["_detail"].get("m", {})
        dm     = bot["_detail"].get("dm", {})
        raw_s  = bot["_detail"].get("raw", {})

        sid    = bot.get("strategyId") or bot.get("strategySlug")
        existing_ids.add(sid)

        if not m:
            continue

        existing_types.append(sf.detect_type(raw_s))
        lc     = sf.lifecycle_verdict(m, dm, regime)
        lbt    = m["sharpe_live"] / m["sharpe_bt"] if m.get("sharpe_bt", 0) > 0 else 0
        decay  = _decay_score(m, dm)
        zone   = _classify_zone(lc["verdict"], decay)

        bot_health.append(BotHealth(
            bot_id        = bot.get("id", ""),
            name          = bot.get("name", sid),
            strategy_slug = bot.get("strategySlug", ""),
            verdict       = lc["verdict"],
            zone          = zone,
            decay_score   = decay,
            lbt_ratio     = round(lbt, 3),
            pct7          = dm.get("pct7", 0),
            pct30         = dm.get("pct30", 0),
            sharpe_live   = m.get("sharpe_live", 0),
            dd            = m.get("dd", 0),
            flags         = lc["flags"],
            action        = lc["action"],
        ))

    # ── add candidates (broad search — use wider criteria for watchlist) ──────
    candidates = sf.find_add_candidates(pool, existing_ids, existing_types, regime, top_n=20)

    # ── swap recommendations ──────────────────────────────────────────────────
    rotate_out = sorted(
        [b for b in bot_health if b.zone == "ROTATE"],
        key=lambda b: b.decay_score
    )
    swaps = []
    for i, drop in enumerate(rotate_out):
        replacement = candidates[i] if i < len(candidates) else None
        swaps.append({
            "drop":  {"name": drop.name, "bot_id": drop.bot_id, "decay_score": drop.decay_score,
                      "reason": "; ".join(drop.flags[:2])},
            "add":   {
                "name":  replacement["name"] if replacement else None,
                "id":    replacement["id"]   if replacement else None,
                "score": replacement["score"] if replacement else None,
                "metrics": replacement["metrics"] if replacement else None,
            } if replacement else None,
        })

    # ── diversity report ──────────────────────────────────────────────────────
    n_active = len(bot_health)
    type_counts: dict[str, int] = {}
    for b in bot_health:
        t = sf.detect_type({"slug": b.strategy_slug})
        type_counts[t] = type_counts.get(t, 0) + 1

    size_ok = TARGET_MIN <= n_active <= TARGET_MAX
    size_verdict = "OK" if size_ok else ("TOO_FEW" if n_active < TARGET_MIN else "TOO_MANY")
    type_concentrated = {t: c for t, c in type_counts.items() if c > MAX_PER_TYPE}

    diversity = {
        "active_count":     n_active,
        "target_range":     f"{TARGET_MIN}-{TARGET_MAX}",
        "size_verdict":     size_verdict,
        "type_distribution": type_counts,
        "concentrated_types": type_concentrated,
        "recommendation": _diversity_recommendation(n_active, size_verdict, type_concentrated, candidates),
    }

    return {
        "regime":     regime,
        "bots": [asdict(b) for b in bot_health],
        "swaps":      swaps,
        "candidates": candidates[:10],
        "diversity":  diversity,
        "summary": {
            "core":   sum(1 for b in bot_health if b.zone == "CORE"),
            "watch":  sum(1 for b in bot_health if b.zone == "WATCH"),
            "rotate": sum(1 for b in bot_health if b.zone == "ROTATE"),
        },
    }


def _diversity_recommendation(n: int, size_verdict: str, concentrated: dict, candidates: list) -> str:
    parts = []
    if size_verdict == "TOO_FEW":
        add_n = TARGET_MIN - n
        parts.append(
            f"Portfolio has {n} strategies — below optimal minimum of {TARGET_MIN}. "
            f"Add {add_n} more to reduce single-strategy impact."
        )
    elif size_verdict == "TOO_MANY":
        parts.append(
            f"Portfolio has {n} strategies — above optimal maximum of {TARGET_MAX}. "
            f"Each strategy has less than {100//n}% impact; prune the weakest."
        )
    else:
        parts.append(f"Portfolio size ({n}) is within the optimal {TARGET_MIN}-{TARGET_MAX} range.")

    if concentrated:
        for t, c in concentrated.items():
            parts.append(f"Over-concentrated in {t} ({c} strategies — max recommended {MAX_PER_TYPE}).")

    if not candidates:
        parts.append("No add candidates currently pass all selection gates — regime may not be favourable for adds.")

    return " ".join(parts)
