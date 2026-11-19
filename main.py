"""Hierarchical growth-curve quantile forecaster for GTA VI trophy rarity (Python 3.13, stdlib only).

WHAT IT FORECASTS
    For every trophy of Grand Theft Auto VI on PS5, five quantiles (5%, 25%, 50%, 75%, 95%) of that
    trophy's Exophase earn rate (percent of the players Exophase tracks who have earned it) at the
    case's target, typically 4-5 days after the case's checkpoint.

WHY THIS SHAPE
    A trophy's earn rate is a ratio, earners / tracked players. Both terms grow during a launch week:
    tracked players sync in as they buy the game, and the players already tracked keep playing. So the
    rate grows, but it grows *decelerating*, and the observed history is a staircase: Exophase only
    changes a number when it refreshes its counts, so consecutive reads repeat and then jump.

    The rate is bounded and multiplicative, so the natural scale is the logit, L = ln(p / (1 - p)),
    exactly the scale the scorer uses. On that scale a rate that multiplies by k moves a fixed
    distance, and a rate near 100 cannot overshoot. Growth in a launch week is close to a power law in
    time (both earners and the pool grow like t^a and t^b with a > b), and ln(1 + t) is the power law's
    linear form, so

        L(t) = a + b * ln(1 + t),    t = hours since 2026-11-19T00:00Z

    is the per-trophy model. Its slope b is the log-odds growth per e-fold of time.

    One trophy's staircase is far too noisy to pin b down: on launch night the pool held 638 players,
    so a rate of 0.63% is four people and one more earner moves it by 25%. But all 55 trophies share
    one progression curve, so the slopes are pooled by tier and points band and each trophy's slope is
    shrunk towards its pool (empirical Bayes). The *level* is always anchored on the trophy's own last
    observed rate: the forecast is that value plus the pooled growth over the extrapolation distance.
    That keeps a quiet or missing trophy from drifting off its own reading.

    The spread has three parts, added in quadrature:
      * a random-walk floor c * sqrt(days ahead), the price of the curve's shape being wrong -- the
        dominant term for a 4.5-day extrapolation, and the term the backtest tunes;
      * the fitted slope's standard error times the extrapolation distance;
      * the per-trophy residual scatter around its own fit, plus the counting noise of the anchor
        (a rate backed by 4 earners is worth less than one backed by 400).
    Then q_tau = logistic(L_hat + z_tau * sigma), which is the quantile of a logit-normal belief.

READ / WRITE CONTRACT
    Reads every *.in in $MAKEFUN_INPUT_DIR (default /input), in name order, and writes
    <id>.out to $MAKEFUN_OUTPUT_DIR (default /output):
        {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}
    one row per trophy of the case, ids unique, five finite non-decreasing numbers in 0..100.
    Deterministic: no randomness anywhere, so two runs write the same bytes.

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

# ------------------------------------------------------------------------------------------- constants

# The contest's clock: every time in a case is measured in hours from here.
BASE_TIME = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# The five levels each row reports, in the order the row lists them, and the standard normal's
# quantile at each (exact values: the scorer compares against the true quantiles).
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722)

# The scorer clamps a printed rate to 0.01..99.99 before taking its logit (0% and 100% have no logit);
# clamping here keeps every internal number finite and every output inside the legal range.
RATE_MIN = 0.01
RATE_MAX = 99.99

# Recency weighting of the reads in the per-trophy fit: a read this many hours before the last one
# counts half as much. 24 h keeps the last two days of a three-day history in charge of the slope,
# which is what the next 4.5 days should be extrapolated from; MIN_WEIGHT stops the launch-night
# reads from vanishing entirely (they carry the only evidence of the very steep early phase).
HALFLIFE_H = 24.0
MIN_WEIGHT = 0.15

# Empirical-Bayes shrinkage of a trophy's slope towards its tier's pooled slope: the trophy's own
# slope counts n_eff times, the pool k times. A trophy with two independent refreshes is mostly pool.
SHRINK_K = 2.5
# A tier's pooled slope is itself shrunk towards the all-trophy pool with this many pseudo-trophies,
# so the one platinum borrows the whole field's curve and the 46 bronzes keep their own.
TIER_PRIOR_TROPHIES = 3.0
# How much of the pooled slope comes from the cross-sectional slope-vs-level regression rather than
# from the tier median. 0.0 is tier pooling only, 1.0 is the level regression only.
LEVEL_BLEND = 0.5
# Floor on the between-trophy spread of slopes (logit units per ln-unit of time) used by the
# inverse-variance shrinkage. Too small and a well-measured slope is dragged onto the prior; too large
# and a noisy one is never pulled in.
TAU_MIN = 0.12

# Slopes are clamped non-negative: an earn rate does not fall in the long run (a refresh that adds
# tracked players can dilute one for an hour, which the pooled slope absorbs; a sustained fall would
# be a data error). B_MAX caps the extrapolation of a fit that saw only a very steep early phase.
B_MIN = 0.0
B_MAX = 6.0

# Random-walk floor of the spread, in logit units per sqrt(day). A launch week's logit moves roughly
# half a unit a day for the trophies that move at all; sqrt-scaling is the honest way to say that a
# 4.5-day extrapolation is ~2.1x as uncertain as a one-day one.
FLOOR_SIGMA = 0.30
# Even a 90-minute horizon has a floor: the answering read may follow a refresh the checkpoint has not
# seen yet, which moves rates without any time to grow. Small, but not zero.
FLOOR_MIN = 0.12

# The fitted slope's standard error is inflated by this much before it enters the spread, because the
# reads of a staircase are serially correlated: a run of identical reads is not a run of independent
# observations, and the plain least-squares standard error would claim more precision than exists.
AUTOCORR = 1.6

# Bounds on the total spread, in logit units.
SIGMA_MIN = 0.05
SIGMA_MAX = 2.20

# Guard rails on the forecast itself.
MAX_LOGIT_GAIN = 3.0   # no trophy's median may move more than this many logit units in one forecast
SATURATED = 99.5       # a trophy at or above this rate is treated as pinned at the top
EMPTY_RATE = 0.01      # the rate assumed for a trophy a case never shows

# How much of the fitted power-law extrapolation to spend, as a fraction of the full ln(1+t) distance.
# 1.0 is the honest reading of the fit. The backtest's synthetic family says whether a launch week
# decelerates like ln(1+t) or faster; this is the knob that says so, and backtest.py --tune prints the
# loss/coverage table it is set from.
EXTRAP_SCALE = 1.0


# ------------------------------------------------------------------------------------------ utilities


def parse_time(text: str) -> datetime:
    """Parse an ISO 8601 UTC timestamp ('2026-11-19T03:00:00.309Z' or with a +00:00 offset)."""
    s = text.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_from_base(dt: datetime) -> float:
    """Hours from 2026-11-19T00:00Z to dt (negative for the reads before launch night's midnight)."""
    return (dt - BASE_TIME).total_seconds() / 3600.0


def clamp_rate(pct: float) -> float:
    """Clamp a printed rate into the range that has a logit, as the scorer does."""
    if pct < RATE_MIN:
        return RATE_MIN
    if pct > RATE_MAX:
        return RATE_MAX
    return pct


def logit_pct(pct: float) -> float:
    """Logit of an earn rate given in percent."""
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def pct_from_logit(lg: float) -> float:
    """Inverse of logit_pct, in percent, clamped so the result always has a logit."""
    if lg >= 9.2:            # logit(99.99) = 9.2101
        return RATE_MAX
    if lg <= -9.21:          # logit(0.01) = -9.2102
        return RATE_MIN
    return 100.0 / (1.0 + math.exp(-lg))


def weighted_median(values: list[float], weights: list[float]) -> float:
    """The weighted median: the value where the cumulative weight first reaches half of the total."""
    pairs = sorted(zip(values, weights), key=lambda vw: vw[0])
    total = 0.0
    for _, w in pairs:
        total += w
    if total <= 0.0:
        return 0.0
    half = total / 2.0
    run = 0.0
    for value, weight in pairs:
        run += weight
        if run >= half:
            return value
    return pairs[-1][0]


def round4(x: float) -> float:
    """x rounded to 4 decimals, half up, as the house baseline rounds (keeps outputs compact)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    return whole / 10000.0


def pinball_loss(y_pct: float, q_pct: float, tau: float) -> float:
    """The contest's pinball loss for one (trophy, level) item, on the logit scale.

    2 * max(tau * d, (tau - 1) * d) with d = logit(y) - logit(q). Exposed here so backtest.py can
    score exactly as the rules define it, without re-deriving the formula.
    """
    d = logit_pct(y_pct) - logit_pct(q_pct)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


# ---------------------------------------------------------------------------------------- case parsing


def parse_case(path: Path) -> dict:
    """Read one case file (the raw makefun/gta6-trophies-case/v1 JSON) into the model's flat form."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return case_from_obj(raw, str(raw.get("id") or path.stem))


def case_from_obj(raw: dict, case_id: str = "") -> dict:
    """Turn one case object into the flat structure the model works on.

    Returns a dict with:
      id            the case id (names the output)
      trophies      [{"id", "tier", "points"}] in the case's trophy order (the order of `rates`)
      reads         [(t_hours, [rate_pct or None, ...])] sorted by time, duplicate stamps collapsed
      t_last        the last read's time, in hours from BASE_TIME
      t_target      the target's time, in hours from BASE_TIME
      tracked_last  players tracked at the last read (None when the field is missing)
      horizon_days  (t_target - t_last) / 24, the extrapolation distance in days
    """
    case_id = case_id or str(raw.get("id") or "case")

    trophies = []
    for t in raw.get("trophies") or []:
        trophies.append({
            "id": t.get("id"),
            "tier": str(t.get("tier") or "bronze"),
            "points": t.get("points"),
        })

    reads: list[tuple[float, list]] = []
    for rec in raw.get("history") or []:
        try:
            t = hours_from_base(parse_time(rec["at"]))
        except (KeyError, TypeError, ValueError):
            continue
        rates = rec.get("rates") or []
        row = []
        for v in rates:
            row.append(None if v is None else float(v))
        reads.append((t, row))
    # Oldest first. A repeated timestamp (a retried read) keeps the later entry, which is the whole
    # page as it parsed, and a stable sort keeps the file's order among equal stamps.
    reads.sort(key=lambda r: r[0])
    deduped: list[tuple[float, list]] = []
    for t, row in reads:
        if deduped and abs(deduped[-1][0] - t) < 1e-9:
            deduped[-1] = (t, row)
        else:
            deduped.append((t, row))
    reads = deduped

    # The target: prefer the case's own `target`, fall back to `at` + horizon_minutes.
    t_target = None
    if raw.get("target"):
        try:
            t_target = hours_from_base(parse_time(raw["target"]))
        except (TypeError, ValueError):
            t_target = None
    if t_target is None:
        base = hours_from_base(parse_time(raw["at"])) if raw.get("at") else (reads[-1][0] if reads else 0.0)
        t_target = base + float(raw.get("horizon_minutes") or 0.0) / 60.0

    t_last = reads[-1][0] if reads else t_target
    tracked_last = None
    if reads:
        for rec in reversed(raw.get("history") or []):
            if rec.get("players_tracked") is not None:
                tracked_last = float(rec["players_tracked"])
                break

    return {
        "id": case_id,
        "trophies": trophies,
        "reads": reads,
        "t_last": t_last,
        "t_target": t_target,
        "tracked_last": tracked_last,
        "horizon_days": (t_target - t_last) / 24.0,
    }


def series_for_trophy(case: dict, index: int) -> list[tuple[float, float]]:
    """Every read that carries trophy `index`, as (t_hours, rate_pct), oldest first.

    Reads that lack the trophy (null) are dropped, so a trophy that appears late in the history is
    fitted on the reads that saw it. A rate is clamped into 0.01..99.99, the range that has a logit.
    """
    out = []
    for t, rates in case["reads"]:
        if index < len(rates) and rates[index] is not None:
            out.append((t, clamp_rate(rates[index])))
    return out


def last_rate(case: dict, index: int) -> tuple[float, bool]:
    """The trophy's rate in the most recent read that carries it, and whether any read carried it."""
    for t, rates in reversed(case["reads"]):
        if index < len(rates) and rates[index] is not None:
            return clamp_rate(rates[index]), True
    return EMPTY_RATE, False


# ----------------------------------------------------------------------------------------- growth fit


def _compress_staircase(points: list[tuple[float, float]], limit: int = 40) -> list[tuple[float, float]]:
    """Collapse a staircase to its change points: the first read at each new level.

    Exophase changes a rate only when it refreshes its counts, so most reads repeat the one before.
    Keeping only the first read at each level turns ~250 reads into ~10-40 genuinely distinct
    observations, which makes the robust slope cheap and stops a long plateau from dominating a
    weighted fit. If a series still has more change points than `limit`, keep the most recent ones:
    the near future is extrapolated from the recent past.
    """
    out: list[tuple[float, float]] = []
    for t, pct in points:
        if not out or abs(pct - out[-1][1]) > 1e-9:
            out.append((t, pct))
    if len(out) > limit:
        out = out[-limit:]
    return out


def _theil_sen(xs: list[float], ys: list[float], ws: list[float]) -> float:
    """Weighted median of all pairwise slopes: a robust line through a noisy staircase.

    A plain least-squares slope is dragged by a single bad read; the median of pairwise slopes is not,
    and on a staircase (where most points repeat) it reports the slope the refreshes actually imply.
    """
    slopes: list[float] = []
    weights: list[float] = []
    n = len(xs)
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[j] - xs[i]
            if dx <= 1e-12:
                continue
            slopes.append((ys[j] - ys[i]) / dx)
            weights.append(ws[i] * ws[j])
    if not slopes:
        return 0.0
    return weighted_median(slopes, weights)


def fit_growth(points: list[tuple[float, float]], t_last: float) -> dict:
    """Fit L(t) = a + b * ln(1 + t) to one trophy's reads, robustly and with recency weighting.

    `points` is [(t_hours, rate_pct)] for the reads that carry the trophy. Returns
        b       the fitted slope (logit units per ln-unit of time), clamped to [B_MIN, B_MAX]
        se_b    its standard error (inflated for serial correlation), or None when unknown
        resid   the robust scatter of the reads around the fitted line, in logit units
        n_eff   how many independent refreshes the slope rests on (drives the shrinkage)
        a       the fitted intercept, for diagnostics and the backtest
    """
    if len(points) < 2:
        return {"b": 0.0, "se_b": None, "resid": 0.0, "n_eff": 0.0, "a": None}

    cp = _compress_staircase(points)
    # x = ln(1 + t). t is clamped at -0.9 so a pre-midnight read cannot push the log to -inf.
    xs = [math.log1p(max(t, -0.9)) for t, _ in cp]
    ys = [logit_pct(p) for _, p in cp]
    ws = [max(MIN_WEIGHT, 0.5 ** ((t_last - t) / HALFLIFE_H)) for t, _ in cp]

    b = _theil_sen(xs, ys, ws)
    if not math.isfinite(b):
        b = 0.0
    b = min(B_MAX, max(B_MIN, b))

    # The intercept is a weighted median of (y - b x): robust to the read that jumped the most.
    intercept = weighted_median([y - b * x for x, y in zip(xs, ys)], ws)

    # Robust residual scale: 1.4826 * median |residual| is a standard deviation for a normal sample
    # and is not inflated by the one outlier a refresh glitch can leave.
    abs_res = [abs(y - (intercept + b * x)) for x, y in zip(xs, ys)]
    resid = 1.4826 * weighted_median(abs_res, ws) if abs_res else 0.0

    # Standard error of the slope from the weighted scatter of x around its weighted mean.
    wsum = sum(ws)
    x_bar = sum(w * x for w, x in zip(ws, xs)) / wsum if wsum > 0 else 0.0
    sxx = sum(w * (x - x_bar) ** 2 for w, x in zip(ws, xs))
    if sxx > 1e-9 and len(cp) >= 3:
        se_b = AUTOCORR * resid / math.sqrt(sxx)
    else:
        se_b = None

    # Independent refreshes: each change point is one new fact about the slope. The cap stops a long
    # history from claiming more certainty than a 3-day launch window can support.
    n_eff = float(min(12, max(1, len(cp))))

    return {"b": b, "se_b": se_b, "resid": resid, "n_eff": n_eff, "a": intercept}


def pool_slopes(fits: list[dict], trophies: list[dict], levels: list[float]) -> list[float]:
    """Shrink each trophy's slope towards a pooled slope for its kind.

    Two pools, averaged, because a staircase tells two different things about how a trophy grows:

    tier   fifty-five trophies share one progression curve, and a single staircase is noisy. Pooling by
           tier (platinum / gold / silver / bronze, which is also the points band: 300 / 90 / 30 / 15
           PSN points) then shrinking the tier pools towards the all-trophy pool means the platinum --
           a tier of one -- borrows the field's curve while the 46 bronzes keep their own.
    level  a slope in logit units is not the same size for every trophy: a trophy at 60% is close to
           its ceiling and can gain a few points a day, while a trophy at 0.05% multiplies. Regressing
           the fitted slopes on the trophies' own last levels captures that, and it is the only thing
           that can inform a trophy so rare its staircase is mostly counting noise.

    `levels` holds logit(last rate) per trophy. Returns b_hat, in the case's trophy order.
    """
    n = len(fits)
    if n == 0:
        return []

    all_b = [f["b"] for f in fits]
    all_w = [max(1.0, f["n_eff"]) for f in fits]
    global_pool = weighted_median(all_b, all_w)

    by_tier: dict[str, list[int]] = {}
    for i, t in enumerate(trophies):
        by_tier.setdefault(t["tier"], []).append(i)

    tier_pool: dict[str, float] = {}
    for tier, idx in by_tier.items():
        own = weighted_median([fits[i]["b"] for i in idx], [max(1.0, fits[i]["n_eff"]) for i in idx])
        m = float(len(idx))
        # A tier with many trophies trusts its own median; a tier with one or two borrows the field's.
        tier_pool[tier] = (m * own + TIER_PRIOR_TROPHIES * global_pool) / (m + TIER_PRIOR_TROPHIES)

    # Cross-sectional prior: slope against level, fitted only on trophies whose staircase carries
    # enough refreshes to say anything. Falls back to the tier pool alone when too few do.
    informative = [i for i in range(n) if fits[i]["n_eff"] >= 2.0]
    level_pred: list[float] | None = None
    tau2 = None
    if len(informative) >= 4:
        xs = [levels[i] for i in informative]
        ys = [fits[i]["b"] for i in informative]
        ws = [fits[i]["n_eff"] for i in informative]
        beta = _theil_sen(xs, ys, ws)
        alpha = weighted_median([y - beta * x for x, y in zip(xs, ys)], ws)
        if math.isfinite(alpha) and math.isfinite(beta):
            level_pred = [min(B_MAX, max(B_MIN, alpha + beta * levels[i])) for i in range(n)]
            # Between-trophy scatter of the slopes around that line: how far a real trophy's own
            # slope is from the prior, which is what sets how much a precise fit should be trusted.
            resid = [fits[i]["b"] - level_pred[i] for i in informative]
            mean_res = sum(r * w for r, w in zip(resid, ws)) / sum(ws)
            var = sum(w * (r - mean_res) ** 2 for r, w in zip(resid, ws)) / sum(ws)
            tau2 = max(TAU_MIN ** 2, var)

    b_hat: list[float] = []
    for i, f in enumerate(fits):
        pool = tier_pool.get(trophies[i]["tier"], global_pool)
        if level_pred is not None:
            pool = LEVEL_BLEND * level_pred[i] + (1.0 - LEVEL_BLEND) * pool
        # Inverse-variance shrinkage: a slope measured precisely keeps most of its own weight, one
        # measured off a nearly flat or nearly empty staircase is pulled onto the prior. This is what
        # stops the platinum -- whose rate is a handful of earners -- from being read as "no growth".
        se = f.get("se_b")
        if se is None or not math.isfinite(se) or se <= 1e-9 or tau2 is None:
            n_eff = f["n_eff"]
            shrunk = (n_eff * f["b"] + SHRINK_K * pool) / (n_eff + SHRINK_K)
        else:
            w_own = 1.0 / se ** 2
            w_prior = 1.0 / tau2
            shrunk = (w_own * f["b"] + w_prior * pool) / (w_own + w_prior)
        b_hat.append(min(B_MAX, max(B_MIN, shrunk)))
    return b_hat


# -------------------------------------------------------------------------------------------- forecast


def _extrapolation(case: dict) -> tuple[float, float]:
    """The distance the curve is extrapolated, in ln(1 + t) units, and in days."""
    t_last = case["t_last"]
    t_target = case["t_target"]
    x_last = math.log1p(max(t_last, -0.9))
    x_target = math.log1p(max(t_target, -0.9))
    return (x_target - x_last) * EXTRAP_SCALE, (t_target - t_last) / 24.0


def _spread(fit: dict, b_hat: float, dx: float, days_ahead: float, earners: float,
            use_floor: bool = True) -> float:
    """Total predictive spread of a trophy's logit at the target, in logit units.

    Three independent parts, added in quadrature:

    floor   the curve's shape may be wrong. Growth decelerates, but whether it decelerates like
            ln(1+t) or faster is not knowable from 2.5 days of launch data; a random walk in the
            logit at FLOOR_SIGMA per sqrt-day prices that in. It dominates at a 4.5-day horizon.
    slope   the fitted slope itself has a standard error; extrapolating dx ln-units multiplies it.
            The shrunk slope is more precise than the raw one, so the raw error is scaled by the
            share of the raw slope that survived the shrinkage.
    scatter the trophy's own reads wander around its curve (refresh jumps, counting noise), and the
            answering read is one draw of that wander. Plus the counting noise of the anchor itself:
            a rate backed by 4 earners is worth less than one backed by 400.
    """
    s_floor = max(FLOOR_SIGMA * math.sqrt(max(days_ahead, 0.0)), FLOOR_MIN) if use_floor else 0.0

    s_slope = 0.0
    se = fit.get("se_b")
    if se is not None and math.isfinite(se):
        raw = fit["b"]
        survived = (b_hat / raw) if raw > 1e-9 else 0.0
        s_slope = se * min(1.0, max(0.0, survived)) * abs(dx)

    s_scatter = 0.8 * min(fit.get("resid") or 0.0, 0.75)
    # Counting noise of the anchor: the relative standard error of a count is 1/sqrt(E).
    s_count = min(0.45, 1.0 / math.sqrt(earners)) if earners > 0 else 0.45

    sigma = math.sqrt(s_floor ** 2 + s_slope ** 2 + s_scatter ** 2 + s_count ** 2)
    return min(SIGMA_MAX, max(SIGMA_MIN, sigma))


def _persistence_row(pct: float, horizon_days: float) -> list[float]:
    """The house baseline's row: the rate stays put, with a random walk's spread of 0.5 per sqrt-day.

    Used when a trophy has no usable history (no read carries it, or the case has fewer than two
    reads), where there is nothing to fit and persistence is the honest answer.
    """
    lg = logit_pct(pct)
    spread = 0.5 * math.sqrt(max(horizon_days, 0.0))
    return [pct_from_logit(lg + z * spread) for z in Z]


def _finish_row(qs: list[float]) -> list[float]:
    """Clamp, round and enforce the output's shape: five finite, non-decreasing numbers in 0.01..99.99."""
    out: list[float] = []
    for q in qs:
        value = q if math.isfinite(q) else RATE_MIN
        value = min(RATE_MAX, max(RATE_MIN, value))
        value = round4(value)
        if out and value < out[-1]:
            value = out[-1]
        out.append(value)
    return out


def forecast_quantiles(case: dict, use_pool: bool = True, use_floor: bool = True) -> list[dict]:
    """Forecast every trophy of a parsed case: one {"id", "q"} row per trophy, in the case's order.

    `use_pool` and `use_floor` exist for backtest.py, which reports the ablation: the curve model
    without the pooled slope and without the random-walk floor is a different, worse forecast, and
    the backtest shows by how much. Production always calls with the defaults.
    """
    trophies = case["trophies"]
    dx, days_ahead = _extrapolation(case)
    tracked = case["tracked_last"]

    series = [series_for_trophy(case, i) for i in range(len(trophies))]
    fits = [fit_growth(s, case["t_last"]) for s in series]
    levels = [logit_pct(last_rate(case, i)[0]) for i in range(len(trophies))]
    b_hat = pool_slopes(fits, trophies, levels) if use_pool else [f["b"] for f in fits]

    rows: list[dict] = []
    for i, trophy in enumerate(trophies):
        pct_last, seen = last_rate(case, i)
        fit = fits[i]

        # Nothing to fit: persistence with the house baseline's spread.
        if not seen or len(series[i]) < 2 or len(case["reads"]) < 2:
            qs = _persistence_row(pct_last, days_ahead)
            rows.append({"id": trophy["id"], "q": _finish_row(qs)})
            continue

        lg_last = logit_pct(pct_last)
        b = b_hat[i]

        # A trophy pinned at the top cannot grow in logit terms, and one the case never showed above
        # the floor cannot fall. Both are handled by clamping the gain, not by trusting the slope.
        if pct_last >= SATURATED:
            b = 0.0
        gain = b * dx
        gain = min(MAX_LOGIT_GAIN, max(-MAX_LOGIT_GAIN, gain))

        lg_hat = lg_last + gain

        # Counting noise of the anchor: earners behind the last printed rate.
        earners = 1.0
        if tracked and tracked > 0:
            earners = max(1.0, pct_last / 100.0 * tracked)

        sigma = _spread(fit, b, dx, days_ahead, earners, use_floor=use_floor)
        qs = [pct_from_logit(lg_hat + z * sigma) for z in Z]
        rows.append({"id": trophy["id"], "q": _finish_row(qs)})

    return rows


def forecast_case(case: dict) -> dict:
    """The output object for one parsed case."""
    return {"trophies": forecast_quantiles(case)}


# ------------------------------------------------------------------------------------------------ main


def write_output(path: Path, forecast: dict) -> None:
    """Write one case's forecast: compact JSON, one object, trailing newline (as the starter writes)."""
    path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


def fallback_case(case: dict) -> dict:
    """A valid forecast from a case the model could not handle: persistence, the house baseline.

    A crash, a missing file or a malformed shape scores as if every quantile were the trophy's last
    rate -- the worst possible row. Writing the house baseline's row instead is strictly better than
    that, and it is the only thing left to say when the model has nothing to work with.
    """
    days = case.get("horizon_days") or 0.0
    rows = []
    for i, trophy in enumerate(case.get("trophies") or []):
        pct, _ = last_rate(case, i)
        rows.append({"id": trophy.get("id"), "q": _finish_row(_persistence_row(pct, days))})
    return {"trophies": rows}


def main() -> None:
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes in the same order.
    for path in sorted(input_dir.glob("*.in")):
        try:
            case = parse_case(path)
            forecast = forecast_case(case)
        except Exception:                                  # noqa: BLE001 - a valid row beats a crash
            try:
                forecast = fallback_case(parse_case(path))
            except Exception:                              # noqa: BLE001 - nothing readable at all
                continue
        write_output(output_dir / f"{path.stem}.out", forecast)


if __name__ == "__main__":
    main()
