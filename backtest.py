"""Backtest and tuning harness for the trophy-rarity forecaster in main.py.

WHAT IT DOES
    For a case file (a practice case, or the final once it exists) it cuts the history at several
    checkpoints, forecasts from each cut, and scores the forecast against the real read that follows,
    exactly as the rules score it: pinball loss on the logit scale at the five levels, plus the 90%
    coverage and the typical miss. Three models are compared:

      persistence  the house baseline: the last rate, spread 0.5 * sqrt(days) on the logit
      curve        the per-trophy growth curve alone: no pooled slope, no random-walk floor
      final        main.py as it ships: pooled slopes, full three-part spread

WHAT THE PRACTICE CASES CAN AND CANNOT SHOW
    Both practice cases (p00, p01) are cut from launch night, and every read in their histories is
    the same unchanged page: Exophase had not refreshed the counts since before launch, so the whole
    history is one flat staircase. On a flat history every model collapses to persistence, every
    within-history backtest scores 0 (the reads it predicts are identical to the reads it fit), and
    the only thing the practice can check is that the program reads a case and writes a valid file.
    Their answers (the refreshed page 90 minutes later) are not in the file at all, so no cut of the
    history can reach them. This backtest therefore reports the practice cases as a smoke test and
    says so, rather than pretending they calibrate anything.

WHAT CARRIES THE CALIBRATION
    A synthetic launch week, built to look like the real recording: a tracked-player pool that grows
    like a power law (638 players at launch night, ~4,000 four hours later, a few hundred thousand by
    day seven), a per-trophy curve that is a probit in log time (slow at launch, saturating later),
    Exophase refreshing the whole page on its own clock so the reads are a staircase, and each printed
    rate rounded to 2 decimals with binomial counting noise. The model is fitted on that staircase and
    scored on the read at the target, which the simulator produces independently of the model's
    functional form. That is a real out-of-form test, but it is a test against MY priors, not against
    the world: it can show that the spread is not absurd, that pooling beats a per-trophy fit, and
    that the floor is in the right range. It cannot show that a launch week's rates really do grow
    like this, and the constants below were chosen to be robust across the whole scenario family
    rather than to win any one of them.

RUN IT
    python3 backtest.py                          # the practice cases + the synthetic family
    python3 backtest.py path/to/f00.in           # also backtest a real case file
    python3 backtest.py --tune                   # sweep the spread floor, print the loss/coverage table
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone

import main as model

BASE = model.BASE_TIME


# ------------------------------------------------------------------------------------------ scoring


def score(truth: list[float], rows: list[dict]) -> dict:
    """Score a forecast against a real read, exactly as the rules define the loss.

    truth: the answering read's rates, in the case's trophy order.
    rows:  [{"id", "q"}] from the model.
    Returns the mean pinball loss over every (trophy, level) item, the 90% coverage and the typical
    miss (mean |y - median| in percentage points).
    """
    by_id = {row["id"]: row["q"] for row in rows}
    total = 0.0
    items = 0
    covered = 0
    miss = 0.0
    trophies = 0
    for i, y in enumerate(truth):
        if y is None:
            continue
        row = by_id.get(i + 1)          # trophy id is 1-based position
        if row is None:
            continue
        for tau, q in zip(model.LEVELS, row):
            total += model.pinball_loss(y, q, tau)
            items += 1
        if row[0] <= model.clamp_rate(y) <= row[-1]:
            covered += 1
        miss += abs(model.clamp_rate(y) - row[2])
        trophies += 1
    if items == 0:
        return {"loss": float("nan"), "coverage": float("nan"), "miss": float("nan"), "n": 0}
    return {
        "loss": total / items,
        "coverage": covered / trophies,
        "miss": miss / trophies,
        "n": trophies,
    }


# ---------------------------------------------------------------------------------------- truncation


def truncate(case: dict, t_cut: float, t_target: float) -> dict:
    """A copy of a parsed case whose history stops at t_cut and whose target is t_target."""
    reads = [(t, r) for t, r in case["reads"] if t <= t_cut + 1e-9]
    out = dict(case)
    out["reads"] = reads
    out["t_last"] = reads[-1][0] if reads else t_cut
    out["t_target"] = t_target
    out["horizon_days"] = (t_target - out["t_last"]) / 24.0
    out["tracked_last"] = case.get("tracked_last")
    return out


def predict(case: dict, which: str) -> list[dict]:
    """Run one of the three models on a parsed case."""
    if which == "persistence":
        rows = []
        for i, trophy in enumerate(case["trophies"]):
            pct, _ = model.last_rate(case, i)
            rows.append({"id": trophy["id"],
                         "q": model._finish_row(model._persistence_row(pct, case["horizon_days"]))})
        return rows
    if which == "curve":
        return model.forecast_quantiles(case, use_pool=False, use_floor=False)
    return model.forecast_quantiles(case)


MODELS = ("persistence", "curve", "final")


# ------------------------------------------------------------------------------------- real-case run


def read_at(case: dict, t_want: float, tol_h: float = 0.5):
    """The read closest to t_want, if one exists within tol_h hours."""
    best = None
    for t, rates in case["reads"]:
        if abs(t - t_want) <= tol_h and (best is None or abs(t - t_want) < abs(best[0] - t_want)):
            best = (t, rates)
    return best


def backtest_real(path: str) -> list[dict]:
    """Within-history backtest of a real case: cut at fractions of the history, score the next read.

    Only reads that exist in the file can be scored, so the horizon is limited by how much history
    the case holds. On the practice cases that horizon is minutes and the history is flat.
    """
    case = model.parse_case(model.Path(path))
    times = [t for t, _ in case["reads"]]
    if len(times) < 4:
        print(f"{path}: only {len(times)} reads, nothing to cut")
        return []
    span = times[-1] - times[0]
    results = []
    print(f"\n== real case {case['id']}: {len(times)} reads, "
          f"{times[0]:.2f} h .. {times[-1]:.2f} h from base, "
          f"{len(case['trophies'])} trophies, horizon {case['horizon_days'] * 24:.0f} min")
    for frac in (0.5, 0.75, 0.9):
        t_cut = times[0] + span * frac
        for ahead_h in (0.5, 1.5, 24.0):
            hit = read_at(case, t_cut + ahead_h)
            if hit is None or hit[0] <= t_cut:
                continue
            t_true, truth = hit
            sub = truncate(case, t_cut, t_true)
            line = f"  cut {t_cut:7.2f} h -> read {t_true:7.2f} h (+{t_true - t_cut:5.2f} h)"
            for which in MODELS:
                s = score(truth, predict(sub, which))
                line += f" | {which}: loss {s['loss']:.3f} cov {s['coverage']:.2f} miss {s['miss']:.3f}"
                results.append({"case": case["id"], "model": which, "ahead_h": t_true - t_cut, **s})
            print(line)
    return results


# -------------------------------------------------------------------------------- synthetic launch week


def _phi(z: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun 7.1.26 via erf, exact enough at 1e-7)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


TIERS = (["platinum"] + ["gold"] * 2 + ["silver"] * 6 + ["bronze"] * 46)
POINTS = {"platinum": 300, "gold": 90, "silver": 30, "bronze": 15}


def simulate_case(seed: int = 7, *, decel: float = 0.643, decel_late: float | None = None,
                  break_h: float = 60.0, pool_q: float = 1.8, pool_tau: float = 4.0,
                  refresh_min: int = 75, depth_scale: float = 1.0,
                  dilution_at: float | None = None, dilution: float = 0.6,
                  noise: bool = True) -> dict:
    """Simulate a launch week of Exophase reads, in the shape of a real case file.

    The world it draws from is deliberately NOT the model's functional form:

      * the tracked pool grows as N(t) = 638 * (1 + t/pool_tau)^pool_q (638 at launch night, ~4,000
        by 04:30, a few hundred thousand by day seven);
      * trophy i needs `depth` hours of play, and the share of tracked players who have reached it
        is  p_i(t) = p_inf_i * Phi(z_i(t))  with  z_i(t) = decel * ln((1+t)/(1+depth_i)) + z0,
        z0 = Phi^-1(0.01): a probit in log time, 1% of its eventual rate when a player has played
        `depth` hours, saturating as the pool matures. `decel` sets how fast it decelerates, and
        `decel_late` (from `break_h` on, kept continuous) lets the world change regime after the
        checkpoint, which is the model's real risk: a power law fitted to the history cannot know;
      * `dilution_at` / `dilution`: a refresh that adds a rush of players and drops every rate by
        that factor, as launch night's first refresh did (3,868 players tracked and every rate but
        the 100% one lower than before);
      * Exophase refreshes the whole page every `refresh_min` minutes, so a read shows the counts as
        of the last refresh and the series is a staircase;
      * each printed rate is binomial counting noise on N tracked players, rounded to 2 decimals.

    Returns a case object (the same shape parse_case produces) plus "truth", the rates at the
    target read, and "all_reads" so different checkpoints can be cut from one simulation.
    """
    rng = random.Random(seed)
    n_trophies = len(TIERS)
    late = decel if decel_late is None else decel_late

    # Depth (hours of play) and eventual reach of each trophy.
    depth, p_inf = [], []
    for i, tier in enumerate(TIERS):
        if tier == "platinum":
            d, p = 42.0, 0.020
        elif tier == "gold":
            d, p = 22.0 + 6.0 * i, 0.32
        elif tier == "silver":
            d, p = 9.0 + 1.5 * i, 0.48
        else:
            d, p = 1.5 + 0.28 * i, 0.55 + 0.30 * rng.random()
        depth.append(max(0.5, d * depth_scale))
        p_inf.append(min(0.97, p))

    z0 = -2.3263478740408408          # Phi^-1(0.01)

    def pool(t: float) -> float:
        return 638.0 * (1.0 + max(t, 0.0) / pool_tau) ** pool_q

    def true_rate(t: float, i: int) -> float:
        """The underlying share of tracked players holding trophy i at time t."""
        tt = max(t, 0.0)
        early = min(tt, break_h)
        z = decel * math.log((1.0 + early) / (1.0 + depth[i])) + z0
        if tt > break_h:
            z += late * math.log((1.0 + tt) / (1.0 + break_h))
        p = p_inf[i] * _phi(z)
        if dilution_at is not None and tt >= dilution_at:
            p *= dilution
        return p

    # Refresh schedule: every refresh_min minutes from t = -0.2 h.
    refresh_times = []
    t = -0.2
    while t <= 200.0:
        refresh_times.append(t)
        t += refresh_min / 60.0

    def rate_at_read(t: float, i: int) -> float:
        """The printed rate a read at time t shows: the value at the last refresh, with noise."""
        last = refresh_times[0]
        for rt in refresh_times:
            if rt <= t:
                last = rt
            else:
                break
        p = true_rate(last, i)
        n = pool(last)
        if noise:
            mean = n * p
            sd = math.sqrt(max(mean * (1.0 - p), 0.0))
            earners = max(0, int(round(mean + sd * rng.gauss(0.0, 1.0))))
            return round(100.0 * earners / n, 2)
        return round(100.0 * p, 2)

    # Reads every 10 minutes from just before midnight to 200 h, plus one exactly at 168 h.
    read_times = []
    t = -0.1667
    while t <= 200.0:
        read_times.append(round(t, 4))
        t += 10.0 / 60.0
    read_times.append(168.0)

    history = []
    for t in sorted(read_times):
        rates = [rate_at_read(t, i) for i in range(n_trophies)]
        history.append({
            "at": (BASE + timedelta(hours=t)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "slot": (BASE + timedelta(hours=t)).strftime("%Y-%m-%dT%H:%M:00Z"),
            "players_tracked": int(round(pool(t))),
            "trophies_earned": 0,
            "rates": rates,
        })

    truth_idx = len(history) - 1
    trophies = [
        {"id": i + 1, "name": f"trophy {i + 1}", "tier": TIERS[i], "points": POINTS[TIERS[i]],
         "secret": False, "description": ""}
        for i in range(n_trophies)
    ]
    return {
        "id": f"sim-s{seed}-d{decel}",
        "trophies": trophies,
        "history": history,
        "truth_index": truth_idx,
        "truth": history[truth_idx]["rates"],
        "all_reads": history,
    }


def _case_from_sim(sim: dict, t_cut: float, t_target: float) -> tuple[dict, list[float]]:
    """Cut a simulated week at t_cut and return a parsed case plus the truth at t_target."""
    rows = []
    truth = None
    for rec in sim["all_reads"]:
        t = model.hours_from_base(model.parse_time(rec["at"]))
        if t <= t_cut + 1e-9:
            rows.append(rec)
        if abs(t - t_target) < 1e-6:
            truth = rec["rates"]
    raw = {
        "id": sim["id"],
        "at": (BASE + timedelta(hours=t_cut)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target": (BASE + timedelta(hours=t_target)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_minutes": int(round((t_target - t_cut) * 60)),
        "levels": list(model.LEVELS),
        "trophies": sim["trophies"],
        "history": rows,
    }
    return model.case_from_obj(raw, sim["id"]), truth


SCENARIOS = (
    # name, kwargs for simulate_case
    ("slow growth (strong deceleration)", dict(decel=0.45, pool_q=1.6, refresh_min=90)),
    ("central", dict(decel=0.643)),
    ("fast growth (weak deceleration)", dict(decel=0.90, pool_q=2.1, refresh_min=60)),
    ("shallow trophies (short game)", dict(decel=0.643, depth_scale=0.6)),
    ("deep trophies (long game)", dict(decel=0.643, depth_scale=1.5)),
    # The three that matter for the spread: the world changes after the checkpoint, so the power law
    # fitted to the history is structurally wrong and only the floor can pay for it.
    ("saturates after day 2.5", dict(decel=0.643, decel_late=0.25, break_h=59.5)),
    ("re-accelerates after day 2.5", dict(decel=0.643, decel_late=1.15, break_h=59.5)),
    ("dilution refresh on launch night", dict(decel=0.643, dilution_at=4.5, dilution=0.55)),
)


def backtest_synthetic(seeds=(7, 11), cuts=(36.0, 48.0, 59.5), aheads=(24.0, 48.0, 108.5)) -> list[dict]:
    """Fit on a simulated launch week cut at `cut` hours, score the read `ahead` hours later."""
    results = []
    print("\n== synthetic launch week: pinball loss / 90% coverage / typical miss (points)")
    for name, kw in SCENARIOS:
        print(f"\n  -- {name}")
        for seed in seeds:
            sim = simulate_case(seed, **kw)
            for cut in cuts:
                for ahead in aheads:
                    target = cut + ahead
                    if target > 168.0:
                        continue
                    case, truth = _case_from_sim(sim, cut, target)
                    if truth is None:
                        continue
                    line = f"    cut {cut:5.1f} h  +{ahead:6.1f} h ({ahead / 24:.1f} d)"
                    for which in MODELS:
                        s = score(truth, predict(case, which))
                        line += f" | {which[:5]}: {s['loss']:5.3f}/{s['coverage']:.2f}/{s['miss']:5.2f}"
                        results.append({"scenario": name, "seed": seed, "model": which,
                                        "cut": cut, "ahead_h": ahead, **s})
                    print(line)
    return results


def summarise(results: list[dict], key: str = "model") -> None:
    """Mean loss and coverage per model, over the synthetic family only."""
    sims = [r for r in results if "scenario" in r]
    if not sims:
        return
    print("\n== summary over the synthetic family (mean of every scenario, seed and horizon)")
    print(f"   {'model':12s} {'mean loss':>10s} {'90% coverage':>14s} {'typical miss':>13s}")
    for which in MODELS:
        sel = [r for r in sims if r["model"] == which]
        if not sel:
            continue
        print(f"   {which:12s} {sum(r['loss'] for r in sel) / len(sel):10.3f} "
              f"{sum(r['coverage'] for r in sel) / len(sel):14.2f} "
              f"{sum(r['miss'] for r in sel) / len(sel):13.2f}")


# ------------------------------------------------------------------------------------------- tuning


def tune(floors=(0.0, 0.25, 0.35, 0.45, 0.55, 0.70), scales=(0.0, 0.5, 1.0, 1.5)) -> None:
    """Sweep the spread floor (logit per sqrt-day) and the extrapolation scale.

    Loss and coverage are averaged over the whole synthetic family. A floor that is too small loses
    coverage and pays at the 5% and 95% quantiles; one that is too large pays everywhere. The scale
    multiplies the extrapolation distance, so 1.0 is the honest power law and < 1 damps it.
    """
    sims = []
    for name, kw in SCENARIOS:
        for seed in (7, 11):
            sims.append(simulate_case(seed, **kw))
    old_floor, old_scale = model.FLOOR_SIGMA, getattr(model, "EXTRAP_SCALE", 1.0)
    print("\n== tuning the spread floor and the extrapolation scale (synthetic family)")
    print(f"   {'floor':>6s} {'scale':>6s} {'mean loss':>10s} {'90% cov':>8s} {'mean |err|':>11s}")
    try:
        for scale in scales:
            model.EXTRAP_SCALE = scale
            for floor in floors:
                model.FLOOR_SIGMA = floor
                losses, covs, errs = [], [], []
                for sim in sims:
                    for cut in (36.0, 48.0, 59.5):
                        for ahead in (24.0, 48.0, 108.5):
                            if cut + ahead > 168.0:
                                continue
                            case, truth = _case_from_sim(sim, cut, cut + ahead)
                            if truth is None:
                                continue
                            s = score(truth, predict(case, "final"))
                            losses.append(s["loss"])
                            covs.append(s["coverage"])
                            errs.append(s["miss"])
                print(f"   {floor:6.2f} {scale:6.2f} {sum(losses) / len(losses):10.3f} "
                      f"{sum(covs) / len(covs):8.2f} {sum(errs) / len(errs):11.2f}")
    finally:
        model.FLOOR_SIGMA, model.EXTRAP_SCALE = old_floor, old_scale


# ---------------------------------------------------------------------------------------------- main


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cases", nargs="*", help="real case files to backtest as well")
    ap.add_argument("--tune", action="store_true", help="sweep the spread floor and extrapolation scale")
    ap.add_argument("--no-synthetic", action="store_true", help="skip the simulated launch week")
    args = ap.parse_args(argv[1:])

    results: list[dict] = []
    for path in args.cases:
        results.extend(backtest_real(path))
    if not args.no_synthetic:
        results.extend(backtest_synthetic())
        summarise(results)
    if args.tune:
        tune()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
