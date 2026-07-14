#!/usr/bin/env python3
"""B3 — CharlieChaplinAI survival across seeds.

The demo (`run_demo.py`) shows a single frozen-VLA-vs-megan-tk survival run on the
rocking ship. A single run is not a citable result. This sweep re-runs the SAME
real pipeline across N seeds and reports the survival / uptime delta as a mean ±
95% CI, so the "megan-tk keeps the pyramid standing" claim is a distribution, not
one lucky roll.

No mocks: for each seed it calls the real `shipyard.demo.run(..., render=False)`,
which runs the full thing under MuJoCo — the efficiency governor learns the max
safe rebuild speed, the NS governor diagnoses the OSCILLATION and kicks, the
anticipator learns the lurch rhythm, the agent discovers the best brace, and the
trained VLA policy then drives a frozen arm and a megan-tk arm through the same
lurches. We aggregate the two survival numbers it returns:

  * mean_integrity — average fraction of the pyramid standing over the run
  * uptime         — fraction of time the tower is essentially whole

Run (cadenza venv — needs mujoco + torch):
    PYTHONPATH=. /Users/akshparekh/Documents/cadenza/.venv/bin/python sweep_survival.py \
        [--seeds 5] [--episodes 8] [--duration 60]
    -> prints the aggregated table, writes out/survival_sweep.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from shipyard.demo import run

HERE = Path(__file__).resolve().parent


def _ci95(xs: list[float]) -> tuple[float, float]:
    a = np.asarray(xs, dtype=np.float64)
    mean = float(a.mean())
    if len(a) < 2:
        return mean, 0.0
    sem = float(a.std(ddof=1) / math.sqrt(len(a)))
    return mean, 1.96 * sem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    per_seed = []
    with tempfile.TemporaryDirectory() as tmp:
        for s in seeds:
            # draw a distinct sea state per seed — the periodic ship motion is
            # deterministic, so varying the disturbance (lurch cadence/strength,
            # sway) is what makes each seed a genuinely different run. Seeded, so
            # the whole sweep is reproducible.
            rng = np.random.default_rng(1000 + s)
            sea = dict(
                lurch_every=float(rng.uniform(8.0, 12.0)),
                lurch_amp=float(rng.uniform(0.20, 0.30)),
                sway_period=float(rng.uniform(1.9, 2.6)),
                sway_amp=float(rng.uniform(0.035, 0.055)),
            )
            print(f"[seed {s}] running full real pipeline "
                  f"(episodes={args.episodes}, duration={args.duration}s, "
                  f"lurch~{sea['lurch_every']:.1f}s @ {sea['lurch_amp']:.2f})...",
                  flush=True)
            # the demo prints three acts of narration per run — silence it so the
            # sweep output is just the per-seed result line and the final table.
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                res = run(Path(tmp) / f"seed{s}", episodes=args.episodes,
                          duration=args.duration, render=False, seed=s, **sea)
            fr, gv = res["survival_frozen"], res["survival_megantk"]
            row = {
                "seed": s, **{f"sea_{k}": round(v, 3) for k, v in sea.items()},
                "frozen_integrity": fr["mean_integrity"], "frozen_uptime": fr["uptime"],
                "megantk_integrity": gv["mean_integrity"], "megantk_uptime": gv["uptime"],
                "lurches": fr["lurches"],
            }
            per_seed.append(row)
            print(f"  seed {s}: frozen {fr['mean_integrity']:.2f}/"
                  f"{fr['uptime']:.2f}   megan-tk {gv['mean_integrity']:.2f}/"
                  f"{gv['uptime']:.2f}   ({fr['lurches']} lurches)", flush=True)

    fi = _ci95([r["frozen_integrity"] for r in per_seed])
    fu = _ci95([r["frozen_uptime"] for r in per_seed])
    gi = _ci95([r["megantk_integrity"] for r in per_seed])
    gu = _ci95([r["megantk_uptime"] for r in per_seed])
    di = _ci95([r["megantk_integrity"] - r["frozen_integrity"] for r in per_seed])
    du = _ci95([r["megantk_uptime"] - r["frozen_uptime"] for r in per_seed])

    print(f"\nB3 — survival across {args.seeds} seeds "
          f"(duration {args.duration}s, mean ± 95% CI)\n")
    print(f"  {'':14s} {'mean integrity':>20s} {'uptime':>20s}")
    print("  " + "-" * 56)
    print(f"  {'frozen VLA':14s} {fi[0]:8.2f} ± {fi[1]:5.2f}      "
          f"{fu[0]:8.2f} ± {fu[1]:5.2f}")
    print(f"  {'with megan-tk':14s} {gi[0]:8.2f} ± {gi[1]:5.2f}      "
          f"{gu[0]:8.2f} ± {gu[1]:5.2f}")
    print("  " + "-" * 56)
    print(f"  {'Δ (megan-tk −':14s} {di[0]:+8.2f} ± {di[1]:5.2f}      "
          f"{du[0]:+8.2f} ± {du[1]:5.2f}")
    print(f"  {'  frozen)':14s}")

    summary = {
        "seeds": args.seeds, "duration": args.duration, "episodes": args.episodes,
        "per_seed": per_seed,
        "frozen":  {"mean_integrity": fi[0], "mean_integrity_ci95": fi[1],
                    "uptime": fu[0], "uptime_ci95": fu[1]},
        "megantk": {"mean_integrity": gi[0], "mean_integrity_ci95": gi[1],
                    "uptime": gu[0], "uptime_ci95": gu[1]},
        "delta":   {"mean_integrity": di[0], "mean_integrity_ci95": di[1],
                    "uptime": du[0], "uptime_ci95": du[1]},
    }
    out_dir = HERE / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "survival_sweep.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
