#!/usr/bin/env python3
"""Calibrate the simulation's generative parameters to the real sample.

Randomized search over SIGMA_Q, SIGMA_P, ETA_CAP, M and THETA_LAMBDA of
simulate.py (the other parameters stay fixed), minimizing the weighted relative
error between the simulated year-T statistics and those of the real sample
(data.year_t_statistics). Each candidate is scored on the single run with seed
simulate.SEED, so the fit is to that one realization. Prints the best parameters
to paste into PARAMS in simulate.py.

    python scripts/calibrate_sim.py [--trials 400]
"""
import argparse

import numpy as np
import simulate as sim
from data import load_arrays, year_t_statistics

TARGETS = year_t_statistics(load_arrays())
KEYS = ["h_mean", "h_median", "h_std", "total_mean", "total_median",
        "papers_mean", "max_cite_over_h", "total_over_h2"]
# Weight of each statistic in the loss.
WEIGHTS = {"h_mean": 1.0, "h_median": 1.0, "h_std": 0.9, "total_mean": 0.7,
           "total_median": 0.7, "papers_mean": 0.5, "max_cite_over_h": 1.8,
           "total_over_h2": 1.0}
SEARCH_SEED = 2026


def year_t_stats(params):
    """Year-T statistics of one simulated run (seed simulate.SEED) with these parameters."""
    return sim.final_year_statistics(sim.simulate(params, years=[sim.T_CAREER]))


def loss(params):
    stats = year_t_stats(params)
    err = sum(WEIGHTS[k] * abs(stats[k] - TARGETS[k]) / abs(TARGETS[k]) for k in KEYS)
    return err, stats


def sample(rng, base):
    p = dict(base)
    p["SIGMA_Q"] = float(rng.uniform(0.30, 0.55))
    p["SIGMA_P"] = float(rng.uniform(0.55, 1.20))
    p["ETA_CAP"] = float(rng.uniform(6.0, 10.0))
    p["M"] = float(rng.uniform(5.0, 14.0))
    p["THETA_LAMBDA"] = float(rng.uniform(1.9, 2.2))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=400)
    args = ap.parse_args()

    rng = np.random.default_rng(SEARCH_SEED)
    best = None
    for i in range(args.trials):
        cand = sample(rng, sim.PARAMS)
        err, stats = loss(cand)
        if best is None or err < best[0]:
            best = (err, cand, stats)
            print(f"[{i:4d}] new best err={err:.4f}  "
                  f"h_mean={stats['h_mean']:.1f} h_median={stats['h_median']:.0f} "
                  f"total={stats['total_mean']:.0f} max/h={stats['max_cite_over_h']:.1f} "
                  f"total/h2={stats['total_over_h2']:.1f}")

    err, cand, stats = best
    print("\n=== BEST PARAMS (paste into PARAMS in simulate.py, full precision) ===")
    for k in ["MU_Q", "SIGMA_Q", "SIGMA_P", "ETA_CAP", "MU_MU", "SIGMA_MU",
              "MU_SIGMA", "SIGMA_SIGMA", "M", "K_LAMBDA", "THETA_LAMBDA"]:
        print(f'    "{k}": {cand[k]!r},')
    print(f"\nweighted err = {err:.4f}")
    print(f"{'stat':18s}{'sim':>12s}{'target':>12s}")
    for k in KEYS:
        print(f"{k:18s}{stats[k]:12.1f}{TARGETS[k]:12.1f}")


if __name__ == "__main__":
    main()
