#!/usr/bin/env python3
"""
Simulate careers: citation arrays of 100 researchers, snapshotted each year for 20 years.

    python scripts/simulate.py [--replicates 20]

Model (Wang-Song-Barabasi 2013 + Sinatra Q-model 2016):
  - Researcher quality Q_j ~ N(MU_Q, SIGMA_Q^2)
  - Publication rate lambda_j ~ Gamma(K_LAMBDA, THETA_LAMBDA) papers/year; the
    number of papers is Poisson(lambda_j * T) and publication times are uniform
  - Per-paper fitness eta_i = min(exp(Q_j + p_i), ETA_CAP), p_i ~ N(0, SIGMA_P^2)
  - Per-paper aging: immediacy mu_i ~ N(MU_MU, SIGMA_MU^2),
                     longevity sigma_i ~ |N(MU_SIGMA, SIGMA_SIGMA^2)| (floored at 0.01)
  - Cumulative citations c_i(t) = M * [exp(eta_i * Phi((ln t - mu_i) / sigma_i)) - 1]
  - Integer citations by rounding at integer years.

Replicate r uses seed SEED + r and is written to outputs/simulation/sim_seed<seed>.json.
Seed SEED is the run the generative parameters were calibrated on (calibrate_sim.py);
the script prints its year-T statistics, and their mean over all replicates, against
the real sample's.
"""

import argparse
import json
import math
import os

import numpy as np
from scipy.stats import norm

from data import OUTPUT_DIR, load_arrays, year_t_statistics

# === Generative parameters ===
# SIGMA_Q, SIGMA_P, ETA_CAP, M and THETA_LAMBDA were found by calibrate_sim.py
# (randomized search); the remaining parameters are fixed.
PARAMS = {
    # Researcher quality Q ~ N(MU_Q, SIGMA_Q^2)
    "MU_Q": 0.0,
    "SIGMA_Q": 0.32589182080547513,
    # Per-paper fitness noise p ~ N(0, SIGMA_P^2); eta = min(exp(Q+p), ETA_CAP)
    "SIGMA_P": 0.5671871073582164,
    "ETA_CAP": 6.492034463363131,
    # Lognormal aging (log-years): immediacy mu, longevity sigma
    "MU_MU": 1.5,
    "SIGMA_MU": 0.5,
    "MU_SIGMA": 0.3,
    "SIGMA_SIGMA": 0.2,
    # Global citation scale
    "M": 13.477908727209371,
    # Publication rate lambda ~ Gamma(K_LAMBDA, THETA_LAMBDA); mean papers/yr = K*THETA
    "K_LAMBDA": 6.0,
    "THETA_LAMBDA": 1.9823441386465899,
}

N_RESEARCHERS = 100
T_CAREER = 20  # years
SEED = 42
N_REPLICATES = 20

SIM_DIR = os.path.join(OUTPUT_DIR, "simulation")


def wsb_cumulative(t, eta, mu, sigma, M):
    """Expected cumulative citations at time t after publication."""
    if t <= 0:
        return 0.0
    z = (math.log(t) - mu) / sigma
    return M * (math.exp(eta * norm.cdf(z)) - 1.0)


def simulate(params=PARAMS, seed=SEED, n_researchers=N_RESEARCHERS, t_career=T_CAREER, years=None):
    """Simulate careers; `years` lists the snapshot years to record (default: 1..t_career)."""
    p = params
    years = range(1, t_career + 1) if years is None else years
    rng = np.random.default_rng(seed)
    researchers = []

    for j in range(n_researchers):
        # --- Researcher-level draws ---
        Q_j = rng.normal(p["MU_Q"], p["SIGMA_Q"])
        lambda_j = rng.gamma(p["K_LAMBDA"], p["THETA_LAMBDA"])

        # Publication times via a Poisson process
        n_papers = max(1, rng.poisson(lambda_j * t_career))
        pub_times = np.sort(rng.uniform(0, t_career, size=n_papers))

        # --- Per-paper draws ---
        p_i = rng.normal(0, p["SIGMA_P"], size=n_papers)
        eta_i = np.minimum(np.exp(Q_j + p_i), p["ETA_CAP"])
        mu_i = rng.normal(p["MU_MU"], p["SIGMA_MU"], size=n_papers)
        sigma_i = np.maximum(np.abs(rng.normal(p["MU_SIGMA"], p["SIGMA_SIGMA"], size=n_papers)), 0.01)

        # --- Yearly snapshots ---
        snapshots = {}
        for year in years:
            paper_citations = []
            for k in range(n_papers):
                age = year - pub_times[k]
                if age <= 0:
                    continue
                paper_citations.append(
                    round(wsb_cumulative(age, eta_i[k], mu_i[k], sigma_i[k], p["M"])))
            paper_citations.sort(reverse=True)
            snapshots[str(year)] = {"citation_array": paper_citations}

        researchers.append({
            "name": f"Researcher_{j + 1:03d}",
            "Q": round(Q_j, 4),
            "lambda": round(lambda_j, 4),
            "n_papers": int(n_papers),
            "snapshots": snapshots,
        })

    return researchers


def final_year_statistics(researchers, t_career=T_CAREER):
    """Year-T summary statistics, defined exactly as for the real sample."""
    return year_t_statistics([r["snapshots"][str(t_career)]["citation_array"] for r in researchers])


def main():
    ap = argparse.ArgumentParser(description="Simulate careers.")
    ap.add_argument("--replicates", type=int, default=N_REPLICATES,
                    help=f"independent runs, seeds {SEED}..{SEED}+R-1 (default {N_REPLICATES})")
    args = ap.parse_args()

    os.makedirs(SIM_DIR, exist_ok=True)
    stats = []
    for r in range(args.replicates):
        seed = SEED + r
        researchers = simulate(seed=seed)
        stats.append(final_year_statistics(researchers))
        path = os.path.join(SIM_DIR, f"sim_seed{seed}.json")
        with open(path, "w") as f:
            json.dump(researchers, f)
        print(f"  replicate {r + 1}/{args.replicates}: seed {seed} -> {os.path.basename(path)}")

    real = year_t_statistics(load_arrays())
    print(f"\nYear-{T_CAREER} statistics ({N_RESEARCHERS} simulated careers) against the real sample")
    print(f"{'':18s}{'seed ' + str(SEED):>12s}{'mean of R':>12s}{'real':>12s}")
    print("-" * 54)
    for k in real:
        mean = float(np.mean([s[k] for s in stats]))
        print(f"{k:18s}{stats[0][k]:12.1f}{mean:12.1f}{real[k]:12.1f}")


if __name__ == "__main__":
    main()
