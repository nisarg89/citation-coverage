#!/usr/bin/env python3
"""
Longitudinal metrics on the simulated careers, aggregated over independent
replicates with 95% confidence intervals.

    python scripts/simulate.py --replicates 20
    python scripts/simulation_cis.py

For every candidate index and component this computes, on each replicate:
coverage (mean / median / min over the researchers) by career year; the mean
component value by year; stability (fraction of the 19 year-over-year
transitions in which the component changes, and the mean maximum single-year
change); the fraction of careers containing a strict decrease; ranking
stability (Kendall tau between the rankings at years x and x+1); and, at year 20,
Kendall tau with the generative parameters Q and lambda and with h. Each quantity
is reported as the mean over replicates with a 95% confidence interval for that
mean (t-based). Results go to outputs/simulation/replicate_cis.json; the coverage
curves are also printed as pgfplots coordinate streams.
"""

import glob
import json
import os
import sys

import numpy as np
from scipy.stats import kendalltau, t as tdist

from data import OUTPUT_DIR
from indices import (
    CANDIDATE_INDICES, h_index, i10_index, compute_component,
    coverage_3tuple, h_only_coverage, i10_coverage,
)

REPLICATE_DIR = os.path.join(OUTPUT_DIR, "simulation")
OUT_PATH = os.path.join(REPLICATE_DIR, "replicate_cis.json")
YEARS = list(range(1, 21))


def ci95(samples):
    """Mean and half-width of a 95% CI for the mean, over replicates."""
    a = np.asarray([s for s in samples if np.isfinite(s)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1:
        return float(a[0]), 0.0
    half = tdist.ppf(0.975, a.size - 1) * a.std(ddof=1) / np.sqrt(a.size)
    return float(a.mean()), float(half)


def safe_tau(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(kendalltau(x, y)[0])


def metrics_for_run(researchers):
    """All longitudinal metrics for one replicate. Returns nested dicts."""
    n = len(researchers)
    Q = np.array([r["Q"] for r in researchers])
    lam = np.array([r["lambda"] for r in researchers])

    out = {}
    for idx in CANDIDATE_INDICES:
        name = idx["name"]
        h_a = np.zeros((len(YEARS), n))
        top_a = np.zeros((len(YEARS), n))
        right_a = np.zeros((len(YEARS), n))
        cov_a = np.zeros((len(YEARS), n))
        for yi, yr in enumerate(YEARS):
            for ri, r in enumerate(researchers):
                arr = r["snapshots"][str(yr)]["citation_array"]
                h = h_index(arr)
                h_a[yi, ri] = h
                if name == "h-index":
                    cov_a[yi, ri] = h_only_coverage(arr, h)
                elif name == "i10-index":
                    v = i10_index(arr)
                    top_a[yi, ri] = v
                    cov_a[yi, ri] = i10_coverage(arr, v)
                else:
                    tp = compute_component(arr, idx["top"], h, "top")
                    rt = compute_component(arr, idx["right"], h, "right")
                    top_a[yi, ri] = tp
                    right_a[yi, ri] = rt
                    cov_a[yi, ri] = coverage_3tuple(arr, tp, h, rt, idx["top"], idx["right"])

        rec = {
            "cov_mean": [float(cov_a[y].mean()) for y in range(len(YEARS))],
            "cov_median": [float(np.median(cov_a[y])) for y in range(len(YEARS))],
            "cov_min": [float(cov_a[y].min()) for y in range(len(YEARS))],
        }
        comps = {"h": h_a}
        if name == "i10-index":
            comps = {"i10": top_a}
        elif name != "h-index":
            comps = {"top": top_a, "h": h_a, "right": right_a}
        for cname, data in comps.items():
            chg, mx, viol = [], [], []
            for ri in range(n):
                d = np.diff(data[:, ri])
                chg.append(float((np.abs(d) > 0.5).mean()))
                mx.append(float(np.abs(d).max()) if d.size else 0.0)
                viol.append(1.0 if (d < -0.5).any() else 0.0)
            rec[f"meanval_{cname}"] = [float(data[y].mean()) for y in range(len(YEARS))]
            # ranking stability: tau between consecutive years, per year x
            rec[f"rankstab_{cname}"] = [
                safe_tau(data[y], data[y + 1]) for y in range(len(YEARS) - 1)
            ]
            rec[f"chgfreq_{cname}"] = float(np.mean(chg))
            rec[f"maxchg_{cname}"] = float(np.mean(mx))
            rec[f"violfrac_{cname}"] = float(np.mean(viol))
            rec[f"tauQ_{cname}"] = safe_tau(data[-1], Q)
            rec[f"taulam_{cname}"] = safe_tau(data[-1], lam)
            rec[f"tauh_{cname}"] = safe_tau(data[-1], h_a[-1])
        out[name] = rec
    return out


def main():
    files = sorted(glob.glob(os.path.join(REPLICATE_DIR, "sim_seed*.json")))
    if not files:
        sys.exit(f"No replicates found in {REPLICATE_DIR}.\n"
                 f"Run: python scripts/simulate.py --replicates 20")
    print(f"Aggregating {len(files)} replicates from {REPLICATE_DIR}\n")

    runs = []
    for i, f in enumerate(files, 1):
        with open(f) as fh:
            runs.append(metrics_for_run(json.load(fh)))
        print(f"  [{i}/{len(files)}] {os.path.basename(f)}")

    R = len(runs)
    agg = {"n_replicates": R, "years": YEARS, "candidates": {}}
    for idx in CANDIDATE_INDICES:
        name = idx["name"]
        rec = {}
        for k, v0 in runs[0][name].items():
            if isinstance(v0, list):
                means, halves = [], []
                for yi in range(len(v0)):
                    m, hw = ci95([r[name][k][yi] for r in runs])
                    means.append(m); halves.append(hw)
                rec[k] = {"mean": means, "ci95": halves}
            else:
                m, hw = ci95([r[name][k] for r in runs])
                rec[k] = {"mean": m, "ci95": hw}
        agg["candidates"][name] = rec

    with open(OUT_PATH, "w") as f:
        json.dump(agg, f, indent=1)

    print(f"\n=== Year-20 coverage over R={R} replicates (mean +/- 95% CI) ===\n")
    print(f"{'candidate':<14}{'mean cov':>20}{'median cov':>20}{'min cov':>20}")
    for idx in CANDIDATE_INDICES:
        c = agg["candidates"][idx["name"]]
        cells = [f"{c[k]['mean'][-1]:.4f} +/- {c[k]['ci95'][-1]:.4f}"
                 for k in ("cov_mean", "cov_median", "cov_min")]
        print(f"{idx['name']:<14}" + "".join(f"{x:>20}" for x in cells))

    print(f"\n=== Year-20 component metrics (mean +/- 95% CI over R={R}) ===\n")
    print(f"{'candidate':<14}{'comp':<7}{'chg freq':>18}{'max |d|':>18}"
          f"{'tau(Q)':>18}{'tau(lambda)':>18}{'% careers w/ drop':>20}")
    for idx in CANDIDATE_INDICES:
        c = agg["candidates"][idx["name"]]
        for cname in ("top", "h", "right", "i10"):
            if f"chgfreq_{cname}" not in c:
                continue
            row = [f"{c[f'{m}_{cname}']['mean']:.3f} +/- {c[f'{m}_{cname}']['ci95']:.3f}"
                   for m in ("chgfreq", "maxchg", "tauQ", "taulam", "violfrac")]
            print(f"{idx['name']:<14}{cname:<7}" + "".join(f"{x:>18}" for x in row[:4])
                  + f"{row[4]:>20}")

    print("\n=== pgfplots coordinate streams for the coverage figure "
          "(years 3-20, with +- 95% CI) ===")
    for stat in ("cov_mean", "cov_median", "cov_min"):
        print(f"\n% {stat}")
        for idx in CANDIDATE_INDICES:
            c = agg["candidates"][idx["name"]][stat]
            stream = "".join(
                f"({y},{c['mean'][y-1]:.4f}) +- (0,{c['ci95'][y-1]:.4f})"
                for y in range(3, 21))
            print(f"% {idx['name']}\n    coordinates {{{stream}}};")

    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
