#!/usr/bin/env python3
"""
The beta sweeps of the power and multiplier families.

    python scripts/sweeps.py

Peak and tail components of the power family P_beta and the multiplier family
M_beta are swept independently over fixed grids; because coverage decomposes as
h^2/C + peak gain + tail gain, each component's per-researcher gain over h alone
depends on that component only. For each grid point the script prints the mean,
median and minimum gain across researchers, then the best peak and tail
component by each criterion, and the best pairing by mean and by minimum
coverage. The full per-grid-point table is also written to outputs/sweeps.json.
"""

import json
import os

import numpy as np

from data import OUTPUT_DIR, load_arrays
from indices import h_index, sr_pow, sr_mult, threshold_height, peak_gain_area, tail_gain_area

# Peak side.
PEAK_POW = ([("pow", round(1.05 + 0.05 * i, 2)) for i in range(40)]      # 1.05 .. 3.00
            + [("pow", float(b)) for b in range(4, 21)])                 # 4 .. 20
PEAK_MULT = [("mult", float(b)) for b in range(2, 21)]                   # 2 .. 20
# Tail side: a common grid for both families.
TAIL = ([("pow", round(0.10 + 0.05 * i, 2)) for i in range(18)]          # 0.10 .. 0.95
        + [("mult", round(0.10 + 0.05 * i, 2)) for i in range(18)])


def label(spec):
    fam, beta = spec
    return f"{'P' if fam == 'pow' else 'M'}_{beta:g}"


def sweep(arrays, H, TOT, specs, role):
    """Per-researcher gains over h alone, and their summary, for each grid point."""
    gain_fn = peak_gain_area if role == "top" else tail_gain_area
    rows = []
    for spec in specs:
        fam, beta = spec
        comp = sr_pow if fam == "pow" else sr_mult
        gains = np.array([
            gain_fn(k, threshold_height(spec, k, h), h) / tot
            for a, h, tot in zip(arrays, H, TOT)
            for k in [comp(a, beta)]
        ])
        rows.append({"label": label(spec), "spec": spec, "gains": gains,
                     "mean": float(gains.mean()), "median": float(np.median(gains)),
                     "min": float(gains.min())})
    return rows


def print_sweep(title, rows):
    print(f"--- {title}")
    print(f"{'component':10s} {'mean':>8s} {'median':>8s} {'min':>8s}")
    for r in rows:
        print(f"{r['label']:10s} {r['mean']:8.4f} {r['median']:8.4f} {r['min']:8.4f}")
    print()


def best(rows, key):
    return max(rows, key=lambda r: r[key])


def main():
    arrays = load_arrays()
    H = np.array([h_index(a) for a in arrays])
    TOT = np.array([float(sum(a)) for a in arrays])
    h_cov = H ** 2 / TOT
    print(f"n = {len(arrays)}; h alone: mean coverage {h_cov.mean():.4f}\n")

    top_rows = sweep(arrays, H, TOT, PEAK_POW + PEAK_MULT, "top")
    right_rows = sweep(arrays, H, TOT, TAIL, "right")
    print_sweep("peak-side gain over h", top_rows)
    print_sweep("tail-side gain over h", right_rows)

    for name, rows in (("peak", top_rows), ("tail", right_rows)):
        print(f"best {name} component by criterion (gain mean / median / min):")
        for key in ("mean", "median", "min"):
            r = best(rows, key)
            print(f"  by {key:6s}: {r['label']:8s} {r['mean']:.4f} {r['median']:.4f} {r['min']:.4f}")
    print()

    # Best pairing by mean coverage: the best peak and best tail components by mean gain.
    bt, br = best(top_rows, "mean"), best(right_rows, "mean")
    cov = h_cov + bt["gains"] + br["gains"]
    print(f"best pairing by mean coverage: ({bt['label']}, h, {br['label']})  "
          f"mean {cov.mean():.4f}  median {np.median(cov):.4f}  min {cov.min():.4f}")

    # Best pairing by minimum coverage: joint search over all peak x tail pairs.
    best_min = None
    for tr in top_rows:
        for rr in right_rows:
            cov = h_cov + tr["gains"] + rr["gains"]
            if best_min is None or cov.min() > best_min[0]:
                best_min = (cov.min(), tr["label"], rr["label"], cov.mean(), np.median(cov))
    mn, tl, rl, mean, med = best_min
    print(f"best pairing by min coverage : ({tl}, h, {rl})  "
          f"mean {mean:.4f}  median {med:.4f}  min {mn:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "sweeps.json")
    with open(out, "w") as f:
        json.dump({"peak": [{k: v for k, v in r.items() if k != "gains"} for r in top_rows],
                   "tail": [{k: v for k, v in r.items() if k != "gains"} for r in right_rows]},
                  f, indent=1)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
