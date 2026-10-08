#!/usr/bin/env python3
"""
Cross-sectional results for h, i10 and the four triples (power, multiplier,
area, and the two-sided h-index of Garcia-Perez (2012)).

    python scripts/triples.py

For each index it prints
  * coverage: mean / median / min, and the box-plot five numbers;
  * peak gain and tail gain over h alone: mean / median / min;
  * mean coverage and mean gain by h-bin;
  * Kendall tau with h and rectangle IoU with the h-square for each component.
The decomposition of coverage into h^2/C plus the two gains relies on the peak
component lying at or left of h and the tail component at or right of it; this
is asserted for every researcher and triple.
"""

import numpy as np
from scipy.stats import kendalltau

from data import load_arrays
from indices import (
    THREE_TUPLE_INDICES, comp_label,
    h_index, i10_index, i10_coverage, compute_component, threshold_height,
    coverage_3tuple, peak_gain_area, tail_gain_area,
)

# Researchers with h < 10 fall outside every bin and appear only in the
# whole-sample statistics.
BINS = [(10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 79), (80, 10 ** 9)]
LABELS = ["10--19", "20--29", "30--39", "40--49", "50--59", "60--69", "70--79", "80+"]


def rect_iou(w1, h1, w2, h2):
    """IoU of two origin-anchored rectangles (w1,h1) and (w2,h2)."""
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0.0
    inter = min(w1, w2) * min(h1, h2)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def five_numbers(x):
    return (x.min(), np.percentile(x, 25), np.median(x), np.percentile(x, 75), x.max())


def print_coverage(cov):
    lo, q1, med, q3, hi = five_numbers(cov)
    print(f"coverage  mean {cov.mean():.4f}  median {med:.4f}  min {lo:.4f}")
    print(f"boxplot   lower {lo:.4f}  q1 {q1:.4f}  median {med:.4f}  q3 {q3:.4f}  upper {hi:.4f}")


def main():
    arrays = load_arrays()
    n = len(arrays)
    H = np.array([h_index(a) for a in arrays])
    TOT = np.array([float(sum(a)) for a in arrays])
    h_cov = H ** 2 / TOT
    in_bin = [(H >= lo) & (H <= hi) for lo, hi in BINS]
    print(f"n = {n}")
    print("researchers per h-bin:", dict(zip(LABELS, (int(m.sum()) for m in in_bin))),
          f"(h < 10, outside every bin: {int((H < 10).sum())})\n")

    print("=" * 78)
    print("h-index")
    print("-" * 78)
    print_coverage(h_cov)
    print("by h-bin (mean coverage):",
          " ".join(f"({l}, {h_cov[m].mean():.4f})" for l, m in zip(LABELS, in_bin)))
    print()

    i10 = np.array([i10_index(a) for a in arrays])
    i10_cov = np.array([i10_coverage(a, v) for a, v in zip(arrays, i10)])
    print("=" * 78)
    print("i10-index")
    print("-" * 78)
    print_coverage(i10_cov)
    print()

    for idx in THREE_TUPLE_INDICES:
        tspec, rspec = idx["top"], idx["right"]
        tw, rw, th, rh, cov, gt, gr = [], [], [], [], [], [], []
        for a, h, tot in zip(arrays, H, TOT):
            t = compute_component(a, tspec, h, "top")
            rt = compute_component(a, rspec, h, "right")
            assert t <= h <= rt, (idx["name"], t, h, rt)
            ft = threshold_height(tspec, t, h)
            fr = threshold_height(rspec, rt, h)
            tw.append(t); rw.append(rt); th.append(ft); rh.append(fr)
            cov.append(coverage_3tuple(a, t, h, rt, tspec, rspec))
            gt.append(peak_gain_area(t, ft, h) / tot)
            gr.append(tail_gain_area(rt, fr, h) / tot)
        tw, rw, th, rh, cov, gt, gr = map(np.array, (tw, rw, th, rh, cov, gt, gr))
        gain = cov - h_cov

        print("=" * 78)
        print(f"{idx['name']}  {idx['label']}")
        print("-" * 78)
        print_coverage(cov)
        print(f"peak gain mean {gt.mean():.4f}  median {np.median(gt):.4f}  min {gt.min():.4f}"
              f"      tail gain mean {gr.mean():.4f}  median {np.median(gr):.4f}  min {gr.min():.4f}")
        print("by h-bin  (mean coverage | mean gain over h):")
        print("  coverage:", " ".join(f"({l}, {cov[m].mean():.4f})" for l, m in zip(LABELS, in_bin)))
        print("  gain    :", " ".join(f"({l}, {gain[m].mean():.4f})" for l, m in zip(LABELS, in_bin)))
        for role, ws, hs in (("top", tw, th), ("right", rw, rh)):
            tau = float(kendalltau(ws, H)[0]) if np.std(ws) > 1e-12 else float("nan")
            iou = float(np.mean([rect_iou(ws[i], hs[i], H[i], H[i]) for i in range(n)]))
            print(f"component {comp_label(idx, role):10s} vs h:  Kendall tau {tau:.4f}   rect IoU {iou:.4f}")
        print()


if __name__ == "__main__":
    main()
