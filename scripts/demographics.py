#!/usr/bin/env python3
"""
Dataset summary statistics and the simulation's calibration targets.

    python scripts/demographics.py

Prints the medians and histogram counts of the h-index, i10-index and total
citations, and the summary statistics of the dataset that the simulation is
calibrated against.
"""

import numpy as np

from data import load_arrays, year_t_statistics
from indices import h_index, i10_index

H_EDGES = [0, 20, 40, 60, 80, 100, 120, 10 ** 9]
H_LABELS = ["0--19", "20--39", "40--59", "60--79", "80--99", "100--119", "120+"]
I10_EDGES = [0, 20, 40, 60, 80, 100, 120, 200, 10 ** 9]
I10_LABELS = ["0--19", "20--39", "40--59", "60--79", "80--99", "100--119", "120--199", "200+"]
TOTAL_EDGES = [0, 1_000, 5_000, 10_000, 20_000, 50_000, 10 ** 12]
TOTAL_LABELS = ["<1k", "1k--5k", "5k--10k", "10k--20k", "20k--50k", "50k+"]


def histogram(values, edges, labels):
    counts = np.histogram(values, edges)[0]
    return " ".join(f"({l},{c})" for l, c in zip(labels, counts))


def main():
    arrays = load_arrays()
    H = np.array([h_index(a) for a in arrays])
    I10 = np.array([i10_index(a) for a in arrays])
    TOT = np.array([sum(a) for a in arrays])
    print(f"n = {len(arrays)}")
    print(f"h:     median {np.median(H):g}  max {H.max()}   {histogram(H, H_EDGES, H_LABELS)}")
    print(f"i10:   median {np.median(I10):g}   {histogram(I10, I10_EDGES, I10_LABELS)}")
    print(f"total: median {np.median(TOT):,.0f}   {histogram(TOT, TOTAL_EDGES, TOTAL_LABELS)}")
    print(f"researchers with h > 100: {(H > 100).sum()}; with more than 50,000 citations: {(TOT > 50_000).sum()}")

    print("\nCalibration targets (real data, exact):")
    for k, v in year_t_statistics(arrays).items():
        print(f"  {k:18s} {v:12.4f}")


if __name__ == "__main__":
    main()
