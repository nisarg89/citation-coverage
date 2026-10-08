#!/usr/bin/env python3
"""
The g-index and g^2 / C, where C is a researcher's total citations.

g = max{k : sum_{j<=k} c_j >= k^2}. This prints g compared with h, the
distribution of g^2 / C on the dataset, and a check that the papers beyond rank h
within the top g hold at most h(g-h) citations.

    python scripts/g_index.py
"""

import numpy as np

from data import load_arrays
from indices import h_index


def g_index(a):
    """g = max{k <= n : sum of the top k counts >= k^2} (no padding with zero-cited papers)."""
    cum, g = 0, 0
    for k, c in enumerate(a, start=1):
        cum += c
        if cum >= k * k:
            g = k
    return g


def main():
    arrays = load_arrays()
    H = np.array([h_index(a) for a in arrays])
    G = np.array([g_index(a) for a in arrays])
    C = np.array([sum(a) for a in arrays], float)
    cov = G ** 2 / C
    print(f"n = {len(arrays)}")
    print(f"g >= h for every researcher: {bool((G >= H).all())}; g > h for {(G > H).mean():.1%}; "
          f"median g/h = {np.median(G / H):.2f}")
    print(f"g^2 / C   mean {cov.mean():.4f}  median {np.median(cov):.4f}  min {cov.min():.4f}   "
          f"(h^2/C mean {np.mean(H ** 2 / C):.4f})")
    viol = sum(sum(a[h:g]) > h * (g - h) + 1e-9 for a, h, g in zip(arrays, H, G))
    print(f"top-g mass beyond rank h exceeds h(g-h) for {viol} researchers (must be 0)")


if __name__ == "__main__":
    main()
