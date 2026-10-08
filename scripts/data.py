"""Loader for the released dataset, and the summary statistics derived from it."""

import json
import os

import numpy as np

from indices import h_index

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "citation_arrays.json")
OUTPUT_DIR = os.path.join(ROOT, "outputs")


def load_arrays(path=DATA_PATH):
    """Per-researcher citation arrays, each sorted in non-increasing order."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    arrays = [sorted(r["citation_array"], reverse=True) for r in records]
    assert all(h_index(a) >= 1 for a in arrays)
    return arrays


def year_t_statistics(arrays):
    """Summary statistics of a set of citation arrays: the statistics the
    simulation parameters are calibrated against."""
    h = np.array([h_index(a) for a in arrays])
    total = np.array([sum(a) for a in arrays])
    return {
        "h_mean": float(h.mean()),
        "h_median": float(np.median(h)),
        "h_std": float(h.std()),
        "total_mean": float(total.mean()),
        "total_median": float(np.median(total)),
        "papers_mean": float(np.mean([len(a) for a in arrays])),
        "max_cite_over_h": float(np.mean([a[0] / hh for a, hh in zip(arrays, h)])),
        "total_over_h2": float(np.mean(total / h ** 2)),
    }
