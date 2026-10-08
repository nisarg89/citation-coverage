#!/usr/bin/env python3
"""
Decisiveness: how many pairs of researchers each index leaves tied.

    python scripts/decisiveness.py

Prints, for every index and component, the number of distinct values taken and
the fraction of researcher pairs left tied, grouped by how many integers the
object reports (only within-group comparisons are meaningful: three coordinates
admit more distinct values than one). It then prints the tie rate as a function
of the gap in total citations, |log10(total citations of x) - log10(total
citations of y)|.
"""

import numpy as np

from data import load_arrays
from indices import (
    h_index, i10_index, sr_pow, sr_mult, area_minus, area_plus, lin_top, lin_right,
)

# Upper edges of the stature-gap bins; the first bin holds pairs with nearly equal totals.
GAP_EDGES = np.array([0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 10.0])
CURVES = ["h", "i10", "(P_2,h,P_0.5)", "(M_10,h,M_0.25)", "(A^-,h,A^+)", "(h_-1,h,h_+1)"]


def main():
    arrays = load_arrays()
    H, I10, Pt, Pr, Mt, Mr, AT, AR, LT, LR, TOT = ([] for _ in range(11))
    for a in arrays:
        h = h_index(a)
        H.append(h); I10.append(i10_index(a))
        Pt.append(sr_pow(a, 2.0));   Pr.append(sr_pow(a, 0.5))
        Mt.append(sr_mult(a, 10.0)); Mr.append(sr_mult(a, 0.25))
        AT.append(area_minus(a, h)); AR.append(area_plus(a, h))
        LT.append(lin_top(a, h));    LR.append(lin_right(a, h))
        TOT.append(sum(a))
    H, I10, Pt, Pr, Mt, Mr, AT, AR, LT, LR = map(np.array, (H, I10, Pt, Pr, Mt, Mr, AT, AR, LT, LR))
    TOT = np.array(TOT, float)
    n = len(H)

    a_, b_ = np.triu_indices(n, 1)

    def tie_scalar(v):
        return v[a_] == v[b_]

    def tie_triple(t1, t2, t3):
        return (t1[a_] == t1[b_]) & (t2[a_] == t2[b_]) & (t3[a_] == t3[b_])

    def n_scalar(v):
        return len(np.unique(v))

    def n_triple(t1, t2, t3):
        return len({*map(tuple, np.stack([t1, t2, t3], 1))})

    groups = [
        ("scalars (1 integer)", [
            ("h", tie_scalar(H), n_scalar(H)),
            ("i10", tie_scalar(I10), n_scalar(I10)),
        ]),
        ("peak components (1 integer)", [
            ("P_2", tie_scalar(Pt), n_scalar(Pt)),
            ("M_10", tie_scalar(Mt), n_scalar(Mt)),
            ("A^-", tie_scalar(AT), n_scalar(AT)),
            ("h_-1", tie_scalar(LT), n_scalar(LT)),
        ]),
        ("tail components (1 integer)", [
            ("P_0.5", tie_scalar(Pr), n_scalar(Pr)),
            ("M_0.25", tie_scalar(Mr), n_scalar(Mr)),
            ("A^+", tie_scalar(AR), n_scalar(AR)),
            ("h_+1", tie_scalar(LR), n_scalar(LR)),
        ]),
        ("triples (3 integers)", [
            ("(P_2,h,P_0.5)", tie_triple(Pt, H, Pr), n_triple(Pt, H, Pr)),
            ("(M_10,h,M_0.25)", tie_triple(Mt, H, Mr), n_triple(Mt, H, Mr)),
            ("(A^-,h,A^+)", tie_triple(AT, H, AR), n_triple(AT, H, AR)),
            ("(h_-1,h,h_+1)", tie_triple(LT, H, LR), n_triple(LT, H, LR)),
        ]),
    ]
    ties = {lbl: t for _, rows in groups for lbl, t, _ in rows}

    print(f"n = {n} researchers; {len(a_):,} pairs\n")
    for gname, rows in groups:
        print(f"  {gname}")
        print(f"    {'index':22s} {'distinct':>9s} {'tie rate':>10s}")
        for lbl, t, d in rows:
            print(f"    {lbl:22s} {d:9d} {t.mean():10.4f}")
        print()

    # ---- tie rate vs gap in total citations ----
    ds = np.abs(np.log10(TOT[a_]) - np.log10(TOT[b_]))
    bin_idx = np.digitize(ds, GAP_EDGES) - 1

    print("=== tie rate by gap in total citations |delta log10(total citations)| ===")
    print("gap bin        " + "".join(f"{k.split('(')[0].strip()[:16]:>18s}" for k in CURVES) + f"{'#pairs':>10s}")
    for bi in range(len(GAP_EDGES) - 1):
        m = bin_idx == bi
        row = f"{GAP_EDGES[bi]:.3f}-{GAP_EDGES[bi + 1]:<6.3f}"
        for k in CURVES:
            row += f"{ties[k][m].mean() if m.sum() else float('nan'):18.4f}"
        print(row + f"{m.sum():10d}")

    m0 = bin_idx == 0
    print(f"\nNearly equal totals (gap < {GAP_EDGES[1]}: totals within "
          f"~{(10 ** GAP_EDGES[1] - 1) * 100:.0f}% of each other; {m0.sum():,} pairs):")
    for k in CURVES:
        print(f"   {k:30s} tie rate {ties[k][m0].mean():.4f}")


if __name__ == "__main__":
    main()
