"""
Index definitions and coverage arithmetic, shared by every analysis script.

An index is evaluated on a researcher's citation array `arr`, sorted in
non-increasing order, so that c_k = arr[k-1]. The height of a rectangle is the
threshold f(k).
"""


# === Index components ===

def h_index(arr):
    h = 0
    for i, c in enumerate(arr):
        if c >= i + 1:
            h = i + 1
        else:
            break
    return h


def i10_index(arr):
    return sum(1 for c in arr if c >= 10)


def sr_pow(arr, beta):
    """Power index P_beta = max {k : c_k >= k^beta}."""
    r = 0
    for i, c in enumerate(arr):
        k = i + 1
        if c >= k ** beta:
            r = k
        else:
            break
    return r


def sr_mult(arr, beta):
    """Multiplier index M_beta = max {k : c_k >= beta * k}."""
    r = 0
    for i, c in enumerate(arr):
        k = i + 1
        if c >= beta * k:
            r = k
        else:
            break
    return r


def area_minus(arr, h):
    """A^- = min {k : k * c_k >= h^2}."""
    t = h * h
    for i, c in enumerate(arr):
        if (i + 1) * c >= t:
            return i + 1
    return 0


def area_plus(arr, h):
    """A^+ = max {k : k * c_k >= h^2}."""
    t = h * h
    r = 0
    for i, c in enumerate(arr):
        if (i + 1) * c >= t:
            r = i + 1
    return r


def lin_top(arr, h):
    """Two-sided h-index, top square: h_{-1} = max {k : c_k >= k + h}."""
    r = 0
    for i, c in enumerate(arr):
        if c >= (i + 1) + h:
            r = i + 1
        else:
            break
    return r


def lin_right(arr, h):
    """Two-sided h-index, right square: h_{+1} = max {k : c_k >= k - h}."""
    r = 0
    for i, c in enumerate(arr):
        if c >= (i + 1) - h:
            r = i + 1
        else:
            break
    return r


def compute_component(arr, spec, h=None, role=None):
    """Evaluate a component from a (family, beta) spec."""
    fam, beta = spec
    if fam == "mult":
        return sr_mult(arr, beta)
    if fam == "pow":
        return sr_pow(arr, beta)
    if fam == "area":
        return area_minus(arr, h) if role == "top" else area_plus(arr, h)
    if fam == "lin":
        return lin_top(arr, h) if beta > 0 else lin_right(arr, h)
    raise ValueError(f"Unknown family: {fam}")


def threshold_height(spec, k, h):
    """Rectangle height f(k) for a component."""
    if k <= 0:
        return 0.0
    fam, beta = spec
    if fam == "pow":
        return k ** beta
    if fam == "mult":
        return beta * k
    if fam == "area":
        return (h * h) / k
    if fam == "lin":
        return k + beta * h          # beta = +1: k + h (top); beta = -1: k - h (right)
    raise ValueError(f"Unknown family: {fam}")


# === Coverage ===

def peak_gain_area(k, f, h):
    """Area of the peak rectangle [0,k] x [0,f] above the h-square."""
    return k * (f - h) if k > 0 and f > h else 0.0


def tail_gain_area(k, f, h):
    """Area of the tail rectangle [0,k] x [0,f] right of the h-square."""
    return (k - h) * f if k > h else 0.0


def coverage_3tuple(arr, top, h, right, top_spec, right_spec):
    """
    Coverage of the union of the peak, h, and tail rectangles.

    The peak region (above the h-square) and the tail region (right of it) are
    disjoint, so coverage is h^2/total plus the two independent gains, all
    measured with threshold heights.
    """
    total = sum(arr)
    if total == 0 or h == 0:
        return 0.0
    gain_top = peak_gain_area(top, threshold_height(top_spec, top, h), h)
    gain_right = tail_gain_area(right, threshold_height(right_spec, right, h), h)
    return (h * h + gain_top + gain_right) / total


def h_only_coverage(arr, h):
    total = sum(arr)
    if total == 0 or h == 0:
        return 0.0
    return (h * h) / total


def i10_coverage(arr, i10):
    total = sum(arr)
    if total == 0:
        return 0.0
    return min(i10 * 10 / total, 1.0)


# === The indices evaluated ===
#
# The h-index, the i10-index, and four triples (power, multiplier, area, and the
# two-sided h-index of Garcia-Perez (2012)). A triple is (peak, h, tail);
# `top`/`right` are the (family, beta) specs of its peak and tail components.

CANDIDATE_INDICES = [
    {"name": "h-index", "label": r"$h$-index", "top": None, "right": None},
    {"name": "i10-index", "label": r"$i_{10}$-index", "top": None, "right": None},
    {"name": "power", "label": r"$(P_{2},\, h,\, P_{0.5})$",
     "top": ("pow", 2.0), "right": ("pow", 0.5)},
    {"name": "multiplier", "label": r"$(M_{10},\, h,\, M_{0.25})$",
     "top": ("mult", 10.0), "right": ("mult", 0.25)},
    {"name": "area", "label": r"$(A^{-},\, h,\, A^{+})$",
     "top": ("area", None), "right": ("area", None)},
    # Thresholds f(k) = k + h (peak) and f(k) = k - h (tail).
    {"name": "twosided", "label": r"$(h_{-1},\, h,\, h_{+1})$",
     "top": ("lin", 1), "right": ("lin", -1)},
]

THREE_TUPLE_INDICES = [d for d in CANDIDATE_INDICES if d["top"] is not None]


def comp_label(idx_def, comp_name):
    """Label of one component of a candidate."""
    spec = idx_def["top"] if comp_name == "top" else idx_def["right"]
    fam, beta = spec
    if fam == "area":
        return "$A^{-}$" if comp_name == "top" else "$A^{+}$"
    if fam == "lin":
        return "$h_{-1}$" if comp_name == "top" else "$h_{+1}$"
    b = f"{beta:g}"
    return f"$P_{{{b}}}$" if fam == "pow" else f"$M_{{{b}}}$"
