#!/usr/bin/env python3
"""
Select the researchers: the top faculty at the top departments in CSRankings.

Follows the live csrankings.org computation with all 27 sub-areas selected:
  - Department "Count" = smoothed geometric mean over the sub-areas,
        Count(d) = ( prod_s (A[s][d] + 1) ) ^ (1 / 27),
    where A[s][d] is the sum of `adjustedcount` over rows of department d,
    sub-area s, within the year window.
  - Faculty "# Pubs" = sum of `count` over their rows within the window.
  - The top departments in North America by Count are kept, and within each the
    top faculty by # Pubs, with all faculty tied at the cutoff.
  - The year window defaults to the last ten years, as on csrankings.org.

The venue-to-area mapping is taken from emeryberger/CSrankings (src/config.ts).
The CSRankings data files are downloaded into build_output/raw/.

    python 01_select_researchers.py [--year-from 2016] [--year-to 2026]
                                    [--top-depts 20] [--per-dept 50]

Writes build_output/researchers_list.json.
"""

import argparse
import csv
import datetime as dt
import io
import math
import os
from collections import defaultdict

from common import Api, out_path, save_json

BASE_URL = "https://raw.githubusercontent.com/emeryberger/CSrankings/gh-pages"
FILES = {"institutions": "institutions.csv", "author_info": "generated-author-info.csv"}

# === CSRankings configuration (from src/config.ts) ===
PARENT_MAP = {
    # AI
    "aaai": "ai", "ijcai": "ai",
    "cvpr": "vision", "eccv": "vision", "iccv": "vision",
    "icml": "mlmining", "iclr": "mlmining", "kdd": "mlmining", "nips": "mlmining",
    "acl": "nlp", "emnlp": "nlp", "naacl": "nlp",
    "sigir": "inforet", "www": "inforet",
    # Systems
    "asplos": "arch", "isca": "arch", "micro": "arch", "hpca": "arch",
    "ccs": "sec", "oakland": "sec", "usenixsec": "sec", "ndss": "sec",
    "vldb": "mod", "sigmod": "mod", "icde": "mod", "pods": "mod",
    "dac": "da", "iccad": "da",
    "emsoft": "bed", "rtas": "bed", "rtss": "bed",
    "sc": "hpc", "hpdc": "hpc", "ics": "hpc",
    "mobicom": "mobile", "mobisys": "mobile", "sensys": "mobile",
    "imc": "metrics", "sigmetrics": "metrics",
    "osdi": "ops", "sosp": "ops", "eurosys": "ops", "fast": "ops", "usenixatc": "ops",
    "popl": "plan", "pldi": "plan", "oopsla": "plan", "icfp": "plan",
    "fse": "soft", "icse": "soft", "ase": "soft", "issta": "soft",
    "nsdi": "comm", "sigcomm": "comm",
    # Interdisciplinary
    "siggraph": "graph", "siggraph-asia": "graph", "eurographics": "graph",
    "focs": "act", "soda": "act", "stoc": "act",
    "crypto": "crypt", "eurocrypt": "crypt",
    "cav": "log", "lics": "log",
    "ismb": "bio", "recomb": "bio",
    "ec": "ecom", "wine": "ecom",
    "chiconf": "chi", "ubicomp": "chi", "uist": "chi",
    "icra": "robotics", "iros": "robotics", "rss": "robotics",
    "vis": "visualization", "vr": "visualization",
    "sigcse": "csed",
}

AI_AREAS = ["ai", "vision", "mlmining", "nlp", "inforet"]
SYSTEMS_AREAS = ["arch", "comm", "sec", "mod", "da", "bed", "hpc", "mobile",
                 "metrics", "ops", "plan", "soft"]
THEORY_AREAS = ["act", "crypt", "log"]
INTERDISCIPLINARY_AREAS = ["bio", "graph", "csed", "ecom", "chi",
                          "robotics", "visualization"]

ALL_SUBAREAS = AI_AREAS + SYSTEMS_AREAS + THEORY_AREAS + INTERDISCIPLINARY_AREAS  # 27

assert set(PARENT_MAP.values()) <= set(ALL_SUBAREAS)

# Venues that csrankings.org excludes by default (config.ts: nextTier).
NEXT_TIER = {
    "ase", "issta", "icde", "pods", "hpca", "ndss", "eurosys",
    "eurographics", "fast", "usenixatc", "icfp", "oopsla", "kdd",
}


# === Reading the data ===

def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = csv.DictReader(io.StringIO(f.read()))
        return [{k.strip(): (v or "").strip() for k, v in r.items()} for r in rows]


def download_data():
    paths = {}
    for key, fname in FILES.items():
        paths[key] = out_path("raw", fname)
        if os.path.exists(paths[key]):
            continue
        os.makedirs(os.path.dirname(paths[key]), exist_ok=True)
        with open(paths[key], "wb") as f:
            f.write(Api(BASE_URL, 0).get(fname, raw=True))
    return paths


def number(text):
    try:
        return float(text)
    except ValueError:
        return 0.0


def north_american_institutions(rows):
    return {r["institution"] for r in rows
            if r["region"].lower().replace(" ", "") in ("northamerica", "usa", "canada")}


# === Selection ===

def select_researchers(year_from, year_to, n_depts, per_dept):
    paths = download_data()
    author_rows = read_csv(paths["author_info"])
    na_institutions = north_american_institutions(read_csv(paths["institutions"]))

    area_adj = defaultdict(lambda: defaultdict(float))  # sub-area -> dept -> adjusted count
    pubs = defaultdict(float)                           # faculty -> # Pubs
    adj = defaultdict(float)                            # faculty -> adjusted count
    dept_faculty = defaultdict(set)

    for row in author_rows:
        venue = row["area"]
        area = PARENT_MAP.get(venue)
        dept = row["dept"]
        if area is None or venue in NEXT_TIER or dept not in na_institutions:
            continue
        try:
            year = int(row["year"])
        except ValueError:
            continue
        if not year_from <= year <= year_to:
            continue
        count, adjusted = number(row["count"]), number(row["adjustedcount"])
        name = row["name"]
        area_adj[area][dept] += adjusted
        pubs[name] += count
        adj[name] += adjusted
        dept_faculty[dept].add(name)

    dept_count = {
        d: math.exp(sum(math.log(area_adj[a].get(d, 0.0) + 1.0) for a in ALL_SUBAREAS) / len(ALL_SUBAREAS))
        for d in dept_faculty
    }
    top_depts = sorted(dept_count, key=lambda d: -dept_count[d])[:n_depts]

    selected = []
    for rank, dept in enumerate(top_depts, 1):
        ranked = sorted(dept_faculty[dept], key=lambda n: (-pubs[n], -adj[n], n))
        cutoff = pubs[ranked[min(per_dept, len(ranked)) - 1]]
        chosen = [n for n in ranked if pubs[n] >= cutoff]  # includes everyone tied at the cutoff
        print(f"{rank:2d}  {dept:<40} count {dept_count[dept]:7.2f}  selected {len(chosen):3d}")
        for i, name in enumerate(chosen, 1):
            selected.append({
                "name": name,
                "affiliation": dept,
                "csrankings_pubs": int(round(pubs[name])),
                "dept_rank": rank,
                "researcher_rank_in_dept": i,
            })
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-from", type=int, default=dt.date.today().year - 10)
    ap.add_argument("--year-to", type=int, default=dt.date.today().year)
    ap.add_argument("--top-depts", type=int, default=20)
    ap.add_argument("--per-dept", type=int, default=50)
    args = ap.parse_args()

    selected = select_researchers(args.year_from, args.year_to, args.top_depts, args.per_dept)
    save_json(out_path("researchers_list.json"), selected)
    print(f"\n{len(selected)} researchers -> {out_path('researchers_list.json')}")


if __name__ == "__main__":
    main()
