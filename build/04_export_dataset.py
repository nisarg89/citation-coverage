#!/usr/bin/env python3
"""
Write the dataset: for each researcher, the author ID and the citation array.

    python 04_export_dataset.py [--arrays ../build_output/citation_arrays.json]
                                [--out ../data/citation_arrays.json]

OpenAlex author IDs are URLs (https://openalex.org/A...); Semantic Scholar author
IDs are numeric strings. Records are ordered by author ID.
"""

import argparse
import json
import os

from common import ROOT, load_json


def build_records(arrays):
    records = sorted(({"author_id": r["author_id"], "citation_array": sorted(r["citation_array"], reverse=True)}
                      for r in arrays), key=lambda r: r["author_id"])
    assert len({r["author_id"] for r in records}) == len(records), "author IDs must be unique"
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", default=os.path.join(ROOT, "build_output", "citation_arrays.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "citation_arrays.json"))
    args = ap.parse_args()

    records = build_records(load_json(args.arrays))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, separators=(",", ":"))
    print(f"{len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
