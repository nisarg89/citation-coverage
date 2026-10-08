#!/usr/bin/env python3
"""
Fetch the per-paper citation counts of each matched profile.

For every matched profile in build_output/profiles.json, the citation count of
each paper is read from OpenAlex (works attributed to the author) or from
Semantic Scholar (the author's papers).

If the array is shorter than the researcher's CSRankings publication count, the
name is searched again, first in OpenAlex and then in Semantic Scholar, and the
longest array found is kept.

    python 03_fetch_citation_arrays.py [--restart]

Needs OPENALEX_EMAIL. Progress is checkpointed and an interrupted run resumes;
--restart discards the checkpoint. Writes build_output/citation_arrays.json.
"""

import argparse
import os
from datetime import datetime, timezone

from common import (
    NAME_MATCH_MIN, load_json, name_similarity, openalex, openalex_citations, out_path,
    require_email, save_json, semantic_scholar, semantic_scholar_citations, strip_name_suffix,
)

CHECKPOINT_EVERY = 10


def fetch_array(source, author_id):
    return (openalex_citations if source == "openalex" else semantic_scholar_citations)(author_id)


def search_longer_array(profile, current):
    """Search the researcher's name again; returns (source, author_id, array) for the
    longest array found, stopping once it reaches the CSRankings publication count."""
    target = profile["csrankings_pubs"]
    name = strip_name_suffix(profile["name"])
    best = (profile["source"], profile["author_id"], current)

    def consider(source, author_id):
        nonlocal best
        array = fetch_array(source, author_id)
        if len(array) > len(best[2]):
            best = (source, author_id, array)
        return len(best[2]) >= target

    def ranked(candidates, name_key, works_key):
        scored = [(name_similarity(name, c.get(name_key)), c.get(works_key) or 0, c) for c in candidates]
        return [(w, c) for s, w, c in sorted(scored, key=lambda t: t[:2], reverse=True)
                if s >= NAME_MATCH_MIN]

    try:
        authors = openalex.get("authors", {"search": name, "per_page": 10})["results"]
        for works, c in ranked(authors, "display_name", "works_count"):
            if works >= target and consider("openalex", c["id"]):
                return best
    except Exception as e:
        print(f"    OpenAlex search failed: {e}")

    try:
        authors = semantic_scholar.get("author/search", {
            "query": name, "limit": 10, "fields": "name,paperCount"}).get("data") or []
        for papers, c in ranked(authors, "name", "paperCount"):
            if (papers >= target or papers > len(best[2])) and consider("semantic_scholar", c["authorId"]):
                return best
    except Exception as e:
        print(f"    Semantic Scholar search failed: {e}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", action="store_true", help="discard the checkpoint")
    args = ap.parse_args()
    require_email()

    profiles = [p for p in load_json(out_path("profiles.json")) if p["status"] == "matched"]
    checkpoint = out_path("citation_arrays_partial.json")
    records = [] if args.restart or not os.path.exists(checkpoint) else load_json(checkpoint)
    done = {r["name"] for r in records}
    if done:
        print(f"Resuming: {len(done)} researchers already fetched")

    for i, p in enumerate(profiles, 1):
        if p["name"] in done:
            continue
        try:
            array = fetch_array(p["source"], p["author_id"])
            source, author_id = p["source"], p["author_id"]
            if len(array) < p["csrankings_pubs"]:
                source, author_id, array = search_longer_array(p, array)
        except Exception as e:
            print(f"[{i}/{len(profiles)}] {p['name']}: {e}")
            continue
        records.append({
            "name": p["name"], "dept": p["dept"], "source": source, "author_id": author_id,
            "csrankings_pubs": p["csrankings_pubs"], "citation_array": array,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        })
        print(f"[{i}/{len(profiles)}] {p['name']:<36} {source:<17} papers={len(array):>4}", flush=True)
        if len(records) % CHECKPOINT_EVERY == 0:
            save_json(checkpoint, records)

    save_json(out_path("citation_arrays.json"), records)
    if os.path.exists(checkpoint):
        os.remove(checkpoint)
    short = sum(len(r["citation_array"]) < r["csrankings_pubs"] for r in records)
    print(f"\n{len(records)} records; {short} with fewer papers than CSRankings publications")


if __name__ == "__main__":
    main()
