#!/usr/bin/env python3
"""
Match each selected researcher to an author profile.

For every researcher in build_output/researchers_list.json:

  1. OpenAlex: search the name and keep the candidates whose name matches and
     whose affiliation history includes the researcher's department; the one
     with the most works is the match. The other candidates are kept as
     alternatives. Failed lookups are retried once, sequentially.
  2. Semantic Scholar: researchers with no OpenAlex match are searched there,
     preferring candidates affiliated with the department.
  3. Cross-check: if the matched profile has fewer works than the researcher has
     publications in the CSRankings data, look for a profile at the department
     with enough works, among the alternatives or by re-querying OpenAlex with
     an institution filter.

    python 02_match_profiles.py [--limit N] [--workers 8]

Needs OPENALEX_EMAIL. Writes build_output/profiles.json.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from common import (
    NAME_MATCH, NAME_MATCH_LOOSE, affiliation_matches, load_json, name_similarity,
    normalize_name, openalex, out_path, require_email, save_json, semantic_scholar,
    strip_name_suffix,
)

# Department names that need rewriting to search cleanly in OpenAlex.
DEPT_SEARCH_NAMES = {
    "Univ. of California - San Diego": "University of California San Diego",
    "Univ. of California - Berkeley": "University of California Berkeley",
    "Univ. of California - Los Angeles": "University of California Los Angeles",
    "Univ. of California - Irvine": "University of California Irvine",
    "Univ. of California - Santa Barbara": "University of California Santa Barbara",
    "Univ. of California - Santa Cruz": "University of California Santa Cruz",
    "Univ. of California - Riverside": "University of California Riverside",
    "Univ. of Maryland - College Park": "University of Maryland College Park",
    "Univ. of Massachusetts Amherst": "University of Massachusetts Amherst",
    "Univ. of Illinois at Urbana-Champaign": "University of Illinois Urbana-Champaign",
    "Massachusetts Inst. of Technology": "Massachusetts Institute of Technology",
    # The top hit for "Purdue University" is the Northwest campus.
    "Purdue University": "Purdue University West Lafayette",
}

# Two or more matching OpenAlex profiles with at least this many works make a match ambiguous.
AMBIGUOUS_WORKS = 50


# === OpenAlex ===

def resolve_department(dept):
    """OpenAlex institution ID of a department's university, or None."""
    query = DEPT_SEARCH_NAMES.get(dept, dept)
    results = openalex.get("institutions", {"search": query, "per_page": 5})["results"]

    def score(c):
        s = name_similarity(query, c["display_name"])
        s += 0.5 * (normalize_name(query) == normalize_name(c["display_name"]))
        return s - 0.2 * (c.get("country_code") not in ("US", "CA"))

    return max(results, key=score)["id"] if results else None


def has_affiliation(author, inst_id):
    """True if the institution is in the author's affiliation history or last known institutions."""
    return (any((a.get("institution") or {}).get("id") == inst_id for a in author.get("affiliations") or [])
            or any(i.get("id") == inst_id for i in author.get("last_known_institutions") or []))


def summarize(author):
    inst = (author.get("last_known_institutions") or [{}])[0]
    return {
        "id": author["id"],
        "display_name": author.get("display_name"),
        "works_count": author.get("works_count") or 0,
        "last_known_institution_id": inst.get("id"),
        "last_known_institution_name": inst.get("display_name"),
    }


def match_openalex(researcher, inst_id):
    """Match one researcher to an OpenAlex author. Returns a profile dict."""
    name = strip_name_suffix(researcher["name"])
    base = {"name": researcher["name"], "dept": researcher["affiliation"],
            "csrankings_pubs": researcher["csrankings_pubs"], "dept_inst_id": inst_id}
    try:
        candidates = openalex.get("authors", {"search": name, "per_page": 25})["results"]
    except Exception as e:
        return {**base, "status": "error", "error": str(e)}

    def matches(require_dept):
        found = [c for c in candidates
                 if name_similarity(name, c.get("display_name")) >= NAME_MATCH
                 and not (require_dept and inst_id and not has_affiliation(c, inst_id))]
        return sorted(found, key=lambda c: -(c.get("works_count") or 0))

    found = matches(require_dept=True)
    ambiguous = sum((c.get("works_count") or 0) >= AMBIGUOUS_WORKS for c in found) >= 2
    if not found and inst_id:  # the department may be missing from OpenAlex's metadata
        found, ambiguous = matches(require_dept=False), True
    if not found:
        return {**base, "status": "unresolved"}
    return {**base, "status": "matched", "source": "openalex", "ambiguous": ambiguous,
            **{k: v for k, v in summarize(found[0]).items() if k != "id"},
            "author_id": found[0]["id"],
            "alternatives": [summarize(c) for c in found[1:]]}


# === Semantic Scholar ===

def match_semantic_scholar(profile):
    """Match a researcher with no OpenAlex profile to a Semantic Scholar author."""
    name, dept = strip_name_suffix(profile["name"]), profile["dept"]
    profile = {k: v for k, v in profile.items() if k != "error"}
    try:
        candidates = semantic_scholar.get("author/search", {
            "query": name, "limit": 25,
            "fields": "name,affiliations,citationCount,paperCount"}).get("data") or []
    except Exception as e:
        return {**profile, "status": "error", "error": str(e)}

    def dept_ok(c):
        return affiliation_matches(" ".join(c.get("affiliations") or []), dept)

    found = [c for c in candidates if name_similarity(name, c.get("name")) >= NAME_MATCH_LOOSE]
    if not found:
        return {**profile, "status": "unresolved"}
    found.sort(key=lambda c: (dept_ok(c), c.get("paperCount") or 0, c.get("citationCount") or 0),
               reverse=True)
    best = found[0]
    return {**profile, "status": "matched", "source": "semantic_scholar",
            "ambiguous": not dept_ok(best) or len(found) > 1,
            "author_id": best["authorId"], "display_name": best.get("name"),
            "works_count": best.get("paperCount") or 0}


# === Cross-check against the CSRankings publication count ===

def rematch_at_department(profile):
    """Find an OpenAlex profile at the department with at least as many works as the
    researcher has CSRankings publications; returns it as a profile update, or None."""
    target, dept = profile["csrankings_pubs"], profile["dept"]

    def update(c):
        return {"source": "openalex", "ambiguous": False, "author_id": c["id"],
                **{k: v for k, v in c.items() if k != "id"}}

    known = sorted((a for a in profile.get("alternatives") or []
                    if affiliation_matches(a.get("last_known_institution_name"), dept)),
                   key=lambda a: -a["works_count"])
    if known and known[0]["works_count"] >= target:
        return update(known[0])

    if not profile.get("dept_inst_id"):
        return None
    name = strip_name_suffix(profile["name"])
    results = openalex.get("authors", {
        "search": name, "per_page": 25,
        "filter": "affiliations.institution.id:" + profile["dept_inst_id"].rsplit("/", 1)[-1],
    })["results"]
    found = sorted((c for c in results if name_similarity(name, c.get("display_name")) >= NAME_MATCH_LOOSE),
                   key=lambda c: -(c.get("works_count") or 0))
    if found and (found[0].get("works_count") or 0) >= target:
        return update(summarize(found[0]))
    return None


# === Driver ===

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only process the first N researchers")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    require_email()

    researchers = load_json(out_path("researchers_list.json"))[:args.limit]
    depts = list(dict.fromkeys(r["affiliation"] for r in researchers))
    inst_ids = {}
    for dept in depts:
        try:
            inst_ids[dept] = resolve_department(dept)
        except Exception as e:
            print(f"  could not resolve {dept!r}: {e}")
            inst_ids[dept] = None
    print(f"Resolved {sum(v is not None for v in inst_ids.values())}/{len(depts)} departments")

    print(f"Matching {len(researchers)} researchers in OpenAlex...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        profiles = list(pool.map(lambda r: match_openalex(r, inst_ids[r["affiliation"]]), researchers))

    failed = [i for i, p in enumerate(profiles) if p["status"] == "error"]
    if failed:
        print(f"Retrying {len(failed)} failed lookups...")
        time.sleep(30)
        openalex.min_interval = 2.0
        for i in failed:
            profiles[i] = match_openalex(researchers[i], inst_ids[researchers[i]["affiliation"]])

    todo = [i for i, p in enumerate(profiles) if p["status"] != "matched"]
    print(f"Searching Semantic Scholar for {len(todo)} researchers...")
    for i in todo:
        profiles[i] = match_semantic_scholar(profiles[i])

    short = [i for i, p in enumerate(profiles)
             if p["status"] == "matched" and p["works_count"] < p["csrankings_pubs"]]
    print(f"Cross-checking {len(short)} profiles with fewer works than CSRankings publications...")
    for i in short:
        try:
            profiles[i].update(rematch_at_department(profiles[i]) or {})
        except Exception as e:
            print(f"  {profiles[i]['name']}: {e}")

    save_json(out_path("profiles.json"), profiles)
    counts = {s: sum(p["status"] == s for p in profiles) for s in ("matched", "unresolved", "error")}
    print(f"\n{counts}; "
          f"{sum(p.get('source') == 'semantic_scholar' for p in profiles)} matched via Semantic Scholar")
    print(f"Wrote {out_path('profiles.json')}")


if __name__ == "__main__":
    main()
