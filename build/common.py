"""Shared helpers for the dataset-building scripts: HTTP access, name matching, file I/O."""

import json
import os
import re
import sys
import threading
import time
import unicodedata

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "build_output")

EMAIL = os.environ.get("OPENALEX_EMAIL", "")  # contact address for the OpenAlex polite pool


def require_email():
    if not EMAIL:
        sys.exit("Set OPENALEX_EMAIL to a contact address (OpenAlex polite pool).")


# === HTTP ===

class Api:
    """A rate-limited JSON API client that retries transient failures.

    `min_interval` is the minimum time in seconds between requests, enforced
    across threads. `get` accepts a path relative to `base` or a full URL.
    """

    RETRY_STATUS = (429, 500, 502, 503, 504)

    def __init__(self, base, min_interval, params=None, max_retries=6):
        self.base = base.rstrip("/")
        self.min_interval = min_interval
        self.params = params or {}
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self._last = 0.0

    def _throttle(self):
        with self._lock:
            wait = self._last + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()

    def get(self, path, params=None, raw=False):
        url = path if path.startswith("http") else f"{self.base}/{path}"
        query = {**self.params, **(params or {})}
        for attempt in range(self.max_retries + 1):
            self._throttle()
            wait = min(2.0 ** attempt, 60.0)
            try:
                resp = requests.get(url, params=query, timeout=45,
                                    headers={"User-Agent": "citation-index-build"})
            except requests.exceptions.RequestException:
                if attempt == self.max_retries:
                    raise
            else:
                if resp.status_code not in self.RETRY_STATUS:
                    resp.raise_for_status()
                    return resp.content if raw else resp.json()
                if attempt == self.max_retries:
                    resp.raise_for_status()
                retry_after = resp.headers.get("Retry-After", "")
                if resp.status_code == 429 and retry_after.isdigit():
                    wait = min(float(retry_after), 60.0)
            time.sleep(wait)


openalex = Api("https://api.openalex.org", 0.1, params={"mailto": EMAIL} if EMAIL else None)
semantic_scholar = Api("https://api.semanticscholar.org/graph/v1", 1.2)


def openalex_citations(author_id):
    """Citation counts of all works attributed to an OpenAlex author, in non-increasing order."""
    bare = author_id.rsplit("/", 1)[-1]
    counts, cursor = [], "*"
    while cursor:
        data = openalex.get("works", {
            "filter": f"authorships.author.id:{bare}",
            "select": "id,cited_by_count",
            "sort": "cited_by_count:desc",
            "per_page": 200,
            "cursor": cursor,
        })
        if not data.get("results"):
            break
        counts += [w.get("cited_by_count") or 0 for w in data["results"]]
        cursor = data.get("meta", {}).get("next_cursor")
    return sorted(counts, reverse=True)


def semantic_scholar_citations(author_id):
    """Citation counts of all papers of a Semantic Scholar author, in non-increasing order."""
    counts, offset, limit = [], 0, 500
    while True:
        data = semantic_scholar.get(f"author/{author_id}/papers",
                                    {"fields": "citationCount", "offset": offset, "limit": limit})
        papers = data.get("data") or []
        counts += [p.get("citationCount") or 0 for p in papers]
        if len(papers) < limit:
            break
        offset += limit
    return sorted(counts, reverse=True)


# === Files ===

def out_path(*parts):
    return os.path.join(OUT_DIR, *parts)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    """Write JSON atomically, creating the directory if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# === Names ===

# Characters that Unicode decomposition leaves alone.
_FOLD = str.maketrans({
    "ı": "i", "İ": "I", "ş": "s", "Ş": "S", "ğ": "g", "Ğ": "G", "ç": "c", "Ç": "C",
    "ö": "o", "Ö": "O", "ü": "u", "Ü": "U", "ł": "l", "Ł": "L", "ø": "o", "Ø": "O",
    "æ": "ae", "Æ": "AE", "–": "-", "‐": "-", "‑": "-", "—": "-",
})

# Name-similarity thresholds (token Jaccard similarity).
NAME_MATCH = 0.7        # primary OpenAlex lookup
NAME_MATCH_LOOSE = 0.6  # Semantic Scholar lookup; OpenAlex re-query by institution
NAME_MATCH_MIN = 0.2    # last resort: a shared last name is enough (including a nickname, as in Ken and Kenneth)


def strip_name_suffix(name):
    """Drop DBLP's disambiguation suffix: "Bo Li 0026" -> "Bo Li"."""
    return re.sub(r"\s+0\d{3}$", "", name).strip()


def normalize_name(s):
    """Lowercase, strip accents, drop punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFKD", (s or "").translate(_FOLD))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", " ", s)).strip()


def name_similarity(a, b):
    """Jaccard similarity of the name tokens (single letters ignored)."""
    ta = {t for t in normalize_name(a).split() if len(t) > 1}
    tb = {t for t in normalize_name(b).split() if len(t) > 1}
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


# Lower-case substrings that identify each department in an affiliation string.
DEPT_KEYWORDS = {
    "Univ. of California - San Diego": ["san diego", "ucsd"],
    "Univ. of California - Berkeley": ["berkeley"],
    "University of Michigan": ["michigan", "ann arbor"],
    "Univ. of Maryland - College Park": ["maryland", "college park"],
    "Northeastern University": ["northeastern"],
    "University of Toronto": ["toronto"],
    "Purdue University": ["purdue"],
    "University of Wisconsin - Madison": ["wisconsin", "madison"],
    "Princeton University": ["princeton"],
    "University of Waterloo": ["waterloo"],
    "University of Pennsylvania": ["pennsylvania", "penn"],
    "Carnegie Mellon University": ["carnegie mellon", "cmu"],
    "Univ. of Illinois at Urbana-Champaign": ["illinois", "urbana", "uiuc", "champaign"],
    "Georgia Institute of Technology": ["georgia"],
    "Massachusetts Inst. of Technology": ["mit", "massachusetts institute"],
    "University of Washington": ["university of washington", "u of washington"],
    "Cornell University": ["cornell"],
    "Stanford University": ["stanford"],
    "New York University": ["new york university", "nyu"],
    "University of Texas at Austin": ["university of texas", "ut austin", "austin"],
}


def affiliation_matches(text, dept):
    """True if an affiliation string names the department."""
    text = (text or "").lower()
    return any(k in text for k in DEPT_KEYWORDS.get(dept, [dept.lower()]))
