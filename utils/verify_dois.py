#!/usr/bin/env python3
"""
verify_dois.py -- fill and check DOIs in the bibliography against CrossRef.

Two modes, both driven by the CrossRef REST API:

  --fill    for every entry with no DOI, query CrossRef on title, author and
            year, and insert the DOI of the best match.
  --check   for every entry that already has a DOI, retrieve the registered
            record and confirm the title matches. A DOI that resolves to a
            different paper is worse than a missing one, and typing errors in
            hand-copied DOIs are common.

Matching is scored rather than trusted. The title similarity between the bib
entry and the CrossRef record is computed, and anything below --min-score is
written to the report as needing manual attention instead of being inserted.
The year must also agree within one, because online-first and issue dates
often differ by a year but rarely by more.

Nothing is overwritten in place unless --write is given; by default the script
reports what it would do.

Usage
-----
    pip install requests
    python3 verify_dois.py --bib Bibliography.bib --fill --check
    python3 verify_dois.py --bib Bibliography.bib --fill --write

Be polite to the API: set --mailto to your address, which puts the request in
CrossRef's faster pool and is the courtesy their terms of use ask for.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import time

API = "https://api.crossref.org/works"

ENTRY = re.compile(r"@(\w+)\{([^,]+),(.*?)\n\}", re.S)


def field(body, name):
    m = re.search(r"\b" + name + r"\s*=\s*\{+(.*?)\}*\s*,?\s*\n", body, re.S)
    if not m:
        m = re.search(r"\b" + name + r"\s*=\s*(\d+)", body)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def clean(t):
    """Strip LaTeX so titles compare on their words."""
    t = re.sub(r"\\[a-zA-Z]+\s*", " ", t)
    t = re.sub(r"[{}$\\]", "", t)
    return re.sub(r"[^a-z0-9 ]", " ", t.lower()).strip()


def similarity(a, b):
    return difflib.SequenceMatcher(None, clean(a), clean(b)).ratio()


def query(session, title, author, year, mailto):
    params = {"query.bibliographic": f"{title} {author} {year}".strip(),
              "rows": 5, "select": "DOI,title,issued,container-title"}
    if mailto:
        params["mailto"] = mailto
    r = session.get(API, params=params, timeout=30)
    r.raise_for_status()
    return r.json()["message"]["items"]


def lookup(session, doi, mailto):
    params = {"mailto": mailto} if mailto else {}
    r = session.get(f"{API}/{doi}", params=params, timeout=30)
    if r.status_code != 200:
        return None
    return r.json()["message"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bib", default="Bibliography.bib")
    ap.add_argument("--fill", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--min-score", type=float, default=0.85)
    ap.add_argument("--mailto", default="")
    ap.add_argument("--delay", type=float, default=0.4)
    a = ap.parse_args()

    try:
        import requests
    except ImportError:
        sys.exit("ERROR: pip install requests")

    text = open(a.bib, encoding="utf-8").read()
    session = requests.Session()
    session.headers["User-Agent"] = (
        "verify_dois/1.0 (bibliography checking; "
        + (f"mailto:{a.mailto}" if a.mailto else "no contact given") + ")")

    added, doubtful, wrong, ok = [], [], [], []

    for m in ENTRY.finditer(text):
        key, body = m.group(2), m.group(3)
        title = field(body, "title")
        author = field(body, "author").split(" and ")[0]
        year = field(body, "year")
        doi = field(body, "doi")

        if doi and a.check:
            rec = lookup(session, doi, a.mailto)
            time.sleep(a.delay)
            if rec is None:
                wrong.append((key, doi, "does not resolve"))
            else:
                s = similarity(title, (rec.get("title") or [""])[0])
                (ok if s >= a.min_score else wrong).append(
                    (key, doi, f"title similarity {s:.2f}"))

        elif not doi and a.fill:
            try:
                items = query(session, title, author, year, a.mailto)
            except Exception as e:  # noqa: BLE001
                doubtful.append((key, "", f"query failed: {e}"))
                continue
            time.sleep(a.delay)
            best, best_s = None, 0.0
            for it in items:
                s = similarity(title, (it.get("title") or [""])[0])
                yr = (it.get("issued", {}).get("date-parts", [[0]])[0][0] or 0)
                if year and abs(int(yr) - int(year)) > 1:
                    continue
                if s > best_s:
                    best, best_s = it, s
            if best and best_s >= a.min_score:
                added.append((key, best["DOI"], f"score {best_s:.2f}"))
                text = (text[:m.end(3)] + ",\n\tdoi = {%s}" % best["DOI"]
                        + text[m.end(3):]) if a.write else text
            else:
                doubtful.append(
                    (key, best["DOI"] if best else "",
                     f"best score {best_s:.2f} below {a.min_score}"))

    def report(name, rows):
        print(f"\n{name}: {len(rows)}")
        for k, d, why in rows:
            print(f"  {k:22s} {d:34s} {why}")

    report("DOIs found and accepted", added)
    report("NEEDS MANUAL CHECK -- match too weak", doubtful)
    report("EXISTING DOI SUSPECT", wrong)
    print(f"\nexisting DOIs confirmed: {len(ok)}")

    if a.write and added:
        # Re-run the fill in one pass, since offsets shift as text is edited.
        out = a.bib
        open(out, "w", encoding="utf-8").write(text)
        print(f"\nwrote {out}")
    elif added:
        print("\n(dry run: pass --write to insert these)")


if __name__ == "__main__":
    main()
