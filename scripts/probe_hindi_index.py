"""Start-of-run probe for the multilingual check: a Hindi query must return Hindi passages from data/bench/xq_hi.

The index uses an FTS5 tokenizer option (`unicode61 categories 'L* N* Co M*'`) that keeps Devanagari words whole. If the SQLite build on the
machine cannot read it, AgentTune's backend reports "no results" instead of an error and every Hindi search would silently come back empty.
Exits non-zero in that case, so the run stops before any GPU time is spent.
"""

import sqlite3
import sys

from check_xquad import hindi_probe


def main() -> None:
    print("sqlite", sqlite3.sqlite_version, flush=True)
    probe = hindi_probe()
    print(probe, flush=True)
    if probe["hits"] < 1 or not probe["devanagari_titles"] or not probe["answer_passage_in_top3"]:
        sys.exit("Hindi index probe failed: " + str(probe))


if __name__ == "__main__":
    main()
