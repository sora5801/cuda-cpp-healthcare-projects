#!/usr/bin/env python3
# =============================================================================
# tools/new_pushnote.py  --  Create a dated push-note stub + index it
# -----------------------------------------------------------------------------
# WHY THIS EXISTS
#   CLAUDE.md section 7.1 makes one rule load-bearing: EVERY push to origin/main
#   ships with a push-note under push-notes/ that explains what was added, so
#   the repo always documents its own latest state. This tool stamps the dated
#   stub (with all eight required sections) and prepends a one-line entry to the
#   root CHANGELOG.md so the changelog stays an index into push-notes/.
#
# NAMING:  push-notes/YYYY-MM-DD-NN-short-title.md
#   NN = that day's push counter, zero-padded (00 for the first push of the day,
#   01 for the second, ...). Computed by counting existing notes for that date.
#
# USAGE
#   python tools/new_pushnote.py "bootstrap"
#   python tools/new_pushnote.py "flagship 1.12 tanimoto" --date 2026-06-29
# =============================================================================

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = ROOT / "push-notes"
CHANGELOG = ROOT / "CHANGELOG.md"


def slug(text: str) -> str:
    """Filesystem-safe short title: lowercase, non-alnum runs -> single '-'."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "push"


def next_counter(date_str: str) -> int:
    """How many push-notes already exist for this date -> the next NN."""
    if not NOTES_DIR.exists():
        return 0
    rx = re.compile(rf"^{re.escape(date_str)}-(\d\d)-")
    used = [int(m.group(1)) for f in NOTES_DIR.glob(f"{date_str}-*.md")
            if (m := rx.match(f.name))]
    return (max(used) + 1) if used else 0


# The stub mirrors the eight mandatory sections in CLAUDE.md section 7.1.
STUB = """\
# Push {date} #{nn:02d} -- {title}

> Push-note (CLAUDE.md section 7.1). Fill every section before pushing; the repo
> must explain its own latest state. Delete these blockquote hints when done.

## 1. Summary
<!-- One paragraph: what this push adds and why it matters to the learner. -->
TODO

## 2. What changed
<!-- New/edited projects and files, grouped and linked with relative paths. -->
TODO

## 3. New projects (didactic blurb each)
<!-- For each new project: 3-5 sentences -- concept taught, the CUDA pattern,
     and the single most interesting thing to look at. -->
TODO

## 4. How to build & run
<!-- Exact commands: open build/<slug>.sln (Release|x64), Build; run demo. -->
TODO

## 5. What to study here
<!-- A suggested reading path through the new material + 1-2 exercises. -->
TODO

## 6. Verification
<!-- What was checked: build passed? demo matched expected_output? on which
     GPU / compute capability / CUDA + VS version? verify_project.py result? -->
TODO

## 7. Known limitations / TODOs
<!-- Honest notes: what is simplified, synthetic, or deferred. -->
TODO

## 8. Next push preview
<!-- What is planned next. -->
TODO
"""


def main():
    ap = argparse.ArgumentParser(description="Create a dated push-note stub and index it.")
    ap.add_argument("title", help="short title, e.g. \"flagship 1.12 tanimoto\"")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--force", action="store_true", help="overwrite if the file exists")
    args = ap.parse_args()

    if args.date:
        try:
            _dt.date.fromisoformat(args.date)  # validate format
        except ValueError:
            sys.exit(f"[pushnote] FATAL: --date must be YYYY-MM-DD, got '{args.date}'.")
        date_str = args.date
    else:
        date_str = _dt.date.today().isoformat()

    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    nn = next_counter(date_str)
    title_slug = slug(args.title)
    fname = f"{date_str}-{nn:02d}-{title_slug}.md"
    path = NOTES_DIR / fname

    if path.exists() and not args.force:
        sys.exit(f"[pushnote] FATAL: {path.name} already exists (use --force).")

    path.write_text(STUB.format(date=date_str, nn=nn, title=args.title), encoding="utf-8")

    # Prepend a one-line index entry to CHANGELOG.md (newest first).
    entry = f"- {date_str} #{nn:02d} -- [{args.title}](push-notes/{fname})\n"
    if CHANGELOG.exists():
        existing = CHANGELOG.read_text(encoding="utf-8")
    else:
        existing = "# Changelog\n\nOne line per push; each links to its push-note in `push-notes/`.\n\n"
    # Insert directly after the header block (first blank line after a heading).
    lines = existing.splitlines(keepends=True)
    insert_at = len(lines)
    for i, ln in enumerate(lines):
        if i > 0 and ln.strip() == "" and any(l.startswith("#") for l in lines[:i]):
            insert_at = i + 1
            break
    lines.insert(insert_at, entry)
    CHANGELOG.write_text("".join(lines), encoding="utf-8")

    print(f"[pushnote] Created push-notes/{fname}")
    print(f"[pushnote] Indexed in CHANGELOG.md")
    print(path)


if __name__ == "__main__":
    main()
