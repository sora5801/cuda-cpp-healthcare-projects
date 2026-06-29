#!/usr/bin/env python3
# =============================================================================
# tools/scaffold.py  --  Stamp all 301 project skeletons from PROJECT_TEMPLATE
# -----------------------------------------------------------------------------
# WHAT THIS DOES
#   Reads catalog.json and, for every project, copies docs/PROJECT_TEMPLATE/
#   into projects/<domain-slug>/<id_padded>-<slug>/ while:
#     * replacing __TOKENS__ in every text file with that project's catalog data,
#     * renaming files containing __PROJECT_SLUG__ to the real slug,
#     * writing a .project_status.json marker (status="todo", skeleton=true).
#
#   The result is 301 BUILDABLE skeletons -- each is the SAXPY placeholder from
#   the template (so the whole repo compiles from day one), clearly marked with
#   TODO(impl)/TODO(theory) for the author to replace (CLAUDE.md §11 Phase 0).
#
# IDEMPOTENT & NON-CLOBBERING
#   By default a project folder that ALREADY EXISTS is skipped, so re-running
#   after the catalog grows only fills in the new ones and never overwrites a
#   worker's real content. Use --force to re-stamp (re-stamps ALL; use with
#   care) or --only <id> to (re)stamp a single project.
#
# USAGE
#   python tools/scaffold.py                 # create any missing skeletons
#   python tools/scaffold.py --only 1.12     # (re)create one project
#   python tools/scaffold.py --force         # overwrite every project (danger)
# =============================================================================

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "catalog.json"
TEMPLATE = ROOT / "docs" / "PROJECT_TEMPLATE"
SLUG_TOKEN = "__PROJECT_SLUG__"
MARKER = ".project_status.json"

# Stable namespace so each project's MSBuild ProjectGuid is deterministic
# (same id -> same GUID on every machine and every run): uuid5(NS, id).
GUID_NS = uuid.UUID("a1b2c3d4-0001-4abc-8def-001122334455")

DIFF_EMOJI = {"Beginner": "\U0001F7E2", "Intermediate": "\U0001F7E1", "Advanced": "\U0001F534"}


def shield(label: str) -> str:
    """Encode a label for a shields.io static badge path.

    shields rules: a literal '-' must be doubled to '--'; spaces and other
    special characters are percent-encoded. We percent-encode everything unsafe
    first, then double any remaining dashes.
    """
    return quote(label, safe="").replace("-", "--")


def tokens_for(rec: dict) -> dict:
    """Build the full __TOKEN__ -> value substitution map for one project."""
    guid = str(uuid.uuid5(GUID_NS, rec["id"])).upper()
    return {
        "__PROJECT_ID__": rec["id"],
        "__PROJECT_ID_PADDED__": rec["id_padded"],
        "__PROJECT_NAME__": rec["project"],
        "__PROJECT_SLUG__": rec["project_slug"],
        "__DOMAIN__": rec["domain"],
        "__DOMAIN_SLUG__": rec["domain_slug"],
        "__SECTION__": str(rec["section"]),
        "__DIFFICULTY__": rec["difficulty"],
        "__DIFFICULTY_EMOJI__": DIFF_EMOJI.get(rec["difficulty"], ""),
        "__DIFFICULTY_BADGE__": shield(rec["difficulty"]),
        "__MATURITY__": rec["maturity"],
        "__MATURITY_BADGE__": shield(rec["maturity"]),
        "__DOMAIN_BADGE__": shield(rec["domain"]),
        "__DEEP_DIVE__": rec["deep_dive"],
        "__ALGORITHMS__": rec["algorithms"],
        "__DATASETS__": rec["datasets"],
        "__REPOS__": rec["repos"],
        "__CUDA_GPU__": rec["cuda_gpu"],
        "__DEEPDIVE_MD__": rec["deepdive_md"] or "_(no catalog deep-dive text found for this ID)_",
        "__PROJECT_GUID__": guid,
    }


def substitute(text: str, tokens: dict) -> str:
    for k, v in tokens.items():
        text = text.replace(k, v)
    return text


def stamp_project(rec: dict, force: bool) -> str:
    """Create one project folder from the template. Returns 'created'|'skipped'."""
    dest = ROOT / rec["folder_path"]
    if dest.exists() and not force:
        return "skipped"
    if dest.exists() and force:
        shutil.rmtree(dest)

    tokens = tokens_for(rec)

    # Walk every file in the template (rglob includes dotfiles like .gitignore).
    for src in sorted(TEMPLATE.rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(TEMPLATE)
        # Rename any path component carrying the slug token (the build/ files).
        rel_str = str(rel).replace(SLUG_TOKEN, rec["project_slug"])
        out_path = dest / rel_str
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Every template file is UTF-8 text -> substitute tokens, then write.
        raw = src.read_text(encoding="utf-8")
        out_path.write_text(substitute(raw, tokens), encoding="utf-8")

    # The status marker (also the idempotency/skeleton flag) -- see status.py.
    marker = {
        "id": rec["id"], "id_padded": rec["id_padded"], "project": rec["project"],
        "status": "todo", "owner": "", "branch": "", "skeleton": True,
    }
    (dest / MARKER).write_text(json.dumps(marker, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
    return "created"


def main():
    ap = argparse.ArgumentParser(description="Stamp project skeletons from PROJECT_TEMPLATE.")
    ap.add_argument("--only", metavar="ID", help="only (re)stamp this project id, e.g. 1.12")
    ap.add_argument("--force", action="store_true", help="overwrite existing folders (danger)")
    args = ap.parse_args()

    if not CATALOG.exists():
        sys.exit("[scaffold] FATAL: catalog.json not found. Run tools/catalog.py first.")
    if not TEMPLATE.exists():
        sys.exit(f"[scaffold] FATAL: template not found at {TEMPLATE}.")

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    if args.only:
        catalog = [r for r in catalog if r["id"] == args.only]
        if not catalog:
            sys.exit(f"[scaffold] FATAL: no project with id '{args.only}'.")

    created = skipped = 0
    for rec in catalog:
        result = stamp_project(rec, force=args.force or bool(args.only))
        created += (result == "created")
        skipped += (result == "skipped")

    print(f"[scaffold] done: {created} created, {skipped} skipped "
          f"(of {len(catalog)} considered).")
    if skipped and not args.force:
        print("[scaffold] (existing folders were left untouched; use --force to re-stamp.)")


if __name__ == "__main__":
    main()
