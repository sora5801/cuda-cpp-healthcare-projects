#!/usr/bin/env python3
# =============================================================================
# tools/status.py  --  Generate docs/STATUS.md (the work-queue dashboard)
# -----------------------------------------------------------------------------
# WHAT THIS IS
#   The single dashboard that answers "what is done, what is in progress, what
#   is left?" across all 301 projects. It is GENERATED, never hand-edited:
#       catalog.json            (the immutable list of projects)
#     + each project's marker    (projects/.../.project_status.json)
#     -> docs/STATUS.md
#
# WHY A PER-PROJECT MARKER (and not editable rows in STATUS.md)
#   CLAUDE.md section 10's cardinal rule is "one agent owns one project folder;
#   agents never edit files outside their own folder." So the source of truth
#   for a project's status lives INSIDE that project's folder, in
#   `.project_status.json`, which only that project's worker touches. This tool
#   (run by the lead) aggregates those markers into the shared STATUS.md. That
#   way concurrent workers never collide on a single shared status file.
#
#   Marker schema (created by scaffold.py, updated via `status.py --set`):
#       { "id": "1.12", "id_padded": "1.12",
#         "project": "Molecular Fingerprint Similarity Search",
#         "status": "todo" | "in-progress" | "done",
#         "owner": "<agent-or-empty>", "branch": "<branch-or-empty>",
#         "skeleton": true }
#
# USAGE
#   python tools/status.py                          # regenerate docs/STATUS.md
#   python tools/status.py --set 1.12 in-progress --owner alice --branch proj/1.12-...
#   python tools/status.py --set 1.12 done          # then regenerate
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "catalog.json"
OUT = ROOT / "docs" / "STATUS.md"
MARKER = ".project_status.json"

VALID_STATUS = ("todo", "in-progress", "done")
# Emoji rendered in the dashboard for each status / difficulty.
STATUS_BADGE = {"todo": "[ ] todo", "in-progress": "[~] in-progress",
                "done": "[x] done", "(missing)": "[!] MISSING"}
DIFF_BADGE = {"Beginner": "\U0001F7E2 Beginner",
              "Intermediate": "\U0001F7E1 Intermediate",
              "Advanced": "\U0001F534 Advanced"}


def load_catalog():
    if not CATALOG.exists():
        sys.exit("[status] FATAL: catalog.json not found. Run tools/catalog.py first.")
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def marker_path(rec) -> Path:
    return ROOT / rec["folder_path"] / MARKER


def read_marker(rec) -> dict:
    """Return the project's status marker, or a synthetic '(missing)' record
    when the folder/marker has not been scaffolded yet."""
    p = marker_path(rec)
    if not p.exists():
        return {"status": "(missing)", "owner": "", "branch": ""}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "(missing)", "owner": "", "branch": f"<unreadable: {exc}>"}


def set_status(rec_by_id, project_id, status, owner, branch):
    """Update one project's marker file in place (the only mutation this tool
    performs, and only ever within that project's own folder)."""
    if status not in VALID_STATUS:
        sys.exit(f"[status] FATAL: status must be one of {VALID_STATUS}, got '{status}'.")
    rec = rec_by_id.get(project_id)
    if not rec:
        sys.exit(f"[status] FATAL: unknown project id '{project_id}'.")
    p = marker_path(rec)
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {
        "id": rec["id"], "id_padded": rec["id_padded"], "project": rec["project"],
        "skeleton": True,
    }
    data["status"] = status
    if owner is not None:
        data["owner"] = owner
    if branch is not None:
        data["branch"] = branch
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[status] {project_id} -> {status}"
          f"{' owner=' + owner if owner else ''}{' branch=' + branch if branch else ''}")


def render(catalog) -> str:
    """Build the STATUS.md markdown text from catalog + markers."""
    # Pull each project's live status from its marker.
    rows = []
    for rec in catalog:
        m = read_marker(rec)
        rows.append({**rec, "status": m.get("status", "todo"),
                     "owner": m.get("owner", ""), "branch": m.get("branch", "")})

    total = len(rows)
    by_status = {k: 0 for k in (*VALID_STATUS, "(missing)")}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    done = by_status.get("done", 0)
    pct = (100.0 * done / total) if total else 0.0

    out = []
    out.append("# Build status dashboard")
    out.append("")
    out.append("> **Generated file** -- do not hand-edit. Regenerate with "
               "`python tools/status.py`. A project's status lives in its own "
               "`projects/.../.project_status.json` marker (see the tool header "
               "for why), and is changed with `python tools/status.py --set <id> <status>`.")
    out.append("")
    out.append(f"**Progress:** {done}/{total} projects done ({pct:.1f}%) | "
               f"todo {by_status.get('todo', 0)} | "
               f"in-progress {by_status.get('in-progress', 0)} | "
               f"missing {by_status.get('(missing)', 0)}")
    out.append("")

    # Per-difficulty rollup (Beginner-first is the build order within a domain).
    out.append("| Difficulty | Total | done | in-progress | todo |")
    out.append("|---|---:|---:|---:|---:|")
    for diff in ("Beginner", "Intermediate", "Advanced"):
        sub = [r for r in rows if r["difficulty"] == diff]
        d = sum(1 for r in sub if r["status"] == "done")
        ip = sum(1 for r in sub if r["status"] == "in-progress")
        td = sum(1 for r in sub if r["status"] in ("todo", "(missing)"))
        out.append(f"| {DIFF_BADGE.get(diff, diff)} | {len(sub)} | {d} | {ip} | {td} |")
    out.append("")

    # Per-domain detailed tables. Domains appear in section order (1..14).
    seen_domains = []
    for r in rows:
        if r["domain"] not in seen_domains:
            seen_domains.append(r["domain"])

    for dom in seen_domains:
        sub = [r for r in rows if r["domain"] == dom]
        sec = sub[0]["section"]
        d = sum(1 for r in sub if r["status"] == "done")
        out.append(f"## {sec}. {dom}  ({d}/{len(sub)} done)")
        out.append("")
        out.append("| ID | Project | Difficulty | Maturity | Status | Owner | Branch |")
        out.append("|---|---|---|---|---|---|---|")
        for r in sub:
            out.append(
                f"| {r['id_padded']} | {r['project']} | "
                f"{DIFF_BADGE.get(r['difficulty'], r['difficulty'])} | {r['maturity']} | "
                f"{STATUS_BADGE.get(r['status'], r['status'])} | "
                f"{r['owner'] or '-'} | {r['branch'] or '-'} |"
            )
        out.append("")

    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Generate docs/STATUS.md from catalog + markers.")
    ap.add_argument("--set", nargs=2, metavar=("ID", "STATUS"),
                    help="update one project's status marker, then regenerate")
    ap.add_argument("--owner", default=None, help="owner to set with --set")
    ap.add_argument("--branch", default=None, help="branch to set with --set")
    args = ap.parse_args()

    catalog = load_catalog()
    rec_by_id = {r["id"]: r for r in catalog}

    if args.set:
        set_status(rec_by_id, args.set[0], args.set[1], args.owner, args.branch)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(catalog), encoding="utf-8")
    print(f"[status] Wrote {OUT.relative_to(ROOT)} for {len(catalog)} projects.")


if __name__ == "__main__":
    main()
