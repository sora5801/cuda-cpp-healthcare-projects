#!/usr/bin/env python3
# =============================================================================
# tools/verify_project.py  --  Check a project against the Definition of Done
# -----------------------------------------------------------------------------
# WHAT THIS IS
#   The automated gatekeeper from CLAUDE.md §9. It checks the STRUCTURAL and
#   COMMENTING gates that a machine can judge, and prints a pass/fail checklist.
#   It does NOT compile or run kernels -- building + demoing are local steps that
#   need the GPU (CI on hosted runners has none). A project is "done" only when
#   every gate here passes AND a human has built it, run the demo, and confirmed
#   it matches expected_output.txt.
#
#   Gates checked here:
#     [structure]  the §4 file layout exists (README, THEORY, src/*, data, demo,
#                  scripts, build/.sln+.vcxproj(+filters), .gitignore, status marker)
#     [readme]     all §4.1 sections present
#     [theory]     the core §4.2 sections present
#     [demo]       demo/expected_output.txt present and non-empty
#     [comments]   src/ comment density >= floor (default 0.40, non-trivial lines)
#     [todos]      no scaffold TODO(impl)/TODO(theory) markers remain
#
#   The first five are "structure ready"; [todos] (and a human build/demo) is
#   what separates a SKELETON from a DONE project. A fresh skeleton therefore
#   reports "NOT DONE: scaffold TODOs remain" -- which is expected.
#
# USAGE
#   python tools/verify_project.py projects/01-drug-discovery/1.12-...   # one
#   python tools/verify_project.py --all                                 # sweep
#   python tools/verify_project.py --all --quiet                         # summary only
# Exit code: 0 if the target project (or all, with --all) is DONE; else 1.
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "catalog.json"

SRC_EXTS = {".cu", ".cuh", ".cpp", ".h", ".hpp", ".cc", ".cxx", ".c"}
DEFAULT_MIN_RATIO = 0.40

# Section markers we expect to find (substring match against the file text).
README_SECTIONS = [
    "Summary", "What this computes", "The algorithm in brief", "## Build",
    "Run the demo", "## Data", "Expected output", "Code tour",
    "Prior art", "Exercises", "Limitations",
]
THEORY_SECTIONS = [
    "The science", "The math", "The algorithm", "GPU mapping",
    "Numerical considerations", "verify correctness", "real world",
]


class Result:
    def __init__(self):
        self.checks = []   # list of (level, name, ok, detail)

    def add(self, name, ok, detail="", level="required"):
        self.checks.append((level, name, ok, detail))

    def required_failures(self):
        return [c for c in self.checks if c[0] == "required" and not c[2]]

    def warnings(self):
        return [c for c in self.checks if c[0] == "warn" and not c[2]]


def is_nontrivial_comment(stripped_inner: str) -> bool:
    """A comment counts only if it carries real words (>= 3 letters), so rulers
    like '// --------' and '////' do not inflate the density."""
    letters = sum(ch.isalpha() for ch in stripped_inner)
    return letters >= 3


def comment_density(src_dir: Path):
    """Return (comment_lines, code_lines, ratio) over src/ source files.
    Blank lines are ignored; trailing-line comments are not separately counted
    (a conservative, simple heuristic)."""
    comm = code = 0
    for f in sorted(src_dir.rglob("*")):
        if f.suffix not in SRC_EXTS or not f.is_file():
            continue
        in_block = False
        for raw in f.read_text(encoding="utf-8", errors="replace").splitlines():
            s = raw.strip()
            if not s:
                continue
            if in_block:
                if is_nontrivial_comment(s.replace("*", "")):
                    comm += 1
                else:
                    comm += 0  # ruler line inside block: ignore
                if "*/" in s:
                    in_block = False
                continue
            if s.startswith("//"):
                if is_nontrivial_comment(s[2:]):
                    comm += 1
            elif s.startswith("/*"):
                if is_nontrivial_comment(s.replace("/*", "").replace("*/", "")):
                    comm += 1
                if "*/" not in s:
                    in_block = True
            else:
                code += 1
    ratio = (comm / code) if code else float("inf")
    return comm, code, ratio


def verify(project_dir: Path, min_ratio: float) -> Result:
    r = Result()
    P = project_dir

    # ---- [structure] required files -------------------------------------
    required = {
        "README.md": P / "README.md",
        "THEORY.md": P / "THEORY.md",
        "src/main.cu": P / "src" / "main.cu",
        "src/kernels.cu": P / "src" / "kernels.cu",
        "src/kernels.cuh": P / "src" / "kernels.cuh",
        "src/reference_cpu.cpp": P / "src" / "reference_cpu.cpp",
        "data/README.md": P / "data" / "README.md",
        "demo/expected_output.txt": P / "demo" / "expected_output.txt",
        ".project_status.json": P / ".project_status.json",
        ".gitignore": P / ".gitignore",
    }
    for name, path in required.items():
        r.add(f"structure: {name}", path.exists())

    # util/ must exist and be non-empty (shared CUDA_CHECK + timer + io).
    util = P / "src" / "util"
    r.add("structure: src/util/* present",
          util.is_dir() and any(util.iterdir()))

    # data/sample must contain at least one committed file (offline demo).
    sample = P / "data" / "sample"
    r.add("structure: data/sample/ non-empty",
          sample.is_dir() and any(f.is_file() for f in sample.rglob("*")))

    # at least one data download script.
    r.add("structure: scripts/download_data.(ps1|sh)",
          (P / "scripts" / "download_data.ps1").exists()
          or (P / "scripts" / "download_data.sh").exists())

    # at least one demo runner.
    r.add("structure: demo/run_demo.(ps1|sh)",
          (P / "demo" / "run_demo.ps1").exists()
          or (P / "demo" / "run_demo.sh").exists())

    # build/: a .sln, a .vcxproj, and a .filters (the required VS deliverable).
    build = P / "build"
    r.add("structure: build/*.sln", build.is_dir() and any(build.glob("*.sln")))
    r.add("structure: build/*.vcxproj", build.is_dir() and any(build.glob("*.vcxproj")))
    r.add("structure: build/*.vcxproj.filters",
          build.is_dir() and any(build.glob("*.vcxproj.filters")))

    # CMake is a nice-to-have (warn only).
    r.add("structure: CMakeLists.txt (optional)", (P / "CMakeLists.txt").exists(),
          level="warn")

    # ---- [readme] sections ----------------------------------------------
    readme = (P / "README.md")
    readme_txt = readme.read_text(encoding="utf-8", errors="replace") if readme.exists() else ""
    for sec in README_SECTIONS:
        r.add(f"readme: '{sec}' section", sec in readme_txt)

    # ---- [theory] sections ----------------------------------------------
    theory = (P / "THEORY.md")
    theory_txt = theory.read_text(encoding="utf-8", errors="replace") if theory.exists() else ""
    for sec in THEORY_SECTIONS:
        r.add(f"theory: '{sec}' section", sec in theory_txt)

    # ---- [demo] expected output non-empty -------------------------------
    eo = P / "demo" / "expected_output.txt"
    r.add("demo: expected_output.txt non-empty",
          eo.exists() and eo.stat().st_size > 0)

    # ---- [comments] density ---------------------------------------------
    src = P / "src"
    if src.is_dir():
        comm, code, ratio = comment_density(src)
        r.add(f"comments: src ratio {ratio:.2f} >= {min_ratio:.2f} "
              f"({comm} comment / {code} code)", ratio >= min_ratio)
    else:
        r.add("comments: src/ exists", False)

    # ---- [todos] no scaffold TODOs remain (skeleton vs done) ------------
    todo_hits = []
    for f in sorted(P.rglob("*")):
        if f.is_file() and f.suffix in (SRC_EXTS | {".md", ".txt", ".py", ".ps1", ".sh"}):
            try:
                txt = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if "TODO(impl)" in txt or "TODO(theory)" in txt:
                todo_hits.append(str(f.relative_to(P)))
    r.add(f"todos: no scaffold TODO markers remain"
          + (f" ({len(todo_hits)} files still have them)" if todo_hits else ""),
          len(todo_hits) == 0)

    return r


def print_result(project_dir: Path, r: Result, quiet: bool):
    fails = r.required_failures()
    warns = r.warnings()
    done = not fails
    if not quiet:
        print(f"\n=== verify: {project_dir.relative_to(ROOT)} ===")
        for level, name, ok, detail in r.checks:
            mark = "PASS" if ok else ("WARN" if level == "warn" else "FAIL")
            line = f"  [{mark}] {name}"
            if detail:
                line += f"  -- {detail}"
            print(line)
    status = "DONE" if done else f"NOT DONE ({len(fails)} required gate(s) failing)"
    print(f"  -> {project_dir.name}: {status}"
          + (f", {len(warns)} warning(s)" if warns else ""))
    return done


def main():
    ap = argparse.ArgumentParser(description="Verify a project against the Definition of Done.")
    ap.add_argument("path", nargs="?", help="path to a project folder")
    ap.add_argument("--all", action="store_true", help="verify every project in catalog.json")
    ap.add_argument("--quiet", action="store_true", help="print only the per-project verdict")
    ap.add_argument("--min-comment-ratio", type=float, default=DEFAULT_MIN_RATIO)
    args = ap.parse_args()

    if args.all:
        if not CATALOG.exists():
            sys.exit("[verify] FATAL: catalog.json not found. Run tools/catalog.py first.")
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        done = 0
        missing = 0
        for rec in catalog:
            pdir = ROOT / rec["folder_path"]
            if not pdir.exists():
                missing += 1
                if not args.quiet:
                    print(f"  [FAIL] MISSING {rec['folder_path']}")
                continue
            res = verify(pdir, args.min_comment_ratio)
            if print_result(pdir, res, quiet=True):
                done += 1
        total = len(catalog)
        print(f"\n[verify --all] {done}/{total} DONE | "
              f"{total - done - missing} not-done | {missing} missing folders.")
        sys.exit(0 if done == total else 1)

    if not args.path:
        sys.exit("[verify] give a project path, or use --all.")
    pdir = Path(args.path).resolve()
    if not pdir.exists():
        sys.exit(f"[verify] FATAL: no such folder: {pdir}")
    res = verify(pdir, args.min_comment_ratio)
    ok = print_result(pdir, res, quiet=args.quiet)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
