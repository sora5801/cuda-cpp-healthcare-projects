# Push 2026-06-29 #08 -- phase1 standards review patterns-cookbook

> Push-note (CLAUDE.md §7.1). The post-Phase-1 standards/template review (CLAUDE.md §11): fold the lessons
> learned building the 14 flagships back into the docs, so the Phase-2 build-out is consistent.

## 1. Summary

Phase 1 surfaced a set of reusable idioms (a shared `__host__ __device__` core for exact CPU/GPU parity,
integer atomics for determinism, honest floating-point tolerances, "no black box" library docs). This push
captures them in a new **GPU patterns cookbook** so every Phase-2 project can find the closest flagship and
follow it — the key to keeping 287 more projects consistent.

## 2. What changed

- [`docs/PATTERNS.md`](../docs/PATTERNS.md) — **new**: a pattern→flagship map, the shared-header idiom,
  determinism rules (stdout/stderr split, integer/fixed-point atomics), verification-tolerance guidance, the
  "use a library but document it" rule, synthetic-data tips, and the honest-timing rule.
- [`docs/BUILD_GUIDE.md`](../docs/BUILD_GUIDE.md) — new §7b: how to link a CUDA library (cuFFT/cuSOLVER/…)
  in the `.vcxproj` + `CMakeLists.txt`.
- [`README.md`](../README.md) — points contributors to `PATTERNS.md` and the flagships as worked examples.

No code changed; the SAXPY `PROJECT_TEMPLATE` held up across all 14 flagships and needs no edits — the
improvements are documentation that captures *how* to specialize it.

## 3. New projects (didactic blurb)

None — this is a standards push. `docs/PATTERNS.md` is the artifact to read: it turns the 14 flagships into a
lookup table ("my project is a stencil → study 6.04") plus the cross-cutting techniques that made them all
verify cleanly.

## 4. How to build & run

No build. Read `docs/PATTERNS.md`, then for any project: copy `docs/PROJECT_TEMPLATE/`, find the pattern,
open that flagship, and build to the Definition of Done (`tools/verify_project.py`).

## 5. What to study here

`docs/PATTERNS.md` end to end (it is short), especially §2 (shared `__host__ __device__` core) and §4
(tolerance guidance) — the two ideas that most affect whether a new project verifies on the first try.

## 6. Verification

- ✅ Docs only; the repo still builds (no source changed). `verify --all` unchanged: 14/301 DONE.
- ✅ Links checked; PATTERNS referenced from README + BUILD_GUIDE.

## 7. Known limitations / TODOs

- The cookbook covers the 13 patterns seen so far; Phase 2 may surface new ones (e.g. cuBLAS GEMM, Thrust
  scan/sort, warp-shuffle reductions) — extend `PATTERNS.md` as they appear.

## 8. Next push preview

**Phase 2 begins.** Domain-by-domain, easiest-first, in parallel batches (CLAUDE.md §10): one worker agent
per project implements it to the Definition of Done in its scaffolded folder; the lead reviews, sets status,
regenerates `STATUS.md`, and pushes one push-note per batch. Starting with a pilot batch in domain 1
(drug discovery) to validate the workflow before scaling.
