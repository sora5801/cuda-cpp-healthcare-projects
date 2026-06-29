# Push 2026-06-28 #02 -- flagship 3.01 smith-waterman

> Push-note (CLAUDE.md §7.1). Second Phase 1 flagship — the genomics domain.

## 1. Summary

The genomics flagship is done: **3.01 Smith-Waterman / Needleman-Wunsch Alignment**, a complete CUDA local
sequence aligner. It is the deliberate *contrast* to flagship 1.12: where Tanimoto search was
"embarrassingly parallel" (independent jobs), this dynamic-programming problem has hard data dependencies,
and the project teaches how to extract parallelism anyway via the **anti-diagonal wavefront**. It also models
honest GPU engineering: for a single small alignment the GPU is *slower* than the CPU, and the project says
so and explains why.

## 2. What changed

- [`projects/03-genomics/3.01-smith-waterman-needleman-wunsch-alignment/`](../projects/03-genomics/3.01-smith-waterman-needleman-wunsch-alignment) — fully implemented:
  - `src/kernels.cu` — `sw_diagonal_kernel` (one cell per thread on an anti-diagonal) + host wavefront sweep.
  - `src/reference_cpu.cpp` / `.h` — scoring constants, `SeqPair`, serial DP fill, and host traceback.
  - `src/main.cu` — load → CPU + GPU fill → compare full matrices → traceback → print score + alignment.
  - `THEORY.md`, `README.md`, `data/` (synthetic motif sequences), `scripts/`, `demo/`.
- `docs/STATUS.md` — `3.01` → **done** (2/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb)

**3.01 Smith-Waterman** teaches the **anti-diagonal wavefront**: the DP recurrence `H[i][j]` depends on its
top/left/diagonal neighbours and looks fatally serial, but all cells on one anti-diagonal `i+j = d` are
independent, so each diagonal is one parallel kernel launch reading only the two prior diagonals — no
atomics, no `__syncthreads`. The most interesting thing to study is `src/kernels.cu`: why launching per
diagonal is correct by construction, and why (honestly) it is launch-overhead-bound for a single small pair.

## 4. How to build & run

```powershell
cd projects/03-genomics/3.01-smith-waterman-needleman-wunsch-alignment
msbuild build/smith-waterman-needleman-wunsch-alignment.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> score + alignment + RESULT: PASS (GPU matrix == CPU)
```

## 5. What to study here

Reading path: `THEORY.md` (§2 the recurrence, §4 the wavefront diagram + the occupancy/launch honesty) →
`src/kernels.cu` → `src/reference_cpu.cpp` (serial DP + traceback). Then try README **Exercises**: convert
to Needleman-Wunsch (global), add affine gaps (Gotoh), or batch many pairs (the genuinely GPU-favorable
workload).

## 6. Verification

- ✅ `Release|x64` **and** `Debug|x64` build with **zero errors / zero warnings**.
- ✅ Demo **PASS**: deterministic score + alignment match `expected_output.txt`.
- ✅ **GPU matrix == CPU matrix exactly** (`matrix mismatches = 0`, integer DP — no float).
- ✅ `verify_project.py` → **DONE** (comment ratio **0.65**, no TODOs).
- **Sample result:** local score 143 over 115 columns (76.5% identity) on M=120/N=150 synthetic DNA.
- **Environment:** RTX 2080 (`sm_75`), CUDA 13.3, VS 2026 (`v145`). CPU fill ~0.21 ms vs GPU wavefront
  ~3.06 ms — the GPU is **slower** here (per-diagonal launch overhead); this is an intended honest lesson.

## 7. Known limitations / TODOs

- **DNA + linear gaps only** (no substitution matrix, no affine gaps).
- **Per-diagonal launches** ⇒ launch-overhead-bound for a single small pair; the wavefront wins on large
  matrices and the production path batches many pairs / uses a single persistent kernel with grid sync.
- Traceback is serial on the host; the full matrix is materialized.

## 8. Next push preview

Next flagship: **4.01 CT filtered backprojection (FDK)** (medical imaging) — a third distinct pattern:
per-output-pixel gather with the ramp-filtered projection data, a staple of CT reconstruction.
