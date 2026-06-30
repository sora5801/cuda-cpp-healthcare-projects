# 2.18 — NMR Structure Refinement

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.18`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Solution NMR does not give you a 3-D protein structure directly. It gives you a
list of **restraints** — pairs of atoms that NOE (Nuclear Overhauser Effect)
cross-peaks say are close (an *upper bound* on their distance, typically < 5–6 Å),
plus the covalent geometry every chain must obey. **Structure refinement** is the
search for coordinates that satisfy as many restraints as possible, and the
workhorse of that search is **restrained simulated annealing (SA)**: start hot and
random, make random moves, accept good ones always and bad ones with a
temperature-dependent probability, and cool slowly so the structure settles into a
low-violation conformation. Because the data is sparse and noisy, NMR pipelines run
**hundreds of independent SA trajectories** and keep the lowest-energy ones — that
*ensemble* is the published "NMR structure." This project builds a small,
heavily-commented version of that workflow and gives **each SA trajectory its own
GPU thread**.

## What this computes & why the GPU helps

NMR structure determination satisfies distance restraints (NOE), dihedral-angle
restraints (J-couplings), and RDC data via simulated-annealing MD. The defining
feature for the GPU is that an NMR structure is an **ensemble**: you run many
independent annealing trajectories from different random seeds and keep the best.
Production tools (XPLOR-NIH, CYANA, ARIA, AMBER `pmemd.cuda`) exploit exactly this
to "run hundreds of independent SA trajectories simultaneously."

**The parallel bottleneck:** the ensemble itself. Each trajectory is a long,
sequential Monte-Carlo loop, but the trajectories are **mutually independent** — no
trajectory reads another's data. That is embarrassingly parallel: we map **one
replica → one GPU thread**, so 512 annealing runs proceed at once instead of one
after another. (The same "ensemble of independent integrators" shape as the 9.02
SEIR and 13.02 PBPK flagships; here the per-thread loop is a Metropolis annealer
instead of an RK4 integrator.)

## The algorithm in brief

- **Energy (the surface SA minimises):** a flat-bottom **NOE penalty** per restraint
  (zero while the distance is within the upper bound, harmonic past it) plus a
  harmonic **bond restraint** keeping consecutive Cα beads near 3.8 Å.
- **Trial move:** perturb one randomly chosen bead by a Gaussian displacement.
- **Metropolis acceptance:** accept if the energy drops; otherwise accept with
  probability `exp(−ΔE / T)`.
- **Geometric cooling:** `T` descends from `T_hot` to `T_cold` over the run.
- **Ensemble:** repeat the whole annealer for many random seeds; report the
  lowest-energy structure and how many replicas satisfied every restraint.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/nmr-structure-refinement.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/nmr-structure-refinement.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\nmr-structure-refinement.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/restraints.txt`, prints the result,
shows the GPU-vs-CPU agreement check, and prints a timing line to stderr.

## Data

- **Sample (committed):** `data/sample/restraints.txt` — a tiny, **synthetic**
  restraint list derived from a known α-helix target, so the demo runs offline with
  zero downloads and a satisfying structure is known to exist.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent;
  prints source links only — never bypasses registration).
- **Regenerate the synthetic sample:** `python scripts/make_synthetic.py`.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: BMRB — Biological Magnetic Resonance Bank
(https://bmrb.io); PDB NMR-derived structures (https://www.rcsb.org); RECOORD —
recalculated NMR structures; CASD-NMR automated structure determination benchmarks.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): the best
of 512 replicas reaches a near-zero restraint energy and satisfies all 19 NOE
restraints, and a few hundred replicas satisfy every restraint. The program runs the
**GPU ensemble** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`)
and asserts they agree: the integer restraint-satisfaction counts match **exactly**,
and the continuous best-energy differs by less than `1e-4` (in practice ~`1e-14`).
That agreement is the correctness guarantee. Timings print to stderr and are *not*
diffed (they vary run to run).

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the job, runs CPU + GPU ensembles, verifies, reports.
2. [`src/nmr_refine.h`](src/nmr_refine.h) — the shared `__host__ __device__` core: RNG, restraint energy, and the simulated-annealing loop (`anneal_one`). **The heart of the project.**
3. [`src/kernels.cuh`](src/kernels.cuh) / [`src/kernels.cu`](src/kernels.cu) — the GPU interface and the one-thread-per-replica kernel.
4. [`src/reference_cpu.h`](src/reference_cpu.h) / [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader and the serial CPU twin.
5. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, host I/O helpers.

## Prior art & further reading

- **XPLOR-NIH** (https://nmr.cit.nih.gov/xplor-nih/) — the reference restrained-MD
  engine for NMR. Study its **NOE / dihedral / RDC energy terms**; our flat-bottom
  NOE penalty is a stripped-down version of its `NOE` term.
- **CYANA** (http://www.cyana.org) — torsion-angle dynamics for NMR. Study how it
  anneals in **dihedral space** (fewer degrees of freedom) instead of Cartesian.
- **AMBER NMR refinement** (https://ambermd.org) — `pmemd.cuda` runs full GPU MD
  with NMR restraints. Study how a real force field replaces our toy energy.
- **ARIA** (http://aria.pasteur.fr) — automated NOE assignment + iterative
  refinement. Study the assignment/refinement loop our single pass omits.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Ensemble of independent annealers — one thread per replica.** Each GPU thread
runs the entire Monte-Carlo SA loop for one trajectory in per-thread local memory;
there is no shared memory and no atomics because replicas never touch each other's
data. The per-replica physics (RNG, energy, accept/reject) lives in one
`__host__ __device__` header so the CPU reference and the GPU kernel execute
identical math and agree to round-off (PATTERNS.md §2). Production NMR codes go
further — full Cartesian or torsion-angle MD per replica, distributed via MPI+CUDA —
but the parallel decomposition is the same.

## Exercises

1. **More replicas, better best.** Re-run `make_synthetic.py --replicas 4096` and
   watch the fraction of all-satisfied replicas and the best energy. How does the
   GPU/CPU timing ratio change as the ensemble grows? (See the honest-timing note in
   THEORY §3.)
2. **Cooling schedule.** Make `T_cold` larger (e.g. 1.0) so the run never freezes.
   What happens to the satisfied-restraint counts? Now make `T_hot` tiny (e.g. 0.1)
   so it starts cold — does it get trapped? This is the classic SA trade-off.
3. **Local energy update.** `total_energy` is recomputed in full after each move
   (`O(restraints)` per step). Moving one bead only changes the terms that touch it.
   Implement an `O(degree)` local ΔE and confirm the trajectory is unchanged.
4. **Add lower bounds.** Real restraints have a lower bound too (van der Waals
   repulsion). Extend `noe_energy` to a double-flat-bottom well and regenerate data
   with both bounds.
5. **RMSD to target.** The generator knows the true α-helix coordinates. Have it
   also emit them, and add a (verification-only) RMSD of the best structure to the
   target after a rigid superposition — does low energy imply low RMSD?

## Limitations & honesty

- **Reduced-scope teaching version.** This is *not* a molecular-dynamics engine. It
  anneals **Cα beads** with a toy energy (flat-bottom NOE + harmonic bonds), not all
  atoms with a real force field, solvent, dihedral/RDC terms, or chemical-shift
  back-calculation. THEORY §7 maps each simplification to what production tools do.
- **Cartesian Metropolis MC, not torsion-angle MD.** Real refinement integrates
  Newton's equations (XPLOR/AMBER) or moves in dihedral space (CYANA). We use
  single-bead Metropolis moves because they are transparent and need no gradients.
- **Synthetic data.** The committed restraints are generated from an idealised
  α-helix, not measured from a spectrum. They are labelled synthetic everywhere and
  imply **no** clinical or biological validity.
- **Determinism caveat.** Reproducibility relies on a shared integer RNG
  (`splitmix64`), not cuRAND, specifically so CPU and GPU histories are bit-identical
  for verification (THEORY §5). A production code would use cuRAND and verify
  statistically instead.
