# 2.7 — Monte Carlo Protein Structure Sampling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.7`
>
> _Educational only — not for clinical use (see CLAUDE.md §8). This is a
> **reduced-scope teaching version**: the 2-D HP lattice-protein model._

## Summary

This project samples protein-like conformations with **Metropolis Monte Carlo**,
the workhorse of computational structural biology. To keep the focus on the GPU
pattern and the math, it uses the classic **HP lattice protein** (Lau & Dill,
1989): a chain of **H** (hydrophobic) and **P** (polar) residues on a 2-D grid,
whose energy rewards burying H residues next to each other — a toy model of the
*hydrophobic collapse* that folds real proteins. We launch a large array of
**independent Monte Carlo walkers** (replicas) across a temperature ladder, one
**GPU thread per replica**, and report the lowest-energy (best-folded) state the
ensemble discovers. The whole point: many independent random walks map perfectly
onto the GPU, and a carefully-shared RNG makes the GPU result **bit-identical** to
a plain CPU reference.

## What this computes & why the GPU helps

Monte Carlo methods sample protein conformational space by proposing random moves
and accepting/rejecting them via the Metropolis criterion. Two things are
GPU-accelerated in practice: (i) **batch-running many independent MC walkers** in
parallel, and (ii) the **energy evaluation** of each trial move. Parallel
tempering scales naturally to a GPU as an array of independent temperature
replicas. (Applications: loop modeling, side-chain packing, protein–ligand pose
sampling.)

**The parallel bottleneck:** a useful MC run needs *thousands* of long,
independent walks (replicas / random restarts) to explore a rugged energy
landscape. Each walk is sequential, but the walks **do not interact**, so the
dominant cost — running the whole ensemble — is embarrassingly parallel. We give
each replica its own GPU thread; with `R` replicas the GPU does `R` walks at once.
There is **no inter-thread communication and no atomics** (each thread owns a
private chain and a private output slot), which makes this one of the cleanest
parallelizations in the collection.

## The algorithm in brief

- **Metropolis–Hastings MC:** propose a random local move, accept with
  probability `min(1, exp(-ΔE/T))`, repeat.
- **HP energy function:** `E = -(number of non-bonded H–H lattice contacts)` —
  an integer, which is what makes exact verification possible.
- **Local move set:** end moves at the chain termini + interior corner/crankshaft
  moves, each checked for connectivity and self-avoidance.
- **Replica temperature ladder:** geometric spacing from `T_min` to `T_max`
  (hot replicas cross barriers, cold replicas refine) — the structure of parallel
  tempering, minus the swaps (described in THEORY).
- **Shared RNG + precomputed Boltzmann table:** a counter-based splitmix64 stream
  and a discrete acceptance table make every accept/reject identical on CPU & GPU.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/monte-carlo-protein-structure-sampling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/monte-carlo-protein-structure-sampling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\monte-carlo-protein-structure-sampling.sln /p:Configuration=Release /p:Platform=x64
```

This project links only `cudart` (the CUDA runtime) — no extra CUDA libraries —
because the RNG is hand-rolled on purpose (see THEORY §4 on cuRAND).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/hp_problem.txt`, prints the
deterministic result, shows the GPU-vs-CPU agreement check, and prints a timing
line (on stderr).

## Data

- **Sample (committed):** `data/sample/hp_problem.txt` — a tiny **synthetic** HP
  sequence + run parameters, so the demo runs offline with zero downloads.
- **Regenerate / resize:** `python scripts/make_synthetic.py [--replicas N ...]`.
- **Full datasets / further data:** `scripts/download_data.ps1` / `.sh` print
  links (this reduced model needs none).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: CASP benchmarks (<https://predictioncenter.org>); PDB
structures (<https://www.rcsb.org>); Dunbrack rotamer library
(<https://dunbrack.fccc.edu/bbdep2010/>); CAMEO (<https://www.cameo3d.org>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
2.7 -- Monte Carlo Protein Structure Sampling
[reduced-scope teaching model: 2-D HP lattice protein, Metropolis MC]
sequence (n=18, 10 H): HPHPPHHPHHPHHPPHPH
replicas = 256, sweeps = 600, T in [0.30, 3.00]
best energy found = -8 (8 H-H contacts) by replica 15
ensemble mean best energy = -1114/256
RESULT: PASS (GPU per-replica energies match CPU exactly)
```

The program runs the ensemble on the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts that **every replica's energies
match exactly** (tolerance = 0, because energies are integers computed by the same
shared code). That bit-for-bit agreement is the correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the problem, builds the Boltzmann tables,
   runs CPU + GPU, verifies (exact), reports.
2. [`src/mc_moves.h`](src/mc_moves.h) — **the heart**: the shared
   `__host__ __device__` RNG + Metropolis walk that both paths run, and the
   comment explaining why CPU and GPU are bit-identical.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-
   replica mapping.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel (calls the shared walk) + host
   wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader and the trusted
   serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, host I/O.

## Prior art & further reading

- **Rosetta** (<https://github.com/RosettaCommons/rosetta>) — the reference protein
  MC sampling/design suite; study its fragment moves and score function (GPU
  extensions are experimental).
- **FoldX** (<https://foldxsuite.crg.eu>) — fast empirical energy evaluation used
  in MC-based design; learn how a cheap, accurate energy term is built.
- **OpenMM** (<https://github.com/openmm/openmm>) — GPU MD/MC via custom
  integrators; a model for how production device kernels are structured.
- **ProteinMPNN** (<https://github.com/dauparas/ProteinMPNN>) — GPU sequence design
  that complements MC backbone sampling.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Ensemble of independent histories** (PATTERNS.md §1): one GPU thread per Monte
Carlo replica, each with its own RNG stream and a private conformation; read-only
shared Boltzmann tables; **no atomics** (independent outputs). Contrast project
5.01 (Monte-Carlo *dose*), where many threads tally into shared bins and *do* need
`atomicAdd` — here the outputs are disjoint, so the parallelization is contention-
free.

## Exercises

1. **Add replica-exchange swaps** to turn the temperature *ladder* into true
   parallel tempering: periodically attempt to swap configurations between
   adjacent temperatures, accepting with probability `min(1, exp((βᵢ−βⱼ)(Eᵢ−Eⱼ)))`.
   What is the minimal inter-thread synchronization this needs?
2. **Make energy incremental.** Replace the `O(n²)` `count_contacts` recompute with
   an `O(n)` (or `O(1)`) update that only re-scores the moved residue's
   neighbourhood. Measure the speed-up; keep the result bit-identical.
3. **Try harder sequences.** Use `make_synthetic.py --sequence ...` with longer
   chains; how does the best energy found scale with `sweeps` and `n_replicas`?
4. **Simulated annealing.** Instead of a fixed temperature per replica, cool each
   replica's `T` over its sweeps. Compare the best energy to the fixed-T ladder.
5. **Go 3-D.** Extend the lattice to `Z³` (6 neighbours). What changes in the move
   set, the energy, and the register footprint per thread?

## Limitations & honesty

- **Reduced-scope by design.** This is the 2-D **HP lattice** model, *not* a
  full-atom engine. Real MC (Rosetta/OpenMM) uses 3-D continuous backbone and
  side-chain angles and physics-based energies (Lennard-Jones, electrostatics,
  solvation). See THEORY §7.
- **The data is synthetic.** `data/sample/hp_problem.txt` is a generated HP
  sequence, labeled synthetic everywhere — it is not a real protein and the output
  has **no clinical or predictive validity**.
- **Simplifications:** naive `O(n²)` energy recompute (clarity over speed); a fixed
  temperature per replica with **no replica-exchange swaps**; a hand-rolled RNG
  chosen for exact CPU/GPU reproducibility rather than statistical pedigree (a real
  run would use cuRAND).
- **Timing is a teaching artifact**, never a benchmark claim (CLAUDE.md §12); it
  varies run to run and lives on stderr.
