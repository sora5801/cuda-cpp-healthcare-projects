# THEORY — 1.1 Molecular Dynamics Engine

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

This is a **reduced-scope teaching version** (CLAUDE.md §13): a complete, correct
molecular-dynamics engine for the simplest physically meaningful force field (a
single Lennard-Jones pair term) with the standard velocity-Verlet integrator and
periodic boundaries. §7 describes what a production engine adds on top.

---

## 1. The science

**Molecular dynamics (MD)** answers a deceptively simple question: *if I know
where every atom is and how it interacts with its neighbours, where will the atoms
be a moment later?* By repeatedly answering it — millions of tiny time-steps — MD
turns a static molecular structure into a movie of how the molecule *moves*. That
movie is how computational chemists watch a drug bind a protein pocket, a membrane
ripple, or a peptide fold. Production MD (GROMACS, OpenMM, NAMD, AMBER) is one of
the workhorses of modern drug discovery and structural biology.

The atoms obey **Newton's second law**, F = m·a. The only physics input is the
**force field**: a function that, given all atom positions, returns the force on
each atom. Real force fields sum many terms — bonds, angles, dihedrals, Coulomb
electrostatics, and van der Waals (dispersion/repulsion). This project models the
last one alone, the **Lennard-Jones (LJ) 12-6 potential**, which already captures
the essential physics of a simple fluid (e.g. liquid argon): atoms attract weakly
at long range and repel hard when they overlap. It is the canonical first MD model
and the right place to learn the *machinery* — the inner loop is identical to the
production case; only the number of force terms grows.

We work in **reduced (LJ) units**: mass m = 1, well depth ε = 1, diameter σ = 1.
Lengths are then in units of σ, energy in units of ε, time in units of σ·√(m/ε).
This strips away unit bookkeeping so the algorithm is what stands out.

## 2. The math

The Lennard-Jones potential between two atoms a distance `r` apart is

```
U(r) = 4ε [ (σ/r)¹² − (σ/r)⁶ ]
```

- the `+(σ/r)¹²` term is steep short-range **Pauli repulsion** (overlapping
  electron clouds),
- the `−(σ/r)⁶` term is the **London dispersion** attraction,
- U has a minimum of depth −ε at r = 2^(1/6)·σ and crosses zero at r = σ.

The **force** on atom *i* from atom *j* is the negative gradient, which (computed
from `r² = |rᵢ − rⱼ|²` to avoid a square root) is

```
F_ij = ( 24ε / r² ) [ 2(σ/r)¹² − (σ/r)⁶ ] · (rᵢ − rⱼ)
```

The total force on atom *i* is the sum over all other atoms,
`Fᵢ = Σ_{j≠i} F_ij`. Newton's law gives the acceleration `aᵢ = Fᵢ/m`, and we
integrate the coupled ODEs `dxᵢ/dt = vᵢ`, `dvᵢ/dt = aᵢ`.

**Periodic boundary conditions (PBC)** make a small box behave like bulk matter:
the cubic box of edge `L` is tiled infinitely, and each pair interacts through its
**nearest image**, so every coordinate difference `d` is wrapped into `(−L/2, L/2]`
via `d − L·round(d/L)` (the *minimum-image convention*). We also truncate the
interaction at a **cutoff** `rcut` (the LJ tail is negligible far out).

Inputs: `n` atoms with initial positions/velocities, box `L`, timestep `dt`, step
count, and `(ε, σ, rcut, m)`. Outputs (the verified observables): total energy at
start/end, the maximum energy drift, the final temperature, and a position
checksum. Energy is `E = Σ ½ m vᵢ² (kinetic) + Σ_{i<j} U(r_ij) (potential)`.

## 3. The algorithm

We integrate with **velocity-Verlet**, the standard MD integrator. Per step of
size `dt` (the "kick-drift-kick" form):

```
1. half-kick : v ← v + (dt/2)·F/m          using current forces F(t)
2. drift     : x ← x + dt·v                 (then wrap into the box)
3. force eval: recompute F(t+dt) from new x  ← the expensive O(N²) part
4. half-kick : v ← v + (dt/2)·F/m          using the new forces F(t+dt)
```

Velocity-Verlet is **time-reversible** and **symplectic**: it conserves a discrete
"shadow" energy, so the true total energy stays bounded (oscillates, never drifts
away) for very long runs. That conservation is *the* reason MD trajectories are
trustworthy, and it is the property we measure to prove the engine works.

**Complexity.** The force evaluation is the cost. Computed directly (all pairs) it
is **O(N²)** interactions per step, so **O(steps · N²)** overall. The kick and
drift are O(N). Serial depth per step is O(N²) for the direct sum; the parallel
*work* is the same O(N²) but the parallel *depth* collapses to O(N) (each of the N
threads does an O(N) loop, all concurrently). Production codes drop the per-step
cost to **O(N)** with a **neighbour list** (only nearby pairs interact within
`rcut`); we keep the direct O(N²) sum because it is the clearest thing to learn and
verify, and we note the optimization (§7).

## 4. The GPU mapping

The force sum is "embarrassingly parallel in *i*": the force on atom *i* does not
depend on the force on atom *k*. So we map **one GPU thread to one atom**:

- **Thread-to-data:** thread `i = blockIdx.x·blockDim.x + threadIdx.x` owns atom
  *i*. It loops over all *j*, accumulates `Fᵢ` and its share of the potential in
  **registers**, and writes one force vector + one energy value to global memory.
- **Launch config:** `block = 128` threads (a warp multiple, good occupancy on
  sm_75…sm_89); `grid = ceil(n/128)` blocks. The last block is *ragged* (n is
  rarely a multiple of 128), so out-of-range threads still help load shared memory
  but skip the writes.
- **Shared-memory tiling — the key optimization.** A naive kernel has every one of
  the N threads read all N positions from DRAM: N² global loads, hopelessly
  bandwidth-bound. Instead each block **cooperatively loads a tile** of 128
  positions into **shared memory** once, and all 128 threads in the block reuse
  that tile before moving to the next. Global traffic drops from ~N² to ~N²/128.
  This is exactly the classic CUDA *n-body* tiling pattern.

```
 grid of blocks (one block shown), block = 128 threads, atom i per thread:

   for base = 0, 128, 256, ... < N:        # sweep all atoms in tiles
       tile[threadIdx] = pos[base+threadIdx]   # 1 cooperative global load each
       __syncthreads()                         # tile fully built
       for t in 0..127:                        # every thread vs the whole tile
           Fi += LJ(ri, tile[t])               # reads from fast shared memory
       __syncthreads()                         # done before tile is overwritten

   pos[i] ─┐                         ┌─ registers: ri, Fi, Ui (per thread)
   pos[..] ┼─► [ shared tile (3 KiB) ]┘
   pos[N-1]┘     reused by 128 threads
```

- **Memory hierarchy used & why:** *registers* hold each thread's `ri`, running
  force and energy (fastest, private); *shared memory* stages the position tile
  (block-wide reuse — the whole point); *global memory* holds the N-sized state
  arrays. The integrator's kick/drift kernels are pure O(N) streaming over global
  memory (one thread per atom, no sharing needed).
- **No CUDA library is needed here**: the force kernel is a custom hand-written
  reduction in registers. The catalog notes that production MD uses **cuFFT** for
  the Particle-Mesh-Ewald reciprocal sum and **Thrust/CUB** to build sorted
  neighbour lists — both are deliberately out of scope for this LJ teaching engine
  (§7 explains what writing them entails, keeping the "no black boxes" rule).
- **Keeping data on the device:** positions/velocities/forces live on the GPU for
  the *entire* run; only the small per-step energy diagnostics come back. Shuttling
  the full state to the host each step would be the classic PCIe bottleneck.

## 5. Numerical considerations

- **Precision: FP64 (double) throughout.** MD energy conservation is sensitive;
  in single precision the all-pairs sum loses too many bits and the drift balloons.
  Double precision also lets the CPU and GPU agree to near machine epsilon, which
  makes verification meaningful. The shared `Vec3` and all physics in `md.h` are
  `double`.
- **Determinism & floating-point summation.** Floating-point addition is *not*
  associative, so a parallel reduction that sums in a hardware-dependent order can
  give run-to-run-varying totals (PATTERNS.md §3). We therefore **do not** reduce
  the energy with `atomicAdd`. The force kernel writes a *per-atom* energy array;
  we copy it back and sum it on the host in a **fixed index order**, so stdout is
  byte-identical every run (the demo diffs it).
- **CPU↔GPU are not bit-identical, and that is honest.** The GPU fuses
  multiply-adds (FMA: `a*b+c` in one rounding step) and sums the force over *j* in a
  different order than the serial CPU loop. Each operation differs by ~1e-16; over
  50 steps these accumulate to ~1e-13 in the energies here. This is the
  "long-iterative-solver" case of PATTERNS.md §4 — real and worth teaching, not a
  bug. We verify to a small **absolute tolerance** instead of pretending equality.
- **No race conditions:** each thread owns its atom's outputs; `__syncthreads()`
  guards the shared tile so no thread reads a half-written tile or overwrites one
  still in use.

## 6. How we verify correctness

`src/reference_cpu.cpp` is an independent, obviously-correct **serial** velocity-
Verlet driver. It and the GPU kernels call the **same** `__host__ __device__`
physics in `md.h` (the LJ force, minimum image, the kicks), so any disagreement is
pure parallel round-off, not a different algorithm (PATTERNS.md §2). `main.cu` runs
both and asserts the GPU observables match the CPU's:

- energies (`E0`, `E_final`, `max_drift`) to **1e-6 absolute**, and
- the final-position checksum to **1e-4 absolute** (the most chaos-sensitive
  observable — a tiny early divergence amplifies over the trajectory).

These tolerances comfortably hold (the observed CPU↔GPU gaps are ~1e-13; see the
demo's stderr) while honestly accounting for FMA/ordering differences.

A **second, physical check** beyond CPU==GPU: the integrator must **conserve
energy**. The demo reports a *relative* energy drift around `2.5e-8` over the run —
the hallmark of a correct symplectic integrator. A buggy force or a too-large `dt`
would show up immediately as energy blow-up, independent of whether CPU and GPU
agree. Edge cases handled: the ragged last thread block, self-interaction
(`r²=0`), pairs beyond the cutoff, and negative coordinates in the box-wrap.

## 7. Where this sits in the real world

Production engines (GROMACS, OpenMM, NAMD, AMBER `pmemd.cuda`) keep this exact
inner loop but add, roughly in order of impact:

- **Neighbour lists / cell lists** — turn the O(N²) force sum into **O(N)** by only
  considering pairs within `rcut` (+ a skin). Built on the GPU with a sort
  (Thrust/CUB) into spatial cells; this is the single biggest scaling change.
- **Full force field** — bonded terms (bonds, angles, dihedrals) and **Coulomb
  electrostatics**. Long-range electrostatics use **Particle-Mesh Ewald (PME)**,
  whose reciprocal-space sum is an FFT on a charge grid (**cuFFT**) — replacing our
  single LJ term. Writing PME by hand means spreading charges to a grid, a 3-D FFT,
  a reciprocal-space convolution, an inverse FFT, and gathering forces.
- **Constraints** (LINCS/SHAKE) to freeze fast bond vibrations so `dt` can grow
  from our ~0.004 to ~2 fs.
- **Thermostats / barostats** (velocity rescaling, Nosé-Hoover; Berendsen,
  Parrinello-Rahman) to sample constant-temperature/pressure ensembles instead of
  the constant-energy run here.
- **Multi-GPU domain decomposition** with halo exchange (**NCCL**) for 10–100M-atom
  systems.

Our engine is the honest *core* of all of these: same integrator, same pairwise
non-bonded math, same GPU thread-per-atom + shared-memory-tiling pattern.

---

## References

- M. P. Allen & D. J. Tildesley, *Computer Simulation of Liquids* — the canonical
  textbook; chapters on LJ fluids, Verlet integration, and PBC underpin this code.
- L. Verlet (1967) and W. C. Swope et al. (1982) — the (velocity-)Verlet
  integrator and why it conserves energy so well.
- **GROMACS** (https://github.com/gromacs/gromacs) — read how a production engine
  structures the non-bonded kernels and neighbour search on the GPU.
- **OpenMM** (https://github.com/openmm/openmm) — clean, well-documented CUDA
  platform; a great source for how PME and constraints are organized.
- **NAMD** (https://www.ks.uiuc.edu/Research/namd/) and **AMBER `pmemd.cuda`**
  (https://ambermd.org/GPUSupport.php) — large-scale, multi-GPU MD; study their
  domain-decomposition and PME strategies.
- The classic CUDA **n-body** sample — the shared-memory tiling pattern this force
  kernel mirrors.
