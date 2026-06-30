# THEORY — 2.19 Membrane Protein Simulation

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use. This is a **reduced-scope teaching
> version** (CLAUDE.md §13): it captures the essential physics and the GPU
> pattern, and describes the full production approach in §7._

---

## 1. The science

A **biological membrane** is a two-layer sheet (a *bilayer*) of **lipid**
molecules. Each lipid is *amphipathic*: a **polar head group** that likes water
and two **hydrophobic tails** that flee it. Drop lipids in water and they
self-assemble — heads facing the water on both sides, tails buried in an oily
core between them — because hiding the tails from water lowers the free energy.
That spontaneous arrangement is the **hydrophobic effect**, and it is the single
most important idea in this project.

**Membrane proteins** — GPCRs, ion channels, transporters, integrins — are
embedded in this bilayer and make up **>50% of current drug targets**. To study
how such a protein moves (a channel opening, a receptor switching state) we run
**molecular dynamics (MD)**: place every atom, compute the forces between them,
and step Newton's equations forward in tiny time increments. Before a production
run, the membrane must be **equilibrated** — relaxed from an artificial starting
geometry into a stable, physically sensible bilayer. A common, fast way to do
that is a **coarse-grained (CG)** pre-equilibration: lump ~4 atoms into one
"bead", which smooths the energy landscape and lets you take bigger timesteps,
then later **backmap** to all-atom detail.

This project models exactly that **CG pre-equilibration** in miniature: a tiny
patch of 3-bead lipids with a short protein column, relaxed under a thermostat,
with the bilayer staying intact. It is the smallest thing that still teaches the
real lesson — *the hydrophobic effect builds the membrane, and MD is force
evaluation in a loop*.

## 2. The math

We track `N` beads with positions **rᵢ**, velocities **vᵢ**, mass `mᵢ`. The
total potential energy is a sum of **non-bonded** and **bonded** terms.

**Lennard-Jones (non-bonded), for each pair (i, j) within a cutoff `r_c`:**

```
U_LJ(r) = 4 ε_ij [ (σ/r)¹² − (σ/r)⁶ ]
```

- `r = |rᵢ − rⱼ|` is the pair distance (minimum-image in the periodic x,y plane).
- `σ` is the bead diameter (the distance scale); in reduced units σ = 1.
- `ε_ij` is the **well depth** for the *types* of beads i and j — the strength of
  their mutual attraction. The `r⁻¹²` term is steep **repulsion** (beads can't
  overlap); the `r⁻⁶` term is the attractive **van der Waals** well, of depth
  `ε_ij` at `r = 2^{1/6} σ`. Making **tail–tail** ε the largest is how we encode
  the hydrophobic effect.
- We **truncate** at `r_c` (default 2.5 σ) and **shift** so `U(r_c) = 0` (energy
  stays continuous). The force is `F = −dU/dr`:

```
F_LJ(r) = (24 ε_ij / r) [ 2 (σ/r)¹² − (σ/r)⁶ ] · r̂        (vector along r̂ = (rᵢ−rⱼ)/r)
```

**Harmonic bonds (bonded), for each spring (i, j) with rest length `r₀`:**

```
U_bond(r) = ½ k (r − r₀)²        F_bond = −k (r − r₀) · r̂
```

This wires a lipid into a 3-bead rod (head–tail–tail) and the protein into a
column. `k` is the bond stiffness.

**Equations of motion** (Newton, with a thermostat):

```
mᵢ d²rᵢ/dt² = Fᵢ_conservative + Fᵢ_Langevin
Fᵢ_Langevin = −γ mᵢ vᵢ + √(2 γ mᵢ kT / dt) · ξ,     ξ ~ N(0,1) per component
```

- `Fᵢ_conservative` = sum of LJ + bond forces on bead i.
- The **Langevin** term is a thermostat: a **friction** `−γ m v` that bleeds
  energy plus a **random kick** whose variance is fixed by the
  *fluctuation-dissipation theorem* so the system relaxes to temperature `kT`
  (the NVT ensemble). `γ` is the friction coefficient.

**Inputs:** the parameters in `data/sample/membrane_sample.txt` (§Data in
README) + the geometry built by `build_system()`. **Output:** the equilibrated
positions/velocities, summarized by **bilayer thickness** (head-to-head z
separation) and **total potential energy**.

## 3. The algorithm

**Velocity-Verlet** is the workhorse MD integrator: symplectic (conserves energy
over long runs), time-reversible, and it needs only **one force evaluation per
step**. In split ("kick-drift-kick") form, one step is:

```
(A)  v ← v + (F/m)·(dt/2)        # half-kick with the current force
     r ← r + v·dt                # drift
(B)  F ← compute_forces(r)       # recompute at the new positions
(C)  v ← v + (F/m)·(dt/2)        # half-kick with the new force
```

We apply the Langevin force alongside the conservative force in **both**
half-kicks (a simple, stable Langevin-Verlet splitting).

**Per-step work / complexity:**

- **Force evaluation** is the cost center. Naively every bead checks every other:
  **O(N²)** distance computations per step. (We skip pairs past `r_c`, so the
  *useful* work is O(N · ⟨neighbours⟩), but the loop still visits all pairs — see
  Exercise 4 for the cell-list fix that makes it truly O(N).)
- **Bonds** are O(B) for B bonds (B ≈ 2·n_lipids).
- **Integration** (the two kicks + drift) is **O(N)** per step.
- Over `S` steps: **O(S · N²)** total. For our tiny sample (N = 59, S = 200)
  that is trivial; the point is the *shape*, not the size.

**Data-access pattern:** the force loop reads all positions for every bead — high
*reuse* of the position array, which is exactly what makes it a good parallel
(and, with neighbour lists, cache-friendly) kernel. Arithmetic intensity is
moderate: a handful of flops per pair against one position read.

## 4. The GPU mapping

**Thread-to-data mapping:** **one thread per bead**. Thread
`i = blockIdx.x·blockDim.x + threadIdx.x` owns bead `i`. It:

1. loops over all `j` and accumulates the LJ force on bead i (independent reads),
2. scans the bond list and adds any spring where i is an endpoint,
3. writes `f[i]` — **only its own slot**, so there are **no races and no atomics**.

The integration kernels are likewise one-thread-per-bead and update `pos[i]`,
`vel[i]` independently. The host drives the per-step sequence
`[kick_drift → compute_forces → kick]`, mirroring the CPU exactly.

**Launch configuration:** block = **256 threads** (a multiple of the 32-lane warp;
gives the scheduler 8 warps to hide global-memory latency; fits the register
budget on sm_75–sm_89). Grid = `ceil(N / 256)` blocks; the last block is ragged,
so every kernel begins with `if (i >= N) return;`.

**Memory hierarchy:**

- **Global memory** holds positions, velocities, forces, masses, types, bonds. We
  store them as **Structure-of-Arrays** (separate arrays) so neighbouring threads
  touch neighbouring addresses → **coalesced** loads (the access pattern global
  memory likes).
- **Registers** hold the per-thread accumulator `fi` and the bead's own state —
  the force sum never touches global memory until the final write.
- `SimParams` is passed **by value** into each kernel; being small and read-only,
  it lives in each thread's registers/constant bank rather than being chased
  through a pointer.
- A production code would additionally use **shared memory** to *tile* the
  position array (load a block of neighbour positions once, reuse across the
  block) — the classic N-body optimization. We keep the simple global-memory
  loop for readability and note the tiling win here.

**CUDA libraries:** none needed in this reduced model — the forces and integrator
are hand-rolled (no black boxes). The deterministic random kick uses a small
counter-based hash (SplitMix64) instead of **cuRAND**; §5 explains why, and what
cuRAND's Philox would give you. The *production* pattern (PME) would use
**cuFFT**; see §7.

```
            grid of blocks (each 256 threads)
   ┌──────────────┬──────────────┬─── ... ───┐
   │  block 0     │  block 1     │           │
   │ t0 t1 ... t255│ t0 ...      │           │
   └──────┬───────┴──────────────┴───────────┘
          │  thread i owns bead i
          ▼
   compute_forces_kernel(i):
      fi = 0
      for j in 0..N-1, j!=i:  fi += LJ(min_image(r_i - r_j))   # all-pairs
      for each bond touching i: fi ± bond_force(...)
      f[i] = fi                                                # only my slot
```

## 5. Numerical considerations

**Precision — double (FP64).** MD force sums are sensitive: many small pair
forces add up, and a stiff `r⁻¹²` repulsion amplifies tiny position errors. We
use `double` throughout so the CPU and GPU stay in lock-step. (Production GPU MD
often runs mixed FP32/FP64 for speed and accepts the resulting non-determinism;
we trade that speed for an exact, teachable comparison.)

**Determinism — the heart of the verification (PATTERNS.md §3).** Two design
choices make the GPU's stdout reproducible *and* matchable to the CPU:

1. **No floating-point reductions across threads.** Each thread sums *its own*
   force in a fixed index order and writes one slot — there is no `atomicAdd`
   into a shared accumulator, so there is no order-dependent float sum. The CPU
   walks beads/bonds in the *same* order, so the partial sums round identically.
2. **A stateless, counter-based PRNG for the Langevin kick.** A normal
   pseudo-random sequence has hidden state and would diverge between CPU and GPU.
   Instead we **hash** the tuple `(seed, step, bead, axis)` into a number
   (SplitMix64 → uniform → Box-Muller normal). Same inputs → same draw, on CPU
   and GPU, in any order, with no shared state. This is precisely the idea behind
   cuRAND's **Philox** counter-based generator; we inline a tiny hash so the
   shared header stays dependency-free and host-includable.

**Race conditions:** none. Every kernel reads shared state and writes only its
own bead's element. The force kernel reads positions that no kernel in the same
launch is writing.

**Stability:** velocity-Verlet is stable for `dt` below the fastest vibrational
period. The stiff bonds (`k = 30`) set that limit; `dt = 0.005 τ` is comfortably
inside it. A too-large `dt`, or beads built overlapping (tiny `r`, huge `r⁻¹²`),
makes the integrator explode — which is exactly what happened in development
before the bilayer geometry was fixed so no two beads coincide.

## 6. How we verify correctness

`src/reference_cpu.cpp` is an independent, obviously-correct **serial** MD: one
readable loop per phase, no parallelism. `main.cu` builds the *same* initial
system twice, runs the CPU reference and the GPU kernels, and compares the
**final position and velocity of every bead**.

- **Tolerance: `1e-4`** on `|Δpos|` and `|Δvel|`, chosen per PATTERNS.md §4 for a
  *long iterative* integrator. The two paths run identical FP64 math in identical
  order, so in practice they agree to **~`1e-14`** (round-off) on this sample —
  but a GPU's fused-multiply-add (FMA) can contract `a·b+c` differently from the
  host compiler, and that ~`1e-15`/op difference can compound over many steps and
  on other hardware. `1e-4` is the honest ceiling that stays *physically
  negligible* while not pretending the result is bit-identical everywhere.
- **A second, physics-level check:** the **bilayer thickness** and **total
  potential energy** are computed for both final states and reported; they agree
  to round-off too. This validates the *science observable*, not just that
  CPU == GPU array-for-array.
- **Edge cases:** the ragged last thread block (guarded), the `n_bonds == 0`
  case (guarded mallocs), and degenerate distances (`r < 1e-12` skipped in both
  LJ and bond force) are all handled identically on both sides.

Why is "an independent serial implementation agrees with the GPU" convincing? A
bug in the parallel decomposition (a missing neighbour, a wrong index, a race)
would change the trajectory and break the agreement *immediately*; matching to
round-off across 59 beads over 200 chaotic, thermostatted steps is strong
evidence the kernels compute the intended physics.

## 7. Where this sits in the real world

A production membrane simulation (GROMACS, NAMD, OpenMM via HTMD) differs from
this teaching model in every direction that matters:

- **All-atom force field:** **CHARMM36** lipids (POPE/POPC/cholesterol, with
  asymmetric leaflets) instead of generic 3-bead lipids — thousands of atoms per
  lipid-patch, parameterized against experiment.
- **System building:** **CHARMM-GUI Membrane Builder** or **packmol-memgen** place
  the protein, pack lipids around it, add water and ions, and assign the force
  field — the non-trivial setup this project fakes with a deterministic raster.
- **Long-range electrostatics — PME:** charged head groups need
  **Particle-Mesh Ewald**, which splits Coulomb into a real-space part + a
  reciprocal-space part solved with **FFTs** (this is where **cuFFT** enters, with
  custom corrections for the 2-D-periodic *slab* geometry of a membrane). We omit
  electrostatics entirely.
- **Pressure coupling:** a **semi-isotropic barostat** (NPT-xy) lets the membrane
  area relax to the correct *area-per-lipid* while the normal pressure is held —
  our box is rigid.
- **Scale:** 10⁵–10⁶ atoms for **microseconds**, with **neighbour/cell lists**
  (O(N) not O(N²)) and **multi-GPU domain decomposition** along the bilayer
  normal. The CG-MARTINI pre-equilibration (1–10 μs) is then **backmapped** to
  all-atom.
- **Analysis:** e.g. **k-means clustering of ion-channel gate states** to find
  conformational basins — the kind of downstream step this project's `11.09`
  sibling demonstrates on the GPU.

Everything above is the "next step up". This project deliberately stops at the
*coarse-grained equilibration with a hand-rolled force loop* because that is the
piece that teaches the GPU MD pattern with nothing hidden.

---

## References

- **MARTINI force field** — Marrink et al., *J. Phys. Chem. B* (2007). The CG
  bead model this project imitates; read it for bead types and the 4-to-1 mapping.
  <http://cgmartini.nl>
- **Velocity-Verlet & Langevin dynamics** — Frenkel & Smit, *Understanding
  Molecular Simulation*. The canonical text for the integrator and thermostat math.
- **Particle-Mesh Ewald** — Darden, York, Pedersen, *J. Chem. Phys.* (1993). What
  the real bilayer electrostatics use (and where cuFFT plugs in).
- **GROMACS** — <https://github.com/gromacs/gromacs>. Study its neighbour-list and
  PME kernels to see the O(N²)→O(N) and the FFT-based electrostatics in practice.
- **CHARMM-GUI Membrane Builder** — <https://charmm-gui.org>. The standard system
  setup tool; shows what "building a membrane system" really involves.
- **HTMD** — <https://github.com/Acellera/htmd>. A GPU membrane-protein pipeline
  (setup → simulate → analyze) worth reading end to end.
- **Counter-based RNGs (Philox)** — Salmon et al., *SC'11*. The basis for the
  deterministic, stateless random kicks (cuRAND's Philox generator).
