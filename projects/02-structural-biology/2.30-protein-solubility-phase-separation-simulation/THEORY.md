# THEORY — 2.30 Protein Solubility & Phase Separation Simulation

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use. The data here is synthetic._

---

## 1. The science

Most textbook proteins fold into one rigid shape. A large fraction of the human
proteome does **not**: **intrinsically disordered proteins (IDPs)** and the
disordered regions of RNA-binding proteins stay as floppy, ever-wiggling chains.
Some of these — the low-complexity domains of **FUS, TDP-43, hnRNPA1** and many
others — can do something striking: above a threshold concentration they
spontaneously **demix from the surrounding solution into dense liquid droplets**,
much as oil separates from water. This is **liquid-liquid phase separation
(LLPS)**, and the droplets are **biomolecular condensates** — stress granules,
P-bodies, the nucleolus, and more.

Condensates organize the cell without membranes. Their misregulation is linked to
ALS, frontotemporal dementia, and cancer (e.g. mutations in FUS/TDP-43 that make
condensates harden into pathological aggregates). So two practical questions
drive the field:

1. **Will a given sequence (or mutant) phase-separate, and under what conditions?**
2. **Can we design a molecule that disrupts or stabilizes a specific condensate?**

Answering these needs simulations large enough to contain *both* a dense and a
dilute phase (hundreds of chains, millions of beads) and long enough for them to
separate (micro- to milliseconds). All-atom MD cannot reach that scale, so the
field uses **residue-level coarse-grained (CG) models**: one bead per amino acid,
with a "stickiness" that encodes how hydrophobic/aromatic that residue is. The
**HPS** (hydrophobicity scale, Dignon–Mittal 2018) and **CALVADOS** models are
the workhorses. This project implements the essential physics of that family — at
a tiny, didactic scale — and runs it on the GPU.

## 2. The math

We model the protein as **beads on a string** in a periodic cubic box of side
`L`. The total potential energy is a sum of three contributions.

**(a) Bonded (backbone) term — consecutive residues of a chain:**

$$ U_\text{bond}(r) = \tfrac{1}{2}\,k_\text{bond}\,(r - r_0)^2 $$

a simple harmonic spring of stiffness `k_bond` and rest length `r0` between
neighbours `i, i+1`. (Real models add angle/dihedral terms; we keep only the bond
for clarity.)

**(b) Non-bonded HPS term — every other pair, the heart of the model.** Start
from the 12-6 **Lennard-Jones** potential

$$ U_\text{LJ}(r) = 4\varepsilon\left[\left(\tfrac{\sigma}{r}\right)^{12} - \left(\tfrac{\sigma}{r}\right)^{6}\right], $$

then apply the **Ashbaugh-Hatch** modulation by a pair stickiness
`λ_ij ∈ [0,1]`:

$$
U_\text{AH}(r)=
\begin{cases}
U_\text{LJ}(r) + (1-\lambda_{ij})\,\varepsilon, & r \le r_\text{min}=2^{1/6}\sigma \quad(\text{repulsive core}) \\
\lambda_{ij}\,U_\text{LJ}(r), & r > r_\text{min} \quad(\text{attractive tail})
\end{cases}
$$

The trick: the **repulsive core is always full strength** (excluded volume — two
residues cannot overlap), but the **attractive well is scaled by λ**. `λ = 1`
is fully sticky (a hydrophobic/aromatic residue), `λ = 0` leaves pure repulsion
(a charged/polar residue). The pair value is the mean
`λ_ij = (λ_i + λ_j)/2`. Beyond a cutoff `r_cut` the interaction is truncated to
zero. *This one knob is how a sequence's hydrophobicity pattern decides whether it
condenses.*

**(c) Dynamics.** Each bead obeys Newton's second law `m a_i = F_i = -∇_i U`. We
integrate with **velocity-Verlet** (see §3). The directly observable signal of
phase separation is a structural **order parameter**; we use the simplest
interpretable one: a bead's **local density** = the number of other beads within
`r_cut`. A condensate is a cluster of high-local-density beads against a low-
density background.

Symbols: `r` = pair distance (length); `σ` = bead diameter (length); `ε` = well
depth (energy); `m` = bead mass; `dt` = step (time); `box = L` (length). We work
in **reduced LJ units** (`σ = ε = m = 1`) so every number is O(1).

## 3. The algorithm

Per velocity-Verlet step (the standard, time-reversible, energy-conserving MD
integrator):

```
1.  v ← v + (dt/2)(F/m)          half "kick"   (uses current forces)
2.  r ← r + dt·v                 "drift"        (move with half-kicked velocity)
        wrap r into [0, L)        periodic bookkeeping
3.  F ← forces(r)                recompute at the NEW positions
4.  v ← v + (dt/2)(F/m)          half "kick"   (uses the new forces)
```

The cost is dominated by step 3, `forces(r)`. The honest, simplest correct
version is **all-pairs**: the force on bead `i` sums contributions from every
other bead `j`.

- **Serial complexity:** `O(N²)` per step, `O(N² · n_steps)` total. For `N`
  beads, `N(N-1)/2` pairs.
- **Parallel structure:** with one thread per bead, step 3 has **work** `O(N²)`
  and **depth** `O(N)` (the inner sum), so `N` threads cut wall-time by ~`N`.
- **Arithmetic intensity:** each pair is a handful of flops over a few loads; the
  kernel is compute-bound only for large `N`, otherwise memory/launch-bound.

The integration steps (1,2,4) are `O(N)` element-wise updates — trivially
parallel. Production codes replace the `O(N²)` force with a **cutoff cell /
neighbour list** to reach `O(N)`; we keep all-pairs because it is the clearest
thing to *learn* from, and the cutoff inside the pair function still zeroes far
interactions (so the physics is identical, only the bookkeeping differs).

## 4. The GPU mapping

**One thread per bead** is the whole idea. Two kernels per step:

- **`force_kernel`** — thread `i = blockIdx.x·blockDim.x + threadIdx.x` owns bead
  `i`. It **gathers**: loops `j = 0..N-1`, reads `x/y/z/λ/chain[j]` from global
  memory, and accumulates `F_i` and the half-pair energy into **registers**. It
  writes only its own `f[i]`, `u_half[i]`. Because every thread writes a *distinct*
  slot, there are **no atomics and no races** — the cleanest possible parallel
  pattern. (Contrast project 5.01, where many threads add into shared dose bins
  and *must* use atomics.)
- **`integrate_kernel`** — thread `i` half-kicks + drifts (+ wraps) bead `i`. We
  call it twice per step (phase 0 before the force recompute, phase 1 after), so
  the GPU's update order matches the CPU's exactly.

**Launch config:** `block = 256` threads (a multiple of the 32-lane warp; enough
warps to hide global-memory latency on sm_75..sm_89), `grid = ceil(N/256)` blocks.
A `if (i >= N) return;` guards the ragged last block.

**Memory hierarchy:** positions live in **global** memory and are streamed (each
thread's read of `x[i]` is coalesced across the warp). Force accumulators stay in
**registers**. We deliberately do *not* tile here.

```
grid:   [ block 0 ][ block 1 ] ... [ block ceil(N/256)-1 ]
block:  256 threads;  thread t -> bead i = block*256 + t
force:  bead i  reads  ALL beads j=0..N-1  (the O(N) inner sum)  -> writes f[i]
        (no two threads write the same slot => no atomics)
```

**The optimization we left as an exercise:** every block re-reads all `N`
positions from global memory. The standard fix is **shared-memory tiling**: each
block cooperatively loads a tile of `j`-positions into `__shared__` memory, all
256 threads reuse that tile, then advance to the next tile. That turns `O(N²)`
global loads into `O(N²/256)` and is how a real all-pairs N-body kernel is
written. We keep the plain global-memory version because the goal here is to see
the gather pattern clearly. No CUDA library is used: the HPS force is a few lines
of arithmetic, so hand-writing it *is* the lesson (cuRAND would only enter if we
added a Langevin thermostat — see §5).

## 5. Numerical considerations

- **Precision: FP64 throughout.** MD accumulates millions of additions; an `O(N²)`
  force sum per step over many steps drifts much less in double precision, and —
  crucially — it keeps the CPU and GPU paths agreeing far longer (see §6).
- **No atomics, fully deterministic.** The gather pattern means each output is
  written by exactly one thread, so there is no order-dependent floating-point
  reduction. Given the *same fixed pair order*, the GPU computes the same sum the
  CPU does. We enforce that by summing `j = 0..N-1` in the **shared**
  `bead_force()` (in `hps_model.h`), used verbatim by both paths.
- **Stability is real and worth feeling.** Velocity-Verlet is only stable if `dt`
  is small relative to the fastest motion (here the stiff bond, `k_bond = 50`). If
  two non-bonded beads start inside the LJ hard core, the `~r^-13` repulsion
  produces an enormous force and the integrator **explodes** (energy → ∞). The
  synthetic generator therefore lays chains out as non-overlapping rods at the
  bond rest length; an earlier random-walk start blew up — a genuine MD failure
  mode, not a bug. Lower `dt` or soft-core potentials are the production fixes.
- **Chaos vs. determinism.** MD is chaotic: tiny perturbations grow exponentially
  (Lyapunov). The GPU's fused multiply-add (FMA) contracts `a*b+c` differently
  from the host compiler, seeding a ~`1e-15` difference that *would* eventually
  diverge on a long enough run. On this short, stable sample it stays at machine
  precision (see §6). Production thermostatted runs are validated **statistically**
  (matching distributions), not trajectory-by-trajectory — which is why we keep
  the demo short and deterministic.

## 6. How we verify correctness

`src/reference_cpu.cpp` is an independent, obviously-correct **serial**
velocity-Verlet loop. `main.cu` runs both it and the GPU on the same initial
system and compares **final-state summaries** — total potential energy, kinetic
energy, a position checksum, and the integer phase order parameters — rather than
diffing thousands of coordinates.

Why this is convincing: the two implementations share *only* the per-pair physics
header; the loop structure, memory management, and parallelism are written
separately. If an independent serial code and a parallel GPU code agree to
**~1e-15** (we observe `|ΔPE| ≈ 1.8e-15`, `|Δchecksum| = 0`), the most likely
explanation is that both are correct.

**Tolerance:** `1e-6` (absolute) on energies/checksum, **exact** on the integer
order parameters. We use a small non-zero tolerance rather than demand bit-
identity because the FMA difference in §5 is physically meaningless yet real; on a
longer/chaotic run it would grow, and pretending otherwise would teach the wrong
lesson (docs/PATTERNS.md §4). A second, *physical* sanity check beyond CPU==GPU:
the potential energy goes **negative** and almost all beads become high-density —
i.e. the chains actually condensed, which is the science we set out to see.

## 7. Where this sits in the real world

Production LLPS simulation (LAMMPS+HPS, **OpenMM**, **CALVADOS 2**, GROMACS+MARTINI)
differs from this teaching version in several big ways:

- **Force field:** real per-residue λ tables (Kapcha-Rossky / HPS / CALVADOS),
  **Debye-Hückel electrostatics** for charged residues (salt-dependent!), and
  refined size/energy parameters. We use a single synthetic λ and no charges.
- **Thermostat:** a **Langevin** integrator couples each bead to a heat bath
  (`F += -γ m v + √(2γ m k_BT/dt)·ξ`, with `ξ` from cuRAND) to sample the NVT
  ensemble at body temperature. We run plain **NVE** (no noise) precisely so the
  demo is deterministic.
- **Scale:** hundreds of chains / millions of beads in **slab geometry**, with
  **neighbour lists** (`O(N)` forces) and domain decomposition across multiple
  GPUs — reaching the system sizes and micro/millisecond times LLPS needs.
- **Observables:** a real study computes a **coexistence (phase) diagram** by
  running many concentrations/temperatures (an *ensemble of boxes* — the catalog's
  "GPU-parallel concentration ensemble"), measures the dense/dilute densities via
  density profiles, and locates the critical point with **finite-size scaling**.
  Our single-box "local density" order parameter is the first rung of that ladder.

This is a deliberately **reduced-scope** version of a 🔴 frontier topic: the GPU
pattern (one-thread-per-bead gather, shared host/device physics, periodic
minimum-image, velocity-Verlet) is exactly the production pattern, just without
the neighbour list, electrostatics, thermostat, and multi-box ensemble.

---

## References

- **Dignon, Zheng, Kim, Best, Mittal (2018)**, *Sequence determinants of protein
  phase behavior from a coarse-grained model*, PLoS Comput Biol — the HPS model
  and the Ashbaugh-Hatch λ modulation used here.
- **Tesei et al. (2021), CALVADOS** — an improved residue-level IDP model;
  https://github.com/KULL-Centre/CALVADOS — study the λ table and DH electrostatics.
- **OpenMM** (https://github.com/openmm/openmm) — a GPU MD engine with a Python
  API; read its custom-force classes to see how HPS is expressed in practice.
- **LAMMPS** (https://github.com/lammps/lammps) — large-scale MD with GPU package
  and neighbour lists; the reference for slab-geometry LLPS at scale.
- **Ashbaugh & Hatch (2008)**, J. Am. Chem. Soc. — the original modified-LJ
  hydration potential.
- **Allen & Tildesley**, *Computer Simulation of Liquids* — velocity-Verlet,
  minimum-image, neighbour lists, and reduced units, all in one place.
