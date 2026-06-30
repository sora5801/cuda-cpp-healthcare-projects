# THEORY — 2.29 Ion Channel Gating & Permeation Simulation

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A cell holds an electrical potential across its membrane (~ −70 mV inside,
resting). **Ion channels** are membrane proteins with a water-filled pore that
lets specific ions (K⁺, Na⁺, Cl⁻, Ca²⁺) flow down their electrochemical gradient.
Open a Na⁺ channel and Na⁺ rushes in, depolarizing the cell — that is the rising
edge of a nerve action potential. Channels are therefore the hardware of every
thought and heartbeat, and a huge fraction of drugs (local anesthetics,
antiarrhythmics, many analgesics) work by blocking or modulating them. The
catalog names the big families: **Nav** (voltage-gated Na⁺), **Kv** (voltage-gated
K⁺), **CFTR** (a Cl⁻ channel; its mutation causes cystic fibrosis), **VGCC**
(voltage-gated Ca²⁺).

Two questions matter for any channel:

1. **Gating** — *when* is the pore open? (a conformational change triggered by
   voltage, ligand binding, or mechanical stress).
2. **Permeation** — *while open, how fast and how selectively do ions cross?*
   This is the **conductance** an electrophysiologist measures with a patch-clamp.

This project simulates **permeation through an open pore under an applied
voltage** — the catalog's "voltage-clamp conductance" with an "applied electric
field." We do *not* model the gating conformational change (the pore is always
open here); that is a much harder problem discussed in §7. The title says
"gating & permeation"; we ship the **permeation** half as a clean teaching model
and describe gating's full machinery in §7.

The key physical picture: an ion crossing the narrow **selectivity filter** must
shed part of its hydration shell and squeeze through, which costs free energy —
there is a **barrier** in the middle of the pore. The applied voltage tilts the
whole landscape, pushing ions one way. Permeation is thus **diffusion over a
barrier in a tilted potential** — exactly what Brownian dynamics models.

## 2. The math

We reduce the pore to a single coordinate **z ∈ [0, L]** (nm) along the
permeation axis; z = 0 is the intracellular mouth, z = L the extracellular mouth.
A single ion of charge **q** (units of e) feels a free-energy landscape (the
**potential of mean force**, PMF), in reduced units where thermal energy kT = 1:

```
U(z) = U_barrier · exp( −(z − L/2)² / (2 σ²) )   −   q · V · (z / L)
       └─────────── selectivity-filter barrier ──┘   └─ applied field ─┘
```

| Symbol | Meaning | Units |
|---|---|---|
| `U_barrier` | barrier height at the pore centre | kT |
| `σ` | barrier width | nm |
| `q` | ion charge | e |
| `V` | applied transmembrane voltage (reduced e·V) | kT/e |
| `L` | pore length | nm |

The **force** on the ion is `F(z) = −dU/dz`:

```
F(z) = U_barrier · exp(…) · (z − L/2)/σ²   +   q·V/L
```

(the field term `+qV/L` is constant and, with our sign convention, pushes a
positive ion forward when V > 0).

The ion's motion is **overdamped Langevin** (Brownian) dynamics — inertia is
negligible at this scale, so velocity is slaved to force plus thermal noise. The
**Ermak–McCammon** time-discretization (kT = 1, mobility = D) is:

```
z_{n+1} = z_n  +  D · F(z_n) · δt  +  √(2 D δt) · ξ_n ,     ξ_n ~ N(0,1)
                  └── deterministic drift ──┘    └── diffusion ──┘
```

| Symbol | Meaning | Units |
|---|---|---|
| `D` | diffusion coefficient | nm²/step-unit |
| `δt` | time step | step-unit |
| `ξ_n` | independent standard-normal random kick | — |

**Boundary condition / current:** when an ion reaches z ≥ L it has **permeated
forward** (a current-carrying event); we count it and re-inject it at z = 0 (one
ion leaves the cell, one enters from the bath — a steady-state current picture).
Falling back through z < 0 is a **reverse** crossing. Over `n_ions` ions × `n_steps`
steps, the **net flux**

```
J = (N_forward − N_reverse) / (n_ions · n_steps)
```

is proportional to the single-channel current (and, divided by V, the
conductance). We also tally the **occupancy histogram** `H[b]` = number of
ion-steps whose position falls in z-bin `b`; `H` is the steady-state probability
density along the pore, and by Boltzmann `H(z) ∝ exp(−U_eff(z))` it directly
reveals the barrier.

## 3. The algorithm

```
for each ion i in 0..n_ions-1:                 # INDEPENDENT  -> parallel
    seed a private RNG stream from (base_seed, i)
    z = 0
    for step in 0..n_steps-1:                  # sequential within an ion
        F   = pmf_force(z)                      # closed-form gradient
        z  += D*F*dt + sqrt(2*D*dt)*normal()    # Ermak-McCammon update
        if z >= L: forward++; z -= L            # permeation (re-inject)
        elif z < 0: reverse++; z += L
        H[bin_of(z)]++                          # occupancy tally
    add this ion's forward/reverse into global counters
```

**Complexity.** Total work is **O(n_ions · n_steps)** — one transcendental-heavy
update per inner iteration (an `exp`, a `log`, a `cos`, a `sqrt` for the Gaussian).
The inner loop is **sequential per ion** (step n+1 depends on step n), so the
*depth* is O(n_steps); the *parallel width* is n_ions. Arithmetic intensity is
high (each step is many flops on a handful of registers, almost no memory
traffic), so the kernel is **compute-bound**, not bandwidth-bound — ideal for a
GPU. The only memory contention is the atomic updates to the small occupancy
histogram and two counters.

**Data-access pattern.** Each ion's hot state (z, RNG, counters) lives in
registers. There are **no big input arrays** — ions are *generated* on the device
from their indices, so there is essentially nothing to copy to the GPU except the
tiny parameter struct. The only writes are atomics into `H` (n_bins buckets) and
the two crossing counters.

## 4. The GPU mapping

**Thread-to-data mapping.** One thread owns one (or, via a grid-stride loop,
several) ions. Thread `t = blockIdx.x·blockDim.x + threadIdx.x` handles ions
`t, t+stride, t+2·stride, …` where `stride = blockDim.x · gridDim.x`. This is the
canonical **independent-histories / Monte-Carlo** pattern (`docs/PATTERNS.md §1`),
the same shape as flagship `5.01` (Monte-Carlo dose).

**Launch configuration.** `block = 256` threads (a multiple of the 32-lane warp;
8 warps give the scheduler enough work to hide the `exp`/`log` latency). `grid =
1024` blocks, fixed; the grid-stride loop lets that fixed grid cover any `n_ions`,
so we never resize for big inputs and we keep the GPU saturated with resident
warps.

**Memory hierarchy.**

- **Registers** hold each ion's `z`, RNG state, and per-ion crossing counters —
  the hot path, zero contention.
- **Global memory** holds the occupancy histogram `H[n_bins]` and the 2 crossing
  counters, updated with `atomicAdd`. Many threads (different ions) land in the
  same bin on the same step, so the add **must** be atomic.
- **No shared memory** in the teaching version: the histogram is tiny and integer
  global atomics are already exact. The privatization optimization (per-block
  `__shared__` histogram, reduced once per block) is left as Exercise 4 — it cuts
  global-atomic traffic and is how production histogramming is done.
- **No constant/texture memory** needed: the parameters are a small struct passed
  by value as a kernel argument (it lands in constant-bank parameter space
  automatically).

**No CUDA library is linked.** The one place a library *could* slot in is the
RNG: production GPU MC uses **cuRAND**. We deliberately hand-roll a shared
`__host__ __device__` **splitmix64** generator instead, because the whole
verification strategy depends on the CPU and GPU drawing the *same* random numbers
for the same ion — and cuRAND's device streams are not bit-reproducible against a
host generator. Writing the RNG by hand (a few integer mixes) is cheap and buys us
exact CPU/GPU parity. If you only needed device-side randomness and not CPU
parity, `curand_init` + `curand_normal` per thread would be the idiomatic choice.

```
            grid (1024 blocks)             global device memory
   ┌──────────────────────────────┐        ┌───────────────────────────┐
   │ block 0   block 1   …  block N│        │ H[0] H[1] … H[n_bins-1]   │  occupancy
   │ ┌────┐    ┌────┐       ┌────┐ │ atomic │ crossings[0]=fwd          │
   │ │256 │    │256 │   …   │256 │ │───add─►│ crossings[1]=rev          │
   │ │thr │    │thr │       │thr │ │        └───────────────────────────┘
   │ └────┘    └────┘       └────┘ │
   └──────────────────────────────┘
   each thread: grid-stride over ions; per-ion z/RNG/counters in registers
```

## 5. Numerical considerations

- **Precision.** The Langevin step is `double` (FP64) throughout. Brownian
  trajectories are chaotic — a tiny rounding difference (e.g. an FMA contraction)
  diverges exponentially — so to make the CPU and GPU produce the *same*
  trajectory we must use the same precision and the same operation order. FP64
  here is about **reproducibility**, not accuracy of any single noisy step.
- **The determinism trick (the heart of the design).** A floating-point sum of
  per-ion currents would depend on the (nondeterministic) order in which warps
  retire their atomics — so it would *not* reproduce run-to-run, and would not
  equal the CPU. We instead tally only **integers**: occupancy counts and crossing
  counts. **Integer addition is associative and commutative**, so the atomic adds
  give the same total regardless of order → the GPU result is deterministic *and*
  bit-identical to the CPU's serial `+=` (`docs/PATTERNS.md §3`).
- **Shared physics ⇒ identical math.** Both sides call the same
  `bd_step()` / `rng_normal()` from `channel_physics.h` (the HD-macro idiom,
  `docs/PATTERNS.md §2`). The CPU loops it serially; the GPU runs one ion per
  thread. Same RNG seeding from the same ion index ⇒ same `ξ` sequence ⇒ same
  trajectory ⇒ same tally.
- **Box–Muller consumes a fixed number of draws per step** (two uniforms, one
  normal returned). That fixed consumption is essential: if the host and device
  ever consumed a different number of random words per step, their streams would
  desynchronize and the trajectories would drift apart.
- **Stability.** The explicit Euler–Maruyama (Ermak–McCammon) step is stable as
  long as the drift step `D·F·δt` and the diffusion step `√(2Dδt)` are small
  relative to the landscape's features (σ, L). The committed sample is on the
  coarse side (Exercise: shrink `δt`).

## 6. How we verify correctness

Two layers of evidence:

1. **Independent reimplementation agreement.** `src/reference_cpu.cpp` is a plain,
   obviously-correct serial double loop; `src/kernels.cu` is the parallel GPU
   version. `main.cu` runs both and asserts **exact** equality: every occupancy
   bin matches and both crossing counters match. **Tolerance = 0** — not a
   floating-point "close enough," but bit-identical integers (`docs/PATTERNS.md
   §4`, the "exact" row). Two independently written programs (one trivially
   correct, one optimized) producing the *same* integers is strong evidence the
   optimized one is right.
2. **Physical sanity checks** (printed, and checkable by eye):
   - **Conservation:** total occupancy = `n_ions · n_steps` exactly (every step
     bins exactly one ion). The stderr line reports this.
   - **Detailed balance:** with `V = 0` the net flux must be ≈ 0 (no driving
     force ⇒ no steady current) — Exercise 1.
   - **Barrier signature:** the occupancy histogram is **depleted at the centre**
     (the bin nearest z = L/2 has the fewest counts), the direct read-out of the
     PMF barrier.

Edge cases the loader guards: non-positive pore length, zero bins, zero ions,
non-positive D/δt — all rejected with a clear message so the demo fails loudly
rather than dividing by zero.

## 7. Where this sits in the real world

This is a deliberately **reduced** model. Production ion-channel science differs on
several axes:

- **Dimensionality & structure.** Real permeation is 3-D through an atomistic
  protein pore solved from a **PDB structure** (e.g. KcsA `1BL8`). Tools like
  **HOLE2** / **GROMACS** compute the **pore radius profile** R(z) by sphere-packing
  down the channel axis (the catalog's "HOLE pore radius algorithm") — a real
  GPU kernel one could add here.
- **The PMF is measured, not assumed.** Our Gaussian `U(z)` stands in for a PMF
  obtained by **umbrella sampling**: run many restrained MD windows along z,
  reweight with WHAM. That is the catalog's "umbrella sampling + PMF along the
  channel axis," and it is where most of the compute actually goes.
- **Multi-ion, knock-on conduction.** K⁺ channels conduct by a **single-file
  knock-on** mechanism — several ions in the filter pushing each other through.
  That needs the ions' mutual electrostatics (Poisson–Boltzmann or explicit
  charges), which our one-ion-at-a-time model omits.
- **Applied-field all-atom MD.** The catalog's "non-equilibrium MD with applied
  electric field" adds a uniform `E` field to a full MD integrator (NAMD, GROMACS)
  and counts ion crossings over hundreds of nanoseconds — the BD model here is the
  cheap caricature of exactly that experiment.
- **Gating.** The conformational open↔closed transition (voltage-sensor movement
  in Nav/Kv, ligand binding, mechanics) operates on µs–ms and is studied with
  enhanced-sampling MD, Markov-state models, or coarse-grained dynamics — a whole
  research program beyond this permeation demo.

What this teaching version *does* faithfully capture: the core idea that
**conductance = barrier-limited diffusion in a tilted potential**, and the GPU
pattern that makes the statistics affordable (thousands of independent
trajectories, one per thread, integer-atomic tallies).

---

## References

- **Ermak & McCammon (1978)**, *Brownian dynamics with hydrodynamic interactions*,
  J. Chem. Phys. — the integration scheme used here.
- **Kramers (1940)** — barrier-crossing rate theory; explains the `exp(−ΔU)`
  conductance dependence in Exercise 3.
- **HOLE / HOLE2** (Smart et al.; <https://github.com/osmart/hole2>) — the pore-
  radius algorithm named in the catalog; study the sphere-packing idea.
- **GROMACS** (<https://github.com/gromacs/gromacs>) — production GPU membrane MD;
  read its applied-field and analysis tooling.
- **NAMD + VMD** (<https://www.ks.uiuc.edu/Research/vmd/>) — the classic applied-
  field ion-permeation workflow.
- **MDAnalysis** (<https://github.com/MDAnalysis/mdanalysis>) — crossing-count
  conductance estimation from trajectories, mirrored by our fwd/rev counters.
- **CUDA C++ Programming Guide**, §atomics and §occupancy — the basis for the
  integer-atomic determinism argument in §5.
