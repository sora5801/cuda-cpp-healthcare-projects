# THEORY — 5.12 FLASH Radiotherapy GPU Modeling

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use. This document also states, plainly,
> where this teaching model departs from research-grade FLASH simulation._

---

## 1. The science

### 1.1 What FLASH radiotherapy is

Conventional external-beam radiotherapy delivers dose at ~0.03–0.1 Gy/s over
minutes. **FLASH radiotherapy (FLASH-RT)** delivers the *entire* dose in a few
**millisecond pulses at ultra-high dose rate (UHDR, > 40 Gy/s** — often > 10⁴
Gy/s for electrons, > 100 Gy/s for protons). The startling, repeatable
experimental observation (mice, zebrafish, and early large-animal and first
human studies) is the **FLASH effect**: at the *same total dose*, UHDR delivery
**spares normal tissue** (less fibrosis, less lung/skin/gut toxicity) while
achieving the *same tumour control* as conventional delivery. If it holds up in
humans, it widens the therapeutic window — the central prize in radiation
oncology.

### 1.2 Why oxygen is the leading suspect

Radiation kills cells largely by damaging DNA. That damage comes in two flavours:
**direct** ionisation of DNA, and **indirect** damage from water radiolysis
radicals (OH•, e⁻ₐq, H•, and downstream peroxyl species) that diffuse to the DNA
and react with it. A radical-induced lesion can either be chemically **repaired**
(e.g. by hydrogen donation from thiols) or **"fixed"** — made permanent — when
molecular **oxygen (O₂)** reacts with the radical site first. This is the classic
**oxygen fixation hypothesis**, and it is why oxygenated cells are ~3× more
radiosensitive than anoxic ones (the *oxygen enhancement ratio*, OER ≈ 3).

The leading mechanistic explanation of the FLASH effect is **transient radiolytic
oxygen depletion**: at UHDR the radicals are produced almost instantaneously and
locally, and they consume the available O₂ *faster than the vasculature can
resupply it*. During that brief window the tissue is transiently hypoxic → less
oxygen to fix damage → less biological damage. Normal tissue (moderately
oxygenated, ~20–40 mmHg) has "room" to be transiently depleted; the tumour core
(already hypoxic, < 10 mmHg) has little O₂ either way, so it is not further
spared — preserving tumour control. (Oxygen depletion alone probably does not
explain the *entire* effect; radical–radical recombination and inflammation/
immune pathways are also implicated. This project models the oxygen story because
it is the most computationally legible.)

### 1.3 What this project models (and what it does not)

We model, **per tissue voxel**, the coupled radical/oxygen chemistry following a
pulse train, and score damage through the oxygen-enhancement ratio. We **sweep
oxygen tension** (hypoxic → normoxic) and compare **conventional vs FLASH**
delivery of the same dose. We deliberately **do not** simulate particle transport
or track-structure: those are collapsed into a single lumped radical-yield term
`g_rad` [µM/Gy]. See §7 for the research-grade picture.

---

## 2. The math

### 2.1 State and reactions (per voxel)

Two lumped concentrations evolve in time `t` [s]:

- `R(t)`  — reactive-radical concentration [µM],
- `O2(t)` — dissolved molecular oxygen [µM].

Oxygen tension is reported clinically in mmHg; we convert with Henry's law at
body temperature, `[O₂] ≈ 1.4 µM/mmHg · pO₂`.

Three lumped processes act:

1. **Radical–radical recombination** (harmless), 2nd order: rate `k_rr R²`
   (removes two radicals per event).
2. **Radical–oxygen consumption** (the damage-fixing channel), 2nd order:
   rate `k_ro R·O2` (removes one radical and one O₂).
3. **Vascular oxygen resupply** (diffusion from capillaries), 1st order toward a
   local supply level `O2_sup` (= the ambient pO₂): rate `k_diff (O2_sup − O2)`.

This gives the autonomous ODE system:

```
dR/dt  = −2 k_rr R²  −  k_ro R·O2
dO2/dt = − k_ro R·O2  +  k_diff (O2_sup − O2)
```

Radicals are injected in discrete bursts (one per beam pulse): at each pulse,
`R ← R + g_rad · dose_per_pulse`. The pulse *timing* is the only thing that
differs between conventional and FLASH delivery (§3).

### 2.2 The damage score: the OER curve

The biological output is scored with the normalized **Alper–Howard–Flanders**
OER as a function of the oxygen the radicals actually experienced:

```
OER(O2) = (m·O2 + K) / (O2 + K)
```

- `O2 = 0`   → `K/K = 1`      (anoxic: least radiosensitive),
- `O2 = K`   → `(m+1)/2`      (half-way up the S-curve),
- `O2 ≫ K`   → `m`            (fully oxygenated: most radiosensitive, `m ≈ 3`),

with half-maximum at `K ≈ 3 mmHg ≈ 4.2 µM`. The **effective O₂** is the
**radical-weighted time-average**

```
O2_eff = ∫ R·O2 dt  /  ∫ R dt
```

— we weight by `R` because oxygen only matters for damage *while radicals are
present*. Finally,

```
damage = total_dose · OER(O2_eff).
```

### 2.3 Why the FLASH effect emerges

In **conventional** delivery the inter-pulse gap is long compared with the O₂
recovery time `1/k_diff`, so O₂ fully refills between pulses; each pulse's
radicals see near-ambient O₂ → high `O2_eff` → high OER → high damage. In
**FLASH** delivery the gap is far shorter than `1/k_diff`, so successive pulses'
radicals pile up and collectively drive O₂ down *before it can recover* → low
`O2_eff` → lower OER → less damage. The **sparing factor**
`= damage_conv / damage_FLASH > 1`. Because the OER curve is *steepest* at low
O₂, a fixed depletion buys the most sparing at low oxygenation and the least when
the tissue is already O₂-saturated — the shape the demo reproduces.

---

## 3. The algorithm

For one voxel (`integrate_voxel` in `src/flash.h`):

```
O2_sup ← 1.4 · pO2                     # supply level [µM]
state  ← (R=0, O2=O2_sup, wO2=0, wR=0) # wO2,wR = weighted-average accumulators
min_O2 ← O2_sup
inject ← g_rad · dose_per_pulse
for each of n_pulses pulses:
    R ← R + inject                     # deposit this pulse's radicals
    repeat steps_per_gap times:        # relax chemistry across the inter-pulse gap
        RK4 step of dt on (R, O2, wO2, wR)
        track min_O2
repeat relax_steps times:              # post-delivery relaxation
    RK4 step of dt; track min_O2
O2_eff ← wO2 / wR
damage ← total_dose · OER(O2_eff)
```

The **only** difference between the two delivery modes is `steps_per_gap`
(→ the inter-pulse gap `steps_per_gap · dt`): large for conventional, `1` for
FLASH. Everything else — dose, pulses, chemistry constants — is identical, so any
difference in the output is due to timing alone.

### 3.1 The ensemble

The full run is a 2-D sweep: `n_po2` oxygen levels × `{conventional, FLASH}` =
`2·n_po2` **independent** per-voxel solves. Flat index
`idx = po2_index·2 + mode` (see `member_axes`/`member_job` in
`reference_cpu.h`).

### 3.2 Complexity

Let `M = 2·n_po2` members and `S` the total RK4 steps per member
(`n_pulses·steps_per_gap + relax_steps`). Each RK4 step is O(1) work (four
derivative evaluations of a fixed-size system). So:

- **Serial (CPU):** `O(M · S)` — members done one after another.
- **Parallel (GPU):** `O(S)` *wall-clock* with `M` threads in flight (plus launch
  overhead) — the members run concurrently. The speed-up grows with `M`; a real
  FLASH map has millions of voxels, where the GPU wins decisively. On the tiny
  16-member demo the GPU is *slower* than the CPU (launch-bound) — an honest
  teaching point, not a benchmark (see §5.3 and PATTERNS.md §7).

---

## 4. The GPU mapping

This is the **ensemble-ODE-integration** pattern (PATTERNS.md §1, exemplified by
flagships `9.02` SEIR and `13.02` PBPK).

- **Thread ↔ data:** `idx = blockIdx.x·blockDim.x + threadIdx.x` is the flat
  ensemble-member index; thread `idx` integrates member `idx` and writes one
  `VoxelResult`. A guard `if (idx >= ensemble_size) return;` covers the ragged
  last block.
- **Block size:** 128 threads. Each thread runs a long, register-heavy
  double-precision RK4 loop, so a smaller block keeps register pressure per SM
  manageable while still giving the scheduler several warps to hide latency.
- **Memory hierarchy:** inputs are *not* in device arrays — the small
  `EnsembleConfig` is passed **by value**, so each thread reconstructs its
  parameters from registers/local memory via `member_job`. The state
  (`R, O2, wO2, wR`) lives in **registers**. The only device allocation is the
  output array (one `VoxelResult` per thread) in **global memory**, written once.
  No shared memory and **no atomics** are needed — the members are fully
  independent.
- **Divergence:** mild. Every member runs the same fixed number of RK4 steps;
  only the `min_O2` comparison branch differs, which is cheap.
- **Occupancy/bandwidth:** the kernel is *compute-bound* (lots of FLOPs per byte
  moved), the opposite of a memory-bound stencil. There is essentially no global
  traffic during the integration, so occupancy is limited by registers, not
  bandwidth.

```
grid  = ceil(M / 128) blocks
block = 128 threads
thread idx --> member_job(config, idx) --> integrate_voxel(...) --> out[idx]
```

### 4.1 The shared `__host__ __device__` core

`src/flash.h` defines `chem_deriv`, `chem_rk4_step`, `oer`, and `integrate_voxel`
as `FLASH_HD inline` functions, where `FLASH_HD` expands to `__host__ __device__`
under `nvcc` and to nothing under the host compiler (the HD-macro idiom,
PATTERNS.md §2). The CPU reference (`reference_cpu.cpp`) *loops* `integrate_voxel`
over all members; the kernel calls the *same* `integrate_voxel` from one thread.
They therefore execute byte-identical arithmetic, which is what makes
verification exact rather than approximate.

---

## 5. Numerical considerations

### 5.1 Precision and stability

All chemistry is **double precision** (`double`). RK4 is 4th-order accurate
(O(dt⁴) local error) and stable here because the fastest timescale — vascular
recovery `1/k_diff ≈ 3 ms` — is large compared with `dt = 10 µs`
(`k_diff·dt ≈ 3e-3 ≪ 1`). Radical injection is applied as a discrete jump
*between* RK4 sub-steps, keeping the integrated system autonomous (so RK4's
intermediate stages are well-defined).

### 5.2 Non-negativity clamp

Concentrations are clamped to ≥ 0 after each step to absorb tiny RK4 undershoots
(otherwise a following `R²` term could misbehave). **The same clamp runs on CPU
and GPU**, so it does *not* break bit-for-bit parity.

### 5.3 Determinism (why stdout is reproducible)

The model is a **deterministic ODE** — no random sampling — so there are no
atomics and no order-dependent floating-point sums. stdout is therefore
byte-identical every run and across `Release`/`Debug` (verified). Timings and the
run-varying verification number go to **stderr** (shown, not diffed). This is the
determinism contract of PATTERNS.md §3; note we chose the deterministic ODE
*instead of* the catalog's stochastic Gillespie SSA precisely to keep the demo
reproducible — the SSA variant (which would need integer-count accumulation to
stay deterministic) is left as an exercise.

---

## 6. How we verify correctness

1. **CPU vs GPU (exactness).** `main.cu` integrates the ensemble on both sides and
   takes the worst absolute difference over every scalar of every member. Because
   both run the identical `integrate_voxel`, they agree to ~`1e-14`; we verify
   against a documented tolerance of `1e-9` (comfortably above FMA/rounding drift,
   below any physically meaningful scale). This is the "same exact operations both
   sides" case of PATTERNS.md §4.
2. **Physical sanity (the science, not just the code).** The sparing factor must
   be **> 1 at every oxygen level** (FLASH never *increases* damage in this
   model), must be **largest at low pO₂** and monotonically shrink toward 1 as
   pO₂ rises, and the FLASH `min_O2` must be well below the ambient level (real
   depletion occurred). The committed `expected_output.txt` exhibits all three.
3. **Cross-config determinism.** stdout is identical under `Release|x64` and
   `Debug|x64`.

Edge cases handled: `wR = 0` (no radicals ever injected) falls back to
`O2_eff = O2_sup`; the loader rejects non-positive `dt`, dose, pulse counts, and
step counts.

---

## 7. Where this sits in the real world

A research-grade FLASH simulation is far heavier than this ODE:

- **Particle transport + track structure.** Codes like **Geant4-DNA**,
  **TOPAS(-nBio)**, and **GATE 10** track each particle and its secondaries,
  depositing energy and seeding the *initial spatial distribution* of dozens of
  radiolysis species (OH•, e⁻ₐq, H•, H₂, H₂O₂, HO₂•/O₂•⁻ …). We collapse all of
  this into one scalar yield `g_rad`.
- **Full radiolysis reaction networks + spatial diffusion.** The real chemistry is
  a stiff reaction–diffusion system over ~10–10³ species on a fine spatial grid,
  often integrated stochastically (**Gillespie SSA**) or with implicit stiff
  solvers. **MPEXS2.1-DNA** runs exactly this water-radiolysis chemistry under
  UHDR *on the GPU* — the closest analogue to what this project gestures at, but
  with real species and real rate constants.
- **LET-dependent yields and true pulse structure.** Radical yields depend on
  linear energy transfer (LET) and on the machine's actual pulse shape/frequency;
  proton and electron FLASH differ substantially.
- **Biology beyond oxygen.** Immune/inflammatory modulation, mitochondrial and
  peroxidation chemistry, and intrinsic radical recombination all likely
  contribute; oxygen depletion is necessary-but-not-sufficient in current
  understanding.

What **transfers directly** from this teaching version to the real one is the
**GPU pattern**: independent per-voxel kinetics → one thread (or block) per voxel,
with a shared, verifiable numerical core. Scaling the *same* structure from 16
members to millions of voxels — and swapping the lumped ODE for a real
species network — is precisely how production GPU radiolysis codes are built.
```mermaid
flowchart LR
  A[pO2 sweep x mode] --> B[member_job: build VoxelJob]
  B --> C[integrate_voxel: pulse-train RK4]
  C --> D[O2_eff = wO2/wR]
  D --> E[damage = dose * OER O2_eff]
  E --> F[sparing = conv/FLASH]
```
