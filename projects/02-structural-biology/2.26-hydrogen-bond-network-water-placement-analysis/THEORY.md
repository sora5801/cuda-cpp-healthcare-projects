# THEORY — 2.26 Hydrogen Bond Network & Water Placement Analysis

> The deep dive. We model **Grid Inhomogeneous Solvation Theory (GIST)**: turning a
> molecular-dynamics trajectory of explicit water into a 3D map of *where* water
> sits in a protein pocket and *how thermodynamically happy* it is there. Read
> [README.md](README.md) first for the quick tour; this file is the *why*.
>
> **Educational only. Not for clinical or real drug-design decisions.** The data
> here is **synthetic** and the energy model is a deliberately simplified teaching
> stand-in for a real force field.

---

## 1. The science

Proteins do their work in water. When a drug molecule (a ligand) binds in a
protein's pocket, it does not bind to a dry cavity — it must **displace the water
already there**. Some of that water is loosely held and bulk-like; some is
trapped, ordered, and thermodynamically *unhappy*. The central insight of
**WaterMap / GIST** is:

> If a water molecule sits in a pocket where it is **ordered** (low entropy)
> and/or makes **poor interactions** (unfavorable energy), then displacing it with
> a ligand atom **releases free energy** — a predicted gain in binding affinity.

So a medicinal chemist wants a *map* of the binding site answering, for every
small region of space: *how often is a water here, and how happy is it?* The
unhappy, high-occupancy spots are the ones to target with the next analog.

**GIST** builds exactly that map. It overlays a fine 3D grid of cubic **voxels**
on the pocket, runs (or is given) an explicit-solvent MD trajectory, and for every
voxel accumulates — over all frames — the water **occupancy** and the water
**interaction energy**. From those tallies it derives per-voxel thermodynamics: a
number **density** `g`, a mean **energy** `ΔE`, a translational **entropy** penalty
`−TΔS`, and their sum, the **free energy** `ΔG`. High `ΔG` flags a displaceable
water.

The **hydrogen-bond network** is the structural backbone of all this: ordered
waters are usually held by hydrogen bonds to the protein and to each other ("water
bridges"). In this reduced teaching version we capture the *energetic* fingerprint
of that network through a water–solute interaction energy; the full orientational
H-bond bookkeeping is described in §7.

---

## 2. The math

### Per-water interaction energy

For a water oxygen at position **w** and solute atoms `a` at **r**ₐ with partial
charges `qₐ`, we use a pairwise **Lennard-Jones + Coulomb** energy — the same two
terms a real molecular-mechanics force field uses for non-bonded interactions:

```
E_sw(w) =  Σ_a  4·ε·[ (σ/rₐ)¹² − (σ/rₐ)⁶ ]   +   k_C · q_water · qₐ / rₐ
                └────────── Lennard-Jones ─────────┘   └──── Coulomb ────┘
```

- `rₐ = ‖w − rₐ‖` — water–atom distance (Å).
- `ε` (kcal/mol), `σ` (Å) — LJ well depth and contact radius (≈ TIP3P oxygen).
- `k_C = 332.0636` kcal·Å/(mol·e²) — converts e²/Å to kcal/mol (standard MD prefactor).
- `rₐ` is clamped to a physical contact floor `r_min` so the steep `1/r¹²` wall
  cannot explode when a water lands almost on an atom; atoms beyond a cutoff
  `r_cut` are ignored.

### Per-voxel reduction

Let voxel `v` collect `N_v` water observations over `F` frames, with energy sum
`ΣE_v`.

**Number density relative to bulk.** In neat bulk water the expected count in a
voxel of volume `V` over `F` frames is `⟨N⟩ = ρ_bulk · V · F`, with `ρ_bulk ≈
0.0334` waters/Å³. The voxel's density is the enrichment over that:

```
g_v = N_v / (ρ_bulk · V · F)
```

`g = 1` is bulk; `g ≫ 1` is a persistently occupied, structured site.

**Mean energy relative to bulk.** A bulk water's mean interaction energy is
`E_bulk ≈ −9.5` kcal/mol (≈ 3.5 hydrogen bonds). The voxel's excess energy:

```
ΔE_v = (ΣE_v / N_v) − E_bulk
```

`ΔE > 0` means this water is *less* favorably bound than bulk (frustrated,
strained) — easy/beneficial to remove.

**Translational entropy (the leading IFST/GIST term).** Inhomogeneous Fluid
Solvation Theory expands the solvation entropy in correlation orders; the
first-order **translational** density term is

```
s_trans(r) = −k_B · ρ(r) · ln( ρ(r) / ρ_bulk )
```

Per water, the entropic **penalty** of confining it to a region of density `g`:

```
−TΔS_trans = +k_B·T · ln(g)      (for g > 1; clamped to 0 for bulk-or-below)
```

An ordered water (`g > 1`) has `ln g > 0`, so this is a positive cost of keeping
the water localized.

**GIST free energy / displaceability.**

```
ΔG_v = ΔE_v + (−TΔS_trans,v)
```

A large positive `ΔG` = a water that is both enthalpically frustrated and
entropically constrained = the most attractive to displace.

---

## 3. The algorithm

```
INPUT : MD frames (water-oxygen positions), solute atoms, a voxel grid
OUTPUT: per-voxel (g, ΔE, −TΔS, ΔG); a ranked list of hydration sites

1.  zero  count[v] = 0,  Esum[v] = 0   for every voxel v
2.  for each frame f, for each water w:                      ── the SCATTER
        v ← voxel containing w     (floor((w − origin)/spacing); skip if outside)
        e ← E_sw(w)                 (LJ + Coulomb over solute atoms)
        count[v] += 1                                          (atomic on GPU)
        Esum[v]  += e                                          (atomic on GPU)
3.  for each voxel v with count[v] ≥ min_occupancy:          ── the REDUCE
        derive g, ΔE, −TΔS, ΔG
4.  rank sites by occupancy (then ΔG); print the top few
```

**Complexity.** With `S = F·W` water samples and `A` solute atoms, the scatter is
`O(S·A)` (each water energy sums over atoms); the reduce is `O(#voxels)`. The
**parallel depth** is `O(A)` per sample (plus atomic contention) — essentially
constant in `S`, which is the entire reason the GPU wins as `S` grows. In
production GIST `S` reaches `10⁶`–`10⁹` and the grid `10⁴`–`10⁶` voxels; the
scatter dominates and is *embarrassingly parallel over samples*.

**Why rank by occupancy first.** The mean energy of a voxel visited only a handful
of times is statistically meaningless — one stray water can fake a high `ΔG`.
WaterMap/GIST therefore first **identify hydration sites** from the robust
occupancy signal, then annotate them with thermodynamics. We mirror that: a voxel
must clear a minimum-occupancy threshold to count as a site, and sites are ranked
by occupancy (ties broken by `ΔG`). This is both more correct and what makes the
demo's answer stable.

---

## 4. The GPU mapping

This is the **grid-accumulation-with-atomics** pattern (PATTERNS.md §1, shared
with Monte-Carlo dose `5.01` and k-means accumulate `11.09`).

- **Thread ↔ data.** One GPU thread per `(water, frame)` **sample**:
  `t = blockIdx.x·blockDim.x + threadIdx.x` handles sample `t`, whose `(x,y,z)`
  live at `waters[3t … 3t+2]`. Samples are independent — ideal for the GPU.
- **The scatter.** Each thread computes its voxel and energy, then **atomically**
  adds into `count[v]` and `Esum[v]`. Many threads target the same voxel (that is
  what a hydration site *is*), so the writes collide → atomics.
- **Memory hierarchy.**
  - `waters[]` (3 floats/sample) and `atoms[]` (4 floats/atom) live in **global**
    memory; `__restrict__` lets the compiler keep loads in registers. The tiny
    solute atom list stays hot in L2 across the grid. *(A production kernel with
    thousands of atoms would stage them in **shared** memory per block, or tile;
    the atom count here is small enough that global is fine — left as an exercise.)*
  - `count[]` (`unsigned int`) and `Esum[]` (fixed-point `int64`) are the
    accumulators in **global** memory; the atomics serialize per-voxel only.
  - The `GistGrid` (a handful of numbers) is passed **by value** → it sits in
    registers, free to read in every thread.
- **Launch / occupancy.** 256 threads/block — a warp multiple, 8 warps to hide the
  atom-list loads, plenty of resident blocks for occupancy on sm_75…sm_89. The
  kernel is **atomic-bound**, not compute-bound: throughput is set by contention on
  the hottest voxels, not by FLOPs.
- **The reduce** (`derive_voxels`) is cheap and runs on the **host**, reusing the
  exact same function the CPU path calls — so the ranked list is identical.

```
            grid of blocks (256 threads each)
   sample t:  ┌──────────────────────────────────────────────┐
   t0 t1 t2 … │  read w=waters[3t..]  →  v=voxel(w)           │
              │  e = E_sw(w)  over atoms[]                    │
              │  atomicAdd(count[v], 1)                       │   many t's
              │  atomicAdd(Esum[v],  fixed(e))   ────────────────►  same v
              └──────────────────────────────────────────────┘   (collide)
                              │   copy tallies D2H
                              ▼
                     derive_voxels()  (host)  →  ranked hydration sites
```

No CUDA *library* is used here — the scatter is a hand-written atomic kernel, the
heart of the lesson. (cuFFT/cuSOLVER appear in sibling projects; GIST's
nearest-neighbour orientational entropy in §7 is where Thrust/CUB sort or a k-d
tree would enter a fuller implementation.)

### Determinism: the load-bearing trick

Floating-point `atomicAdd` is **not associative**: when many threads add energies
into the same voxel, the *order* varies run-to-run, so a float sum is both
irreproducible and unequal to the serial CPU sum. We therefore accumulate energy
in **fixed-point integers** (`gist_to_fixed`: micro-kcal/mol in an `int64`).
Integer atomic adds **commute**, so:

1. the GPU energy sum is **bit-identical every run**, and
2. it equals the serial CPU sum **exactly** → verification is exact, not fuzzy.

(CUDA has a native `atomicAdd` for `unsigned long long` but not signed; energies
can be negative, so we reinterpret-cast the signed cell to unsigned — two's
complement addition is identical for both — see `device_atomic_add_fixed`.)
Occupancy counts are plain integer atomics, already deterministic.

---

## 5. Numerical considerations

- **Precision.** Energies are computed in `double` on both host and device; only
  the *accumulator* is fixed-point. Coordinates are `float` (MD precision). The
  shared `__host__ __device__` core (`gist.h`) guarantees host and device evaluate
  the **same** formula on the **same** inputs.
- **The `1/r¹²` wall.** Clamping `r` to `r_min` (≈ van der Waals contact) caps the
  LJ repulsion at a finite, physical value; without it a water sampled on top of an
  atom injects a `~10⁶` kcal/mol spike that dominates the map. The synthetic
  generator additionally forbids diffuse waters from overlapping solute atoms (a
  real water cannot occupy an atom's volume).
- **Fixed-point range.** Micro-kcal/mol in an `int64`: a voxel seeing `10⁴` waters
  × `~50` kcal/mol ≈ `5·10¹¹` units, far below the `~9.2·10¹⁸` ceiling — no
  overflow at teaching or realistic scale.
- **Atomic contention vs. determinism.** Atomics serialize per voxel, but
  fixed-point makes the result independent of *which* order they fire — contention
  costs time, never correctness.
- **`constexpr`, not `static const`.** All physical constants in `gist.h` are
  `constexpr` so nvcc accepts them in `__device__` code (a plain file-scope
  `static const` is host-only to nvcc and triggers "identifier undefined in device
  code"). A small but instructive CUDA gotcha.

---

## 6. How we verify correctness

Two independent checks, both in `main.cu`:

1. **GPU == CPU, exactly.** Because both paths share the per-element physics and
   accumulate in fixed-point integers, every voxel's occupancy and energy sum must
   match **bit-for-bit** (tolerance `== 0`; PATTERNS.md §4, the "exact" class).
   The ranked hydration-site list must be identical too (same voxels, same order,
   `max ΔG diff = 0`). Any nonzero mismatch is a real bug, not float noise. This is
   convincing because the serial reference is written to be *obviously* correct —
   one readable loop — so agreement pins down the parallel version.
2. **The embedded known answer.** The synthetic sample is engineered so two voxels
   each hold one ordered, *caged* water **every frame** (occupancy = #frames,
   `g ≈ 240`) with a strained, unfavorable energy. The program must surface those
   two voxels at **ranks 1–2** of the list — recovering the planted answer
   (PATTERNS.md §6). The demo's `expected_output.txt` pins this byte-for-byte.

Edge cases handled: waters outside the grid box (dropped, not clamped onto the
boundary), empty / under-sampled voxels (excluded as noise), divide-by-zero guards
in `g` and `ΔE`.

---

## 7. Where this sits in the real world

Production GIST (AMBER **cpptraj** `gist`, the **GISTPP** post-processor) and the
commercial **WaterMap** (Schrödinger) go substantially further:

- **Full force field.** The complete non-bonded energy of a real water model
  (TIP3P/TIP4P) against *all* solute **and** solvent atoms, with proper LJ
  combining rules, Ewald/PME long-range electrostatics, and per-water orientation —
  not a single LJ+Coulomb term against a reduced atom set.
- **Orientational entropy.** Beyond the translational `−k_B ρ ln g` term we keep,
  GIST estimates **orientational** entropy from the distribution of water
  dipole/H-bond angles using a **nearest-neighbour** (k-NN) estimator on the 6D
  position+orientation distribution rather than a closed-form density term. This is
  the "nearest-neighbor entropy estimation" in the catalog, and where a GPU k-d
  tree / sort (Thrust/CUB) would slot in.
- **Hydrogen-bond network graph.** The real analysis builds the explicit H-bond and
  **water-bridge** graph (which donor/acceptor pairs, how persistent) — the
  "hydrogen bond network" half of the title — typically via MDAnalysis/cpptraj. We
  summarize that network only through its energetic effect.
- **Reference energies & normalization.** `E_bulk`, `ρ_bulk`, and the entropy
  reference come from a separate bulk-water simulation in the same force field, not
  textbook constants.
- **Scale.** Real runs stream `10⁶`–`10⁹` frames×waters over `10⁴`–`10⁶` voxels;
  the GPU scatter is what makes that tractable — exactly the bottleneck this
  project parallelizes.

What carries over verbatim is the **shape**: lay a voxel grid, scatter independent
water observations into it with atomics, reduce to per-voxel thermodynamics, rank.
That is the GIST algorithm and the GPU pattern worth taking away.

---

## References

- **Nguyen, Young & Gilson (2012/2014)** — Grid Inhomogeneous Solvation Theory,
  the formal basis for the per-voxel energy/entropy decomposition used here.
- **Lazaridis (1998)** — Inhomogeneous Fluid Solvation Theory (IFST), the
  entropy-expansion framework GIST grids.
- **Abel, Young, Friesner et al. (2008)** — WaterMap; the "displace the unhappy
  water" affinity argument this project teaches.
- **GISTPP** — https://github.com/liedlgroup/gist-pp — GIST post-processing; study
  how per-voxel quantities are combined and thresholded.
- **AMBER cpptraj `gist`** — https://github.com/Amber-MD/cpptraj — the reference
  production GIST implementation; read its grid accumulation and entropy code.
- **MDAnalysis** — https://github.com/MDAnalysis/mdanalysis — H-bond and
  water-bridge network analysis; the structural complement to the energetics here.
- **SAMPL challenges** — https://github.com/samplchallenges/SAMPL — blinded
  water-placement benchmarks to validate a real implementation against.

(Study these for the production approach; reimplement didactically and credit the
source — do not copy code wholesale. CLAUDE.md §2.)
