# THEORY — 5.11 Microdosimetry & Track-Structure Simulation

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to radiation physics. This project is a
> **reduced-scope teaching model**; the final section says exactly what a
> production code does differently.

---

## The science

### Why microscopic energy deposition matters

Absorbed **dose** (energy per unit mass, grays) is a *macroscopic average*. It
tells you the mean energy deposited in a gram of tissue but hides a crucial fact:
at the scale of a **cell nucleus** (~5–10 µm) and especially at the scale of the
**DNA double helix** (~2 nm wide, base pairs 0.34 nm apart), energy deposition is
violently **stochastic and discrete**. One cell in a uniformly irradiated
population might receive a single ionization; its neighbour might receive a dense
cluster that shatters both DNA strands at once.

That difference is everything for biology. The same dose delivered by **low-LET**
radiation (fast electrons from X-rays: sparse, well-separated ionizations) versus
**high-LET** radiation (alpha particles, carbon ions: dense ionization tracks)
produces very different cell killing. High-LET radiation is more **biologically
effective** per gray — its **RBE** (Relative Biological Effectiveness) can exceed
3 — precisely because its dense tracks create **clustered, hard-to-repair DNA
lesions**, chiefly **double-strand breaks (DSBs)**.

- **Microdosimetry** (Rossi, 1960s) quantifies the *statistics* of energy
  deposited in micrometre volumes. Its central quantity is **lineal energy**
  y = ε / ℓ̄, the energy imparted ε to a small volume divided by the mean chord
  length ℓ̄ of that volume. Measured with a **TEPC** (tissue-equivalent
  proportional counter), it yields a spectrum f(y).
- **Nanodosimetry / track structure** goes one level deeper: it simulates the
  *individual interactions* (ionizations, excitations, elastic scatters) of a
  charged particle and its secondaries, then scores damage on an explicit DNA
  target. Codes: Geant4-DNA, PARTRAC, TOPAS-nBio, MPEXS-DNA.

### DNA damage taxonomy

- **SSB** — single-strand break: one sugar-phosphate backbone cut. Common,
  usually faithfully repaired using the intact complementary strand.
- **DSB** — double-strand break: both strands cut within ~10 base pairs. Rare but
  the key lethal/mutagenic lesion; misrepair causes chromosome aberrations.
- **Clustered / complex lesions** — multiple breaks + base damage in a small
  volume; the signature of high-LET radiation and the hardest to repair.

Applications named in the catalog: carbon-ion RBE modelling, targeted
radionuclide therapy with **alpha emitters** (Ac-225, Ra-223), and predicting
clustered-damage yields in **mixed radiation fields**.

---

## The math

We model the simplest system that still teaches the ideas: a cubic scoring volume
of **liquid water** (the standard tissue surrogate; ~70% of a cell) of side `L`
nanometres, traversed by primary charged particles.

### Free-flight between ionizations

Along its path a charged particle ionizes water at a mean rate Σ (events per nm),
set by its **LET class**. The distance to the next ionization is exponentially
distributed with mean 1/Σ. To sample it from a uniform ξ ∈ (0,1]:

```
s = -ln(ξ) / Σ                (nanometres)
```

This is the **inverse-CDF** (inverse-transform) method applied to the exponential
distribution — the same free-path sampling used by every Monte Carlo transport
code (compare project 5.01's photon steps).

### Mixed field via LET straggling

A truly mono-energetic beam gives a nearly delta-function y-spectrum, which is
boring and unrealistic. Real fields have **straggling** and are often **mixed**
(a spread of particle types/energies). We give each track its own local density

```
Σ_track = Σ · exp(σ_LET · Z),   Z ~ N(0,1)         (lognormal)
```

so a minority of dense tracks form the physically important high-y, DSB-rich
tail. `σ_LET = let_spread` in the code; `σ_LET = 0` recovers a pure mono-LET beam.

### Energy quantisation

Each ionization deposits an integer number of **quanta** of size `quantum_eV`
(≈ mean energy per ionization in water, tens of eV). Working in integer quanta is
a deliberate numerical choice (see *Numerical considerations*): the total energy
imparted by a track,

```
ε = (Σ quanta) · quantum_eV        (eV)
```

is an exact integer count times a constant.

### Lineal energy

For a convex body, **Cauchy's theorem** gives the mean chord length
ℓ̄ = 4V/S. For a cube of side L, V = L³ and S = 6L², so

```
ℓ̄ = 4L³ / 6L² = 2L/3.
```

The track's lineal energy is then

```
y = ε / ℓ̄          (convert ε to keV and ℓ̄ to µm ⇒ y in keV/µm).
```

Binning y over all tracks gives the microdosimetric spectrum f(y). Two summary
means:

```
frequency-mean  yF = Σ y f(y) / Σ f(y)
dose-mean       yD = Σ y² f(y) / Σ y f(y)
```

yD weights by dose (each event contributes energy ∝ y) and is dominated by the
high-LET tail — it is the number radiobiologists quote and correlates with RBE.

### DNA damage (combinatorial model)

DNA is a line of `n_dna_segments` segments along the box y-axis at its centre. An
ionization at radial distance r from the axis is a **break candidate** iff
r ≤ `dna_radius_nm`; it is assigned to a segment and a random strand s ∈ {0,1}.
Given a track's break list, classify:

```
DSB  = # pairs (a,b): same segment, opposite strands, |pos_a − pos_b| ≤ 3.4 nm
SSB  = # breaks not consumed by any DSB
```

3.4 nm ≈ 10 base pairs, the standard proximity threshold for calling two breaks
a DSB.

---

## The algorithm

Per **track** (one Monte Carlo history):

```
1.  seed a private RNG stream from (base_seed, track_index)
2.  sample this track's LET  Σ_track = Σ · exp(σ_LET · N(0,1))
3.  pick a random entry point on the box face; direction ≈ +y
4.  repeat:
        s   = -ln(ξ)/Σ_track            # free flight
        advance position by s along the ray
        if exited the box: stop
        deposit quanta_per_ion quanta   # primary ionization
        if within dna_radius of axis: record a break (segment, random strand, pos)
        with prob p_delta:              # delta ray
            deposit delta_quanta
            maybe record a second nearby break
5.  classify the break list -> (SSB, DSB)         # O(k²), k ≤ 96
6.  compute lineal energy y and its histogram bin
7.  emit TrackResult{ energy_quanta, ssb, dsb, y_bin }   (all integers)
```

Then **reduce** across all tracks: sum energy, SSB, DSB; increment the y-histogram.

**Complexity.** Let N = number of tracks and s̄ = mean steps per track
(≈ Σ·L). Serial cost is **O(N · s̄)** time, O(1) tally space plus O(k) per-track
scratch for the break list. The break classifier is O(k²) but k is small and
capped (`TS_MAX_BREAKS = 96`), so it is a constant factor. The computation is
**embarrassingly parallel over N** — no track depends on another.

---

## The GPU mapping

**Pattern:** *stochastic / Monte-Carlo histories → per-thread RNG + atomic
scoring* (PATTERNS.md §1), the same pattern as flagship **5.01** (Monte Carlo
dose). The shared `__host__ __device__` core is PATTERNS.md §2.

### Thread-to-data map

One GPU thread simulates one or more tracks via a **grid-stride loop**:

```
i = blockIdx.x * blockDim.x + threadIdx.x          # this thread's first track
stride = blockDim.x * gridDim.x                    # total threads
for (t = i; t < N; t += stride) simulate track t
```

A fixed grid (1024 blocks × 256 threads) gives the scheduler plenty of resident
warps to hide the branchy per-track work; the stride loop then covers any N.

### Memory hierarchy

- **Registers / local memory:** each track's state (position, RNG, the small
  `BreakEvent breaks[96]` scratch) lives per-thread. No dynamic allocation — the
  fixed cap is what lets the break list sit on the stack on the device.
- **Global memory:** the tallies (three u64 scalars + the y-histogram). Tiny, but
  written by every thread, so they are updated with `atomicAdd`.
- **Constant memory (production):** the catalog envisions cross-section tables in
  constant memory (read by all threads, never written) and the current material's
  table staged in **shared memory** per step. Our teaching model replaces those
  tables with a single `sigma_ion` parameter, so we don't need them — but this is
  exactly where they would go.

### Integer atomics → determinism (the load-bearing trick)

Many threads `atomicAdd` into the same handful of addresses. Floating-point
addition is **not associative**, so a float tally would depend on the
(nondeterministic) order in which warps retire — the GPU result would drift run to
run and never exactly match the CPU. Because every tallied quantity here is an
**integer count**, the atomic adds **commute**: the sum is order-independent,
deterministic, and **bit-identical to the serial CPU sum**. That is the whole
reason we quantise energy into integer `quanta`.

### Warp divergence — the real track-structure challenge

Different primaries take different numbers of steps and hit different branches
(ionize? delta ray? near DNA?), so lanes in a warp finish at different times and
execute different code paths — classic **thread divergence** that wastes SIMT
lanes. Production GPU codes (MPEXS-DNA) mitigate it by **sorting tracks by
interaction type before each step** so a warp processes coherent work, and by
processing **one warp per track** for the densest tracks. We deliberately keep the
readable one-thread-per-track version and explain the fix here rather than
obscuring the teaching kernel; Exercise 4 has you try the sort.

### Performance note (honest)

On 4000 tiny tracks the kernel is **launch/branch-bound** and roughly ties the
CPU — this is expected (PATTERNS.md §7). The GPU's advantage grows with track
count; real nanodosimetry runs 10⁶–10⁸ tracks where the 50–70× speedups the
catalog cites appear. Timing is a **teaching artifact, never a benchmark claim**.

---

## Numerical considerations

- **Precision:** all transport math is `double`. The RNG uses 53 random bits per
  uniform. Because the CPU and GPU call the *identical* inline functions from
  `ts_physics.h`, they produce the same doubles and the same integer outcomes.
- **Determinism:** guaranteed by (a) counter-based per-track seeding
  `(seed, track)` — independent of execution order — and (b) integer tallies with
  commuting atomics. stdout is therefore byte-identical every run; timings (which
  vary) go to stderr.
- **RNG choice:** splitmix64 is a small, well-mixed counter-based generator,
  chosen so the *same* code compiles and runs identically on host and device.
  Production GPU codes use **cuRAND Philox** (a counter-based generator built for
  exactly this per-thread-stream use case); splitmix64 teaches the idea with zero
  library dependency and exact CPU/GPU parity.
- **Guards:** free paths use ξ ∈ (0,1] so `log` is finite; a step-count guard
  (200000) prevents a runaway history; segment/bin indices are clamped to valid
  ranges; the break list is capped and truncates honestly if exceeded.
- **Overflow:** tallies are `unsigned long long`; 4000 tracks × ~400 quanta ≈ 1.6M
  is nowhere near the 64-bit limit, and the type scales to billions of tracks.

---

## How we verify correctness

Two independent checks:

1. **CPU == GPU, exactly.** `reference_cpu.cpp` replays the same tracks serially
   and `main.cu` asserts every tally (total energy, total SSB, total DSB, and all
   y-histogram bins) is **bit-identical** to the GPU's. Tolerance = **0** — the
   correct choice because both sides run the same integer operations (PATTERNS.md
   §4, the "exact" case). Any mismatch is a real bug (a race, a divergent branch,
   an RNG desync), not floating-point noise.
2. **Physical sanity.** The reported summaries move in the physically expected
   directions: raising `sigma_ion`/`let_spread` (higher LET) *lowers* the SSB/DSB
   ratio and *raises* yD; a sparse mono-LET beam does the opposite. The y-spectrum
   is a broad, right-skewed distribution with a high-y tail — the qualitative
   shape of a real microdosimetric spectrum. These are teaching checks on the
   *shape*, not claims of quantitative accuracy.

The committed sample is engineered so the result is interpretable: a moderately
high-LET mixed field gives SSB/DSB ≈ 2.3, DSB/track ≈ 0.11, yD ≈ 226 keV/µm, and
a populated 12-bin f(y).

---

## Where this sits in the real world

What a production track-structure code does that we do **not**:

| Aspect | This teaching model | Geant4-DNA / MPEXS-DNA / TOPAS-nBio / PARTRAC |
|---|---|---|
| Interactions | one lumped "ionization" with a scalar rate Σ | explicit ionization, excitation, elastic scatter, charge transfer, each with tabulated cross-sections from sub-eV to MeV |
| Secondaries | a stochastic delta-ray "cluster" | every secondary electron tracked to thermalisation |
| Chemistry | none | radiolysis: H₂O⁺, ·OH, e⁻aq, H·, etc., diffusion-reaction on ns–µs timescales |
| DNA target | line of segments + capture radius + random strand | atomistic/coarse-grained double helix in nucleosomes/chromatin |
| Damage | combinatorial SSB/DSB proximity rule | direct + indirect (chemical) damage, base damage, complex-lesion spectra |
| Biology | none | links yields to survival via LEM/MKM/repair kinetics for RBE |
| GPU | one thread per track, atomic tallies | one warp per track, constant-memory cross-sections, shared-memory step tables, tracks sorted by interaction type to cut divergence |

The **architecture** transfers directly: independent histories, per-thread
counter-based RNG, integer atomic scoring, and the divergence problem. Swapping
the scalar `sigma_ion` for real cross-section tables (in constant memory) and
adding a chemistry stage is the path from this teaching kernel toward a research
code — a large but conceptually incremental step.
