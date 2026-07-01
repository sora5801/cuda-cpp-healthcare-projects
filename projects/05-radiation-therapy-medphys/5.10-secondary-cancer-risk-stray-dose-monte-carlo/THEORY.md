# THEORY — 5.10 Secondary Cancer Risk & Stray-Dose Monte Carlo

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use. This is a **reduced-scope teaching
> version** (CLAUDE.md §13); §7 describes the full research-grade problem._

---

## 1. The science

Radiotherapy cures cancer by depositing a large radiation dose (tens of grays) in a
tumour while sparing surrounding tissue. But radiation is not perfectly confined to
the beam:

- **Scatter.** A primary photon travelling through the patient undergoes Compton
  scattering, changing direction and carrying a little energy sideways into tissue
  *outside* the treated field.
- **Leakage.** No collimator is perfect; a small fraction of photons leak through the
  treatment-machine head and irradiate the whole body roughly uniformly.
- **Secondary neutrons.** In proton therapy and high-energy (>10 MV) photon therapy,
  nuclear reactions in high-Z machine parts (and in the patient) produce neutrons,
  which are especially damaging per unit dose and travel far.

These **stray doses** are three to five orders of magnitude below the target dose —
often a few millisieverts to a distant organ. Individually negligible, but they act
on **large volumes of healthy tissue** over a patient's remaining lifetime. For young
patients cured of a first cancer, radiation-induced **secondary cancers** decades
later are a real, quantified concern. The clinical question this project models is:
*given a treatment, what extra lifetime cancer risk does the stray dose create in
each out-of-field organ?* Answering it means (a) computing the stray dose everywhere
in the body, then (b) converting dose to risk with an epidemiological model
(BEIR-VII).

The hard part of (a) is statistics: because the signal is so faint, a naive Monte
Carlo would need **10¹¹–10¹²+ particle histories** to get a few counts in a distant
organ. That is intractable even on a GPU by brute force — which is exactly why
**variance reduction** is the subject of this project.

## 2. The math

**Transport.** A photon of statistical weight `w` travels through tissue with linear
attenuation coefficient `μ` (units: 1/cm). The probability it travels at least a
distance `s` without interacting follows the Beer–Lambert law:

```
P(no interaction over s) = exp(-μ s)
```

so the probability it interacts while crossing one organ slab of thickness `Δ` is
`p_int = 1 - exp(-μ Δ)`. When it interacts, a fraction `f_s` (`scatter_frac`) of its
weight **scatters** (survives, keeps going) and the rest `1 - f_s` is **absorbed**,
depositing dose locally.

**Forced detection (next-event estimator).** Rather than wait for the rare event that
a scattered photon happens to travel to a distant organ `j` and deposit there, we add
its *expected* contribution deterministically at every scatter site `i`:

```
ΔD_j  =  w · f_s · σ_side · exp(-μ · d_ij) · E
```

where `σ_side` (`sidescatter`) is the fraction of scattered weight redirected toward
each downstream organ, `d_ij = (j-i)·Δ` is the path length to organ `j`, and `E` is
the photon's energy unit. This is an **unbiased** estimator: its expectation equals
the analog process's, but with far lower variance because there is no waiting for a
rare hit.

**Survival biasing + Russian roulette.** The scattered photon continues with reduced
weight `w ← w·f_s`. Once `w` falls below a floor `w_min` (`roulette_floor`), we play
roulette: with probability `p_s` (`roulette_survive`) it survives with boosted weight
`w ← w/p_s`, else it is killed. Roulette preserves the expected weight
(`p_s·(w/p_s) + (1-p_s)·0 = w`) so it is unbiased — it just stops us wasting compute
on negligible particles.

**Machine channels.** Leakage adds a uniform weight `w_leak = leakage_frac·E` to every
organ per primary; the neutron surrogate adds `w_n = neutron_frac·E·exp(-α·j)` (a
distance falloff, `α = 0.15` here). These depend on the *machine*, not the stochastic
patient path, so they are deposited deterministically once per history.

**Dose to risk (BEIR-VII, LNT).** For low doses the excess **Lifetime Attributable
Risk** for organ `o` is taken proportional to its equivalent dose `H_o` (the Linear
No-Threshold assumption used in radiation protection):

```
LAR_o = r_o · H_o           (cases per 10^4 persons, illustrative units)
Total secondary-cancer LAR = Σ_{o out-of-field} LAR_o
```

where `r_o` is the organ's radiosensitivity coefficient. Inputs: phantom geometry,
`μ`, the VR parameters, `n_histories`, and per-organ `r_o`. Outputs: per-organ stray
dose and the total out-of-field LAR.

## 3. The algorithm

Per primary history (all in statistical weight):

1. Deposit the machine channels (leakage + neutron surrogate) to every organ.
2. Set `w = 1`. Walk organ by organ `i = 0 … n-1`:
   - Sample `ξ ~ U[0,1)`. If `ξ < p_int`, the photon interacts:
     - deposit absorbed dose `(1-f_s)·w·E` locally (large for in-field organs);
     - **forced detection:** for every downstream organ `j > i`, deposit the
       expected stray contribution `ΔD_j`;
     - continue with `w ← f_s·w`;
     - if `w < w_min`, play **Russian roulette** (kill or boost).
3. Sum all deposits into a per-organ tally; convert to risk (§2).

**Complexity.** Serial cost is `O(n_histories · n²)` in the worst case (each
interaction forced-detects to all downstream organs), with `n = n_organs` tiny
(≈10). Crucially, histories are **independent**: work `O(n_histories·n²)`, depth
`O(n²)` per history. The data-access pattern is: read a handful of scalar parameters,
scatter a few integer adds into a tiny tally. Arithmetic intensity is modest; the
kernel is compute/branch-bound, not bandwidth-bound.

## 4. The GPU mapping

**Thread-to-data mapping.** One thread simulates one (or more) primary history.
Thread `t = blockIdx.x·blockDim.x + threadIdx.x` starts at history `t` and strides by
the total thread count (a **grid-stride loop**), so a fixed grid covers any
`n_histories`:

```
histories:   0    1    2    3    4    5    6    7   8   ...
             |    |    |    |    |    |    |    |   |
thread 0 ----+---------------------------+-----------  (0, stride, 2·stride, ...)
thread 1 ---------+---------------------------+------
thread 2 --------------+---------------------------+-
   ...                                     (grid-stride)
             \_______ each thread: seed RNG, run history, atomicAdd deposits
```

**Launch configuration.** `block = 256` threads (a multiple of the 32-lane warp;
enough warps to hide latency). `grid = 1024` blocks — a fixed, generous grid that
keeps the GPU busy while the grid-stride loop absorbs any history count. This
decouples occupancy from `n_histories`.

**Memory hierarchy.**
- *Registers / local memory:* each thread's RNG state, running weight, and a small
  `DepositList` scratch. No dynamic allocation on the device hot path.
- *Global memory:* the per-organ tally (`n_organs` 64-bit accumulators), updated with
  `atomicAdd`. It is tiny but written by many threads, so atomics — not shared memory
  — are the right tool here.
- *Constant memory:* not needed at this scope (parameters are passed by value in a
  small `SimParams` struct → registers). The catalog's full version would put
  energy-dependent **cross-section tables in constant memory** (broadcast to all
  threads); that is Exercise 3.

**No CUDA library.** Production codes use **cuRAND** for the RNG. We deliberately
hand-roll a splitmix64 counter-based RNG shared between host and device so the CPU
reference reproduces the identical histories and verification is *exact* (a cuRAND
stream would not be trivially reproducible on the host). That is the one "no black
box" trade we make on purpose — documented here rather than hidden.

**Divergence.** Different photons interact, scatter, and roulette differently, so
lanes in a warp take different branches and finish at different times — the classic MC
warp-divergence challenge. Production codes mitigate it by sorting/regenerating
particles by state; we keep it simple and name the effect.

## 5. Numerical considerations

**Precision.** All physics runs in **double precision** (FP64). The weights span many
orders of magnitude (1.0 down to `roulette_floor`), and `exp(-μ d)` for distant organs
is small; FP64 keeps those contributions meaningful.

**The determinism problem and its fix.** Weights are floating point, but a **float**
`atomicAdd` is *not* associative: the order in which thousands of threads add into the
same organ depends on the (nondeterministic) hardware scheduling, so a float tally
would differ run-to-run and would not equal the CPU's serial sum. We therefore convert
every deposit to **fixed-point integers** before accumulation:

```
fixed = trunc(weighted_energy · DOSE_FIXED_SCALE)   // DOSE_FIXED_SCALE = 1e9
```

Integer adds **commute**, so the GPU tally is deterministic *and* bit-identical to the
CPU tally. `1e9` keeps ~9 significant digits of the fraction; `2^63/1e9 ≈ 9.2e9` units
of headroom is far more than any tally here (same idiom as flagship `5.01`'s integer
energy quanta and `11.09`'s fixed-point centroids; PATTERNS.md §3).

**Race conditions.** Only the tally is shared, and only via `atomicAdd`. No other
shared state exists between threads. The RNG is per-thread and stateless across
histories (seeded from the index), so there is no cross-history correlation or race.

## 6. How we verify correctness

`src/reference_cpu.cpp` is an independent, obviously-correct serial implementation:
one `for` loop over histories, calling the **same** `simulate_history()` from
`stray_physics.h` and accumulating with plain `+=`. Because both sides (a) use the same
RNG seeded identically per history and (b) accumulate fixed-point integers, the two
per-organ tallies must be **bit-identical**.

**Tolerance = exact (`== 0` mismatches).** This is the strongest verification class in
PATTERNS.md §4, available precisely because the deposits are integers. `main.cu`
compares the two tallies element-by-element and reports the mismatch count;
`RESULT: PASS` means zero mismatches.

**Edge cases exercised by the sample:** organ 0 (in-field) receives the huge primary
dose while distant organs receive only the tiny stray floor (leakage-dominated),
confirming the out-of-field falloff; roulette fires for low-weight photons; forced
detection reaches the farthest organ. A second, *physical* sanity check the learner
can read off the output: `stray/target` ratios are ~1e-3…1e-5, matching the real-world
"3–4 orders of magnitude below target" statement.

## 7. Where this sits in the real world

This is a **reduced-scope teaching version**. Production stray-dose Monte Carlo differs
in almost every dimension:

| Aspect | This project | Production (TOPAS / GATE / EGSnrc / PHITS) |
|---|---|---|
| Geometry | 1-D organ stack | 3-D **ICRP-110 voxel phantom** (millions of labelled voxels) |
| Cross-sections | single constant `μ` | energy-dependent **NIST XCOM** tables per material |
| Physics | photon scatter/absorb only | full coupled photon–electron transport |
| Neutrons | distance-weighted surrogate | real **hadronic cascade** (INCL/BERT in Geant4) |
| Variance reduction | survival biasing, roulette, forced detection | + geometry importance, splitting with **particle-stack forking** on the GPU |
| RNG | shared splitmix64 (for exact verify) | **cuRAND** correlated streams |
| Risk model | illustrative LNT coefficients | full **BEIR-VII** age/sex-specific risk with DDREF |

The catalog's "one thread per particle with a nested interaction loop, cross-section
tables in constant memory, and splitting via a per-thread particle stack" is the full
GPU design; our version keeps the *shape* (independent per-history threads, atomic
scoring, per-thread variance reduction) while dropping the 3-D geometry, coupled
electron transport, and hadronic physics that make the real problem research-grade.
The variance-reduction techniques you learn here (survival biasing, Russian roulette,
forced detection) are the exact ones EGSnrc/TOPAS use — that transfer of intuition is
the point.

---

## References

- **BEIR VII Phase 2** (National Research Council, 2006) — the lifetime-risk model our
  §2 risk convolution abstracts. Read for the LNT assumption and organ coefficients.
- **ICRP Publication 110** — reference adult voxel phantoms; the real geometry.
  <https://www.icrp.org/publication.asp?id=ICRP%20Publication%20110>
- **NIST XCOM** — photon cross-section database; the real energy-dependent `μ`.
  <https://www.nist.gov/pml/xcom-photon-cross-sections>
- **EGSnrc** (<https://github.com/nrc-cnrc/EGSnrc>) — the canonical source for
  survival biasing, Russian roulette, and forced-detection idioms we reimplement.
- **TOPAS** (<https://github.com/OpenTOPAS/OpenTOPAS>) and **GATE 10**
  (<https://github.com/OpenGATE/opengate>) — Geant4-based full hadronic transport and
  out-of-field scoring; study their scoring/geometry layering.
- **PHITS** (<https://phits.jaea.go.jp/>) — hadronic + neutron transport in radiation
  protection; context for the neutron channel we surrogate.
- Flagship **5.01** (Monte Carlo dose) in this repo — the pattern exemplar: per-thread
  RNG, integer scoring, exact CPU/GPU agreement.
