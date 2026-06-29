# THEORY — 5.01 Monte Carlo Dose Calculation (simplified slab)

> For a reader who knows C++ but is new to CUDA and to radiation transport.
> See [README.md](README.md) for the tour and build. _Educational only; this is a
> deliberately simplified model, not a dose engine._

## 1. The science

When radiation passes through tissue it deposits energy — the **dose** — which is
what radiotherapy aims to control (enough in the tumour, little in healthy
tissue). Photons travel in straight lines until they interact (photoelectric
absorption, Compton scatter, pair production), transferring energy to electrons
that then deposit dose locally. There is no closed-form solution for a realistic
patient, so we **simulate**: follow many individual particle "histories" drawn
from the interaction probabilities and average. That is **Monte Carlo** transport,
the gold standard for dose accuracy.

## 2. The math

A photon's distance to its next interaction is **exponentially distributed**: the
probability of travelling a distance `s` without interacting is `exp(-μs)`, where
`μ` is the linear attenuation coefficient. We sample a free path by inverting the
CDF:

```
s = -ln(ξ) / μ ,   ξ ~ Uniform(0,1]
```

At an interaction the photon is absorbed with probability `p_abs = μ_a/μ_t` (here
all remaining energy is deposited) or scatters with probability `1 - p_abs`. The
**depth-dose** `D(z)` is the expected energy deposited per depth bin, estimated as

```
D(bin) ≈ (1/N) Σ_histories  (energy deposited in bin)
```

with statistical uncertainty falling as `1/sqrt(N)`.

## 3. The algorithm

```
for each of N photon histories:                  # INDEPENDENT -> parallel
    z = 0 ; energy = E0
    loop:
        s = -ln(xi)/mu ;  z += s
        if z >= L: break                          # escaped the slab
        if xi2 < p_abs:  deposit energy at bin(z); break        # absorbed
        else:            deposit a packet at bin(z); energy -= packet   # scatter, continue
```

**Complexity.** Cost is `O(N · steps_per_history)`. Accuracy improves only as
`1/sqrt(N)`, so halving the noise costs 4× the work — which is exactly why MC is
compute-bound and why the GPU matters.

## 4. The GPU mapping

**Decomposition.** One thread per history, grid-stride over `N` so a fixed grid
(here 1024×256 threads) covers any `N`. Each thread:

- **seeds its own RNG stream** from its history index (`rng_seed(seed, i)` in
  `mc_physics.h`), so streams are independent yet reproducible;
- runs the transport loop entirely in **registers/local memory** (no shared mem);
- **`atomicAdd`s** its integer deposits into the global depth-dose tally.

**Why a shared, deterministic RNG (not cuRAND).** Production GPU MC uses cuRAND.
We instead use a small splitmix64 counter-based RNG defined `__host__ __device__`
in one header, so the **CPU reference runs the identical histories** and the two
dose tallies can be compared **exactly**. The `RNG_HD` macro makes the same code
compile under nvcc (device) and the host compiler. (Exercise 1 swaps in cuRAND
and switches to statistical verification.)

**Why integer energy quanta.** Many threads `atomicAdd` into the same few bins.
Floating-point addition is **not associative**, so a float tally would depend on
the (non-deterministic) order of atomic operations — different every run, and
different from the CPU. By depositing **integer** quanta, the atomic adds
**commute**: the tally is exact, reproducible, and equal to the CPU's. This is a
deliberate, important design choice for a verifiable teaching demo.

**Divergence — the headline MC challenge.** Photons take different numbers of
steps and different branches, so threads in a warp finish at different times and
execute different code paths (`if absorbed … else …`). This **warp divergence**
is the main inefficiency in GPU MC; production codes mitigate it by **sorting or
compacting particles by state/material** between transport steps so a warp does
uniform work. Memory-side, real codes also keep the CT geometry and cross-section
tables in **constant/texture memory**.

## 5. Numerical considerations

- **Determinism:** guaranteed here by integer quanta + reproducible per-history
  RNG (Section 4). The float-associativity pitfall is the key lesson.
- **Statistics:** the result is an estimate; uncertainty `∝ 1/sqrt(N)`. Variance
  reduction (Russian roulette, splitting) lowers it per unit time.
- **`-ln(ξ)`:** we draw `ξ ∈ (0,1]` (as `1 - U[0,1)`) so `ln` never sees 0.

## 6. How we verify correctness

`main.cu` runs `dose_cpu` and `dose_gpu` on the same parameters and compares the
two integer histograms bin-by-bin; they are **identical** (`0 mismatches`)
because of the shared RNG and integer scoring. The depth-dose also makes physical
sense for this absorption model: it is highest near the entrance and falls off
with depth as attenuation removes photons.

## 7. Where this sits in the real world

A clinical engine (EGSnrc, GATE/Geant4, gDPM, FRED, MC-GPU) replaces every
simplification here: **sampled interaction physics** (Klein-Nishina Compton
angles, photoelectric, pair production), **condensed-history electron transport**
(which creates the dose **buildup region** and `d_max` that our absorption model
lacks), **3-D patient CT geometry** with material-dependent cross sections, and
**variance reduction**. The structure you learn here — independent histories,
per-thread RNG, atomic scoring, divergence management — is exactly the skeleton
those codes are built on.

## References

- Bielajew, *Fundamentals of the Monte Carlo Method for Radiation Transport* — the standard text.
- Jia et al., *GPU-based fast Monte Carlo dose calculation* (gDPM) — GPU MC for therapy.
- Badal & Badano, **MC-GPU** — open CUDA photon MC.
- NVIDIA cuRAND documentation — production GPU RNG; CUDA Programming Guide — atomics.
