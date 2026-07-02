# THEORY — 6.25 Liver & Kidney Perfusion Modeling

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See
> [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use. All data are synthetic._

---

## 1. The science

The liver and the kidney are both built from **repeating functional units** that
process blood in parallel:

- The **liver lobule** is a hexagonal prism, roughly 1 mm across. Blood enters at
  the corners (the **portal triads**, called the **periportal** region, *zone 1*),
  seeps through thousands of leaky capillaries called **sinusoids** lined with
  **hepatocytes**, and drains out the middle at the **central vein** (the
  **centrilobular** region, *zone 3*). As blood streams past, hepatocytes extract
  and metabolize drugs, toxins, and metabolites.
- The **kidney nephron** is an analogous tube: blood is filtered, then solutes are
  reabsorbed and secreted along a sequence of tubular segments, with a
  **countercurrent** arrangement in the medulla that concentrates urine.

Two features make this more than a stirred tank:

1. **Metabolic zonation.** Enzyme expression is *not* uniform along the sinusoid.
   Oxygen and hormone gradients drive periportal cells to favor some pathways and
   centrilobular cells others (visible in Human Protein Atlas liver data). So the
   local clearance capacity `Vmax` varies with position `x`.
2. **Heterogeneous perfusion.** Different sinusoids carry blood at different
   velocities, so they clear the drug by different amounts. The *organ's* behavior
   is the population statistic over all of them.

The real-world question this answers: **given a drug's kinetics and the organ's
zonation, what fraction of the drug is cleared in one pass, and how does that
depend on blood flow?** That single-pass **extraction ratio** is the microscopic
basis of hepatic/renal **clearance** — the quantity PBPK models and organ-on-chip
"digital twins" ultimately need. Simulating millions of segments is exactly a
"virtual pharmacotoxicology" workload, and it is embarrassingly parallel.

We build a clearly-labeled **reduced-scope teaching model**: a single lobule as an
ensemble of independent 1-D sinusoids with zonal Michaelis-Menten clearance. §7
describes the full multi-physics version.

## 2. The math

Model one sinusoid as a 1-D **plug-flow** tube of length `L` (mm), with blood
moving at velocity `v` (mm/s). Let `C(x)` be the drug concentration (µM) at
position `x ∈ [0, L]`, with `x = 0` periportal (inlet) and `x = L` centrilobular
(outlet). At **steady state**, convection of the drug downstream balances its
consumption by the wall enzymes:

```
    v · dC/dx  =  −R(C, x)                         (convection = −reaction)
    R(C, x)    =  Vmax(x) · C / (Km + C)           (Michaelis-Menten kinetics)
    Vmax(x)    =  Vmax_pp + (Vmax_cl − Vmax_pp)·(x/L)   (linear zonation)
```

Symbols (all in a consistent unit system so `v·dC/dx` [µM/s] balances `R` [µM/s]):

| Symbol | Meaning | Units |
|---|---|---|
| `C(x)` | drug concentration at position `x` | µM |
| `v` | blood velocity in this sinusoid | mm/s |
| `L` | sinusoid length | mm |
| `Km` | Michaelis constant (half-saturation conc.) | µM |
| `Vmax(x)` | local maximal clearance rate | µM/s |
| `Vmax_pp`, `Vmax_cl` | periportal / centrilobular capacity | µM/s |

- **Input:** the inlet condition `C(0) = C_in`, the constants above, and the
  velocity `v`.
- **Output per sinusoid:** the outlet concentration `C_out = C(L)` and the
  **extraction ratio** `E = (C_in − C_out) / C_in ∈ [0, 1]`.
- **Objective for the lobule:** the distribution of `E` across `nsin` sinusoids
  whose velocities span `[v_lo, v_hi]` — in particular the mean `E`.

**Two regimes of the Michaelis-Menten term.** When `C ≪ Km` the reaction is
*first-order*: `R ≈ (Vmax/Km)·C` (unsaturated). When `C ≫ Km` it *saturates* at
`Vmax` (zero-order, enzymes maxed out). The saturation is what makes the ODE
**nonlinear**, so in general there is no closed-form `C(x)` and we integrate
numerically. (In the first-order limit there *is* a closed form — we use it in §6
to validate the numerics.)

## 3. The algorithm

For **one** sinusoid we integrate the ODE `dC/dx = −R(C,x)/v` from `x=0` to `x=L`
with classical **4th-order Runge-Kutta (RK4)** in *space*. RK4 evaluates the slope
at four sub-points per step and combines them with weights `1/6, 2/6, 2/6, 1/6`;
its truncation error is `O(h⁴)`, so even a coarse grid (`nseg = 200`) resolves the
smooth washout profile. We march in the fractional coordinate `ξ = x/L ∈ [0,1]`
with step `h = 1/nseg`, so the physical increment is `ΔC = (dC/dx)·(L·h)`.

- Cost per sinusoid: `nseg` steps × 4 slope evaluations = **`O(nseg)`** work,
  `O(1)` memory (a handful of `double`s in registers).
- The **lobule** is `nsin` such solves. Serially that is `O(nsin · nseg)` and each
  solve is completely **independent** of the others — the defining property that
  makes this the *ensemble-ODE* pattern (PATTERNS.md §1): the parameter that
  differs across members is the inlet velocity `v`.

Arithmetic intensity is high (many flops per byte moved: each thread reads a tiny
config and writes two `double`s), so this is **compute-bound**, not
bandwidth-bound — a good fit for the GPU.

## 4. The GPU mapping

The mapping is the eighth flagship pattern, **ensemble ODE integration** (`9.02`,
`13.02`): **one thread integrates one sinusoid**.

- **Thread-to-data map:** `idx = blockIdx.x * blockDim.x + threadIdx.x` owns
  sinusoid `idx`; it derives its velocity `v = v_lo + idx/(nsin−1)·(v_hi−v_lo)`
  from the `LobuleConfig` passed **by value** (so there is *no per-member input
  array to copy* — the only device buffer is the output).
- **Launch config:** `block = 128` threads, `grid = ceil(nsin/128)`. 128 is a warp
  multiple that keeps register pressure low for a register-heavy per-thread
  integrator while giving the scheduler enough warps to hide latency; a guard
  `if (idx >= nsin) return;` handles the ragged last block.
- **Memory hierarchy:** the entire RK4 march lives in **registers / local memory**
  — no shared memory, no atomics, because members never interact. The only global
  write is one `SinusoidResult` (two `double`s) per thread. This is *embarrassing
  parallelism*.
- **Divergence:** minimal — every thread runs the same `nseg` steps; only the
  `C < 0` physical-floor branch can differ, and it rarely fires.

```
lobule (LobuleConfig, passed by value)
   |
   |   grid = ceil(nsin / 128) blocks
   v
[ block 0 ][ block 1 ] ... [ block G-1 ]      each block = 128 threads
    | | |      | | |            | | |
    thread idx  ->  v(idx)  ->  RK4 march C(0..L) in registers  ->  out[idx]
   (sinusoid 0)                (nseg steps, 4 slopes each)     (C_out, E)
```

**No CUDA library is used here.** The catalog mentions cuSPARSE for a *coupled*
lobular network linear system; our teaching model keeps sinusoids independent, so a
hand-written kernel is both sufficient and more transparent (no black box). §7
explains where cuSPARSE would enter for the full networked model.

**The "hierarchical parallelism" the catalog names** (blocks per lobule, threads
per segment) is the natural extension: to simulate *many* lobules at once, make
`blockIdx.y` the lobule index and keep `threadIdx.x` the sinusoid/segment — the
current kernel is the inner level of that hierarchy.

## 5. Numerical considerations

- **Precision: FP64 (double).** Extraction ratios and the exponential washout are
  sensitive to accumulated round-off over `nseg` steps; `double` keeps the CPU and
  GPU in lock-step and makes the analytic cross-check meaningful. FP32 would be
  faster but would blur the GPU/CPU agreement below our tolerance.
- **No atomics, no reductions on the device.** Each thread writes its own output
  slot, so there is **no floating-point summation-order problem** (PATTERNS.md §3).
  The lobule-mean is computed on the host, once, in a fixed order — deterministic.
- **Determinism.** Because the per-element physics lives in one shared
  `__host__ __device__` header (`perfusion.h`, PATTERNS.md §2), the CPU loop and
  the GPU kernel execute *byte-identical* arithmetic. Stdout is therefore identical
  every run; timing (which varies) is confined to stderr.
- **Stability.** RK4 is stable for this smooth, monotonically-decaying profile at
  `nseg = 200`; the concentration is clamped at `0` to prevent a tiny negative
  round-off from producing an unphysical negative rate.

## 6. How we verify correctness

Two independent checks (PATTERNS.md §4):

1. **GPU vs CPU, to round-off.** `reference_cpu.cpp` integrates every sinusoid
   serially using the *same* `integrate_sinusoid()` from `perfusion.h`. `main.cu`
   compares per-sinusoid `C_out` and `E`; the worst difference observed is
   `~2.2e-16` (one ULP), far under the documented tolerance `1e-9`. Agreement
   between an obviously-correct serial loop and the parallel kernel is strong
   evidence the GPU code is right.
2. **Against an analytic limit — validating the *science*, not just CPU==GPU.** In
   the first-order regime `C ≪ Km`, `R = (Vmax(x)/Km)·C`, and integrating the
   linear ODE with the linear zonation gives a closed form:

   ```
   C_out = C_in · exp( −(Vmax_avg / Km) · L / v ),   Vmax_avg = (Vmax_pp + Vmax_cl)/2
   E     = 1 − C_out/C_in
   ```

   The sample is engineered with `C_in = 1 µM ≪ Km = 50 µM`, so the numerical mean
   extraction (`0.0930`) matches the analytic mean (`0.0946`) to within `3.4e-3` —
   well inside the documented **physical** tolerance `1e-2`. The small residual is
   exactly the (real, teachable) nonlinearity that the analytic first-order formula
   ignores; we verify to a physically-negligible tolerance and *say so* rather than
   pretend the numbers are identical.

Edge cases handled: ragged last block (`idx ≥ nsin` guard), `C_in = 0`
(extraction defined as 0), and non-positive parameters (loader throws).

## 7. Where this sits in the real world

Production organ-perfusion / PBPK tools do much more than this single-lobule toy:

- **Open Systems Pharmacology (PK-Sim/MoBi)** and **mrgsolve** solve *whole-body*
  PBPK: the liver and kidney are compartments in a system of coupled ODEs across
  organs, with blood flows, protein binding, and multiple metabolites. Our
  single-pass extraction ratio is the microscopic quantity those models summarize
  into an organ clearance term.
- **The full lobular model is a coupled network, not independent tubes.** Sinusoids
  share nodes (portal inflow, central-vein outflow), so mass conservation couples
  their flows into a **sparse linear system** `A·p = b` for the pressures — this is
  where the catalog's **cuSPARSE** enters (assemble the network Laplacian, solve
  with a sparse iterative solver, then advect the drug). Our model omits the
  coupling to stay legible; adding it is the natural next project.
- **Microvessel flow** (SimVascular's `svFSI` for the portal tree, **HemeLB**'s
  lattice-Boltzmann for sinusoidal flow) resolves the actual 3-D geometry and
  computes `v(x)` from first principles instead of prescribing a sweep. Coupling an
  LBM velocity field (see flagship `6.04`) into this transport kernel is a powerful
  combination.
- **The nephron** adds *filtration–reabsorption–secretion* along the tubule and the
  medullary **countercurrent multiplier**, which our liver-sinusoid model does not
  represent; the same "one thread per segment" mapping applies.

The teaching value here is the clean core: a nonlinear reaction-transport ODE,
solved per functional unit, mapped to the GPU as an ensemble, and validated against
an analytic limit.

---

## References

- **Michaelis-Menten kinetics & hepatic clearance** — any pharmacokinetics text
  (e.g. Rowland & Tozer, *Clinical Pharmacokinetics*): the origin of `Vmax`, `Km`,
  and the extraction-ratio → clearance link.
- **Metabolic zonation** — Human Protein Atlas liver section
  (https://www.proteinatlas.org): the per-enzyme periportal/centrilobular gradient
  that motivates `Vmax(x)`.
- **Open Systems Pharmacology Suite** (https://github.com/Open-Systems-Pharmacology)
  — how a production organ-level PBPK model is structured; study its liver/kidney
  clearance sub-models.
- **mrgsolve** (https://github.com/metrumresearchgroup/mrgsolve) — ODE-based organ
  pharmacokinetics; a clean example of solving many ODE members.
- **SimVascular `svFSI`** (https://github.com/SimVascular/svFSI) — vascular-tree
  flow; where the prescribed velocity sweep would come from in a real model.
- **HemeLB** (https://github.com/hemelb-codes/hemelb) — lattice-Boltzmann
  microvessel flow for sinusoidal geometry; pairs with flagship `6.04`.
- **RK4** — any numerical-methods reference; here applied in *space* to a
  steady-state transport ODE.
