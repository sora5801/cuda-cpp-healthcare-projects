# THEORY — 1.18 Fragment / Combinatorial Library Enumeration

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

Modern drug discovery rarely tests molecules one at a time. Instead, chemists
work with **building blocks** — small, commercially available fragments
("synthons") that can be snapped together by reliable reactions. A
**combinatorial library** is every molecule you can make by choosing one building
block for each *reactant slot* of a reaction. A classic example is the
**Ugi four-component reaction**, but a three-component scheme (amine + aldehyde-
or-acid + isocyanide-style cap) already illustrates the idea.

The power — and the problem — is **multiplicative scale**. With `s₀` choices in
slot 0, `s₁` in slot 1, and `s₂` in slot 2, the library has

```
N = s₀ × s₁ × s₂
```

products. A few hundred building blocks per slot gives **tens of millions** of
products; the commercial **Enamine REAL Space** library reaches **>6 billion**
make-on-demand compounds from ~160 reactions and >130k building blocks. You
cannot synthesize and assay billions of molecules, so you **enumerate them in
silico** and triage: keep only the products that look drug-like, are chemically
diverse, and are worth making.

The first triage gate is **drug-likeness**. Lipinski's "Rule of Five" (1997)
captured an empirical observation: orally bioavailable drugs tend to have
molecular weight ≤ 500, octanol–water log-partition (logP) ≤ 5, ≤ 5 hydrogen-bond
donors, and ≤ 10 hydrogen-bond acceptors. Veber (2002) added that low polar
surface area (TPSA ≤ 140 Å²) and few rotatable bonds further predict good oral
absorption. A product that violates these is unlikely to become an oral drug, so
we can discard it cheaply before ever assembling or docking it.

This project models exactly that gate: **enumerate every product, estimate its
descriptors, and count how many pass Lipinski + Veber.**

## 2. The math

**Inputs.** For each slot `k ∈ {0,1,2}` we are given `sₖ` building blocks. Block
`j` of slot `k` carries a descriptor-contribution vector

```
cₖⱼ = ( MW, cLogP, TPSA, HBD, HBA )   ∈ ℝ⁵
```

with units g/mol, dimensionless, Å², count, count.

**Product indexing.** Number the products `p = 0, 1, …, N−1`. There is a bijection
between `p` and a tuple of per-slot indices `(i₀, i₁, i₂)` via a **mixed-radix**
(odometer) encoding with radices `sₖ`:

```
p = i₀ + s₀·( i₁ + s₁·i₂ )          (slot 0 is the least-significant digit)
```

so the decode is the familiar `/`, `%` peeling:

```
i₀ = p mod s₀ ,   i₁ = (p ÷ s₀) mod s₁ ,   i₂ = (p ÷ (s₀ s₁)) mod s₂ .
```

**Product descriptors (additivity).** We approximate each product's descriptor
vector as the **sum** of its chosen blocks' contributions:

```
d(p) = c₀,i₀ + c₁,i₁ + c₂,i₂       (elementwise in ℝ⁵).
```

This is the **group-contribution** assumption. It is *exact* for additive
descriptors (HBD/HBA counts) and a good approximation for Crippen **cLogP** (a sum
of per-atom contributions; Wildman & Crippen 1999) and Ertl **TPSA** (a sum of
per-fragment polar contributions; Ertl 2000). Molecular weight is additive except
for the small mass lost when a bond forms — we fold an approximate correction into
the block values (see §7 for the honest caveat).

**The filter.** A product passes iff *all* thresholds hold:

```
pass(p) = [ MW(p) ≤ 500 ] ∧ [ cLogP(p) ≤ 5 ] ∧ [ HBD(p) ≤ 5 ]
                          ∧ [ HBA(p) ≤ 10 ] ∧ [ TPSA(p) ≤ 140 ].
```

**Outputs.** The deterministic quantities we report:

```
n_pass  = Σ_p pass(p)                       (an integer count)
S_MW    = Σ_{p : pass(p)} MW(p)             (sum of MW over passers, g/mol)
first_K = the K smallest p with pass(p)=1   (a canonical preview list)
```

## 3. The algorithm

```
for p = 0 .. N-1:                       # one independent job per product
    (i0,i1,i2) = decode(p)              # mixed-radix, O(N_SLOTS)
    d = c[0][i0] + c[1][i1] + c[2][i2]  # additive descriptors, O(N_SLOTS*N_DESC)
    if passes(d):                       # 5 comparisons
        n_pass += 1
        S_MW   += MW(d)
        record p if among first K
```

**Complexity.** Time `Θ(N · N_SLOTS · N_DESC)` = `Θ(N)` since the inner factors
are tiny constants (3 and 5); extra space `O(K)`. The serial **depth** is `Θ(N)`
(one product after another); the parallel **work** is the same `Θ(N)` but the
**depth collapses to `O(1)` per product plus an `O(log N)` reduction** because the
products are independent. That independence is the whole reason the GPU helps.

**Arithmetic intensity.** Each product does ~15 additions + 5 comparisons and
reads three 5-vectors from a *tiny* shared table (a few dozen rows total). The
table fits in cache and is reused by every thread, so the kernel is **compute-/
launch-bound at small N and bandwidth-light** — ideal for constant memory.

## 4. The GPU mapping

**Thread-to-data map.** One logical thread owns one product. Thread
`(blockIdx.x, threadIdx.x)` starts at

```
p = blockIdx.x · blockDim.x + threadIdx.x
```

and, via a **grid-stride loop**, also handles `p + stride, p + 2·stride, …` where
`stride = blockDim.x · gridDim.x`. So a *fixed-size* grid covers any `N` — no
relaunch needed when the library grows.

**Launch configuration.** `blockDim.x = 256` (8 warps: a multiple of the 32-lane
warp, enough resident warps to hide latency, good occupancy on sm_75…sm_89).
`gridDim.x = ceil(N / 256)` capped at 4096 blocks; the grid-stride loop sweeps the
remainder.

**Memory hierarchy — and why.**

- **Constant memory** holds the synthon descriptor tables (`c_desc`, `c_off`,
  `c_sizes`). Every thread reads these; none writes them; they are tiny. Constant
  memory's hardware cache **broadcasts one address to a whole warp in a single
  transaction**, so the table is effectively free to read. (Capacity is 64 KB; a
  `double` row is 40 bytes, so we cap at 256 rows = 10 KB — far more than a
  teaching catalog needs. A 130k-block production catalog would instead stream
  the tables from **global memory**, coalesced — see §7.)
- **Registers** hold the decoded indices, the 5-element descriptor accumulator,
  and the radices — the entire per-product state is register-resident, so the hot
  loop touches no global memory except the one flag write.
- **Global memory** receives only a dense per-product pass-**flag** byte array
  (`d_flag`) and the two scalar accumulators.

**Reduction.** Counting passers and summing MW are *reductions*. We use
`atomicAdd` on `unsigned long long` device scalars. Integer atomics are
**associative regardless of thread order**, so the totals are deterministic (§5).
The `first_K` preview is recovered by a host scan of the flag array in ascending
`p` order, which is canonical and matches the CPU exactly.

```
  flat product index space  [0 .................................. N-1]
                              |     |     |          |
  grid-stride threads:      t0    t1    t2   ...    t_{stride-1}
   (each thread strides by `stride`, covering many products)
                              |
                         decode(p) --> (i0,i1,i2)
                              |
           c_desc[c_off[k]+ik] (constant cache, broadcast)
                              |
                  d = sum of 3 rows  (registers)
                              |
                 passes(d)?  --yes--> atomicAdd(count,1)
                              |        atomicAdd(sum_mw, round(MW*1000))
                              +------> d_flag[p] = pass   (global, dense)
```

**Why not write SMILES strings?** The catalog mentions "GPU-parallel SMARTS
matching over SMILES bytes." That is a *string*-processing kernel (variable-length
output, divergent control flow) and is much harder. The additivity shortcut lets
us stay in fixed-size numeric registers — the right first lesson. SMARTS is §7.

## 5. Numerical considerations

- **Precision.** Descriptors are accumulated in **FP64** (`double`). The values
  are small (MW ≲ 1000), so FP32 would suffice, but FP64 keeps the additive sum
  exact to ~15 digits and makes the CPU↔GPU comparison trivially exact. There is
  no FMA in the accumulation (plain `+=`), so host and device perform the *same*
  IEEE operations in the *same* order → identical bits.
- **Determinism of the reduction.** A naïve `float atomicAdd` reduction is **not
  reproducible**: floating-point addition is not associative, and the order in
  which thousands of threads add is nondeterministic, so the low bits of the sum
  vary run to run (PATTERNS.md §3 rule 2). We avoid this by accumulating MW in
  **fixed point**: round `MW × 1000` to an integer (milli-g/mol) and sum with an
  **integer** atomic. Integer addition *is* associative, so the total is
  bit-identical every run and equals the CPU's. The pass count is naturally
  integer. Both `round`s use the same `llround`, so the quantization matches.
- **Comparisons.** The five filter clauses are pure `double ≤ double`, identical
  on host and device — no tolerance needed.
- **Overflow.** `N` is computed and indexed as `int64_t` (a few hundred blocks per
  slot exceeds 2³¹ products); the milli-g/mol sum stays far inside `int64`.

## 6. How we verify correctness

`src/reference_cpu.cpp::enumerate_cpu` is an independent, single-threaded
implementation: one readable `for p` loop, no parallelism, calling the **same**
`product_core.h` math the kernel calls. `main.cu` runs both and asserts the GPU
and CPU agree on **all three** deterministic outputs.

**Tolerance: exact (`== 0`).** This is the strongest tolerance and the correct one
here (PATTERNS.md §4): the count is an integer, the MW sum is a fixed-point
integer, and the `first_K` list is the same canonical prefix — **no floating-point
rounding enters the comparison**. Any mismatch is a real bug, not numeric drift.

**Why this is convincing.** The CPU reference and the GPU kernel were written
differently (serial loop vs. grid-stride atomics) but share only the per-product
formula. If a transcription error existed in either the decode, the accumulation,
or the filter, the two would disagree on at least one product and the exact-match
check would fail. Edge cases covered by the loader: wrong `N_SLOTS`, malformed
rows, and a catalog too large for constant memory all throw loudly rather than
producing a silent wrong answer.

**A second, science-level check.** The committed sample is engineered so that the
pass fraction (**130/216 = 60.2 %**) and summed MW (**47136.000 g/mol**) are known
ahead of time from an independent Python model (`scripts/make_synthetic.py`'s
design) — so we are validating against an external expectation, not just CPU==GPU.

## 7. Where this sits in the real world

This is a **reduced-scope teaching version** (CLAUDE.md §13). Production virtual-
library tools do several things we deliberately omit:

- **Real product construction via reaction SMARTS.** RDKit's `ChemicalReaction`
  (`AllChem.ReactionFromSmarts`) matches each building block's reactive group and
  *forms the actual bond*, producing a real product molecule (and a real SMILES).
  Only then are descriptors computed on the assembled molecule — capturing the
  mass lost at the bond and any non-additive electronic effects. We assume every
  slot-combination reacts and that descriptors are additive; both are
  approximations that a real pipeline corrects.
- **Property accuracy.** RDKit computes MW/cLogP/TPSA/HBD/HBA on the true molecule
  (`Descriptors.MolWt`, `Crippen.MolLogP`, `rdMolDescriptors.CalcTPSA`, …). Our
  additive estimates are a *fast pre-filter*; production code re-evaluates
  survivors exactly.
- **Scale without enumeration.** At 6×10⁹ products, even one thread per product is
  expensive. **SyntheMol / V-Synthes** instead *navigate* the synthesis graph with
  GPU ML (Monte-Carlo tree search over synthons), scoring promising branches
  without materializing the whole library — a fundamentally different algorithm.
- **Diversity filtering.** After drug-likeness, real workflows cluster survivors
  (GPU k-means on Morgan fingerprints — cf. flagship `11.09`; or shape screening
  with OpenEye FastROCS) and keep diverse representatives. **Thrust**
  (`copy_if`/`count_if`) compacts the passing set; **cuML** does the clustering.
- **Larger libraries.** Beyond ~256 synthons our constant-memory table overflows;
  the production path streams the tables from global memory with coalesced loads,
  trading the broadcast cache for capacity.

The CUDA *pattern* you learn here — independent jobs, constant-memory lookup
tables, deterministic integer-atomic reduction — is exactly what the property-and-
filter stage of those production pipelines uses; we have simply isolated it.

---

## References

- **C. A. Lipinski et al.**, "Experimental and computational approaches to estimate
  solubility and permeability…", *Adv. Drug Deliv. Rev.* 23 (1997) — the Rule of Five.
- **D. F. Veber et al.**, "Molecular properties that influence the oral
  bioavailability of drug candidates", *J. Med. Chem.* 45 (2002) — TPSA + rotatable bonds.
- **S. A. Wildman & G. M. Crippen**, "Prediction of physicochemical parameters by
  atomic contributions", *J. Chem. Inf. Comput. Sci.* 39 (1999) — additive cLogP.
- **P. Ertl et al.**, "Fast calculation of molecular polar surface area…",
  *J. Med. Chem.* 43 (2000) — additive TPSA.
- **RDKit** (<https://github.com/rdkit/rdkit>) — reaction SMARTS, library
  enumeration, and the reference descriptor implementations.
- **SyntheMol** (<https://github.com/swansonk14/SyntheMol>) — ML navigation of
  combinatorial space without explicit enumeration.
- **ASKCOS** (<https://github.com/ASKCOS/ASKCOS>) — reaction-condition prediction;
  how synthesizability is judged in practice.
- **NVIDIA CUDA C++ Programming Guide**, "Constant Memory" and "Atomic Functions" —
  the two hardware features this project teaches.
