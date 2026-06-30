# THEORY — 2.23 Protein-Ligand Interaction Energy Decomposition

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use. All numbers here come from a
> **synthetic** system (see `data/README.md`)._

---

## 1. The science

When a drug molecule (a **ligand**) binds in a protein's pocket, the binding free
energy is not spread evenly over the protein — a handful of **residues** do most
of the work. A positively charged arginine reaching across to a negatively charged
group on the ligand forms a **salt bridge** worth many kcal/mol; a fat hydrophobic
leucine packing snugly against an aromatic ring contributes through shape
(**van der Waals**) complementarity; a distant surface residue contributes almost
nothing.

**Per-residue energy decomposition** answers: *which residues matter, and how?*
For each residue it reports a number (kcal/mol, negative = favourable) split into
physical **components** — electrostatics vs. van der Waals vs. solvation. Medicinal
chemists read this to:

- **find hot spots** to preserve or strengthen during lead optimization;
- **explain selectivity** — why a ligand binds kinase A but not its close relative
  B (often a single residue difference);
- **anticipate resistance** — in oncology, tumours mutate a hot-spot residue to
  escape a kinase inhibitor; if the decomposition says residue *X* carries the
  binding, *X* is exactly where a resistance mutation is likely to appear (the
  "kinase resistance mutation mapping" the catalog calls out).

The standard practical method is **MM-GBSA** (Molecular Mechanics, Generalized
Born, Surface Area): run molecular dynamics to get an ensemble of snapshots
(**frames**), and on each frame evaluate a molecular-mechanics energy with an
**implicit** (continuum) solvent model instead of explicit water. Averaging over
frames gives a smooth per-residue estimate. This project implements the per-residue
energy evaluation — the GPU-parallel inner loop — on a **synthetic** system, as a
deliberately **reduced-scope teaching version** (full caveats in §7).

## 2. The math

We have **M** protein residues (each modelled as one bead — see §7), **L** ligand
atoms, and **F** trajectory frames. Residue *m* and ligand atom *a* in frame *f*
are separated by distance `r = |x_m^f − x_a^f|` (Angstrom). Each carries a partial
charge `q` (in units of the elementary charge `e`), Lennard-Jones parameters
(ε, Rmin/2), and an effective Born radius `R` (how shielded from solvent it is).

For one residue–ligand-atom **pair** we evaluate three energy terms.

**(1) Electrostatics (Coulomb):**
```
E_elec = K · q_m q_a / (ε_in · r)
```
`K = 332.0636 kcal·Å·mol⁻¹·e⁻²` is the Coulomb constant in molecular-mechanics
units; `ε_in ≈ 1` is the solute interior dielectric. Opposite charges → negative
(attractive) energy.

**(2) van der Waals (Lennard-Jones 12-6):**
```
E_vdw = ε_ma · [ (Rmin_ma / r)¹² − 2 (Rmin_ma / r)⁶ ]
```
The `+r⁻¹²` term is Pauli repulsion (atoms cannot overlap); the `−2r⁻⁶` term is
London dispersion (induced-dipole attraction). The minimum is at `r = Rmin_ma`,
depth `ε_ma`. Pair parameters use the **Lorentz–Berthelot** combining rules
`ε_ma = √(ε_m ε_a)` and `Rmin_ma = Rmin_m/2 + Rmin_a/2`.

**(3) Generalized-Born desolvation (implicit solvent, Still's formula):**
```
E_gb = −K · (1/ε_in − 1/ε_out) · q_m q_a / f_GB
f_GB = √( r² + R_m R_a · exp( −r² / (4 R_m R_a) ) )
```
`ε_out ≈ 80` is the water dielectric. `f_GB` smoothly interpolates between `r`
(atoms far apart) and `√(R_m R_a)` (atoms overlapping). This term is the
electrostatic cost of moving charge from high-dielectric water into the
low-dielectric protein interior — it **opposes** burying charge, which is why a
salt bridge's favourable `E_elec` is partly cancelled by an unfavourable `E_gb`.

**Per-residue result.** Sum a residue's three components over all ligand atoms and
average over frames:
```
E_component(m) = (1/F) · Σ_f Σ_a  E_component(m, a, f)      for component ∈ {elec, vdw, gb}
E_total(m)     = E_elec(m) + E_vdw(m) + E_gb(m)
```
A distance **cutoff** `r_c` (here 12 Å) zeroes far pairs, the standard MM cutoff
trick. The headline output is the vector `{E_elec, E_vdw, E_gb, E_total}` per
residue, plus the ranking of residues by `E_total` (most negative = hot spot).

## 3. The algorithm

```
for each residue m in 0..M-1:                 # the parallel axis
    elec = vdw = gb = 0
    for each frame f in 0..F-1:               # accumulate over the trajectory
        for each ligand atom a in 0..L-1:     # the inner pair loop
            r2 = |x_m^f - x_a^f|^2
            if r2 <= cutoff^2:
                add Coulomb + LJ + GB components for (m,a) into elec/vdw/gb
    store (elec/F, vdw/F, gb/F, sum)  for residue m
then sort residues by total ascending -> hot-spot ranking
```

**Complexity.** Exactly `M · F · L` pair evaluations, each O(1) (one sqrt, one
exp, a handful of multiplies). Serial cost is `Θ(M·F·L)`. The work is **embarrassingly
parallel along M**: residue *m*'s accumulation reads only its own coordinates and
the (shared, read-only) ligand — it never touches another residue's state. So the
parallel **depth** is `Θ(F·L)` (one residue's inner loops) and the parallel
**work** is the same `Θ(M·F·L)`. Arithmetic intensity is high (transcendental
sqrt/exp per pair over a few coordinate loads), so this is **compute-bound**, not
memory-bound — a good fit for the GPU once M·F·L is large.

## 4. The GPU mapping

**One thread per residue.** Thread `m = blockIdx.x·blockDim.x + threadIdx.x` owns
residue *m*. It loops over all F frames and L ligand atoms, accumulating its three
energy components in **registers**, and writes exactly one `PerResidueEnergy` at
the end. This mirrors the CPU's outer loop precisely (PATTERNS.md §1, "the same
work for many independent items" — the `1.12 / 12.01` family).

```
            grid  = ceil(M / 128) blocks
            block = 128 threads
  ┌──────────────── block 0 ───────────────┐   ┌──── block 1 ────┐
  │ t0   t1   t2  ...            t127        │   │ t0 ...          │
  │ res0 res1 res2 ...           res127      │   │ res128 ...      │
  └─────────────────────────────────────────┘   └─────────────────┘
        each thread: for f in F: for a in L: accumulate elec/vdw/gb
        no atomics, no shared memory — residues are independent
```

**Why no atomics / no shared memory.** Because outputs are per-residue and
independent, each thread privately accumulates and does a single global write.
That sidesteps the classic float-atomic nondeterminism (PATTERNS.md §3) entirely —
there is *nothing to reduce across threads*.

**Block size = 128 (not 256).** Each thread does a lot of double-precision work
with `sqrt` and `exp`, so register pressure per thread is high. A smaller block
keeps enough blocks resident to hide latency without spilling registers to local
memory. On a real protein (M in the hundreds) the grid is still many blocks, so
occupancy is fine.

**Memory hierarchy.**
- **Registers:** the running `elec/vdw/gb` sums and the residue's own coordinates
  (loaded once, reused for every ligand atom) — the hot path touches no global
  memory for the accumulators.
- **Global memory:** residue/ligand parameters and the flat coordinate arrays
  (`res_xyz [F·M·3]`, `lig_xyz [F·L·3]`, both frame-major row-major so one frame's
  atoms are contiguous). The ligand block is re-read by every residue thread, so
  the L1/L2 cache carries most of that traffic — a natural fit since L is small
  and shared across all threads.
- A production kernel would stage the (shared, hot) ligand into **shared memory**
  per block; we keep it in global+cache for teaching clarity (an exercise).

**Where cuBLAS would fit (the catalog's "cuBLAS for energy matrix accumulation").**
If you formed the full `M × L` pair-energy matrix per frame and wanted per-residue
sums, that row-reduction is a matrix–vector product `E · 1` — one `cublasDgemv`.
We deliberately fuse the reduction into the kernel instead (no `M·L` matrix is ever
materialised), which is faster and simpler here; §7 and the exercises show when the
matrix form (and cuBLAS) pays off. Naming the library call but explaining the
hand-rolled fused version keeps it from being a black box (CLAUDE.md §6.1.6).

## 5. Numerical considerations

- **Precision: FP64 (double) throughout.** Energies span a large dynamic range
  (a near-clash LJ term can be thousands of kcal/mol while a distant Coulomb term
  is a few); double precision keeps the component sums faithful. The synthetic
  geometry is tuned so no pair sits on the steep `r⁻¹²` wall.
- **Determinism: exact and reproducible.** Each thread sums frames and ligand
  atoms in **fixed index order**, identical to the CPU loop. There is no
  cross-thread reduction and no atomic, so there is no floating-point reordering.
  Re-running yields byte-identical stdout (the demo depends on this).
- **The one source of CPU↔GPU divergence** is the GPU's **fused multiply-add**
  (FMA): the device may contract `a*b + c` into a single rounding step inside the
  `sqrt`/`exp` argument chains, where the host compiler rounds twice. For these
  magnitudes that is at the `1e-12` level. We measure it: `max_abs_err` is
  typically ~`1e-15` kcal/mol.
- **Guarded edge cases:** the `r² > cutoff²` test (no sqrt for rejected pairs);
  neutral residues (`q = 0`) give exactly zero electrostatics and GB; the loader
  rejects malformed files with precise messages.

## 6. How we verify correctness

The GPU result is checked against an **independent serial CPU reference**
(`src/reference_cpu.cpp`, `decompose_cpu`). Crucially, both call the **same**
`__host__ __device__` physics functions in `src/mmgbsa.h` (PATTERNS.md §2): there
is exactly one copy of the Coulomb/LJ/GB math, compiled for both the host and the
device. So agreement tests the *parallelisation and memory plumbing*, not a second
re-derivation of the formula.

We compare every component of every residue with `max_abs_err` and require it
below **`1.0e-4 kcal/mol`**. That tolerance is honest about the FMA divergence
described in §5 (PATTERNS.md §4): the true error is ~`1e-15`, so `1e-4` is a wide
safety margin that is still far below any physically meaningful energy difference
(thermal noise `kT ≈ 0.6 kcal/mol`). A **second, stronger check** is built into the
synthetic data (PATTERNS.md §6): the system embeds a known answer — ARG41 must rank
as the #1 (electrostatic) hot spot and LEU88 as the #2 (van der Waals) hot spot —
and the demo output recovers exactly that ranking, validating the *science*, not
just CPU==GPU agreement.

## 7. Where this sits in the real world

This is a **reduced-scope teaching version**. Production per-residue MM-GBSA
(AMBER's `MMPBSA.py` with `idecomp`, or `gmx_MMPBSA` for GROMACS) differs in ways
worth knowing:

- **All-atom, not one-bead-per-residue.** Real decomposition sums over every atom
  of a residue (backbone + side chain), using full force-field parameters from an
  AMBER `prmtop` / GROMACS topology. We use one bead per residue so the example is
  small and readable; the kernel generalises directly (give each residue a
  variable atom count and an inner atom loop).
- **Real trajectories.** Frames come from a proper MD run (often itself
  GPU-accelerated — AMBER `pmemd.cuda`, OpenMM); we jitter a fixed pose to mimic an
  ensemble. The decomposition math on each frame is the same.
- **Better Born radii.** We treat Born radii as fixed inputs; real GB models
  (GB-OBC, GB-Neck2) compute *effective* Born radii from the instantaneous geometry
  each frame (an extra O(N²) pass), and add a non-polar **surface-area** (SA) term
  for the hydrophobic effect — the "SA" in MM-GBSA, omitted here.
- **Water bridges & entropy.** The catalog mentions water-mediated interactions and
  FEP component analysis; explicit-water bridge detection and the configurational
  entropy term (normal-mode or quasi-harmonic) are beyond this teaching scope.
- **The energy-matrix / cuBLAS form.** At very large M·L, forming the per-frame
  pair-energy matrix and reducing with `cublasDgemv` (or batched GEMM across
  frames) can win by using highly-tuned BLAS; our fused kernel avoids
  materialising the matrix and is the right call at teaching scale (§4).

Despite the simplifications, the **GPU pattern is exactly the production one**:
`N frames × M residues` of independent pairwise energy evaluation, parallelised
one residue per thread.

---

## References

- **AMBER MMPBSA.py** — https://ambermd.org/AmberTools.php — the canonical
  per-residue MM-PBSA/GBSA decomposition (`idecomp`); study its component split and
  the GB models it offers.
- **gmx_MMPBSA** — https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA — MM-GBSA
  decomposition for GROMACS trajectories; good worked examples of hot-spot output.
- **MDAnalysis** — https://github.com/MDAnalysis/mdanalysis — trajectory I/O and
  pairwise residue–ligand contact analysis; how to turn a real trajectory into the
  per-frame coordinates this kernel consumes.
- **ProLIF** — https://github.com/chemosim-lab/ProLIF — interaction fingerprints
  (IFP) for binding-mode decomposition; a complementary, contact-based view.
- **Still, Tempczyk, Hawley, Hendrickson (1990)**, *Semianalytical treatment of
  solvation for molecular mechanics and dynamics*, JACS 112:6127 — the original
  Generalized-Born `f_GB` formula used in §2.
- **Datasets:** PDBbind (http://www.pdbbind.org.cn), KLIFS (https://klifs.net),
  ChEMBL (https://www.ebi.ac.uk/chembl/), ClinVar
  (https://www.ncbi.nlm.nih.gov/clinvar/) — real complexes and resistance mutations
  to cross-reference predicted hot spots.
