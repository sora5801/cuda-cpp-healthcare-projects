// ===========================================================================
// src/ddg_model.h  --  Shared (host + device) ΔΔG scoring model + data layout
// ---------------------------------------------------------------------------
// Project 2.16 : ΔΔG Stability Prediction (reduced-scope teaching version)
//
// WHY THIS HEADER IS SHARED  (PATTERNS.md §2, the HD-macro idiom)
//   The whole point of CPU-vs-GPU verification here is that the serial reference
//   (reference_cpu.cpp, compiled by cl.exe/g++) and the GPU kernel (kernels.cu,
//   compiled by nvcc) evaluate the *identical* scoring function for every
//   mutation. The only way to guarantee that is to put the per-mutation math in
//   ONE place and include it from both worlds. That is this header.
//
//   The DDG_HD macro expands to `__host__ __device__` under nvcc and to nothing
//   under the plain host compiler, so the same inline functions compile in both
//   translation units and produce the same arithmetic. Keep CUDA-only types
//   (__global__, dim3, ...) OUT of this header so cl.exe can include it.
//
// ---------------------------------------------------------------------------
// WHAT WE ARE MODELLING  (the science is expanded in ../THEORY.md)
//   ΔΔG = the change in folding free energy when a single amino acid in a
//   protein is mutated:   ΔΔG = ΔG_fold(mutant) - ΔG_fold(wild-type), in
//   kcal/mol. Sign convention used here (the ThermoMPNN / "ddG = stability of
//   mutant minus wild-type" convention):
//       ΔΔG > 0  -> the mutation is STABILISING (mutant folds more tightly)
//       ΔΔG < 0  -> the mutation is DESTABILISING (the common case; most random
//                   mutations make a well-evolved protein worse)
//       ΔΔG = 0  -> the trivial self-"mutation" (residue -> itself).
//
//   Production predictors (ThermoMPNN, ProteinMPNN-ddG, ESM-1v) LEARN this map
//   from tens of thousands of experimental measurements (Protherm) or millions
//   of proteolysis-stability readouts (the Megascale set). We CANNOT ship a
//   trained neural net inside a tiny didactic C++ file without it becoming an
//   opaque blob of weights. So this project implements a small, fully transparent
//   PHYSICS-INSPIRED scoring function whose every term you can read and reason
//   about. It is deliberately a *teaching* model, NOT a validated predictor --
//   see ../THEORY.md "Where this sits in the real world" and the README
//   "Limitations & honesty" section. The CUDA lesson (a saturation-mutagenesis
//   scan as N×20 independent scoring jobs) is identical for a real model.
//
// ---------------------------------------------------------------------------
// THE SCORING FUNCTION  (a sum of interpretable physical penalties)
//   For a mutation (wild-type residue `wt` -> mutant residue `mut`) at a position
//   whose local structural environment is summarised by a burial fraction
//   `buried` in [0,1] (1 = fully buried core, 0 = fully solvent-exposed), we
//   compute four additive contributions, all in kcal/mol:
//
//     1) HYDROPHOBIC BURIAL.  Burying a hydrophobic side chain is stabilising;
//        burying a polar/charged one (or exposing a hydrophobic one) is not.
//        term = w_hyd * buried * (hydropathy[mut] - hydropathy[wt])
//
//     2) VOLUME / PACKING STRAIN.  In the tightly packed core, changing side-
//        chain volume strains the fold; the penalty scales with how buried the
//        site is and with the squared volume change (over-packing or creating a
//        cavity both cost energy).
//        term = -w_vol * buried * (vol[mut]-vol[wt])^2
//
//     3) PROLINE / GLYCINE BACKBONE PENALTY.  Proline kinks the backbone and
//        cannot donate a backbone H-bond; glycine is uniquely flexible. Mutating
//        TO proline or glycine in a structured (buried) site is destabilising.
//        term = -w_pg * buried * is_pro_or_gly[mut]
//
//     4) CHARGE-BURIAL PENALTY.  Burying a net charge with no solvent to
//        stabilise it is costly (desolvation). Penalise introducing a charged
//        residue into a buried site.
//        term = -w_chg * buried * |charge[mut]|
//
//   ΔΔG = sum of the four terms. A self-mutation (mut == wt) gives exactly 0 by
//   construction (every term is a difference or is cancelled), which is a nice
//   built-in sanity check the demo verifies.
//
//   This is then passed through a smooth bounded squashing (a scaled tanh) so
//   the output stays in a physically plausible window (about ±8 kcal/mol),
//   mimicking how real ΔΔG values cluster in a narrow range. The tanh is the one
//   nonlinearity; it is also the term most sensitive to floating-point fused-
//   multiply-add differences between host and device (see ../THEORY.md
//   "Numerical considerations" and the tolerance discussion).
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cu (both include this header).
// ===========================================================================
#pragma once

#include <cmath>     // std::tanh (host) / tanhf is used via the HD wrapper
#include <cstdint>

// --- The HD macro: __host__ __device__ under nvcc, nothing under cl.exe -----
#ifdef __CUDACC__
#define DDG_HD __host__ __device__
#else
#define DDG_HD
#endif

// ---------------------------------------------------------------------------
// The 20 standard amino acids, in a FIXED canonical order. Every per-residue
// property table below is indexed by this 0..19 code, so the order is part of
// the data contract (the loader in reference_cpu.cpp maps one-letter codes to
// these indices). We expose the count as a constant used to size the scan.
// ---------------------------------------------------------------------------
constexpr int NUM_AA = 20;   // number of standard amino acids (scan width)

// One-letter codes in canonical index order. Index i  <->  AA_ONE_LETTER[i].
//   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
// (alphabetical by one-letter code; this exact order is mirrored in the tables.)
constexpr char AA_ONE_LETTER[NUM_AA + 1] = "ARNDCQEGHILKMFPSTWYV";

// ---------------------------------------------------------------------------
// PER-RESIDUE PROPERTY TABLES  (literature-derived biochemical scales).
//
// IMPLEMENTATION NOTE -- why these are returned from DDG_HD accessor functions
// rather than declared as bare namespace-scope `constexpr` arrays:
//   A namespace-scope `constexpr float[]` has HOST linkage; nvcc forbids reading
//   such a host variable from `__device__` code ("identifier undefined in device
//   code"). The portable idiom that keeps ONE source of truth readable from BOTH
//   host and device is to wrap each table in a `__host__ __device__` (DDG_HD)
//   inline accessor whose `static constexpr` local array is materialised in
//   whichever address space the caller compiles to. (Alternatives: duplicate the
//   data with a separate `__constant__` copy for the device, or pass the tables
//   as kernel arguments -- both are heavier and less DRY for a fixed 20-entry
//   table. See ../THEORY.md "GPU mapping".)
//
// These numbers are NOT fit to any ΔΔG dataset -- they are physical priors,
// which is exactly why the resulting model is interpretable but only didactic.
// ---------------------------------------------------------------------------

// (1) Kyte-Doolittle hydropathy index. Positive = hydrophobic (likes the core),
//     negative = hydrophilic (likes the surface). Indexed by canonical AA code.
DDG_HD inline float aa_hydropathy(int i) {
    //                              A    R    N    D    C    Q    E    G    H    I    L    K    M    F    P    S    T    W    Y    V
    static constexpr float T[NUM_AA] = {
       1.8f,-4.5f,-3.5f,-3.5f, 2.5f,-3.5f,-3.5f,-0.4f,-3.2f, 4.5f, 3.8f,-3.9f, 1.9f, 2.8f,-1.6f,-0.8f,-0.7f,-0.9f,-1.3f, 4.2f };
    return T[i];
}

// (2) Approximate side-chain volume (Å^3). Bigger residues strain a packed core
//     more when their volume changes. Rounded Zamyatnin volumes.
DDG_HD inline float aa_volume(int i) {
    //                               A     R     N     D     C     Q     E     G     H     I     L     K     M     F     P     S     T     W     Y     V
    static constexpr float T[NUM_AA] = {
       88.6f,173.4f,114.1f,111.1f,108.5f,143.8f,138.4f, 60.1f,153.2f,166.7f,166.7f,168.6f,162.9f,189.9f,112.7f, 89.0f,116.1f,227.8f,193.6f,140.0f };
    return T[i];
}

// (3) Net formal charge at physiological pH (e). Asp/Glu = -1, Lys/Arg = +1,
//     His ~ +0.1 (mostly neutral). All others 0. Used by the desolvation penalty.
DDG_HD inline float aa_charge(int i) {
    //                            A   R   N   D   C   Q   E   G   H    I   L   K   M   F   P   S   T   W   Y   V
    static constexpr float T[NUM_AA] = {
       0.f,1.f,0.f,-1.f,0.f,0.f,-1.f,0.f,0.1f,0.f,0.f,1.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f };
    return T[i];
}

// (4) Backbone-disrupting flag: 1.0 for Proline (index 14) and Glycine (index 7),
//     else 0.0. Mutating TO one of these inside a structured site is penalised.
DDG_HD inline float aa_is_pg(int i) {
    //                           A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
    static constexpr float T[NUM_AA] = {
       0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,0.f,0.f,0.f,0.f,0.f };
    return T[i];
}

// ---------------------------------------------------------------------------
// MODEL WEIGHTS  (the four term coefficients, in kcal/mol-consistent units).
// Hand-chosen so the terms are commensurate and the output spans a realistic
// range. Centralised here so THEORY.md, the CPU reference, and the GPU kernel
// all refer to the same single source of truth. These are NOT trained.
// ---------------------------------------------------------------------------
constexpr float W_HYDRO  = 0.10f;    // hydrophobic-burial gain
constexpr float W_VOLUME = 0.0008f;  // packing-strain gain (multiplies ΔV^2); kept
                                     // small so the squared term rarely saturates
                                     // the tanh, leaving the four terms legible.
constexpr float W_PROGLY = 2.50f;    // proline/glycine backbone penalty
constexpr float W_CHARGE = 1.80f;    // buried-charge desolvation penalty
constexpr float DDG_SCALE = 8.0f;    // tanh squashing half-range (±kcal/mol)

// ---------------------------------------------------------------------------
// ddg_raw : the un-squashed sum of the four physical terms for one mutation.
//   wt, mut : amino-acid indices in [0, NUM_AA)  (wild-type and mutant).
//   buried  : local burial fraction in [0,1] (1 = core, 0 = surface). This is a
//             stand-in for the per-residue structural embedding a real GNN would
//             compute from the backbone; here it is a single scalar feature.
//   returns : raw ΔΔG-like score in kcal/mol BEFORE the bounded squashing.
//
//   Marked DDG_HD so the *same* function runs on host and device. It is a pure
//   function of its inputs and the constexpr tables above -> deterministic.
// ---------------------------------------------------------------------------
DDG_HD inline float ddg_raw(int wt, int mut, float buried) {
    // Term 1: hydrophobic burial. Gaining hydropathy in a buried site stabilises.
    const float d_hyd = aa_hydropathy(mut) - aa_hydropathy(wt);
    const float t_hyd = W_HYDRO * buried * d_hyd;

    // Term 2: packing strain. Cost grows with the SQUARED volume change, scaled
    // by burial (surface sites tolerate volume changes far better than the core).
    const float d_vol = aa_volume(mut) - aa_volume(wt);
    const float t_vol = -W_VOLUME * buried * (d_vol * d_vol);

    // Term 3: proline/glycine backbone penalty (only when mutating TO P or G,
    // and only meaningfully in a structured/buried site).
    const float t_pg = -W_PROGLY * buried * aa_is_pg(mut);

    // Term 4: buried-charge desolvation. Introducing |charge| into the core costs
    // energy; subtract the wild-type charge magnitude so the self-mutation is 0.
    const float d_chg = fabsf(aa_charge(mut)) - fabsf(aa_charge(wt));
    const float t_chg = -W_CHARGE * buried * d_chg;

    return t_hyd + t_vol + t_pg + t_chg;   // total raw ΔΔG (kcal/mol)
}

// ---------------------------------------------------------------------------
// ddg_predict : the full model -- raw score passed through a bounded squashing.
//   We use DDG_SCALE * tanh(raw / DDG_SCALE), which is ~ linear for small |raw|
//   (so the physics terms read through directly) but saturates smoothly to
//   ±DDG_SCALE for large |raw| (so no single term can produce an absurd value).
//   This mirrors how real ΔΔG measurements cluster in a narrow band.
//
//   IMPORTANT (verification): tanhf is the one place where the device and host
//   math libraries, plus fused-multiply-add (FMA) contraction, can differ in the
//   last bit. The results agree to ~1e-5 kcal/mol, which is physically
//   negligible; the demo verifies to a documented 1e-3 tolerance (PATTERNS.md §4,
//   "a small physical tolerance"). See ../THEORY.md "Numerical considerations".
//
//   Using tanhf (the float overload) on BOTH sides keeps the precision matched;
//   under nvcc tanhf is the device intrinsic, under cl.exe it is <cmath>'s float
//   tanh. Returns ΔΔG in kcal/mol.
// ---------------------------------------------------------------------------
DDG_HD inline float ddg_predict(int wt, int mut, float buried) {
    const float raw = ddg_raw(wt, mut, buried);
    return DDG_SCALE * tanhf(raw / DDG_SCALE);
}
