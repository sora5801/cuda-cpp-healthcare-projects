// ===========================================================================
// src/screen_core.h  --  The ONE shared per-ligand "physics" (CPU == GPU)
// ---------------------------------------------------------------------------
// Project 1.4 : Ultra-Large Virtual Screening
//
// WHY THIS FILE EXISTS  (the single most important idiom in the repo --
// PATTERNS.md sec 2, "the shared __host__ __device__ core")
//   A virtual-screening campaign scores the SAME function on millions of
//   independent ligands. We must run that function in TWO places:
//     * on the CPU (reference_cpu.cpp) -- the trusted, obviously-correct baseline;
//     * on the GPU (kernels.cu)        -- one thread per ligand, the thing taught.
//   If the two implementations drifted apart even slightly, "GPU == CPU" would no
//   longer be a real correctness check. So we write the per-ligand math EXACTLY
//   ONCE, here, as `__host__ __device__` inline functions, and BOTH sides call
//   it. The CPU loops over ligands calling these; the kernel calls them from one
//   thread. The results are then *bit-for-bit identical* (we use integer /
//   fixed-point arithmetic -- see "DETERMINISM" below), so verification is exact.
//
// WHAT A LIGAND IS HERE  (a deliberately REDUCED-SCOPE teaching model)
//   Real ultra-large screening docks a 3-D molecule into a protein pocket with a
//   genetic-algorithm + local-search optimiser (AutoDock-GPU's LGA/BFGS). That is
//   genuinely research-grade and far beyond one teaching kernel (CLAUDE.md sec 13
//   says: ship the simplest correct teaching version, describe the full one in
//   THEORY). So a "ligand" here is a small fixed-size DESCRIPTOR VECTOR -- the
//   kind of cheap 2-D property a real cascade pre-filters on BEFORE expensive
//   docking -- plus a packed PHARMACOPHORE BITMASK standing in for 3-D features:
//       mw        molecular weight              (Daltons)
//       logp      lipophilicity (octanol/water) (x100, signed integer)
//       hbd       hydrogen-bond donors          (count)
//       hba       hydrogen-bond acceptors       (count)
//       rotb      rotatable bonds               (count, flexibility proxy)
//       psa       topological polar surface area(Angstrom^2)
//       feat      32-bit pharmacophore bitmask  (which features the ligand has)
//   The PIPELINE this file encodes is the real shape of a screening campaign:
//       (1) a cheap FILTER CASCADE (Lipinski Rule-of-Five + a PSA/rotbond
//           drug-likeness gate) throws out ligands that can never be oral drugs;
//       (2) survivors get a SURROGATE DOCKING SCORE -- here a fast, deterministic
//           pharmacophore-overlap + property-complementarity score that stands in
//           for the expensive docking a real campaign would run next.
//   This is exactly the "ML/cheap surrogate filters the library, full docking
//   evaluates only the survivors" strategy (HASTEN/REINVENT) from the catalog.
//
// DETERMINISM  (PATTERNS.md sec 3)
//   The surrogate score is computed in INTEGER fixed-point and returned as an int.
//   Integer addition is associative and order-independent, so the CPU and GPU --
//   and any thread ordering -- produce the identical score. That makes stdout
//   byte-for-byte reproducible AND makes the GPU-vs-CPU check pass with tolerance
//   ZERO (the strongest possible kind, PATTERNS.md sec 4).
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh, kernels.cu.
// The science/math/derivation lives in ../THEORY.md.
// ===========================================================================
#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// HD: the host/device decorator macro (PATTERNS.md sec 2).
//   When this header is compiled by nvcc (which defines __CUDACC__), HD expands
//   to `__host__ __device__`, so each function is compiled BOTH for the CPU and
//   for the GPU. When compiled by the plain host C++ compiler (reference_cpu.cpp,
//   which does NOT define __CUDACC__), HD expands to nothing, so the very same
//   source is an ordinary inline function. One body, two targets, identical math.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// Number of pharmacophore feature bits we model (fits one 32-bit word). A real
// pharmacophore has features like "aromatic ring", "H-bond donor at site X",
// "positive ionizable", etc.; here they are abstract bits 0..31.
static constexpr int FEAT_BITS = 32;

// ---------------------------------------------------------------------------
// Ligand: one row of the library. Plain-old-data (POD) so it copies trivially
// to the GPU with a single cudaMemcpy and lives happily in device global memory.
//   * All-integer fields keep the math exact and the struct trivially copyable.
//   * logp is stored x100 (e.g. logP = 3.50 -> 350) so we avoid floats entirely;
//     Lipinski's logP<=5 rule becomes the integer test logp_x100 <= 500.
// ---------------------------------------------------------------------------
struct Ligand {
    int32_t  mw;         // molecular weight, Daltons (integer-rounded)
    int32_t  logp_x100;  // logP * 100, signed (lipophilicity)
    int32_t  hbd;        // # hydrogen-bond donors
    int32_t  hba;        // # hydrogen-bond acceptors
    int32_t  rotb;       // # rotatable bonds (flexibility)
    int32_t  psa;        // topological polar surface area, Angstrom^2
    uint32_t feat;       // 32-bit pharmacophore feature bitmask
};

// ---------------------------------------------------------------------------
// Target: the binding-site "wish list" the campaign screens AGAINST. A real
// campaign derives this from the protein pocket; here it is a small property
// window plus the pharmacophore features the pocket rewards. Stored once and
// (on the GPU) placed in CONSTANT memory because every thread reads it and none
// writes it -- the constant cache then broadcasts it warp-wide (see kernels.cu).
// ---------------------------------------------------------------------------
struct Target {
    int32_t  mw_opt;        // ideal ligand molecular weight for the pocket
    int32_t  logp_opt_x100; // ideal logP * 100
    int32_t  psa_opt;       // ideal polar surface area
    uint32_t feat_required; // pharmacophore features the pocket REWARDS (bitmask)
};

// ---------------------------------------------------------------------------
// popcount32_portable: count set bits in a 32-bit word.
//   On the GPU nvcc lowers this loop to the single-instruction __popc intrinsic;
//   on the CPU it is Brian Kernighan's method (`x &= x-1` clears the lowest set
//   bit, so the loop runs once per set bit). We hand-roll it (rather than call
//   __popc / std::popcount) so the SAME source compiles for both targets and the
//   result is provably identical. Counting overlapping pharmacophore bits is the
//   heart of the surrogate score.
// ---------------------------------------------------------------------------
HD inline int popcount32_portable(uint32_t x) {
    int c = 0;
    while (x) { x &= (x - 1); ++c; }   // clear lowest set bit, count once per bit
    return c;
}

// ---------------------------------------------------------------------------
// abs_i: integer absolute value. We avoid <cstdlib>'s std::abs so the function
// is unambiguously available in device code with no header surprises.
// ---------------------------------------------------------------------------
HD inline int abs_i(int v) { return v < 0 ? -v : v; }

// ---------------------------------------------------------------------------
// passes_filter_cascade: the cheap drug-likeness GATE (stage 1 of the pipeline).
//   Implements Lipinski's "Rule of Five" (the classic oral-bioavailability
//   heuristic) PLUS a Veber-style flexibility/polarity gate:
//       MW   <= 500 Da
//       logP <= 5            (stored as logp_x100 <= 500)
//       HBD  <= 5
//       HBA  <= 10
//       rotatable bonds <= 10   (Veber: too floppy -> poor oral absorption)
//       PSA  <= 140 A^2         (Veber: too polar  -> poor permeability)
//   A ligand that violates ANY rule is rejected (returns false) and never scored
//   -- exactly how a real cascade saves the expensive docking budget for the
//   plausible molecules. Returns true iff the ligand survives all gates.
//
//   Teaching note: real cascades allow "one Lipinski violation"; we use the
//   strict all-must-pass form because it is the clearest to read and to verify.
//   (An exercise in the README relaxes it.)
// ---------------------------------------------------------------------------
HD inline bool passes_filter_cascade(const Ligand& L) {
    if (L.mw        > 500) return false;   // Lipinski: molecular weight
    if (L.logp_x100 > 500) return false;   // Lipinski: logP <= 5
    if (L.hbd       > 5)   return false;   // Lipinski: H-bond donors
    if (L.hba       > 10)  return false;   // Lipinski: H-bond acceptors
    if (L.rotb      > 10)  return false;   // Veber: rotatable bonds
    if (L.psa       > 140) return false;   // Veber: polar surface area
    return true;                           // survived the whole cascade
}

// ---------------------------------------------------------------------------
// surrogate_dock_score: the fast, deterministic stand-in for full docking
// (stage 2). Higher = better predicted binder. It rewards two things a real
// scoring function also rewards, combined into one INTEGER score:
//
//   (A) PHARMACOPHORE OVERLAP -- how many of the pocket's required feature bits
//       the ligand actually presents. This is popcount(feat & feat_required):
//       the same bit-AND-then-popcount motif as Tanimoto search (project 1.12),
//       which is why this project shares 1.12's "independent jobs" GPU pattern.
//       Each matched feature is worth FEAT_WEIGHT points.
//
//   (B) PROPERTY COMPLEMENTARITY -- a penalty for how far the ligand's key
//       properties sit from the pocket's ideal window. We penalise the absolute
//       differences in molecular weight, logP, and PSA (each scaled down so the
//       penalty is commensurate with the feature reward). A ligand whose size
//       and polarity fit the pocket loses fewer points.
//
//   score = BASE + FEAT_WEIGHT * (matched features)
//                - |dMW|/MW_SCALE - |dlogP|/LOGP_SCALE - |dPSA|/PSA_SCALE
//
//   All terms are integers, so the sum is order-independent and identical on CPU
//   and GPU. We clamp at 0 so a hopeless ligand scores 0 rather than going
//   negative (keeps the printed ranking tidy; the clamp is itself deterministic).
//
//   This is intentionally NOT real docking -- it is a transparent surrogate that
//   (a) exercises the exact GPU pattern real campaigns use (score N independent
//   ligands, keep the top-K) and (b) is cheap enough to run on the CPU reference.
//   THEORY sec "Where this sits in the real world" explains the gap honestly.
// ---------------------------------------------------------------------------
HD inline int surrogate_dock_score(const Ligand& L, const Target& T) {
    // Tunable integer constants (kept here so CPU and GPU share them exactly).
    constexpr int BASE        = 1000;  // baseline so most scores stay positive
    constexpr int FEAT_WEIGHT = 60;    // points per matched pharmacophore feature
    constexpr int MW_SCALE    = 10;    // 1 penalty point per 10 Da off-target
    constexpr int LOGP_SCALE  = 50;    // 1 point per 0.5 logP unit off-target
    constexpr int PSA_SCALE   = 4;     // 1 point per 4 A^2 PSA off-target

    // (A) pharmacophore overlap: bits the ligand has AND the pocket rewards.
    const uint32_t overlap = L.feat & T.feat_required;
    const int matched      = popcount32_portable(overlap);

    // (B) property mismatch penalties (integer-divided -> exact, deterministic).
    const int pen_mw   = abs_i(L.mw        - T.mw_opt)        / MW_SCALE;
    const int pen_logp = abs_i(L.logp_x100 - T.logp_opt_x100) / LOGP_SCALE;
    const int pen_psa  = abs_i(L.psa       - T.psa_opt)       / PSA_SCALE;

    int score = BASE + FEAT_WEIGHT * matched - pen_mw - pen_logp - pen_psa;
    if (score < 0) score = 0;          // clamp: no negative scores
    return score;
}

// ---------------------------------------------------------------------------
// score_ligand: the WHOLE per-ligand pipeline in one call (filter then score).
//   Returns the surrogate score for a ligand that PASSES the cascade, or the
//   sentinel REJECTED for one that fails. Both CPU and GPU call exactly this, so
//   "which ligands were filtered out" and "what each survivor scored" are
//   computed by identical code -> identical results. main.cu turns the REJECTED
//   sentinel into the survivor count and the top-K hit list.
// ---------------------------------------------------------------------------
static constexpr int REJECTED = -1;    // sentinel: ligand failed the cascade

HD inline int score_ligand(const Ligand& L, const Target& T) {
    if (!passes_filter_cascade(L)) return REJECTED;  // stage 1: cheap gate
    return surrogate_dock_score(L, T);               // stage 2: surrogate dock
}
