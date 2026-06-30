// ===========================================================================
// src/inverse_folding.h  --  The ONE TRUE per-(residue, amino-acid) score core
// ---------------------------------------------------------------------------
// Project 2.10 : Protein Design / Inverse Folding Inference
//
// WHY THIS HEADER EXISTS  (the "HD-core" idiom, PATTERNS.md sec 2)
//   The single most useful trick in this repo: put the per-element physics in
//   ONE header marked __host__ __device__, so the CPU reference and the GPU
//   kernel call the *exact same* function. Then their results are byte-for-byte
//   identical and verification is EXACT (==), not "within some fuzzy epsilon".
//
//   reference_cpu.cpp (host C++ compiler) AND kernels.cu (nvcc) both include
//   this file. So it must contain NO CUDA-only types and NO __global__ kernels
//   -- only plain data structs, constants, and IF_HD-decorated inline helpers.
//
// WHAT THE SCIENCE IS  (full derivation in ../THEORY.md)
//   "Inverse folding" asks the reverse of structure prediction: given a fixed
//   protein BACKBONE (the chain of Calpha atom positions, one per residue), what
//   amino-acid SEQUENCE would fold into it? The real tool here is ProteinMPNN
//   (Baker Lab): a graph neural network that looks at backbone geometry and
//   autoregressively decodes a sequence, recovering ~50% of the native residues.
//
//   We ship a REDUCED-SCOPE TEACHING VERSION (CLAUDE.md sec 13): instead of a
//   trained 1.6M-parameter GNN, we use a small, transparent, physically-motivated
//   energy that captures the single dominant signal real models learn -- the
//   HYDROPHOBIC / POLAR burial preference:
//
//       * Buried core positions (many neighbors) prefer HYDROPHOBIC residues
//         (Leu, Ile, Val, Phe, ...) -- "oil hides from water".
//       * Exposed surface positions (few neighbors) prefer POLAR / CHARGED
//         residues (Glu, Lys, Arg, Asp, ...) -- they like the solvent.
//
//   The score of placing amino acid `aa` at residue `i` is therefore a function
//   of that residue's BURIAL (its neighbor count) and the amino acid's intrinsic
//   HYDROPHOBICITY. Designing the sequence = choosing, at every position, the
//   amino acid with the best score (argmax). That argmax is exactly the
//   zero-temperature / deterministic limit of ProteinMPNN's sampler.
//
//   This is honest as a *teaching* surface for the GPU pattern (one independent
//   scoring job per residue) without pretending to be a real design tool.
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh. Then read kernels.cu.
// ===========================================================================
#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// IF_HD: the host/device decorator switch.
//   When compiled by nvcc (__CUDACC__ is defined), every helper below becomes
//   callable from BOTH host and device code. When compiled by the plain C++
//   compiler (reference_cpu.cpp), the decorators simply vanish, so the same
//   source is an ordinary inline function. This is what makes CPU == GPU exact.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define IF_HD __host__ __device__
#else
#define IF_HD
#endif

// ---------------------------------------------------------------------------
// The amino-acid alphabet. We use the standard 20 proteinogenic amino acids in
// a FIXED canonical order so an index 0..19 always means the same residue on
// both CPU and GPU (and so the designed sequence string is reproducible).
//   NUM_AA = 20 is a compile-time constant: it sizes the per-residue score
//   vector, lets the inner loop be unrolled, and bounds the argmax.
// ---------------------------------------------------------------------------
constexpr int NUM_AA = 20;

// One-letter codes in our canonical index order. AA_CODES[k] is the letter for
// amino-acid index k. Order chosen to interleave hydrophobic/polar for clarity;
// the exact order is arbitrary but FIXED (changing it would change outputs).
//   Index: 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
//   Code : A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
// (Standard alphabetical-by-3-letter ProteinMPNN ordering: Ala Arg Asn Asp Cys
//  Gln Glu Gly His Ile Leu Lys Met Phe Pro Ser Thr Trp Tyr Val.)
// Defined in reference_cpu.cpp (a single definition shared by all translation
// units); declared here so device code knows the count, not the letters.

// ---------------------------------------------------------------------------
// aa_preferred_burial: the burial level (neighbor count) that amino acid `k`
//   "likes best", as a small INTEGER. This is the single learnable signal we
//   distil from real inverse folding into a transparent teaching constant:
//       HIGH preferred burial  -> hydrophobic residue, happiest deep in the core
//                                  (e.g. Ile/Leu/Val/Phe ~ 22-24 neighbors)
//       LOW  preferred burial  -> polar/charged residue, happiest on the surface
//                                  (e.g. Glu/Lys/Arg/Asp ~ 2-6 neighbors)
//       MID  preferred burial  -> amphipathic / neutral residues in between
//   Values are illustrative teaching constants (loosely tracking Kyte-Doolittle
//   hydropathy and observed burial statistics), NOT a calibrated force field.
//
//   WHY "PREFERRED BURIAL" AND NOT A SINGLE HYDROPHOBICITY SCALAR:
//     A purely bilinear (buriedness x hydrophobicity) energy always picks the
//     single MOST extreme residue at every buried position and the single most
//     polar one at every exposed position -- the whole core collapses to one
//     amino acid. Giving each residue its own preferred burial and scoring by
//     how CLOSE the position's burial is to that preference (a quadratic well,
//     below) makes the design GRADED and DIVERSE -- different burial levels
//     select different residues -- which is both more realistic and a better
//     teaching example. See THEORY sec 2.
//
//   WHY A FUNCTION WITH A LOCAL TABLE (not a global array):
//     A namespace-scope `constexpr` array is host-only; device code (nvcc)
//     cannot read a plain global from a __device__ function without a __device__
//     / __constant__ qualifier. Wrapping the table in an IF_HD function makes the
//     SAME constants visible to BOTH host and device with zero duplication --
//     the table lives in registers/immediate values at the call site. Indexed by
//     amino-acid index 0..19 (same order as AA_CODES above).
// ---------------------------------------------------------------------------
IF_HD inline int aa_preferred_burial(int k) {
    //                              A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
    constexpr int B[NUM_AA] = {    14,  3,  7,  4, 20,  6,  2, 11,  8, 23, 22,  4, 18, 24, 10, 12, 13, 19, 16, 23 };
    return B[k];
}

// ---------------------------------------------------------------------------
// BackboneResidue: the geometry the design depends on. In the full ProteinMPNN
// this is Calpha + virtual Cbeta + backbone dihedrals; in this teaching version
// we keep just the Calpha coordinate per residue (enough to compute burial).
//   x,y,z : Calpha atom position in angstroms (the natural length unit for
//           protein structure). float is plenty: PDB coordinates have ~0.01 A
//           precision and the burial count is integer-valued.
// ---------------------------------------------------------------------------
struct BackboneResidue {
    float x;   // Calpha x-coordinate (angstrom)
    float y;   // Calpha y-coordinate (angstrom)
    float z;   // Calpha z-coordinate (angstrom)
};

// Squared neighbor cutoff (angstrom^2). Two residues are "in contact" when their
// Calpha atoms are within CONTACT_RADIUS angstroms. We compare SQUARED distances
// to avoid a sqrt per pair (faster, and exact-integer-free comparison is fine).
//   10 A is a standard residue-contact cutoff in structural biology; 10^2 = 100.
constexpr float CONTACT_RADIUS    = 10.0f;          // angstrom
constexpr float CONTACT_RADIUS_SQ = CONTACT_RADIUS * CONTACT_RADIUS;  // = 100 A^2

// "Buried" threshold: a residue with at least this many contacts is treated as
// core (interior); fewer -> surface. ~16 neighbors within 10 A is typical for a
// well-packed protein core. This single integer turns a continuous neighbor
// count into the buried/exposed signal that drives the hydrophobic preference.
constexpr int BURIAL_THRESHOLD = 16;                // neighbor count

// ---------------------------------------------------------------------------
// score_aa_at_residue: the ONE TRUE scoring formula (THE shared core).
//   Returns an INTEGER score for placing amino-acid `aa_index` at a residue
//   whose burial (neighbor count) is `neighbor_count`. Higher = better fit.
//
//   The model is a QUADRATIC WELL around each amino acid's preferred burial:
//       score = -(neighbor_count - aa_preferred_burial(aa))^2
//   i.e. the closer this position's actual burial is to what amino acid `aa`
//   likes, the higher (less negative) the score; the best score, 0, occurs at a
//   perfect match. The argmax over `aa` therefore picks the residue whose burial
//   preference best fits this position -- buried positions favour hydrophobic
//   residues, exposed positions favour polar ones, and intermediate positions
//   favour amphipathic ones. This single transparent rule reproduces the
//   dominant signal a trained model learns. THEORY sec 2 derives it; sec 7
//   explains what a real GNN adds (pairwise couplings, learned chemistry, ...).
//
//   PARAMETERS
//     aa_index       : amino-acid index in 0..NUM_AA-1 (caller guarantees range)
//     neighbor_count : number of Calpha neighbors within CONTACT_RADIUS (>= 0)
//   RETURNS
//     an int score (units: negative squared neighbors); pure integer math so the
//     CPU and GPU produce identical values -> EXACT verification. The values are
//     small (|burial| <= a few dozen, squared -> at most a few thousand), so the
//     int never overflows.
//
//   Called by: design_cpu() in reference_cpu.cpp AND design_kernel() in
//   kernels.cu -- the whole point of putting it here.
// ---------------------------------------------------------------------------
IF_HD inline int score_aa_at_residue(int aa_index, int neighbor_count) {
    // How far this position's burial is from what this amino acid prefers.
    const int delta = neighbor_count - aa_preferred_burial(aa_index);
    // Quadratic penalty: a well centred on the preferred burial. Negated so that
    // "closer is better" and the best achievable score is 0 (a perfect match).
    return -(delta * delta);
}

#undef IF_HD
