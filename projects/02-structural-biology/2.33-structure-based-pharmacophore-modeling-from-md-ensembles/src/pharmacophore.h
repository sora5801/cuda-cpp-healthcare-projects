// ===========================================================================
// src/pharmacophore.h  --  The ONE TRUE per-molecule scoring formula (HD core)
// ---------------------------------------------------------------------------
// Project 2.33 : Structure-Based Pharmacophore Modeling from MD Ensembles
//
// WHY THIS HEADER EXISTS  (PATTERNS.md §2: the shared __host__ __device__ core)
//   The CPU reference (reference_cpu.cpp, compiled by cl.exe / g++) and the GPU
//   kernel (kernels.cu, compiled by nvcc) must compute the EXACT SAME number for
//   each molecule so that "GPU == CPU" verification is meaningful. The cleanest
//   way to guarantee that is to write the per-molecule physics ONCE, in this
//   header, as an inline function decorated `__host__ __device__`. Both compilers
//   then emit the same arithmetic (same order of `exp`, `+`, `*`), so the results
//   agree to ~machine precision instead of merely "approximately".
//
//   To make this work the header must contain NO CUDA-only constructs:
//     * no `__global__` kernels  (those live in kernels.cu),
//     * no `<<<...>>>` launches,
//     * only plain structs + the HD-decorated scoring function.
//   The FP_HD macro below expands to `__host__ __device__` under nvcc and to
//   NOTHING under a plain host compiler, so the same line compiles in both.
//
// WHAT IT MODELS  (see ../THEORY.md for the full science -> math derivation)
//   A *pharmacophore* is the 3-D arrangement of chemical features a ligand needs
//   to bind a receptor: hydrogen-bond donors/acceptors, hydrophobic blobs,
//   aromatic rings, positive/negative charges. An *ensemble* pharmacophore is the
//   consensus of those features computed over many MD trajectory frames, so it
//   captures receptor flexibility (induced fit, cryptic pockets) that a single
//   static crystal structure would miss.
//
//   We screen a LIBRARY of candidate molecules, each represented as its own set
//   of typed feature points, against this one query pharmacophore. The match
//   score is a ROCS-style "color" overlap: a sum of Gaussian overlaps between
//   query and library feature points OF THE SAME TYPE, normalized Tanimoto-style
//   so the score lies in [0, 1]. (ROCS = Rapid Overlay of Chemical Structures,
//   the OpenEye tool named in the catalog; "color" = the chemical-feature term,
//   as opposed to plain shape overlap.)
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

// ---------------------------------------------------------------------------
// FP_HD: the host/device portability shim.
//   * Under nvcc (`__CUDACC__` is defined) it becomes `__host__ __device__`, so
//     the function is compiled for BOTH the CPU and the GPU.
//   * Under a plain C++ compiler the decorators do not exist, so it must expand
//     to nothing -- otherwise reference_cpu.cpp would not compile.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define FP_HD __host__ __device__
#else
#define FP_HD
#endif

#include <cmath>   // std::exp (host) / exp (device) -- both visible via this include

// ---------------------------------------------------------------------------
// Feature types. A pharmacophore feature is "typed": a donor only overlaps with
// a donor, an acceptor with an acceptor, etc. We encode the type as a small int
// so the per-pair test is a single integer compare (cheap on the GPU, no string
// handling on the device). The exact set mirrors the common ROCS/Pharmer color
// types; six is plenty to teach the idea.
// ---------------------------------------------------------------------------
enum FeatureType {
    FEAT_DONOR        = 0,   // hydrogen-bond donor      (e.g. -OH, -NH)
    FEAT_ACCEPTOR     = 1,   // hydrogen-bond acceptor   (e.g. =O, ring N)
    FEAT_HYDROPHOBE   = 2,   // hydrophobic contact      (e.g. alkyl, aromatic face)
    FEAT_AROMATIC     = 3,   // aromatic ring centroid   (pi-stacking)
    FEAT_POS_CHARGE   = 4,   // positively ionizable     (e.g. protonated amine)
    FEAT_NEG_CHARGE   = 5,   // negatively ionizable     (e.g. carboxylate)
    FEAT_NUM_TYPES    = 6    // sentinel: how many distinct types exist
};

// ---------------------------------------------------------------------------
// Feature: one pharmacophore point. Center is in angstroms (the natural unit for
// molecular geometry); `type` is one of the FeatureType enum values above.
//
// Memory note: this is a Plain-Old-Data struct (4 floats + 1 int + 1 pad = 24 B)
// so we can `cudaMemcpy` a flat array of them straight to the GPU with no
// serialization. We keep it `float` (FP32) because feature coordinates come from
// MD with ~0.01 A noise -- double precision would be false precision here, and
// FP32 doubles the memory bandwidth, which is what the GPU is bound by.
// ---------------------------------------------------------------------------
struct Feature {
    float x;       // center x  [angstrom]
    float y;       // center y  [angstrom]
    float z;       // center z  [angstrom]
    float weight;  // feature importance in [0,1]; consensus features from many MD
                   // frames get weight ~1, rare/transient ones less. It scales the
                   // feature's contribution to the overlap (see overlap_pair()).
    int   type;    // FeatureType: only same-type features overlap
    int   _pad;    // explicit padding -> 24-byte struct, predictable on host & GPU
};

// ---------------------------------------------------------------------------
// The Gaussian "alpha" that sets how quickly overlap falls off with distance.
//   Each feature is modeled as an isotropic Gaussian g(r) = exp(-alpha * r^2)
//   centered on the feature point. ROCS uses alpha derived from a per-atom
//   radius; for typed pharmacophore points a single tolerance radius is the
//   standard teaching choice. We pick alpha so that two features 1.0 A apart
//   still overlap strongly and ones >3 A apart barely do:
//       alpha = ln(2) / r_half^2,  with r_half = 1.0 A  (half-max at 1 A separation)
//   => alpha = 0.6931.  (Derivation in THEORY.md §The math.)
//
// `constexpr` so the value is a compile-time constant shared verbatim by host and
// device -- no risk of the two sides using different numbers.
// ---------------------------------------------------------------------------
constexpr float PHARM_ALPHA = 0.6931471805599453f;   // ln(2), with r_half = 1 A

// ---------------------------------------------------------------------------
// overlap_pair: the Gaussian overlap of ONE query feature with ONE library
// feature. Returns 0 immediately if the types differ (a donor cannot satisfy an
// acceptor requirement). Otherwise it is
//       w_q * w_l * exp(-alpha * |r_q - r_l|^2)
// i.e. the product of the two feature weights times a Gaussian in the squared
// separation. This is the elementary building block both the CPU and the GPU
// sum over -- defining it ONCE here is what makes their results identical.
//
// Returns a double so the accumulation in score_molecule() does not lose bits;
// the inputs are float (coords) but the running sum wants headroom.
// ---------------------------------------------------------------------------
FP_HD inline double overlap_pair(const Feature& q, const Feature& l) {
    if (q.type != l.type) return 0.0;            // typed: only like overlaps like
    const double dx = (double)q.x - (double)l.x; // separation components [A]
    const double dy = (double)q.y - (double)l.y;
    const double dz = (double)q.z - (double)l.z;
    const double r2 = dx * dx + dy * dy + dz * dz;   // squared distance [A^2]
    // exp() resolves to the host std::exp or the device intrinsic; both follow
    // IEEE-754 closely enough that host and device agree to ~1e-7 relative.
    return (double)q.weight * (double)l.weight * exp(-(double)PHARM_ALPHA * r2);
}

// ---------------------------------------------------------------------------
// score_molecule: the ROCS-style "color Tanimoto" of the query pharmacophore
// against ONE library molecule's feature set. THIS IS THE FUNCTION THE GPU
// KERNEL AND THE CPU REFERENCE BOTH CALL -- one thread runs it per library
// molecule on the GPU; a plain loop runs it per molecule on the CPU.
//
//   Let O_ql = sum over (query feature i, library feature j) of overlap_pair(i,j)
//       O_qq = self-overlap of the query with itself  (a constant per query)
//       O_ll = self-overlap of this library molecule with itself
//   Tanimoto color score:
//       T = O_ql / (O_qq + O_ll - O_ql)
//   T is 1.0 when the library features perfectly coincide with the query's, and
//   ~0 when they are far apart or of the wrong types. The Tanimoto denominator
//   normalizes out molecule size, so a big floppy molecule that happens to cover
//   the query does not automatically win (a known ROCS design choice).
//
// Parameters:
//   query     : pointer to the query pharmacophore features  [n_query]
//   n_query   : number of query features
//   self_qq   : O_qq, the query self-overlap, precomputed ONCE on the host and
//               passed in (it is identical for every library molecule, so we do
//               not recompute it n_lib times -- a small but honest optimization).
//   lib       : pointer to THIS molecule's features          [n_lib_feats]
//   n_lib     : number of features on this molecule
// Returns the Tanimoto color score in [0,1] (or 0 if the denominator vanishes).
//
// Complexity: O(n_query * n_lib) overlap evaluations for one molecule. With both
// counts small (a pharmacophore is typically 4-10 points) this is a handful of
// exp() calls -- the parallelism is across the MANY library molecules, not within
// one. That is exactly the "independent jobs" GPU pattern (PATTERNS.md §1).
// ---------------------------------------------------------------------------
FP_HD inline float score_molecule(const Feature* query, int n_query, double self_qq,
                                  const Feature* lib, int n_lib) {
    double o_ql = 0.0;   // cross-overlap: query vs this library molecule
    double o_ll = 0.0;   // self-overlap of this library molecule (for Tanimoto)

    // Cross term: every query feature against every library feature.
    for (int i = 0; i < n_query; ++i)
        for (int j = 0; j < n_lib; ++j)
            o_ql += overlap_pair(query[i], lib[j]);

    // Library self-overlap. We sum ALL ordered pairs (including i==j and both
    // orders) so it is consistent with how self_qq for the query is computed on
    // the host (same double loop). Consistency of the two self-terms is what
    // keeps the Tanimoto well-defined and host/device-identical.
    for (int i = 0; i < n_lib; ++i)
        for (int j = 0; j < n_lib; ++j)
            o_ll += overlap_pair(lib[i], lib[j]);

    const double denom = self_qq + o_ll - o_ql;   // Tanimoto denominator
    if (denom <= 0.0) return 0.0f;                 // guard: empty/degenerate sets
    return (float)(o_ql / denom);                  // collapse to FP32 result
}
