// ===========================================================================
// src/ifp.h  --  Shared (host + device) interaction-fingerprint + clustering core
// ---------------------------------------------------------------------------
// Project 1.33 : Interaction Fingerprinting & Binding-Mode Clustering
//
// WHAT THIS PROJECT COMPUTES  (two stages, two classic GPU patterns)
//
//   STAGE A -- IFP GENERATION (a geometry kernel over a pose x residue grid)
//     A docking run or an MD trajectory gives many candidate ligand POSES inside
//     one protein binding pocket. For each pose we ask, residue by residue:
//     "does the ligand make a hydrophobic contact / a hydrogen bond / an
//     aromatic (pi) contact / an ionic (salt-bridge) contact with THIS residue?"
//     Each yes/no answer is a BIT. Laying those bits out residue-by-residue,
//     type-by-type, gives the pose's INTERACTION FINGERPRINT (IFP) -- a fixed
//     length bit-string, exactly like a SIFt (Structural Interaction Fingerprint)
//     or a PLIF. Whether a bit is set is decided purely by GEOMETRY (a distance,
//     sometimes a distance + angle), so every (pose, residue) cell is an
//     independent little computation -> perfect for one GPU thread per cell.
//
//   STAGE B -- BINDING-MODE CLUSTERING (Tanimoto k-means on the bit-vectors)
//     Two poses that light up the SAME residues in the SAME way occupy the same
//     "binding mode". We group poses into modes with k-means, but using the
//     TANIMOTO (Jaccard) distance between bit-vectors -- popcount(A&B)/popcount(A|B)
//     -- the structural-biology analogue of chemical-fingerprint Tanimoto
//     (project 1.12). The cluster "centroid" is a CONSENSUS bit-vector: bit b is
//     set iff a MAJORITY of the cluster's members set bit b. Consensus is a pure
//     integer majority vote, so it is order-independent -> the GPU and the CPU
//     reference produce BIT-IDENTICAL results (no floating-point drift at all).
//
// WHY A GPU
//   Real runs have 10^3-10^6 poses (or MD frames) x hundreds of residues. STAGE A
//   is millions of independent distance tests; STAGE B's ASSIGN step is millions
//   of Tanimoto popcounts. Both are embarrassingly parallel. This file holds the
//   PER-ELEMENT math as __host__ __device__ inline functions (the IFP_HD idiom,
//   PATTERNS.md sec 2) so the CPU reference and the CUDA kernels run byte-for-byte
//   identical logic -- making verification EXACT rather than approximate.
//
// READ THIS AFTER: nothing (start here), then reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>

// IFP_HD expands to __host__ __device__ when this header is pulled into a .cu by
// nvcc, and to nothing when the plain C++ host compiler (cl.exe / g++) compiles
// reference_cpu.cpp. One source of truth, two compilers, identical machine code.
#ifdef __CUDACC__
#define IFP_HD __host__ __device__
// `#pragma unroll` is an nvcc-only hint. The plain C++ host compiler (cl.exe)
// would emit a "C4068 unknown pragma" warning, and we treat warnings as defects,
// so we route the unroll through a macro that vanishes off the device.
#define IFP_UNROLL _Pragma("unroll")
#else
#define IFP_HD
#define IFP_UNROLL
#endif

// ---------------------------------------------------------------------------
// PROBLEM DIMENSIONS (fixed at compile time so the bit math fully unrolls)
// ---------------------------------------------------------------------------
// We model a small binding pocket of NUM_RESIDUES residues. For each residue we
// record NUM_ITYPES interaction TYPES (one bit each). The whole IFP is therefore
// NUM_RESIDUES * NUM_ITYPES bits, packed into FP_WORDS 64-bit words.
//
// These are deliberately small (a teaching pocket), but the layout is exactly
// the one ProLIF / SIFt use at full scale -- only the constants grow.
static const int NUM_RESIDUES = 24;   // residues lining the modeled pocket
static const int NUM_ITYPES   = 4;    // interaction types per residue (below)
static const int IFP_BITS      = NUM_RESIDUES * NUM_ITYPES;          // = 96 bits
static const int FP_WORDS      = (IFP_BITS + 63) / 64;               // = 2 words

// The four interaction TYPES, in the bit order used within each residue's nibble.
// (A "nibble" here = NUM_ITYPES consecutive bits belonging to one residue.)
//   0 HYDROPHOBIC : a non-polar contact (any close carbon-like approach)
//   1 HBOND       : a hydrogen bond     (a tighter polar-atom approach)
//   2 AROMATIC    : a pi / ring-stacking contact
//   3 IONIC       : a salt-bridge / charged contact
enum InteractionType { IT_HYDROPHOBIC = 0, IT_HBOND = 1, IT_AROMATIC = 2, IT_IONIC = 3 };

// ---------------------------------------------------------------------------
// GEOMETRIC CRITERIA  (squared distances, in Angstrom^2)
// ---------------------------------------------------------------------------
// We compare SQUARED distances so neither the CPU nor the GPU ever calls sqrt --
// that removes a whole class of last-bit rounding differences and keeps STAGE A
// bit-exact across the two compilers. A bit is set when the ligand's relevant
// atom is closer than the type's cutoff to the residue's interaction center.
//   Cutoffs are the conventional ProLIF/PLIP-style defaults, squared.
//   These are `constexpr` (not just `static const`) so nvcc folds them into a
//   compile-time immediate usable inside __device__ code -- a plain namespace-
//   scope `static const float` is NOT visible in device code (it would error).
constexpr float CUT2_HYDROPHOBIC = 4.5f * 4.5f;   // <= 4.5 A  -> hydrophobic
constexpr float CUT2_HBOND       = 3.5f * 3.5f;   // <= 3.5 A  -> H-bond
constexpr float CUT2_AROMATIC    = 4.0f * 4.0f;   // <= 4.0 A  -> pi contact
constexpr float CUT2_IONIC       = 4.0f * 4.0f;   // <= 4.0 A  -> salt bridge

// A residue's geometry, as seen by STAGE A. Real tools track many atoms per
// residue; for teaching we keep ONE interaction center per residue plus flags
// saying which interaction types this residue's chemistry can even make (an Ala
// cannot H-bond through its side chain; a Lys can form a salt bridge, etc.).
struct Residue {
    float x, y, z;        // interaction-center coordinates (Angstrom)
    int   can_hbond;      // 1 if this residue can donate/accept an H-bond
    int   can_aromatic;   // 1 if this residue has an aromatic ring
    int   can_ionic;      // 1 if this residue is charged (salt-bridge capable)
    // Every residue can make a hydrophobic contact, so there is no flag for it.
};

// A single ligand pose: a handful of pharmacophoric "feature atoms". For a
// teaching pocket we summarize each pose by ONE representative atom of each
// chemistry the ligand carries. The coordinates move from pose to pose (that is
// what "a different binding mode" means); the chemistry flags are constant.
struct Pose {
    float x, y, z;        // representative ligand atom (Angstrom)
    int   has_donor;      // ligand can H-bond here
    int   has_aromatic;   // ligand has an aromatic ring here
    int   has_charge;     // ligand is charged here (for the salt bridge)
};

// ---------------------------------------------------------------------------
// ifp_sqdist : squared Euclidean distance between a pose atom and a residue
//   center. Pure arithmetic, no sqrt -> identical on host and device.
// ---------------------------------------------------------------------------
IFP_HD inline float ifp_sqdist(const Pose& p, const Residue& r) {
    const float dx = p.x - r.x;
    const float dy = p.y - r.y;
    const float dz = p.z - r.z;
    return dx * dx + dy * dy + dz * dz;   // Angstrom^2
}

// ---------------------------------------------------------------------------
// ifp_residue_nibble : compute the NUM_ITYPES interaction bits for ONE
//   (pose, residue) pair and return them packed into the low NUM_ITYPES bits of
//   an int. This is the heart of STAGE A and the single most important function
//   to read. The CPU reference and the GPU kernel BOTH call it, so the IFP they
//   build is guaranteed identical.
//
//   A type's bit is set iff (a) BOTH partners carry the needed chemistry and
//   (b) the squared distance is within that type's cutoff. Using one shared
//   distance keeps the teaching version compact; a production IFP would also
//   check donor-H-acceptor ANGLES for H-bonds and ring-normal angles for
//   pi-stacking (see THEORY "Where this sits in the real world").
// ---------------------------------------------------------------------------
IFP_HD inline int ifp_residue_nibble(const Pose& p, const Residue& r) {
    int bits = 0;
    const float d2 = ifp_sqdist(p, r);

    // HYDROPHOBIC: any sufficiently close contact (both partners always "have"
    // non-polar surface, so there is no chemistry gate -- only the distance).
    if (d2 <= CUT2_HYDROPHOBIC) bits |= (1 << IT_HYDROPHOBIC);

    // HBOND: needs a donor/acceptor on BOTH sides and a tight approach.
    if (r.can_hbond && p.has_donor && d2 <= CUT2_HBOND) bits |= (1 << IT_HBOND);

    // AROMATIC (pi): needs a ring on both sides within the pi cutoff.
    if (r.can_aromatic && p.has_aromatic && d2 <= CUT2_AROMATIC) bits |= (1 << IT_AROMATIC);

    // IONIC (salt bridge): needs opposite-ish charges in range; we model "both
    // are charged" as the gate (sign handling is a documented simplification).
    if (r.can_ionic && p.has_charge && d2 <= CUT2_IONIC) bits |= (1 << IT_IONIC);

    return bits;   // low NUM_ITYPES bits hold this residue's interaction flags
}

// ---------------------------------------------------------------------------
// BIT-VECTOR HELPERS (used by STAGE B, the clustering)
// ---------------------------------------------------------------------------
// Portable 64-bit population count (number of set bits). On the GPU the kernel
// uses the __popcll hardware intrinsic instead (one instruction); on the host
// this loop is the equivalent. We never mix the two on the SAME number, so the
// COUNT is identical either way -- popcount of a fixed bit pattern is exact.
IFP_HD inline int ifp_popcount64(uint64_t v) {
    int c = 0;
    while (v) { v &= (v - 1); ++c; }   // Kernighan: clear lowest set bit each step
    return c;
}

// ---------------------------------------------------------------------------
// ifp_tanimoto_distance : 1 - Tanimoto(A, B) over two FP_WORDS-word bit-vectors.
//   Tanimoto(A,B) = popcount(A & B) / popcount(A | B)  in [0,1]; distance = 1 - it.
//   Two empty fingerprints (union = 0) are defined as distance 0 (identical).
//   Returned as a double so the nearest-centroid argmin compares cleanly; the
//   inputs are integers, so the RATIO is computed from exact integer popcounts.
// ---------------------------------------------------------------------------
IFP_HD inline double ifp_tanimoto_distance(const uint64_t* a, const uint64_t* b) {
    int inter = 0, uni = 0;
    IFP_UNROLL                                   // nvcc: unroll; host: no-op
    for (int w = 0; w < FP_WORDS; ++w) {
        inter += ifp_popcount64(a[w] & b[w]);   // bits set in BOTH
        uni   += ifp_popcount64(a[w] | b[w]);   // bits set in EITHER
    }
    if (uni == 0) return 0.0;                    // empty vs empty -> identical
    const double tan = static_cast<double>(inter) / static_cast<double>(uni);
    return 1.0 - tan;                            // distance in [0,1]
}

// ---------------------------------------------------------------------------
// ifp_nearest_centroid : index of the cluster whose consensus fingerprint is
//   closest (smallest Tanimoto distance) to fingerprint `fp`. Ties resolve to
//   the LOWEST index (strict < update), which makes the assignment deterministic
//   and identical on host and device. This is STAGE B's ASSIGN step for one pose.
// ---------------------------------------------------------------------------
IFP_HD inline int ifp_nearest_centroid(const uint64_t* fp, const uint64_t* centroids, int K) {
    int    best   = 0;
    double best_d = ifp_tanimoto_distance(fp, centroids);
    for (int k = 1; k < K; ++k) {
        const double d = ifp_tanimoto_distance(fp, centroids + static_cast<std::size_t>(k) * FP_WORDS);
        if (d < best_d) { best_d = d; best = k; }
    }
    return best;
}
