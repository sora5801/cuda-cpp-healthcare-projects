// ===========================================================================
// src/shape_overlap.h  --  The ONE shared physics core (CPU == GPU parity)
// ---------------------------------------------------------------------------
// Project 1.13 : Pharmacophore & 3D Shape Screening
//
// WHY THIS HEADER EXISTS  (PATTERNS.md sec 2, the "__host__ __device__" idiom)
//   The single most useful trick in this repo: put the per-element PHYSICS in
//   ONE header as __host__ __device__ inline functions, so the CPU reference
//   (reference_cpu.cpp, compiled by cl.exe) and the GPU kernel (kernels.cu,
//   compiled by nvcc) execute *byte-for-byte identical math*. That turns
//   verification from "are they approximately equal?" into "are they equal to
//   ~machine precision?" -- a far stronger correctness statement.
//
//   To make that work, this header must be includable by BOTH compilers:
//     * It contains NO CUDA-only syntax (no __global__, no <<<>>>). The only
//       CUDA token is the SHAPE_HD decorator, which expands to nothing when the
//       plain C++ host compiler is parsing the file.
//     * All math is done in DOUBLE precision. exp() in double, summed in a
//       FIXED loop order, gives the host and device the same answer to ~1e-12
//       (the only residual difference is FMA contraction; see THEORY sec 5/6).
//
// THE SCIENCE IN ONE PARAGRAPH (full derivation in ../THEORY.md sec 1-2)
//   A molecule's 3D SHAPE is modeled as a sum of spherical Gaussians, one per
//   heavy atom, each centered on the atom and as "fat" as the atom's van der
//   Waals radius. Two molecules are SIMILAR in shape if their Gaussian volumes
//   OVERLAP a lot when superimposed. The overlap integral of two Gaussians has
//   a closed form (no numerical integration needed!), so the whole score is a
//   double sum over atom pairs. The normalized score is the "Shape Tanimoto":
//       ShapeTanimoto = O_AB / (O_AA + O_BB - O_AB)   in [0, 1].
//   This is exactly the quantity OpenEye's ROCS optimizes; we compute it for a
//   rigid (pre-aligned) overlay, which is the GPU-friendly inner kernel.
//
// READ THIS BEFORE: reference_cpu.cpp (loops it on the CPU) and kernels.cu
// (calls it from one GPU thread per conformer). See ../THEORY.md for the math.
// ===========================================================================
#pragma once

#include <cmath>     // std::exp, std::pow  (host); nvcc maps these to device exp/pow
#include <cstddef>   // std::size_t

// ---------------------------------------------------------------------------
// SHAPE_HD: the portability shim. When nvcc compiles this header (it defines
// __CUDACC__), SHAPE_HD becomes "__host__ __device__" so the same inline
// function is emitted for BOTH the CPU and the GPU. When the plain host
// compiler (cl.exe / g++) compiles reference_cpu.cpp, __CUDACC__ is NOT
// defined, so SHAPE_HD expands to nothing and the function is an ordinary
// host inline. One source of truth, two targets.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define SHAPE_HD __host__ __device__
#else
#define SHAPE_HD
#endif

// Maximum atoms per molecule we support in this teaching build. Small molecules
// (the screening use case) have a few dozen heavy atoms; 64 is a generous cap
// that also lets the QUERY fit comfortably in GPU constant memory (64 atoms x
// 4 doubles x 8 bytes = 2 KB, far under the 64 KB constant bank). The data
// loader (reference_cpu.cpp) rejects molecules with more atoms than this.
#define MAX_ATOMS 64

// ---------------------------------------------------------------------------
// Atom: one heavy atom modeled as a single spherical Gaussian.
//   x, y, z : Cartesian coordinates in angstroms (the molecule's frame; the
//             library conformers are assumed PRE-ALIGNED to the query, which is
//             the rigid-overlay simplification -- see THEORY sec 7).
//   alpha   : the Gaussian width parameter (units: 1/angstrom^2). A larger
//             alpha = a NARROWER, more peaked Gaussian = a smaller atom. It is
//             derived from the van der Waals radius in atom_alpha() below.
//   We store alpha (not the radius) so the hot inner loop does zero pow() work.
// ---------------------------------------------------------------------------
struct Atom {
    double x;       // angstrom
    double y;       // angstrom
    double z;       // angstrom
    double alpha;   // 1/angstrom^2  (Gaussian width; see atom_alpha)
};

// A whole molecule = an array of up to MAX_ATOMS atoms plus the count.
//   This POD layout is trivially copyable to the GPU (cudaMemcpy of the bytes)
//   and usable verbatim by the CPU -- exactly what the shared-core idiom needs.
struct Molecule {
    int  n_atoms;            // number of heavy atoms actually used (<= MAX_ATOMS)
    Atom atom[MAX_ATOMS];    // fixed-size storage (no device-side allocation)
};

// ---------------------------------------------------------------------------
// atom_alpha: convert a van der Waals radius (angstrom) to the Gaussian width
// parameter alpha (1/angstrom^2) used everywhere else.
//
//   THE MODEL (Grant & Pickett, J. Comput. Chem. 1996 -- the basis of ROCS):
//   each atom is a single Gaussian rho(r) = p * exp(-alpha * |r - R|^2). The
//   constant p ("partial weight") is fixed at 2.7 by convention so that a
//   single Gaussian reproduces the HARD-SPHERE volume of the atom. Matching the
//   Gaussian's volume to the sphere's volume (4/3) pi r^3 yields:
//
//       alpha = pi * ( 3 p / (4 pi) )^(2/3) * (1 / r^2)
//
//   so alpha scales as 1/r^2: a bigger atom (larger r) is a fatter, lower-alpha
//   Gaussian. We fold the constant prefactor into KAPPA (computed once).
//
//   This is the standard, literature-grounded choice; the comment exists so the
//   number is not a black box (CLAUDE.md sec 6.1.6).
// ---------------------------------------------------------------------------
SHAPE_HD inline double atom_alpha(double radius_angstrom) {
    const double p     = 2.7;                       // Grant-Pickett partial weight
    const double PI    = 3.14159265358979323846;
    // KAPPA = pi * (3p / 4pi)^(2/3). Computed inline (cheap, done once per atom
    // at load time, never in the hot loop).
    const double kappa = PI * std::pow(3.0 * p / (4.0 * PI), 2.0 / 3.0);
    return kappa / (radius_angstrom * radius_angstrom);
}

// ---------------------------------------------------------------------------
// pair_overlap: the CLOSED-FORM overlap integral of two atom-Gaussians.
//
//   The product of two Gaussians is itself a Gaussian, and integrating it over
//   all space gives (THEORY sec 2 derives this):
//
//       V_ij = p^2 * (pi / (a_i + a_j))^(3/2)
//                  * exp( -(a_i a_j)/(a_i + a_j) * d_ij^2 )
//
//   where a_i, a_j are the two alphas and d_ij^2 is the squared distance
//   between the atom centers. The p^2 prefactor (p = 2.7) is the same for every
//   pair; we keep it here so a single atom's self-overlap V_ii has the right
//   magnitude and the Tanimoto ratio is dimensionally consistent.
//
//   Cost: a handful of FLOPs + ONE exp(). This function is called M*K times per
//   conformer pair (M query atoms x K fit atoms), so it is THE hot path -- which
//   is precisely why we hand it to thousands of GPU threads.
//
//   params: ai, aj  = Gaussian widths (1/angstrom^2)
//           d2       = squared center-center distance (angstrom^2)
//   returns: the overlap volume contribution (angstrom^3, up to the p^2 factor)
// ---------------------------------------------------------------------------
SHAPE_HD inline double pair_overlap(double ai, double aj, double d2) {
    const double PI = 3.14159265358979323846;
    const double p  = 2.7;                          // partial weight (matches atom_alpha)
    const double sum = ai + aj;                     // a_i + a_j  (always > 0)
    const double pref = (p * p) * std::pow(PI / sum, 1.5);   // (pi/sum)^{3/2} prefactor
    const double expo = -(ai * aj / sum) * d2;      // Gaussian falloff exponent
    return pref * std::exp(expo);
}

// ---------------------------------------------------------------------------
// molecule_overlap: the first-order Gaussian overlap volume between molecule A
// and molecule B,  O_AB = sum_i sum_j V_ij,  summed over every atom pair.
//
//   "First order" = we add up the pairwise overlaps and STOP. The exact volume
//   would subtract triple-overlap corrections (inclusion-exclusion), but ROCS
//   and most production tools use exactly this first-order sum because the
//   higher terms are tiny for the score's purpose and explode combinatorially.
//   We document that simplification in THEORY sec 3/7 rather than hide it.
//
//   DETERMINISM NOTE: the loop order (i outer, j inner) is FIXED and identical
//   on CPU and GPU, and we accumulate in a double. So the host and device sum
//   the exact same numbers in the exact same order -> the only possible
//   difference is FMA contraction (~1e-12 relative), which our tolerance covers.
//
//   When A == B this returns the SELF-overlap O_AA (needed by the Tanimoto
//   denominator). Self-overlap of an atom with itself uses d2 = 0.
//
//   params: A, B = the two molecules (already in a common coordinate frame)
//   returns: O_AB, the total first-order overlap volume
// ---------------------------------------------------------------------------
SHAPE_HD inline double molecule_overlap(const Molecule& A, const Molecule& B) {
    double total = 0.0;                             // running overlap volume
    for (int i = 0; i < A.n_atoms; ++i) {           // each query atom...
        const double ax = A.atom[i].x;
        const double ay = A.atom[i].y;
        const double az = A.atom[i].z;
        const double ai = A.atom[i].alpha;
        for (int j = 0; j < B.n_atoms; ++j) {       // ...vs each fit atom
            const double dx = ax - B.atom[j].x;
            const double dy = ay - B.atom[j].y;
            const double dz = az - B.atom[j].z;
            const double d2 = dx * dx + dy * dy + dz * dz;   // squared distance
            total += pair_overlap(ai, B.atom[j].alpha, d2);  // add this pair's overlap
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// shape_tanimoto: the normalized, scale-free shape-similarity score in [0, 1].
//
//       ShapeTanimoto(A,B) = O_AB / (O_AA + O_BB - O_AB)
//
//   * 1.0  => the two shapes overlap perfectly (identical volume in space).
//   * 0.0  => the two shapes do not overlap at all (disjoint in space).
//   It is the 3D, continuous analogue of the bit-fingerprint Tanimoto from
//   project 1.12: intersection over union, but of VOLUMES instead of bit sets.
//
//   We pass the three precomputed overlaps so the caller can reuse O_AA (the
//   query self-overlap is the same for every library molecule -- compute once).
//
//   params: o_ab = cross overlap, o_aa = query self-overlap, o_bb = fit self-overlap
//   returns: the Tanimoto ratio, guarded against a zero/negative denominator.
// ---------------------------------------------------------------------------
SHAPE_HD inline double shape_tanimoto(double o_ab, double o_aa, double o_bb) {
    const double denom = o_aa + o_bb - o_ab;        // |A| + |B| - |A^B| = |A v B|
    // Guard the degenerate case (empty molecules): return 0 rather than NaN/inf.
    return (denom > 0.0) ? (o_ab / denom) : 0.0;
}
