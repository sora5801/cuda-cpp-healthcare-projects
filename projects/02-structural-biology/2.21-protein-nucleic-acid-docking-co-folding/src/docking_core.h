// ===========================================================================
// src/docking_core.h  --  The ONE TRUE scoring formula (CPU == GPU, bit-exact)
// ---------------------------------------------------------------------------
// Project 2.21 : Protein-Nucleic Acid Docking & Co-Folding  (reduced-scope
//                teaching version -- see README "Limitations & honesty" and
//                THEORY "Where this sits in the real world").
//
// WHY THIS HEADER EXISTS (PATTERNS.md sec 2, the HD-macro idiom)
//   The single most useful trick in this repository: put the per-element
//   PHYSICS in ONE header marked `__host__ __device__`, so the CPU reference
//   (reference_cpu.cpp, compiled by cl.exe/g++) and the GPU kernel (kernels.cu,
//   compiled by nvcc) call the *same* inline functions and therefore compute
//   BYTE-FOR-BYTE IDENTICAL numbers. Verification then becomes an EXACT integer
//   comparison instead of a fuzzy float tolerance.
//
//   To make that exactness real we made two deliberate modelling choices that
//   are also genuine techniques in rigid-body docking:
//
//     1. INTEGER COORDINATES.  Every atom position is stored in fixed-point
//        "milli-Angstrom" units (1 Angstrom = 1000 units), i.e. as int32. All
//        geometry (translate, rotate, squared distance) is then pure integer
//        arithmetic in int64 -- no floating point, so no fused-multiply-add
//        (FMA) divergence between host and device (PATTERNS.md sec 4). Grid /
//        fixed-point representations are exactly how production FFT dockers
//        (ZDOCK, PIPER/ClusPro, HEX) discretise space.
//
//     2. CUBE-GROUP ROTATIONS.  Candidate orientations are drawn from the 24
//        proper rotations of a cube. Their matrices have entries in {-1,0,+1},
//        so rotating an integer coordinate stays an exact integer -- again no
//        trig, no rounding. (A real docker samples orientations far more finely;
//        THEORY explains how and why we simplified.)
//
//   The scoring core (`pair_score` below) is a 3-shell pairwise potential:
//   a hard CLASH penalty when atoms overlap, a CONTACT bonus in the interface
//   shell (shape complementarity), and an ELECTROSTATIC term for that shell
//   (opposite formal charges attract, like charges repel). Everything is summed
//   as int64, so the total score is an exact, order-independent integer.
//
// READ THIS BEFORE: reference_cpu.h, reference_cpu.cpp, kernels.cuh, kernels.cu.
// The science/maths/derivation lives in ../THEORY.md.
// ===========================================================================
#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// HD: the host/device decorator macro (PATTERNS.md sec 2).
//   * Compiled by nvcc (__CUDACC__ defined) -> `__host__ __device__`, so each
//     inline function is emitted for BOTH the CPU and the GPU.
//   * Compiled by the plain host C++ compiler -> expands to nothing, because
//     `__host__`/`__device__` are not C++ keywords there.
//   Keep this header free of CUDA-only types (no __global__, no dim3) so the
//   host compiler can include it unchanged.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// Units & scale.
//   COORD_SCALE: fixed-point resolution. A coordinate value of 1500 means
//   1.500 Angstrom. int32 then spans +/- ~2.1e9 units = +/- ~2.1e6 Angstrom,
//   far larger than any molecule, with no overflow risk for a single axis.
//   Squared distances and score sums are accumulated in int64 (see below).
// ---------------------------------------------------------------------------
static constexpr int32_t COORD_SCALE = 1000;   // units per Angstrom (milli-A)

// ---------------------------------------------------------------------------
// Atom: a single point charge in fixed-point space.
//   x,y,z  : position in milli-Angstrom (int32, exact).
//   charge : formal charge sign, one of {-1, 0, +1}. Real force fields use
//            fractional partial charges; we quantise to a sign so the
//            electrostatic term stays an exact integer (a documented teaching
//            simplification -- THEORY "Numerical considerations").
// This POD struct is shared by host and device; it contains no pointers, so it
// copies trivially to GPU global memory.
// ---------------------------------------------------------------------------
struct Atom {
    int32_t x, y, z;   // fixed-point coordinates (milli-Angstrom)
    int32_t charge;    // -1, 0, or +1  (formal charge sign)
};

// ---------------------------------------------------------------------------
// ScoreParams: the tunable knobs of the pairwise potential, all in fixed-point
// integer units so the whole computation is integer. These are uploaded to the
// GPU unchanged so host and device score with identical thresholds.
//
//   clash_r2   : squared distance (milli-A^2) below which two atoms "clash".
//   contact_r2 : squared distance below which two atoms are "in contact".
//                The interface shell is clash_r2 <= d2 < contact_r2.
//   clash_pen  : score penalty subtracted per clashing pair (a big negative).
//   contact_w  : score bonus added per contacting (non-clashing) pair.
//   elec_w     : electrostatic weight; contributes  -elec_w * (qi*qj) per
//                contacting pair, so OPPOSITE charges (qi*qj = -1) ADD +elec_w
//                (favourable) and LIKE charges (qi*qj = +1) subtract elec_w.
// ---------------------------------------------------------------------------
struct ScoreParams {
    int64_t clash_r2;     // (milli-A)^2
    int64_t contact_r2;   // (milli-A)^2
    int64_t clash_pen;    // subtracted per clashing pair (positive magnitude)
    int64_t contact_w;    // added per contacting pair
    int64_t elec_w;       // electrostatic weight (per contacting pair)
};

// ---------------------------------------------------------------------------
// Rot3: a 3x3 integer rotation matrix (entries in {-1,0,+1} for the cube group).
//   Row-major: applying it to a vector v gives
//       (m[0]*x + m[1]*y + m[2]*z,
//        m[3]*x + m[4]*y + m[5]*z,
//        m[6]*x + m[7]*y + m[8]*z).
//   Because entries are small integers, the product of an int32 coordinate by
//   the matrix stays exactly representable (it cannot overflow int64).
// ---------------------------------------------------------------------------
struct Rot3 {
    int32_t m[9];
};

// ---------------------------------------------------------------------------
// pair_score: the per-atom-pair contribution -- THE single source of truth.
//   Given a protein atom `p` and an ALREADY-TRANSFORMED ligand atom `l`
//   (rotated + translated by the caller), return this pair's integer score:
//       d2 < clash_r2                 ->  -clash_pen           (steric overlap)
//       clash_r2 <= d2 < contact_r2   ->  +contact_w
//                                          - elec_w*(qp*ql)    (interface shell)
//       d2 >= contact_r2              ->   0                   (too far to matter)
//   All arithmetic is int64; the result is exact and order-independent, so the
//   sum over pairs is identical no matter what order the CPU loop or the GPU
//   threads visit pairs in (PATTERNS.md sec 3: integer accumulation = no float
//   non-associativity, so the demo's stdout is reproducible).
//
//   Complexity: O(1) per pair; called Np*Nl times per pose by both backends.
// ---------------------------------------------------------------------------
HD inline int64_t pair_score(const Atom& p, const Atom& l, const ScoreParams& sp) {
    // Differences are taken in int64 to be safe even for far-apart atoms.
    const int64_t dx = (int64_t)p.x - (int64_t)l.x;
    const int64_t dy = (int64_t)p.y - (int64_t)l.y;
    const int64_t dz = (int64_t)p.z - (int64_t)l.z;
    // Squared Euclidean distance in (milli-Angstrom)^2. Max per-axis term is
    // ~(4.2e6)^2 ~= 1.8e13; times 3 stays well within int64's 9.2e18 range.
    const int64_t d2 = dx * dx + dy * dy + dz * dz;

    if (d2 < sp.clash_r2) {
        // Hard overlap: van-der-Waals spheres interpenetrate. Strongly penalise
        // so clashing poses can never out-score a clean interface.
        return -sp.clash_pen;
    }
    if (d2 < sp.contact_r2) {
        // Interface shell: a favourable contact (shape complementarity) plus an
        // electrostatic term. qp*ql is -1 (opposite, attractive), 0 (a neutral
        // atom involved), or +1 (like, repulsive).
        const int64_t qq = (int64_t)p.charge * (int64_t)l.charge;
        return sp.contact_w - sp.elec_w * qq;
    }
    // Beyond the contact shell this pair contributes nothing.
    return 0;
}

// ---------------------------------------------------------------------------
// apply_pose: transform a ligand atom by a rigid pose (rotate, then translate).
//   in   : the ligand atom in its reference frame.
//   R    : a cube-group rotation (integer matrix).
//   tx,ty,tz : translation in milli-Angstrom (int32).
//   returns the transformed atom (charge is carried through unchanged).
//   Pure integer math -> bit-identical on host and device. This is the other
//   half of the "one true formula": both backends pose atoms the same way.
// ---------------------------------------------------------------------------
HD inline Atom apply_pose(const Atom& in, const Rot3& R,
                          int32_t tx, int32_t ty, int32_t tz) {
    Atom o;
    // Matrix-vector product with the integer rotation, then add the translation.
    o.x = R.m[0] * in.x + R.m[1] * in.y + R.m[2] * in.z + tx;
    o.y = R.m[3] * in.x + R.m[4] * in.y + R.m[5] * in.z + ty;
    o.z = R.m[6] * in.x + R.m[7] * in.y + R.m[8] * in.z + tz;
    o.charge = in.charge;   // a rigid move does not change an atom's charge
    return o;
}

// ---------------------------------------------------------------------------
// score_pose: the full score of ONE candidate pose.
//   protein : array of Np protein atoms (fixed in space).
//   Np      : number of protein atoms.
//   ligand  : array of Nl ligand (nucleic-acid) atoms in their reference frame.
//   Nl      : number of ligand atoms.
//   R, tx,ty,tz : the rigid pose to evaluate.
//   sp      : scoring thresholds/weights.
//   returns the int64 interface score (higher = better complementarity).
//
//   This is the kernel of the whole project: it is the SAME function the CPU
//   reference loops over all poses and the GPU kernel calls once per thread.
//   Cost: O(Np*Nl) integer pair evaluations.
// ---------------------------------------------------------------------------
HD inline int64_t score_pose(const Atom* protein, int Np,
                             const Atom* ligand, int Nl,
                             const Rot3& R, int32_t tx, int32_t ty, int32_t tz,
                             const ScoreParams& sp) {
    int64_t total = 0;
    // Transform each ligand atom once, then test it against every protein atom.
    // (Transforming inside the inner loop would recompute the same atom Np
    // times; we hoist it out -- a small but real optimisation that the GPU
    // kernel mirrors exactly so the two stay bit-identical.)
    for (int j = 0; j < Nl; ++j) {
        const Atom lj = apply_pose(ligand[j], R, tx, ty, tz);
        for (int i = 0; i < Np; ++i) {
            total += pair_score(protein[i], lj, sp);
        }
    }
    return total;
}
