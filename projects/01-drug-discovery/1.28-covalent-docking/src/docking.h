// ===========================================================================
// src/docking.h  --  Shared (host + device) covalent-docking geometry + score
// ---------------------------------------------------------------------------
// Project 1.28 : Covalent Docking
//
// WHAT THIS PROJECT COMPUTES (the reduced-scope teaching model)
//   A COVALENT inhibitor is a drug that forms a real chemical bond to a protein
//   residue -- classically the sulfur (S-gamma) of a catalytic CYSTEINE. Famous
//   examples: the KRAS(G12C) drugs (sotorasib), ibrutinib (BTK), afatinib
//   (EGFR). Docking such a ligand is a TWO-STAGE problem:
//     (1) place the reactive "warhead" atom so it can reach the cysteine, then
//     (2) FORM the covalent bond and score the resulting pose, now constrained
//         by the new bond's ideal length and angle.
//   The expensive, parallel part is stage (2): once the warhead is anchored, the
//   rest of the ligand still has FLEXIBLE rotatable bonds (torsions). We must
//   sample many torsion combinations and keep the lowest-energy one. Each
//   combination is an INDEPENDENT scoring job -> one GPU thread per conformation
//   (the same "score N independent candidates" pattern as project 1.12).
//
//   We model the ligand as a short open chain of atoms hanging off a fixed
//   ANCHOR (the warhead carbon, already bonded to the cysteine S-gamma). Each
//   bond between consecutive ligand atoms past the anchor is a rotatable torsion.
//   A CONFORMATION is a tuple of torsion angles (theta_0, theta_1, ...). Forward
//   kinematics turns those angles into 3-D atom positions; an energy function
//   scores the pose. The DOCKED POSE is the argmin-energy conformation.
//
//   The PER-ELEMENT PHYSICS -- forward kinematics + the energy terms -- lives
//   here as __host__ __device__ inline functions so the CPU reference and the
//   GPU kernel run BYTE-FOR-BYTE identical double-precision math. That makes
//   verification exact (DOCK_HD expands to __host__ __device__ under nvcc and to
//   nothing under the host compiler; see docs/PATTERNS.md section 2).
//
// NOT FOR CLINICAL USE. All geometry, parameters, and the "protein pocket" here
// are SYNTHETIC and didactic -- a real covalent docker (CovDock, AutoDock-GPU's
// covalent mode, GNINA) uses full force fields, real PDB structures, reaction
// SMARTS for the warhead chemistry, and MM-GBSA rescoring. See THEORY.md
// "Where this sits in the real world".
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh. The science/math is in
// ../THEORY.md.
// ===========================================================================
#pragma once

// DOCK_HD marks a function as callable from BOTH host and device. Under nvcc
// (__CUDACC__ defined) it becomes "__host__ __device__"; under the plain C++
// compiler that builds reference_cpu.cpp the decorators do not exist, so it
// expands to nothing. This single idiom is what guarantees CPU/GPU parity.
#ifdef __CUDACC__
#define DOCK_HD __host__ __device__
#else
#define DOCK_HD
#endif

#include <cmath>     // std::sqrt, std::sin, std::cos, std::fabs (host + device)
#include <cstddef>   // std::size_t

// ---------------------------------------------------------------------------
// Compile-time problem dimensions.
//   N_TORSIONS  : number of rotatable bonds we sample (degrees of freedom).
//   N_LIG_ATOMS : ligand atoms past the anchor (= N_TORSIONS + 1; each torsion
//                 adds one downstream atom in our simple linear chain).
//   N_POCKET    : number of fixed protein "pocket" atoms the ligand feels.
//   GRID_PER_DOF: how many discrete angle samples per torsion (the sampling
//                 resolution). The total conformation count is GRID_PER_DOF ^
//                 N_TORSIONS -- this exponential blow-up ("the curse of
//                 dimensionality") is exactly why we want the GPU.
// They are COMPILE-TIME constants so loops unroll and the per-thread state
// (atom coordinates) fits in registers/local memory -- no dynamic allocation.
// ---------------------------------------------------------------------------
constexpr int N_TORSIONS  = 3;                 // 3 rotatable bonds (DOF)
constexpr int N_LIG_ATOMS = N_TORSIONS + 1;    // = 4 atoms past the anchor
constexpr int N_POCKET    = 6;                 // 6 fixed pocket atoms
constexpr int GRID_PER_DOF = 36;               // 36 samples -> 10-degree steps

// A 3-D point / vector in angstroms. Trivial aggregate so it lives happily in
// both host std::vectors and device registers.
struct Vec3 {
    double x, y, z;
};

// ---------------------------------------------------------------------------
// A pocket atom the ligand interacts with: a fixed position plus the two
// parameters of a Lennard-Jones-style nonbonded term and a partial charge.
//   sigma   : LJ distance where the potential crosses zero (atom "size"), A.
//   epsilon : LJ well depth (interaction strength), kcal/mol.
//   charge  : partial charge (in electron units) for the electrostatic term.
// In a real docker these come from a force field (e.g. AMBER) keyed by atom
// type; here they are hand-set synthetic values that make the demo legible.
// ---------------------------------------------------------------------------
struct PocketAtom {
    Vec3   pos;
    double sigma;
    double epsilon;
    double charge;
};

// ---------------------------------------------------------------------------
// The full docking problem: the covalent anchor geometry + the flexible-ligand
// chain template + the rigid protein pocket. One struct passed by value to the
// kernel (it is small and POD, so copying it to the device is trivial).
//
//   anchor        : position of the warhead atom, already covalently bonded to
//                   the cysteine S-gamma. FIXED -- stage (1) chose it.
//   sg            : position of the cysteine S-gamma (the bond partner). FIXED.
//   bond_len_ideal: ideal covalent C-S bond length (~1.81 A). The score
//                   penalizes deviation of |anchor-first_atom-...|... actually
//                   the covalent bond is anchor<->sg and is fixed; the penalty
//                   we enforce is on the warhead's APPROACH GEOMETRY: the first
//                   ligand atom should sit at the ideal bond geometry relative
//                   to the anchor and S-gamma (see covalent_penalty()).
//   bond_len_ideal/angle_ideal/k_bond/k_angle : harmonic covalent constraint.
//   seg_len       : length of each rigid ligand segment (bond length), A.
//   bond_angle    : the fixed valence angle between consecutive segments, rad.
//   first_dir     : unit direction of the first segment from the anchor when all
//                   torsions are zero (the reference frame for the chain).
//   pocket        : the N_POCKET fixed protein atoms.
//   lig_sigma/eps/charge : the ligand atoms' own nonbonded parameters.
// ---------------------------------------------------------------------------
struct DockProblem {
    Vec3   anchor;                 // warhead atom (bonded to cysteine), A
    Vec3   sg;                     // cysteine S-gamma position, A
    double bond_len_ideal;         // ideal warhead-Sgamma covalent length, A
    double angle_ideal;            // ideal Sgamma-anchor-firstatom angle, rad
    double k_bond;                 // harmonic force constant for bond length
    double k_angle;                // harmonic force constant for the bond angle
    double seg_len;                // ligand segment (bond) length, A
    double bond_angle;             // valence angle between segments, rad
    Vec3   first_dir;              // unit dir of segment 0 at zero torsion
    PocketAtom pocket[N_POCKET];   // fixed protein pocket atoms
    double lig_sigma;              // ligand-atom LJ sigma, A
    double lig_epsilon;            // ligand-atom LJ epsilon, kcal/mol
    double lig_charge;             // ligand-atom partial charge, e
};

// ---------------------------------------------------------------------------
// Small vector helpers. All inline + DOCK_HD so they compile identically on
// host and device and are free (the compiler inlines them away).
// ---------------------------------------------------------------------------
DOCK_HD inline Vec3 vadd(const Vec3& a, const Vec3& b) { return Vec3{a.x+b.x, a.y+b.y, a.z+b.z}; }
DOCK_HD inline Vec3 vsub(const Vec3& a, const Vec3& b) { return Vec3{a.x-b.x, a.y-b.y, a.z-b.z}; }
DOCK_HD inline Vec3 vscale(const Vec3& a, double s)    { return Vec3{a.x*s, a.y*s, a.z*s}; }
DOCK_HD inline double vdot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
DOCK_HD inline Vec3 vcross(const Vec3& a, const Vec3& b) {
    return Vec3{a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
DOCK_HD inline double vnorm(const Vec3& a) { return std::sqrt(vdot(a, a)); }
// Normalize to unit length; guards the degenerate zero vector so we never
// divide by zero (returns +x as an arbitrary but deterministic fallback).
DOCK_HD inline Vec3 vunit(const Vec3& a) {
    const double n = vnorm(a);
    return (n > 1e-12) ? vscale(a, 1.0 / n) : Vec3{1.0, 0.0, 0.0};
}

// ---------------------------------------------------------------------------
// rotate_about_axis: rotate vector v by angle theta (radians) about a UNIT axis
// k, using Rodrigues' rotation formula
//     v_rot = v cos(theta) + (k x v) sin(theta) + k (k.v)(1 - cos(theta)).
// This is the workhorse of forward kinematics: each torsion rotates the rest of
// the chain about the current bond axis. We use it instead of building 3x3
// matrices because it is branch-free and maps cleanly to registers on the GPU.
// ---------------------------------------------------------------------------
DOCK_HD inline Vec3 rotate_about_axis(const Vec3& v, const Vec3& k, double theta) {
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const Vec3 term1 = vscale(v, c);
    const Vec3 term2 = vscale(vcross(k, v), s);
    const Vec3 term3 = vscale(k, vdot(k, v) * (1.0 - c));
    return vadd(vadd(term1, term2), term3);
}

// ---------------------------------------------------------------------------
// map_conformation_index: turn a FLAT conformation id into its per-torsion angle
// indices, the mixed-radix decomposition id = sum_j a_j * GRID_PER_DOF^j.
//   id        : 0 .. GRID_PER_DOF^N_TORSIONS - 1
//   angle_idx : output array [N_TORSIONS] of integers in [0, GRID_PER_DOF)
// This is the deterministic "thread idx -> which conformation" mapping; the CPU
// loop and the GPU thread use the SAME mapping so member i means the same pose.
// ---------------------------------------------------------------------------
DOCK_HD inline void map_conformation_index(long long id, int angle_idx[N_TORSIONS]) {
    for (int j = 0; j < N_TORSIONS; ++j) {
        angle_idx[j] = static_cast<int>(id % GRID_PER_DOF);  // this DOF's sample
        id /= GRID_PER_DOF;                                   // shift to the next
    }
}

// Total number of conformations on the torsion grid (the search-space size).
// long long because GRID_PER_DOF^N_TORSIONS can exceed 32-bit (36^3 = 46656 is
// fine, but the type documents that the count is exponential in N_TORSIONS).
DOCK_HD inline long long n_conformations() {
    long long total = 1;
    for (int j = 0; j < N_TORSIONS; ++j) total *= GRID_PER_DOF;
    return total;
}

// Convert a per-DOF sample index to its torsion angle in radians. The grid spans
// the full circle [0, 2*pi); sample a maps to a/GRID_PER_DOF of a turn. Using a
// uniform grid (not random) keeps the result fully deterministic.
DOCK_HD inline double sample_to_angle(int a) {
    const double two_pi = 6.283185307179586476925286766559;
    return two_pi * static_cast<double>(a) / static_cast<double>(GRID_PER_DOF);
}

// ---------------------------------------------------------------------------
// build_conformation: FORWARD KINEMATICS.
//   Given the torsion angles, place the N_LIG_ATOMS ligand atoms in space.
//   atom[0] is the first atom past the anchor; we walk the chain placing each
//   next atom by (a) taking the current bond direction, (b) bending it by the
//   fixed valence angle, and (c) twisting it by this bond's torsion angle.
//
//   We keep an orthonormal frame (dir, perp) as we march:
//     * dir  = direction of the bond we just placed (unit).
//     * perp = a fixed perpendicular used to apply the valence-angle bend.
//   The j-th torsion rotates the growing chain about `dir` (the bond axis),
//   which is exactly what a dihedral angle does physically.
//
//   Output: positions[N_LIG_ATOMS] filled with absolute coordinates (A).
//   Mapping note: this is pure per-thread work -- no shared memory, no atomics.
// ---------------------------------------------------------------------------
DOCK_HD inline void build_conformation(const DockProblem& p,
                                       const double torsion[N_TORSIONS],
                                       Vec3 positions[N_LIG_ATOMS]) {
    // Start at the anchor with the reference first-segment direction.
    Vec3 prev = p.anchor;                  // last placed atom (start: the anchor)
    Vec3 dir  = vunit(p.first_dir);        // current bond direction (unit)

    // A perpendicular to `dir`, used to bend by the valence angle. We build it
    // by crossing `dir` with a non-parallel reference axis; if `dir` happens to
    // be along z we cross with x instead (avoids a zero cross product).
    Vec3 ref = (std::fabs(dir.z) < 0.9) ? Vec3{0.0, 0.0, 1.0} : Vec3{1.0, 0.0, 0.0};
    Vec3 perp = vunit(vcross(dir, ref));   // unit vector orthogonal to dir

    // Place atom 0: one segment along `dir` from the anchor.
    positions[0] = vadd(prev, vscale(dir, p.seg_len));
    prev = positions[0];

    // Place the remaining atoms. For each, bend `dir` by the valence angle in
    // the plane spanned by (dir, perp), then twist about the previous bond axis
    // by this segment's torsion angle.
    for (int a = 1; a < N_LIG_ATOMS; ++a) {
        const double tors = torsion[a - 1];      // torsion of the bond into atom a

        // (b) Bend: rotate `dir` toward `perp` by (pi - bond_angle). Using
        //     pi - angle makes a 180-degree valence angle a straight chain.
        const double bend = 3.14159265358979323846 - p.bond_angle;
        Vec3 bent = rotate_about_axis(dir, perp, bend);

        // (c) Twist: rotate the bent direction about the CURRENT bond axis
        //     `dir` by the torsion angle -- this is the rotatable degree of
        //     freedom we are searching over.
        Vec3 newdir = vunit(rotate_about_axis(bent, dir, tors));

        // Place the atom one segment along the new direction.
        positions[a] = vadd(prev, vscale(newdir, p.seg_len));

        // Advance the frame: the new bond becomes `dir`; recompute a fresh
        // perpendicular so the next bend is well-defined.
        prev = positions[a];
        ref  = (std::fabs(newdir.z) < 0.9) ? Vec3{0.0, 0.0, 1.0} : Vec3{1.0, 0.0, 0.0};
        perp = vunit(vcross(newdir, ref));
        dir  = newdir;
    }
}

// ---------------------------------------------------------------------------
// covalent_penalty: STAGE-2 covalent constraint energy (harmonic).
//   After bond formation, the warhead atom must sit at the ideal bond geometry
//   to the cysteine sulfur. We penalize two deviations with springs:
//     * bond LENGTH: |anchor - sg| should equal bond_len_ideal.
//     * bond ANGLE : the S-gamma -- anchor -- first-ligand-atom angle should
//       equal angle_ideal (sp3 geometry, ~109.5 degrees).
//   E_cov = 0.5*k_bond*(len - len0)^2 + 0.5*k_angle*(ang - ang0)^2.
//   This is the term that makes covalent docking DIFFERENT from ordinary
//   docking: it is what "enforces the covalent bond geometry" (catalog 1.28).
// ---------------------------------------------------------------------------
DOCK_HD inline double covalent_penalty(const DockProblem& p, const Vec3& first_atom) {
    // Bond-length spring on the warhead-sulfur distance.
    const double len = vnorm(vsub(p.anchor, p.sg));
    const double dlen = len - p.bond_len_ideal;
    const double e_len = 0.5 * p.k_bond * dlen * dlen;

    // Bond-angle spring on Sgamma-anchor-firstatom. Compute the angle from the
    // dot product of the two bond vectors at the anchor.
    const Vec3 a_to_s = vsub(p.sg, p.anchor);          // anchor -> sulfur
    const Vec3 a_to_f = vsub(first_atom, p.anchor);    // anchor -> first ligand atom
    const double cosang = vdot(a_to_s, a_to_f) /
                          (vnorm(a_to_s) * vnorm(a_to_f) + 1e-12);
    // Clamp to [-1,1] so acos is finite even with round-off at the extremes.
    const double cc = (cosang > 1.0) ? 1.0 : (cosang < -1.0 ? -1.0 : cosang);
    const double ang = std::acos(cc);
    const double dang = ang - p.angle_ideal;
    const double e_ang = 0.5 * p.k_angle * dang * dang;

    return e_len + e_ang;
}

// ---------------------------------------------------------------------------
// nonbonded_energy: the docking "score" proper -- how well the flexible ligand
// fits the pocket. For every (ligand atom, pocket atom) pair we sum:
//   * Lennard-Jones 12-6:  4*eps*[(s/r)^12 - (s/r)^6]  (steric clash + vdW
//     attraction). The r^12 wall punishes overlapping atoms; the -r^6 well
//     rewards good shape complementarity.
//   * Coulomb electrostatics: k * q_i*q_j / r  (a simplified, distance-
//     dependent-free form; real dockers use a dielectric model).
//   Combining geometric (sigma/eps via the Lorentz-Berthelot mixing rules) lets
//   each atom pair use a sensible combined size/strength.
// Lower energy = better pose. This is summed over all ligand atoms, so a single
// conformation's score is O(N_LIG_ATOMS * N_POCKET) -- tiny per thread.
// ---------------------------------------------------------------------------
DOCK_HD inline double nonbonded_energy(const DockProblem& p,
                                       const Vec3 positions[N_LIG_ATOMS]) {
    const double coulomb_k = 332.0636;   // kcal*A/(mol*e^2): Coulomb constant in
                                         // molecular units (converts q.q/r to kcal/mol)
    double e = 0.0;
    for (int a = 0; a < N_LIG_ATOMS; ++a) {
        const Vec3 la = positions[a];
        for (int q = 0; q < N_POCKET; ++q) {
            const PocketAtom& pa = p.pocket[q];
            const Vec3 d = vsub(la, pa.pos);
            double r = vnorm(d);
            if (r < 0.1) r = 0.1;        // floor the distance so r^12 cannot blow
                                         // up to infinity on a coincident overlap
            // Lorentz-Berthelot combining rules: arithmetic mean of sigmas,
            // geometric mean of epsilons -> the pair's effective LJ parameters.
            const double sig = 0.5 * (p.lig_sigma + pa.sigma);
            const double eps = std::sqrt(p.lig_epsilon * pa.epsilon);
            const double sr  = sig / r;
            const double sr6 = sr * sr * sr * sr * sr * sr;   // (sigma/r)^6
            const double sr12 = sr6 * sr6;                    // (sigma/r)^12
            e += 4.0 * eps * (sr12 - sr6);                    // LJ 12-6
            e += coulomb_k * p.lig_charge * pa.charge / r;    // electrostatics
        }
    }
    return e;
}

// ---------------------------------------------------------------------------
// score_conformation: THE ONE TRUE SCORE used by both CPU and GPU.
//   Given a flat conformation id, decode its torsion angles, build the pose by
//   forward kinematics, and return total energy = covalent constraint penalty +
//   nonbonded interaction with the pocket. Lower is better; the docked pose is
//   the argmin over all ids.
//   Because this function and everything it calls are DOCK_HD and use only
//   double precision and the same operation order, the CPU reference and the GPU
//   kernel return bit-identical scores -> exact verification.
// ---------------------------------------------------------------------------
DOCK_HD inline double score_conformation(const DockProblem& p, long long id) {
    int angle_idx[N_TORSIONS];
    map_conformation_index(id, angle_idx);

    double torsion[N_TORSIONS];
    for (int j = 0; j < N_TORSIONS; ++j) torsion[j] = sample_to_angle(angle_idx[j]);

    Vec3 positions[N_LIG_ATOMS];
    build_conformation(p, torsion, positions);

    const double e_cov = covalent_penalty(p, positions[0]);
    const double e_nb  = nonbonded_energy(p, positions);
    return e_cov + e_nb;
}
