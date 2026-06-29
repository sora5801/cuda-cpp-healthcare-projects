// ===========================================================================
// src/conformer.h  --  The ONE shared physics core (CPU + GPU run identical math)
// ---------------------------------------------------------------------------
// Project 1.14 : Conformer Ensemble Generation
//
// WHY THIS HEADER IS THE MOST IMPORTANT FILE IN THE PROJECT
//   This is the "shared __host__ __device__ core" idiom (docs/PATTERNS.md §2).
//   The per-conformer physics -- how torsion angles become 3D atom coordinates,
//   and how those coordinates become a single scalar energy -- lives here ONCE,
//   as plain inline functions tagged __host__ __device__. The CPU reference
//   (reference_cpu.cpp) and the GPU kernel (kernels.cu) BOTH call these exact
//   functions, so they compute byte-for-byte (to within FMA reordering) the same
//   numbers. That is what makes verification meaningful instead of approximate.
//
//   The file is included by BOTH the plain host compiler (cl.exe, when it builds
//   reference_cpu.cpp) AND nvcc (when it builds kernels.cu). To make that work:
//     * the CONF_HD macro expands to "__host__ __device__" under nvcc and to
//       nothing under the host compiler (which has never heard of those words);
//     * we use ONLY <cmath>/<cstdint> and POD math -- no __global__, no CUDA
//       types, no std::vector -- so the host compiler is happy too.
//
// THE TOY MOLECULE WE GENERATE CONFORMERS FOR  (see ../THEORY.md "The science")
//   A real conformer engine (RDKit ETKDG) embeds an arbitrary molecular graph in
//   3D from a distance-geometry guess and refines it with an MMFF94 force field.
//   That is too much machinery for a first lesson, so we teach the SAME IDEAS on
//   a deliberately simple molecule: an unbranched chain of N_ATOMS atoms
//   (think a short alkane like n-octane's carbon backbone). Its shape is fixed
//   except for rotation about its N_TORSION rotatable single bonds -- exactly the
//   degrees of freedom a conformer generator explores.
//
//     * Bond lengths and bond angles are held FIXED (rigid-geometry approximation
//       -- the dominant flexibility of a drug-like molecule is torsional, which is
//       precisely why ETKDG and torsional-diffusion models sample torsions).
//     * Each rotatable bond contributes one DIHEDRAL (torsion) angle phi_t.
//     * A "conformer" is therefore just a vector of N_TORSION torsion angles.
//
//   READ THIS BEFORE: reference_cpu.h, kernels.cuh.  Pairs with THEORY.md.
// ===========================================================================
#pragma once

#include <cmath>     // std::cos, std::sin, std::sqrt  (host) / device intrinsics
#include <cstdint>   // fixed-width ints for the conformer index

// ---------------------------------------------------------------------------
// CONF_HD : make the SAME function compile for the CPU and the GPU.
//   * Under nvcc, __CUDACC__ is defined, so CONF_HD = "__host__ __device__"
//     and nvcc emits both a CPU version and a GPU version of each function.
//   * Under the plain host compiler, those keywords do not exist, so CONF_HD
//     must vanish. This is the canonical CPU/GPU-parity trick (PATTERNS §2).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define CONF_HD __host__ __device__
#else
#define CONF_HD
#endif

// ---------------------------------------------------------------------------
// Compile-time geometry of the toy molecule. These are constexpr so loop bounds
// are known to the compiler (it can unroll), and so device code can size small
// fixed arrays in registers/local memory with no dynamic allocation.
// ---------------------------------------------------------------------------
constexpr int N_ATOMS   = 8;              // atoms in the backbone chain
constexpr int N_TORSION = N_ATOMS - 3;    // dihedrals in a chain of A atoms = A-3
constexpr int N_ROTAMER = 3;              // discrete torsion choices per bond (below)
// Number of conformers we enumerate = N_ROTAMER ^ N_TORSION = 3^5 = 243.
// We expand it with a small integer-power helper so it stays a compile-time const.
constexpr long ipow_ct(int base, int exp) { return exp == 0 ? 1 : base * ipow_ct(base, exp - 1); }
constexpr long N_CONFORMER = ipow_ct(N_ROTAMER, N_TORSION);   // 243

// Fixed internal coordinates of the chain (Angstrom / radian).
//   BOND_LEN  ~ a C-C single bond (1.53 A).
//   BOND_ANG  ~ the ~111 degree tetrahedral-ish backbone angle of an alkane.
// These never change between conformers; only the torsions do.
constexpr double BOND_LEN = 1.53;                       // Angstrom
constexpr double BOND_ANG = 111.0 * 3.14159265358979323846 / 180.0;  // radians

// The three discrete torsion choices ("rotamers"). In a saturated chain the
// staggered minima of the 3-fold torsion potential sit at the classic
// anti / gauche+ / gauche- angles. We sample exactly those, which is what a
// rotamer-library conformer generator does in spirit.
//   index 0 -> anti     (180 deg) : the extended, lowest-torsion-energy choice
//   index 1 -> gauche+  (+60 deg)
//   index 2 -> gauche-  (-60 deg)
CONF_HD inline double rotamer_angle(int r) {
    const double deg2rad = 3.14159265358979323846 / 180.0;
    // A tiny branch is fine here; it is evaluated N_TORSION times per conformer.
    if (r == 0) return 180.0 * deg2rad;   // anti
    if (r == 1) return  60.0 * deg2rad;   // gauche+
    return              -60.0 * deg2rad;  // gauche-
}

// ---------------------------------------------------------------------------
// decode_torsions : turn a flat conformer index into its N_TORSION torsion angles.
//   We enumerate conformers as a mixed-radix (base N_ROTAMER) number: torsion t
//   uses digit t of the index. Conformer 0 is therefore all-anti (the extended
//   chain), which we expect to be the global energy minimum -- a known answer we
//   can verify against (THEORY "How we verify").
//   index : 0 .. N_CONFORMER-1
//   phi   : output array of N_TORSION angles (radians), caller-owned
// ---------------------------------------------------------------------------
CONF_HD inline void decode_torsions(long index, double phi[N_TORSION]) {
    for (int t = 0; t < N_TORSION; ++t) {
        const int digit = static_cast<int>(index % N_ROTAMER);  // rotamer of torsion t
        index /= N_ROTAMER;                                     // shift to next digit
        phi[t] = rotamer_angle(digit);
    }
}

// A minimal 3-vector. We avoid float3/double3 (CUDA-only types) so the struct is
// legal in the plain host compiler too. Plain POD -> lives in registers on device.
struct Vec3 { double x, y, z; };

CONF_HD inline Vec3 vsub(Vec3 a, Vec3 b) { return Vec3{a.x - b.x, a.y - b.y, a.z - b.z}; }
CONF_HD inline double vdot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
CONF_HD inline Vec3 vcross(Vec3 a, Vec3 b) {
    return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
CONF_HD inline double vnorm(Vec3 a) { return std::sqrt(vdot(a, a)); }
CONF_HD inline Vec3 vscale(Vec3 a, double s) { return Vec3{a.x * s, a.y * s, a.z * s}; }
CONF_HD inline Vec3 vadd(Vec3 a, Vec3 b) { return Vec3{a.x + b.x, a.y + b.y, a.z + b.z}; }
// Normalize, guarding the zero vector so a degenerate geometry cannot divide by 0.
CONF_HD inline Vec3 vunit(Vec3 a) {
    const double n = vnorm(a);
    return n > 1e-12 ? vscale(a, 1.0 / n) : Vec3{0.0, 0.0, 0.0};
}

// ---------------------------------------------------------------------------
// build_coords : the "embedding" step -- internal coordinates -> 3D positions.
//   This is the heart of conformer GENERATION. Given the fixed bond lengths and
//   angles plus this conformer's torsions, we PLACE each atom in 3D using the
//   Natural Extension Reference Frame (NeRF / "SN-NeRF") construction -- the same
//   internal-coordinate-to-Cartesian step that sits inside every distance-geometry
//   embedder and every protein-folding internal-coordinate builder.
//
//   The recipe to place atom i (given the three previous atoms A=i-3, B=i-2, C=i-1):
//     1) Start from the bond (C->B) direction, of length BOND_LEN.
//     2) Bend by the bond angle BOND_ANG in the plane of (A,B,C).
//     3) Twist by the dihedral phi about the B->C axis.
//   We build a local orthonormal frame {bc, n, m} at C and express the new atom's
//   offset in it -- a numerically clean, branch-free formulation.
//
//   The first three atoms have no torsion freedom, so we PIN them to a fixed
//   reference geometry. Because every conformer shares those three atoms in the
//   same lab frame, two conformers' coordinate sets are already aligned -- so a
//   plain coordinate RMSD between them is meaningful WITHOUT a Kabsch
//   superposition (a simplification we exploit in the clustering step; see
//   THEORY "How we verify" and the RMSD exercise).
//
//   phi  : N_TORSION torsion angles (radians) from decode_torsions()
//   pos  : output array of N_ATOMS positions (Angstrom), caller-owned
// ---------------------------------------------------------------------------
CONF_HD inline void build_coords(const double phi[N_TORSION], Vec3 pos[N_ATOMS]) {
    // Atom 0 at the origin; atom 1 along +x at one bond length.
    pos[0] = Vec3{0.0, 0.0, 0.0};
    pos[1] = Vec3{BOND_LEN, 0.0, 0.0};
    // Atom 2 placed by bending the (1->0) direction by BOND_ANG in the xy-plane.
    // Interior angle at atom 1 is BOND_ANG, so the turn from the +x axis is
    // (pi - BOND_ANG). We put atom 2 in the xy-plane to seed the frame.
    {
        const double turn = 3.14159265358979323846 - BOND_ANG;
        pos[2] = Vec3{pos[1].x + BOND_LEN * std::cos(turn),
                      pos[1].y + BOND_LEN * std::sin(turn),
                      0.0};
    }
    // Atoms 3..N_ATOMS-1: each consumes one torsion phi[i-3] via NeRF.
    for (int i = 3; i < N_ATOMS; ++i) {
        const Vec3 A = pos[i - 3];
        const Vec3 B = pos[i - 2];
        const Vec3 C = pos[i - 1];

        // bc : unit vector along the bond we are extending from (B -> C).
        const Vec3 bc = vunit(vsub(C, B));
        // n  : unit normal of the plane (A,B,C) -- the axis the dihedral rotates the
        //      new bond out of. Built from the two bond vectors meeting at B.
        const Vec3 n  = vunit(vcross(vsub(B, A), bc));
        // m  : completes a right-handed orthonormal frame {bc, m, n} at C.
        const Vec3 m  = vcross(n, bc);

        // The new bond direction in the local frame: bend by BOND_ANG, twist by phi.
        // d2 lives in the {bc, m, n} basis; we then rotate it into the lab frame.
        const double phi_i = phi[i - 3];
        const double ca = std::cos(BOND_ANG), sa = std::sin(BOND_ANG);
        const double cp = std::cos(phi_i),    sp = std::sin(phi_i);
        // Local offset (length BOND_LEN): -cos(angle) along bc, and the bend spread
        // across m (cos phi) and n (sin phi). The minus sign makes the chain extend
        // forward rather than fold straight back.
        const Vec3 dlab = vadd(vadd(vscale(bc, -BOND_LEN * ca),
                                    vscale(m,   BOND_LEN * sa * cp)),
                               vscale(n,        BOND_LEN * sa * sp));
        pos[i] = vadd(C, dlab);
    }
}

// ---------------------------------------------------------------------------
// torsion_energy : the bonded part of the force field.
//   A saturated single bond has a 3-fold periodic torsion potential. We use the
//   standard MMFF/OPLS form  V(phi) = 0.5 * V3 * (1 + cos(3*phi)) per torsion.
//   It is MINIMIZED at the anti staggered angle (180 deg, where cos(3*180)=cos540=-1),
//   which is why the all-anti conformer (index 0) is the torsional minimum.
//   V3 is a per-bond barrier height in kcal/mol (a typical alkane value).
// ---------------------------------------------------------------------------
CONF_HD inline double torsion_energy(const double phi[N_TORSION]) {
    const double V3 = 2.0;   // kcal/mol, 3-fold barrier height (alkane-like)
    double e = 0.0;
    for (int t = 0; t < N_TORSION; ++t) {
        e += 0.5 * V3 * (1.0 + std::cos(3.0 * phi[t]));
    }
    return e;
}

// ---------------------------------------------------------------------------
// nonbonded_energy : the steric-clash part of the force field (the O(A^2) work).
//   Atoms that come close in space when the chain folds repel each other. We use
//   the repulsive wall of a Lennard-Jones potential, (sigma/r)^12, summed over all
//   atom pairs separated by more than two bonds (1-2 and 1-3 pairs are governed by
//   the fixed bond/angle geometry, so they are excluded -- the standard force-field
//   convention). This term is what makes folded conformers cost energy and what
//   gives the per-conformer work its O(N_ATOMS^2) cost -- the part most worth
//   parallelizing across conformers on the GPU.
//     EPS   : well depth scale (kcal/mol).
//     SIGMA : the van-der-Waals contact distance (Angstrom).
//
//   SOFT CORE -- why we clamp the minimum distance.
//     A bare (sigma/r)^12 wall is a NUMERICAL TRAP: when a conformer folds two
//     atoms almost on top of each other, r -> 0 and (sigma/r)^12 explodes to
//     ~1e14 kcal/mol. At that magnitude, the last-bit difference between the
//     host libm and the device's fused-multiply-add is amplified to whole
//     kcal/mol -- so the CPU and GPU energies would DISAGREE by a lot for those
//     conformers, and exact verification becomes impossible. Real force fields
//     hit the same wall and fix it with a "soft core": below a minimum
//     separation R_MIN the potential saturates to a finite, well-conditioned
//     value instead of diverging. We do exactly that by flooring r^2 at R_MIN^2.
//     This is physically harmless (those clashed conformers are rejected anyway,
//     being astronomically high in energy) and it makes the computation
//     well-conditioned, so host and device agree to ~1e-9 (THEORY "Numerical").
// ---------------------------------------------------------------------------
CONF_HD inline double nonbonded_energy(const Vec3 pos[N_ATOMS]) {
    const double EPS   = 0.10;   // kcal/mol
    const double SIGMA = 3.40;   // Angstrom (carbon-like vdW contact)
    const double R_MIN = 1.50;   // Angstrom: soft-core floor on the separation
    const double R_MIN2 = R_MIN * R_MIN;
    double e = 0.0;
    for (int i = 0; i < N_ATOMS; ++i) {
        for (int j = i + 3; j < N_ATOMS; ++j) {   // skip 1-2 and 1-3 neighbours
            const Vec3 d = vsub(pos[i], pos[j]);
            double r2 = vdot(d, d);
            if (r2 < R_MIN2) r2 = R_MIN2;          // soft core: saturate the wall
            // (sigma^2 / r^2)^6 = (sigma/r)^12, computed from r2 to avoid a sqrt.
            const double s2 = (SIGMA * SIGMA) / r2;
            const double s6 = s2 * s2 * s2;
            e += EPS * s6 * s6;   // (sigma/r)^12 repulsive wall (soft-cored)
        }
    }
    return e;
}

// ---------------------------------------------------------------------------
// conformer_energy : the single scalar each GPU thread (and the CPU loop) produces.
//   This is the WHOLE per-conformer computation, composed from the pieces above:
//     index -> torsions -> 3D coords -> (torsion energy + clash energy).
//   Identical on host and device because every step is a CONF_HD inline call,
//   which is exactly why GPU and CPU agree to ~machine precision (THEORY "verify").
//   Returns the conformer's potential energy in kcal/mol.
//
//   If `out_pos` is non-null we also hand back the 3D coordinates, because the
//   RMSD-clustering step (CPU, in reference_cpu.cpp) needs them. Passing a pointer
//   keeps the energy-only GPU path from paying to store coordinates it discards.
// ---------------------------------------------------------------------------
CONF_HD inline double conformer_energy(long index, Vec3* out_pos = nullptr) {
    double phi[N_TORSION];
    Vec3   pos[N_ATOMS];
    decode_torsions(index, phi);
    build_coords(phi, pos);
    if (out_pos) {
        for (int i = 0; i < N_ATOMS; ++i) out_pos[i] = pos[i];
    }
    return torsion_energy(phi) + nonbonded_energy(pos);
}

// ---------------------------------------------------------------------------
// coord_rmsd : root-mean-square deviation between two conformers' atom positions.
//   Used by the pruning/clustering step to decide whether two conformers are
//   "the same shape". Because all conformers share the first three atoms in the
//   same lab frame (see build_coords), no superposition (Kabsch) is needed here --
//   a deliberate simplification documented in THEORY (the Kabsch version is an
//   exercise). Returns RMSD in Angstrom.
// ---------------------------------------------------------------------------
CONF_HD inline double coord_rmsd(const Vec3 a[N_ATOMS], const Vec3 b[N_ATOMS]) {
    double s = 0.0;
    for (int i = 0; i < N_ATOMS; ++i) {
        const Vec3 d = vsub(a[i], b[i]);
        s += vdot(d, d);
    }
    return std::sqrt(s / static_cast<double>(N_ATOMS));
}
