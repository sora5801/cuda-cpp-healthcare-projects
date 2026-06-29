// ===========================================================================
// src/reference_cpu.h  --  Data model + shared physics + CPU reference
// ---------------------------------------------------------------------------
// Project 1.27 : MM-GBSA / MM-PBSA Rescoring
//
// WHY THIS HEADER IS THE HEART OF THE PROJECT
//   MM-GBSA estimates a protein-ligand BINDING FREE ENERGY by averaging a
//   per-snapshot energy over many frames of a molecular-dynamics (MD)
//   trajectory. Every snapshot is an INDEPENDENT job, which is exactly the
//   shape a GPU loves: one thread evaluates one snapshot's energy. To make the
//   GPU result verifiable, the CPU reference and the GPU kernel must run the
//   *same arithmetic*. We guarantee that by putting the per-snapshot physics in
//   ONE place -- the GB_HD-decorated inline function `snapshot_dg()` below --
//   that is compiled for BOTH the host (by cl.exe/g++) and the device (by nvcc).
//   This is the "shared __host__ __device__ core" idiom (docs/PATTERNS.md §2):
//   it makes CPU == GPU agreement EXACT instead of approximate.
//
//   This header is pure-ish C++: the only CUDA token it uses is the GB_HD macro,
//   which expands to NOTHING under the host compiler. So reference_cpu.cpp (host)
//   and kernels.cu (device) can both include it with no leakage either way.
//
// WHAT MM-GB(PB)SA COMPUTES  (see ../THEORY.md for the full derivation)
//   For one snapshot of the complex, the binding free energy estimate is
//       dG = E_vdw  +  E_elec  +  dG_GB  +  (-T*dS)
//   where the first three are summed over all receptor-atom / ligand-atom PAIRS:
//     * E_elec : Coulomb electrostatics  332.06 * q_i q_j / r        [kcal/mol]
//     * E_vdw  : Lennard-Jones 12-6       4 eps[(sig/r)^12-(sig/r)^6][kcal/mol]
//     * dG_GB  : Generalized-Born pair (implicit-solvent) cross term,
//                the Still/Hawkins-Cramer-Truhlar form (THEORY §The math).
//   The MM-GBSA binding free energy is the MEAN of dG over all snapshots.
//   (The configurational entropy -T*dS is a separate, expensive estimate; we
//   fold a single constant -T*dS into the result and explain why in THEORY.)
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh, kernels.cu.
// ===========================================================================
#pragma once

#include <cmath>     // std::sqrt, std::exp (host path of the math shims below)
#include <cstddef>   // std::size_t
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// GB_HD : the host/device portability macro (docs/PATTERNS.md §2).
//   * When nvcc compiles a .cu translation unit it defines __CUDACC__, so GB_HD
//     becomes `__host__ __device__` and the function is emitted for BOTH the CPU
//     and the GPU.
//   * When the plain host compiler builds reference_cpu.cpp, __CUDACC__ is NOT
//     defined, so GB_HD expands to nothing and the decorators (which the host
//     compiler does not understand) simply vanish.
//   The net effect: snapshot_dg() is literally the same source on both sides.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define GB_HD __host__ __device__
#else
#define GB_HD
#endif

// ---------------------------------------------------------------------------
// sqrtval / expval : tiny shims so the shared physics compiles on BOTH sides.
//   On the device (when __CUDA_ARCH__ is defined, i.e. we are generating GPU
//   code) we call CUDA's built-in double-precision sqrt/exp; on the host we call
//   std::sqrt/std::exp. Routing through these inline GB_HD wrappers keeps
//   snapshot_dg() free of #ifdefs in its body. They are defined FIRST so they
//   are visible where snapshot_dg() uses them.
//   Determinism note: IEEE-754 sqrt is correctly rounded and identical on host
//   and device; exp is the only term that *could* differ by ~1 ULP between
//   libm and CUDA math, which is why our tolerance is ~1e-9, not exactly 0
//   (THEORY §How we verify correctness; docs/PATTERNS.md §4).
// ---------------------------------------------------------------------------
GB_HD inline double sqrtval(double v) {
#ifdef __CUDA_ARCH__
    return sqrt(v);          // device: CUDA math (double-precision hardware sqrt)
#else
    return std::sqrt(v);     // host: <cmath>
#endif
}
GB_HD inline double expval(double v) {
#ifdef __CUDA_ARCH__
    return exp(v);           // device: CUDA math
#else
    return std::exp(v);      // host: <cmath>
#endif
}

// ---------------------------------------------------------------------------
// Physical constants (kcal/mol/angstrom unit system, the AMBER convention).
// ---------------------------------------------------------------------------
//   COULOMB_K : Coulomb's constant in these units. In SI, E = k q1 q2 / r; in
//     AMBER units with charges in electrons and distance in angstroms, the
//     prefactor that yields kcal/mol is 332.0637. Every MD/MM code uses ~this.
constexpr double COULOMB_K = 332.0637;

//   The Generalized-Born "screening" prefactor is -COULOMB_K*(1/eps_in -
//   1/eps_water). We keep COULOMB_K and the two dielectrics separate so the
//   formula reads like the textbook one (THEORY §The math).
constexpr double EPS_WATER  = 78.5;    // bulk water relative permittivity (~80)
constexpr double EPS_SOLUTE = 1.0;     // interior (vacuum-like) dielectric

// ---------------------------------------------------------------------------
// Atom : a single point charge with a Lennard-Jones size and a GB Born radius.
//   x,y,z   : position in angstroms.
//   q       : partial charge in units of the elementary charge e.
//   sigma   : LJ distance parameter (angstroms) -- where the 12-6 potential is 0.
//   eps     : LJ well depth (kcal/mol) -- the strength of the dispersion well.
//   born    : effective Born radius (angstroms) for the GB solvation model;
//             larger born => the atom is more buried => less solvent screening.
// This 8-double layout is deliberately simple and self-describing so a learner
// can see every parameter that enters the energy. Real force fields (AMBER ff19,
// GAFF2) carry the same quantities per atom; we just expose them plainly.
// ---------------------------------------------------------------------------
struct Atom {
    double x, y, z;   // position [angstrom]
    double q;         // partial charge [e]
    double sigma;     // LJ sigma [angstrom]
    double eps;       // LJ well depth [kcal/mol]
    double born;      // GB effective Born radius [angstrom]
};

// ---------------------------------------------------------------------------
// Complex : the whole problem -- a rigid receptor, plus S snapshots of the
// ligand (each snapshot = one MD frame in which the ligand has moved).
//
//   receptor          : R atoms, held fixed across snapshots (a teaching
//                       simplification; see THEORY §Where this sits). Flattened
//                       as one std::vector<Atom> of length R.
//   ligand_snapshots  : S * L atoms, ROW-MAJOR: snapshot s occupies
//                       ligand_snapshots[s*L .. s*L + L-1]. Each snapshot is a
//                       full copy of the L ligand atoms at frame s.
//   R, L, S           : counts (receptor atoms, ligand atoms, snapshots).
//   minus_TdS         : the single constant entropy penalty -T*dS [kcal/mol]
//                       added to every snapshot's dG (THEORY explains why we use
//                       one constant instead of a per-frame normal-mode estimate).
//
// Why flatten the ligand snapshots into one contiguous array? Because that is
// exactly the layout we upload to the GPU: snapshot s's atoms sit together, so
// the thread that owns snapshot s reads a compact, contiguous block.
// ---------------------------------------------------------------------------
struct Complex {
    int R = 0;                          // receptor atom count
    int L = 0;                          // ligand atom count (per snapshot)
    int S = 0;                          // number of MD snapshots
    double minus_TdS = 0.0;             // constant entropy term [kcal/mol]
    std::vector<Atom> receptor;         // [R]
    std::vector<Atom> ligand_snapshots; // [S * L], row-major by snapshot
};

// ---------------------------------------------------------------------------
// snapshot_dg : THE ONE TRUE FORMULA. Compute the binding free energy estimate
// dG for a SINGLE snapshot, by summing the receptor-ligand interaction over
// every (ligand atom i, receptor atom j) pair. This is GB_HD so the CPU loop and
// the GPU kernel call the identical code (=> bit-near agreement, THEORY §How we
// verify correctness).
//
//   receptor    : pointer to R receptor Atoms (device or host memory).
//   R           : receptor atom count.
//   ligand      : pointer to this snapshot's L ligand Atoms (i.e. the base of
//                 ligand_snapshots + s*L).
//   L           : ligand atom count.
//   minus_TdS   : constant entropy term added once to this snapshot's dG.
//
// Returns dG for the snapshot in kcal/mol. O(R*L) pair work per snapshot.
//
// NOTE ON ORDERING (determinism): we sum pairs in a FIXED nested order
// (i = 0..L-1 outer, j = 0..R-1 inner). Floating-point addition is not
// associative, so a fixed order is what makes the host and device totals match.
// Each thread computes ONE snapshot's whole sum in this order; no cross-thread
// atomics are involved, so there is no reduction nondeterminism.
// ---------------------------------------------------------------------------
GB_HD inline double snapshot_dg(const Atom* receptor, int R,
                                const Atom* ligand,   int L,
                                double minus_TdS) {
    // The GB screening prefactor: -k*(1/eps_in - 1/eps_w). With eps_w ~ 78.5 and
    // eps_in = 1 this is a large NEGATIVE number, so the GB term is a (favorable)
    // solvent screening of the charge-charge interaction. Computed once.
    const double gb_pre = -COULOMB_K * (1.0 / EPS_SOLUTE - 1.0 / EPS_WATER);

    double e_elec = 0.0;   // running Coulomb electrostatic energy [kcal/mol]
    double e_vdw  = 0.0;   // running Lennard-Jones (vdW) energy   [kcal/mol]
    double e_gb   = 0.0;   // running Generalized-Born solvation   [kcal/mol]

    // Outer loop over ligand atoms; inner loop over receptor atoms. Each pair
    // contributes to all three energy components. This double loop is the whole
    // cost of a snapshot: O(R*L). For our teaching sizes (tens of atoms) that is
    // tiny per snapshot, but multiplied by thousands of snapshots it is real work
    // -- which is exactly why we parallelize across snapshots on the GPU.
    for (int i = 0; i < L; ++i) {
        const Atom li = ligand[i];      // this ligand atom (by value, 56 B)
        for (int j = 0; j < R; ++j) {
            const Atom rj = receptor[j]; // this receptor atom

            // --- pairwise geometry -------------------------------------------
            const double dx = li.x - rj.x;
            const double dy = li.y - rj.y;
            const double dz = li.z - rj.z;
            const double r2 = dx * dx + dy * dy + dz * dz;  // squared distance
            // Guard against divide-by-zero if two atoms coincide. Our synthetic
            // geometry never overlaps receptor and ligand, but a real trajectory
            // can momentarily clash, so we clamp r2 to a tiny floor. (A ?: keeps
            // this callable from device code without pulling in <algorithm>.)
            const double r2c = r2 > 1e-12 ? r2 : 1e-12;
            const double r   = sqrtval(r2c);                // |r_ij|

            // --- electrostatics: Coulomb 332*q_i*q_j / r ---------------------
            const double qq = li.q * rj.q;                  // charge product [e^2]
            e_elec += COULOMB_K * qq / r;

            // --- van der Waals: Lennard-Jones 12-6 ---------------------------
            // Combine per-atom sigma/eps with the standard Lorentz-Berthelot
            // mixing rules: sigma_ij = (sig_i+sig_j)/2, eps_ij = sqrt(eps_i eps_j).
            const double sig_ij = 0.5 * (li.sigma + rj.sigma);
            const double eps_ij = sqrtval(li.eps * rj.eps);
            const double sr2    = (sig_ij * sig_ij) / r2c;  // (sigma/r)^2
            const double sr6    = sr2 * sr2 * sr2;          // (sigma/r)^6
            const double sr12   = sr6 * sr6;                // (sigma/r)^12
            e_vdw += 4.0 * eps_ij * (sr12 - sr6);

            // --- Generalized-Born solvation cross term -----------------------
            // Still's effective Born interaction distance:
            //   f_GB = sqrt(r^2 + R_i R_j * exp(-r^2 / (4 R_i R_j))).
            // At short range f_GB -> sqrt(R_i R_j) (self/near limit); at long
            // range f_GB -> r (it reduces to a screened Coulomb). The GB energy
            // of the pair is gb_pre * q_i q_j / f_GB.  (THEORY §The math.)
            const double RR    = li.born * rj.born;         // R_i * R_j
            const double f_gb  = sqrtval(r2c + RR * expval(-r2c / (4.0 * RR)));
            e_gb += gb_pre * qq / f_gb;
        }
    }
    // The snapshot's binding free-energy estimate: enthalpic MM terms + GB
    // solvation + the constant entropy penalty. This single scalar is what each
    // GPU thread writes out, and what the host reference computes per snapshot.
    return e_elec + e_vdw + e_gb + minus_TdS;
}

// ---------------------------------------------------------------------------
// HOST-SIDE declarations: data loading + the CPU reference rescoring.
// These are implemented in reference_cpu.cpp (compiled by the host compiler).
// ---------------------------------------------------------------------------

// Load a Complex from the text format documented in data/README.md. Throws
// std::runtime_error on a missing file or a malformed/short body.
Complex load_complex(const std::string& path);

// CPU reference: fill dg[s] with snapshot_dg() for every snapshot s, in order.
// This is the trusted serial baseline the GPU kernel is verified against, and
// the timing baseline that makes the speed-up legible. dg is resized to S.
void rescore_cpu(const Complex& cx, std::vector<double>& dg);

// mean : the MM-GBSA binding free-energy estimate = average of the per-snapshot
// dG values, accumulated in a fixed order so host and device agree. Pulled out
// so main.cu and the CPU path share one definition.
double mean(const std::vector<double>& v);
