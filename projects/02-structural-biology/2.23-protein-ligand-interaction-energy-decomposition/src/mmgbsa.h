// ===========================================================================
// src/mmgbsa.h  --  Shared physics core + data model (CPU/GPU PARITY header)
// ---------------------------------------------------------------------------
// Project 2.23 : Protein-Ligand Interaction Energy Decomposition
//
// WHY THIS HEADER EXISTS  (PATTERNS.md sec 2 -- the HD-macro idiom)
//   The per-atom-pair energy formula (Coulomb + Lennard-Jones + a Generalized
//   Born desolvation term) is the ONE piece of math that must be IDENTICAL on
//   the CPU reference and the GPU kernel, or verification would compare two
//   subtly different computations. So we put it in this single header as
//   `__host__ __device__` inline functions:
//     * nvcc compiles it for BOTH the host (reference_cpu.cpp includes it through
//       the host compiler) AND the device (kernels.cu includes it for the GPU).
//     * The MMGBSA_HD macro expands to `__host__ __device__` under nvcc and to
//       nothing under a plain C++ compiler, so the SAME source compiles in both
//       worlds. (reference_cpu.cpp is compiled by cl.exe / g++ -- no CUDA syntax
//       may leak in, hence the macro and NO `__global__` here.)
//   Result: CPU and GPU run byte-for-byte the same formula -> verification is a
//   tight tolerance check, not a hand-wave (THEORY.md "How we verify").
//
// WHAT WE MODEL  (a deliberately REDUCED-SCOPE teaching version; THEORY.md
//                 "Where this sits in the real world" lists every simplification)
//   Per-residue MM-GBSA energy decomposition: for each protein residue r we sum
//   its interaction energy with every ligand atom, over a trajectory of frames,
//   and split that energy into physical COMPONENTS:
//       E_elec  : Coulomb electrostatics (screened, distance-dependent)
//       E_vdw   : Lennard-Jones 12-6 van der Waals
//       E_gb    : Generalized Born pairwise desolvation (implicit solvent)
//   The per-residue total  E_r = E_elec + E_vdw + E_gb  (averaged over frames)
//   is the "contribution" that flags HOT-SPOT residues for mutational scanning
//   (the oncology kinase-resistance use case in the catalog deep dive).
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh.  Then kernels.cu.
// The science -> math -> GPU mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <cstddef>   // std::size_t
#include <cmath>     // std::sqrt, std::exp  (host side; device uses the builtins)
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// MMGBSA_HD : the host/device decorator macro.
//   * Under nvcc (__CUDACC__ defined) it becomes `__host__ __device__`, so each
//     inline function is emitted for BOTH the CPU and the GPU.
//   * Under a plain C++ compiler the decorators do not exist, so the macro
//     expands to nothing and the function is an ordinary host inline.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define MMGBSA_HD __host__ __device__
#else
#define MMGBSA_HD
#endif

// ---------------------------------------------------------------------------
// Physical constants (kept in ONE place so CPU and GPU share the exact values).
// Units throughout: distances in Angstrom (A), charges in elementary charge (e),
// energies in kcal/mol. These are the conventional AMBER-style units so the
// numbers are recognisable to anyone who has run MMPBSA.py.
// ---------------------------------------------------------------------------

// Coulomb prefactor in AMBER units: E_coul = COULOMB_K * q_i q_j / r, with q in
// e, r in A, E in kcal/mol. 332.0636 kcal*A/(mol*e^2) is the standard value.
constexpr double COULOMB_K = 332.0636;

// Interior (solute) and exterior (solvent/water) dielectric constants for the
// Generalized Born implicit-solvent model. eps_in ~ 1 (vacuum-like protein
// interior), eps_out ~ 80 (bulk water). The GB term rewards burying charge.
constexpr double EPS_IN  = 1.0;
constexpr double EPS_OUT = 80.0;

// ---------------------------------------------------------------------------
// pair_energy_components: the heart of the model. Returns nothing; writes the
// three energy components for ONE protein-atom / ligand-atom pair into the
// out-params. Called by both the CPU reference and the GPU kernel.
//
//   r2      : squared interatomic distance (A^2). We pass r^2 (not r) so the
//             caller can cheaply reject pairs beyond a cutoff without a sqrt,
//             and so we take exactly one sqrt here.
//   qi, qj  : partial charges of the two atoms (e).
//   eps_ij  : Lennard-Jones well depth for the pair (kcal/mol), already combined
//             via the Lorentz-Berthelot rule eps_ij = sqrt(eps_i * eps_j).
//   rmin_ij : Lennard-Jones distance at the energy minimum (A), combined as
//             rmin_ij = rmin_i/2 + rmin_j/2 (AMBER stores per-atom Rmin/2).
//   born_i,
//   born_j  : effective Born radii of the two atoms (A) -- how buried each atom
//             is. Larger radius = more shielded from solvent.
//   e_elec,
//   e_vdw,
//   e_gb    : OUT params, the three energy components for this pair (kcal/mol).
//
// WHY split into components: medicinal chemists read the decomposition to see
// WHETHER a residue helps via electrostatics (salt bridge / H-bond) or shape
// (vdW packing) -- that distinction guides which mutation to try. Returning the
// three pieces (not just the sum) is the whole point of "decomposition".
// ---------------------------------------------------------------------------
MMGBSA_HD inline void pair_energy_components(double r2,
                                             double qi, double qj,
                                             double eps_ij, double rmin_ij,
                                             double born_i, double born_j,
                                             double& e_elec,
                                             double& e_vdw,
                                             double& e_gb)
{
    // One sqrt for the whole pair: r = |x_i - x_j|.
    const double r = sqrt(r2);

    // --- (1) Coulomb electrostatics -------------------------------------
    // E_elec = K * q_i q_j / (eps_in * r). We screen by the interior dielectric
    // (the *reaction-field* screening by water is handled separately by the GB
    // term below -- a standard split in implicit-solvent MM-GBSA).
    e_elec = COULOMB_K * qi * qj / (EPS_IN * r);

    // --- (2) Lennard-Jones 12-6 van der Waals ---------------------------
    // E_vdw = eps * [ (rmin/r)^12 - 2 (rmin/r)^6 ]. The r^-12 wall is Pauli
    // repulsion (atoms cannot overlap); the -2 r^-6 well is London dispersion
    // (induced-dipole attraction). We build the 6th power once and square it for
    // the 12th -> fewer multiplies and better numerical behaviour than pow().
    const double ratio2 = (rmin_ij * rmin_ij) / r2;   // (rmin/r)^2
    const double ratio6 = ratio2 * ratio2 * ratio2;   // (rmin/r)^6
    const double ratio12 = ratio6 * ratio6;           // (rmin/r)^12
    e_vdw = eps_ij * (ratio12 - 2.0 * ratio6);

    // --- (3) Generalized Born pairwise desolvation ----------------------
    // The GB pair energy approximates the electrostatic solvation free energy:
    //   E_gb = -K * (1/eps_in - 1/eps_out) * q_i q_j / f_GB
    //   f_GB = sqrt( r^2 + R_i R_j * exp( -r^2 / (4 R_i R_j) ) )           (Still)
    // f_GB smoothly interpolates between r (far apart) and the Born radii (when
    // the atoms overlap). The (1/eps_in - 1/eps_out) factor is the cost of
    // moving charge from water into the low-dielectric protein interior -- this
    // is what penalises desolvating a charged residue on binding.
    const double RiRj = born_i * born_j;
    const double f_gb = sqrt(r2 + RiRj * exp(-r2 / (4.0 * RiRj)));
    e_gb = -COULOMB_K * (1.0 / EPS_IN - 1.0 / EPS_OUT) * qi * qj / f_gb;
}

// ===========================================================================
// DATA MODEL  (pure C++: usable by the host compiler and by nvcc alike)
// ---------------------------------------------------------------------------
// A "system" is: M protein residues (each one bead with charge/LJ/Born params),
// L ligand atoms, and F trajectory frames giving 3-D coordinates for everyone.
// We use a one-bead-per-residue coarse graining (the residue's representative
// atom, e.g. the C-alpha or a charge centroid) so the teaching example stays
// small and readable; THEORY.md explains the all-atom generalisation.
// ===========================================================================

// Per-residue force-field parameters (constant across the trajectory).
//   These mirror the columns an AMBER prmtop would store per atom; here one set
//   per residue bead. `name` is for the human-readable report only.
struct ResidueParams {
    double charge;     // partial charge q (e)
    double eps;        // LJ well depth eps (kcal/mol)  [per-atom, pre-combine]
    double rmin_half;  // LJ Rmin/2 (A)                 [per-atom, pre-combine]
    double born;       // effective Born radius (A)
    char   name[8];    // residue label e.g. "ASP45" (display only; fixed-size so
                       // the struct is trivially copyable to the GPU)
};

// Per-ligand-atom force-field parameters (constant across the trajectory).
struct LigandParams {
    double charge;     // partial charge q (e)
    double eps;        // LJ well depth eps (kcal/mol)
    double rmin_half;  // LJ Rmin/2 (A)
    double born;       // effective Born radius (A)
};

// A loaded system + its trajectory. All coordinate arrays are FLAT and
// ROW-MAJOR so they upload to the GPU as a single contiguous block.
//
//   res_xyz : [F * M * 3], frame-major then residue-major: the (x,y,z) of
//             residue m in frame f is at res_xyz[((f*M)+m)*3 + {0,1,2}].
//   lig_xyz : [F * L * 3], same layout for ligand atoms.
// Keeping frames as the OUTERMOST index means "all atoms of one frame" are
// contiguous, which matches how an MD engine writes a trajectory and how the
// kernel reads one frame at a time.
struct MmgbsaSystem {
    int F = 0;                         // number of trajectory frames (snapshots)
    int M = 0;                         // number of protein residues
    int L = 0;                         // number of ligand atoms
    double cutoff = 0.0;               // interaction cutoff (A); pairs beyond -> 0
    std::vector<ResidueParams> res;    // [M]
    std::vector<LigandParams>  lig;     // [L]
    std::vector<double> res_xyz;       // [F*M*3]
    std::vector<double> lig_xyz;       // [F*L*3]
};

// Per-residue decomposition result: the trajectory-AVERAGED energy of each
// residue with the whole ligand, split into the three components plus the sum.
// One PerResidueEnergy per residue is the project's headline output.
struct PerResidueEnergy {
    double elec = 0.0;   // <E_elec> over frames (kcal/mol)
    double vdw  = 0.0;   // <E_vdw>  over frames (kcal/mol)
    double gb   = 0.0;   // <E_gb>   over frames (kcal/mol)
    double total = 0.0;  // elec + vdw + gb (kcal/mol)
};

// ---------------------------------------------------------------------------
// residue_frame_energy: compute residue `m`'s interaction energy with the WHOLE
// ligand in ONE frame `f`, accumulating the three components. This is the unit
// of work that the GPU parallelises (one thread per residue, looping frames) and
// that the CPU reference loops serially -- defined here so BOTH call the exact
// same code (CPU/GPU parity, PATTERNS.md sec 2).
//
//   sys-like inputs are passed as raw pointers/scalars (not the MmgbsaSystem
//   struct) so this function is callable from device code, where we hand it the
//   uploaded GPU arrays. Pointers are to the FLAT arrays described above.
//
//   res_params : [M] residue force-field params
//   lig_params : [L] ligand force-field params
//   res_xyz_f  : pointer to THIS frame's residue coords  = res_xyz + f*M*3
//   lig_xyz_f  : pointer to THIS frame's ligand  coords  = lig_xyz + f*L*3
//   m, L       : residue index and ligand-atom count
//   cutoff2    : squared cutoff (A^2); pairs with r^2 > cutoff2 contribute 0
//   out_elec/vdw/gb : accumulators ADDED INTO (caller zeroes them once)
//
// Determinism note: the ligand loop runs in fixed index order on both sides, and
// double-precision sums of the same terms in the same order match to ~1e-13;
// the small residual we tolerate comes only from FMA contraction (THEORY).
// ---------------------------------------------------------------------------
MMGBSA_HD inline void residue_frame_energy(const ResidueParams* res_params,
                                           const LigandParams*  lig_params,
                                           const double* res_xyz_f,
                                           const double* lig_xyz_f,
                                           int m, int L,
                                           double cutoff2,
                                           double& out_elec,
                                           double& out_vdw,
                                           double& out_gb)
{
    // This residue's bead coordinates and parameters (loaded once, reused for
    // every ligand atom -> registers, not repeated global reads).
    const double rx = res_xyz_f[m * 3 + 0];
    const double ry = res_xyz_f[m * 3 + 1];
    const double rz = res_xyz_f[m * 3 + 2];
    const ResidueParams rp = res_params[m];

    // Walk every ligand atom and accumulate the pair energy components.
    for (int a = 0; a < L; ++a) {
        const double lx = lig_xyz_f[a * 3 + 0];
        const double ly = lig_xyz_f[a * 3 + 1];
        const double lz = lig_xyz_f[a * 3 + 2];
        const double dx = rx - lx, dy = ry - ly, dz = rz - lz;
        const double r2 = dx * dx + dy * dy + dz * dz;

        // Cutoff: ignore pairs that are too far to matter. Comparing r^2 avoids
        // a sqrt for the rejected (majority) pairs -- a standard MD optimisation.
        if (r2 > cutoff2) continue;

        const LigandParams lp = lig_params[a];

        // Lorentz-Berthelot combining rules for the LJ pair parameters:
        //   eps_ij  = sqrt(eps_i * eps_j)      (geometric mean of well depths)
        //   rmin_ij = rmin_i/2 + rmin_j/2      (arithmetic mean of radii)
        const double eps_ij  = sqrt(rp.eps * lp.eps);
        const double rmin_ij = rp.rmin_half + lp.rmin_half;

        // The shared per-pair physics (the one true formula).
        double e_elec, e_vdw, e_gb;
        pair_energy_components(r2, rp.charge, lp.charge, eps_ij, rmin_ij,
                               rp.born, lp.born, e_elec, e_vdw, e_gb);
        out_elec += e_elec;
        out_vdw  += e_vdw;
        out_gb   += e_gb;
    }
}

// ---------------------------------------------------------------------------
// Loader prototype (defined in reference_cpu.cpp). Parses the text format in
// data/README.md into an MmgbsaSystem. Throws std::runtime_error on a bad file.
// ---------------------------------------------------------------------------
MmgbsaSystem load_system(const std::string& path);

// CPU reference prototype (defined in reference_cpu.cpp): fill `out` [M] with the
// trajectory-averaged per-residue decomposition. The trusted baseline the GPU is
// verified against.
void decompose_cpu(const MmgbsaSystem& sys, std::vector<PerResidueEnergy>& out);
