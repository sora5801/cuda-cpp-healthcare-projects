// ===========================================================================
// src/gist.h  --  Shared (host + device) GIST physics + fixed-point reduction
// ---------------------------------------------------------------------------
// Project 2.26 : Hydrogen Bond Network & Water Placement Analysis
//
// WHAT THIS PROJECT COMPUTES
//   GIST = Grid Inhomogeneous Solvation Theory. We lay a fixed 3D grid of cubic
//   VOXELS over a protein binding pocket, then stream many MD snapshots of the
//   explicit waters in that pocket. Each water-oxygen lands in exactly one voxel.
//   For every (water, frame) we accumulate, PER VOXEL:
//       * an occupancy count           (how often a water sits here),
//       * the water<->solute interaction ENERGY of that water.
//   After all frames we turn those tallies into per-voxel THERMODYNAMICS:
//       * number density  g  = (waters seen here) / (frames * bulk waters/voxel),
//       * mean energy     dE = <E_sw>  relative to bulk water,
//       * translational entropy  -T dS_trans  from a density expansion,
//       * free energy     dG = dE - T dS.
//   A voxel with HIGH dG holds a water that is expensive to keep solvated --
//   displacing it with a ligand atom is predicted to GAIN binding affinity. That
//   single ranked list is the deliverable WaterMap/GIST give a medicinal chemist.
//
// WHY A GPU  (the catalog "CUDA pattern" for 2.26)
//   Production GIST integrates over MILLIONS of frames x thousands of waters x a
//   grid of ~10^5 voxels. The work is a SCATTER: each (water, frame) reads a few
//   numbers and ADDS into one voxel. Independent samples, colliding destinations
//   -> the classic GPU "grid accumulation with atomic updates" pattern (one
//   thread per water-sample, atomicAdd into the voxel it occupies). This is the
//   same shape as Monte-Carlo dose scoring (5.01) and k-means accumulation
//   (11.09); see docs/PATTERNS.md.
//
// DETERMINISM TRICK (PATTERNS.md §3, same idea as 5.01 / 11.09)
//   Floating-point atomicAdd is NOT associative: when thousands of threads add
//   energies into the same voxel, the order varies run-to-run, so a float sum is
//   irreproducible AND will not match the serial CPU sum. We therefore accumulate
//   energies in FIXED-POINT integers (atomicAdd on unsigned long long). Integer
//   adds commute, so the GPU tally is bit-identical every run and equals the CPU
//   tally exactly. We only convert back to kcal/mol at the very end.
//
//   Everything below is __host__ __device__ (GIST_HD) so the CPU reference
//   (reference_cpu.cpp) and the GPU kernel (kernels.cu) run BYTE-FOR-BYTE
//   identical math. Keep CUDA-only types (e.g. __global__) OUT of this header so
//   the plain host compiler can include it.
//
// READ THIS AFTER: util/cuda_check.cuh; BEFORE: reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>   // std::uint32_t, std::int64_t
#include <cmath>     // std::sqrt, std::log, std::floor

// __host__ __device__ on the GPU compiler; nothing on the host compiler.
#ifdef __CUDACC__
#define GIST_HD __host__ __device__
#else
#define GIST_HD
#endif

// ---------------------------------------------------------------------------
// PHYSICAL CONSTANTS (teaching values; documented, not tuned to a force field)
// ---------------------------------------------------------------------------
// NOTE: these are `constexpr`, not `static const`. A plain `static const double`
// at file scope is a HOST-ONLY object as far as nvcc is concerned, so referencing
// it from a __device__ function errors ("identifier undefined in device code").
// `constexpr` makes each a compile-time constant usable in both host and device
// code, which is exactly what the shared __host__ __device__ functions below need.
//
// Temperature of the simulated ensemble. 298.15 K = 25 C, standard "room temp".
constexpr double GIST_TEMPERATURE_K = 298.15;
// Boltzmann constant in kcal/(mol*K). Energies in this project are in kcal/mol,
// the chemist's unit, so k_B carries the matching units. k_B*T ~ 0.5925 kcal/mol.
constexpr double GIST_KB_KCAL = 0.0019872041;
// Bulk number density of liquid water: ~0.0334 molecules per cubic Angstrom
// (= 997 kg/m^3). A voxel of volume V holds, on average, RHO_BULK * V waters in
// neat bulk water; g (below) measures enrichment RELATIVE to this.
constexpr double GIST_RHO_BULK = 0.0334;
// Reference energy of a water fully surrounded by bulk water (kcal/mol). A water
// in bulk makes ~3.5 hydrogen bonds; -9.5 kcal/mol is a standard textbook value
// for the mean water-water interaction energy. dE below is measured against it.
constexpr double GIST_E_BULK = -9.533;

// ---------------------------------------------------------------------------
// FIXED-POINT ENERGY ACCUMULATION
//   Energies are a few kcal/mol; we keep ~6 decimal digits by scaling by 1e6 and
//   storing a signed 64-bit integer of "micro-kcal/mol". A voxel might see ~10^4
//   waters * ~50 kcal/mol => ~5e5 kcal/mol => 5e11 micro-units, far below the
//   int64 ceiling (~9.2e18), so there is no overflow risk at teaching scale.
// ---------------------------------------------------------------------------
typedef long long gist_fixed_t;                 // signed 64-bit fixed-point sum
constexpr double GIST_ENERGY_SCALE = 1.0e6;     // micro-kcal/mol per integer unit

// Quantize an energy (kcal/mol) to fixed-point micro-units (round to nearest).
GIST_HD inline gist_fixed_t gist_to_fixed(double e_kcal) {
    // +/-0.5 rounding keeps the CPU and GPU on the SAME integer, so their sums
    // agree exactly even though the inputs are floating point.
    const double scaled = e_kcal * GIST_ENERGY_SCALE;
    return static_cast<gist_fixed_t>(scaled >= 0.0 ? (scaled + 0.5) : (scaled - 0.5));
}
// Convert a fixed-point sum back to kcal/mol.
GIST_HD inline double gist_from_fixed(gist_fixed_t f) {
    return static_cast<double>(f) / GIST_ENERGY_SCALE;
}

// ---------------------------------------------------------------------------
// GRID GEOMETRY
//   The voxel grid is an axis-aligned box: origin (ox,oy,oz) = the (min) corner,
//   nx*ny*nz cubic voxels each of side `spacing` Angstroms. We store it as a tiny
//   POD struct passed by value to kernels (it is read-only, a handful of
//   numbers, so it lives in registers/constant-ish and costs nothing to copy).
// ---------------------------------------------------------------------------
struct GistGrid {
    int nx, ny, nz;          // voxel counts along x, y, z
    double ox, oy, oz;       // world coords of the grid's minimum corner (Angstrom)
    double spacing;          // voxel edge length (Angstrom); voxels are cubes

    GIST_HD int    num_voxels() const { return nx * ny * nz; }
    GIST_HD double voxel_volume() const { return spacing * spacing * spacing; }

    // Flatten a 3D voxel index to the 1D storage index (x fastest, then y, z).
    // This is the layout the accumulation arrays use; keep it in ONE place so
    // host and device never disagree about where a voxel's tally lives.
    GIST_HD int flat_index(int ix, int iy, int iz) const {
        return (iz * ny + iy) * nx + ix;
    }
};

// Map a world-space point to its voxel index. Returns -1 ("outside the grid")
// when the point falls outside the box -- those samples are simply not scored,
// exactly as production GIST ignores waters beyond the analysis region.
//   Thought process: floor((p - origin)/spacing) is the cell that contains p;
//   we clamp by an explicit range check rather than min/max so out-of-box waters
//   are dropped (not piled onto the boundary voxel, which would bias the edges).
GIST_HD inline int gist_voxel_of(const GistGrid& g,
                                 double px, double py, double pz) {
    const int ix = static_cast<int>(std::floor((px - g.ox) / g.spacing));
    const int iy = static_cast<int>(std::floor((py - g.oy) / g.spacing));
    const int iz = static_cast<int>(std::floor((pz - g.oz) / g.spacing));
    if (ix < 0 || ix >= g.nx) return -1;
    if (iy < 0 || iy >= g.ny) return -1;
    if (iz < 0 || iz >= g.nz) return -1;
    return g.flat_index(ix, iy, iz);
}

// ---------------------------------------------------------------------------
// WATER<->SOLUTE INTERACTION ENERGY  (the per-sample physics)
//   For a teaching model we use a single Lennard-Jones + Coulomb term between a
//   water oxygen and each solute atom, summed. This captures the two effects
//   that matter for the GIST story: short-range packing (LJ) and electrostatics
//   (Coulomb). Production GIST uses the full TIP3P/force-field nonbonded energy
//   over all solute AND solvent atoms; we keep the SAME functional FORM but a
//   reduced atom model so the demo runs offline and the math is legible.
//
//   Inputs:
//     wx,wy,wz   water oxygen position (Angstrom)
//     atoms      flat [4*natoms] array: (x, y, z, charge) per solute atom
//     natoms     number of solute atoms
//   Returns: interaction energy in kcal/mol (negative = favorable).
//
//   Units: LJ epsilon in kcal/mol, sigma in Angstrom; Coulomb constant 332.06
//   converts (e^2 / Angstrom) to kcal/mol (the standard MD prefactor).
// ---------------------------------------------------------------------------
constexpr double GIST_LJ_EPS    = 0.152;   // kcal/mol, ~TIP3P oxygen well depth
constexpr double GIST_LJ_SIGMA  = 3.15;    // Angstrom, ~TIP3P oxygen sigma
constexpr double GIST_Q_WATER   = -0.834;  // partial charge on a water oxygen (e)
constexpr double GIST_COULOMB_K = 332.0636;// kcal*Angstrom/(mol*e^2)
constexpr double GIST_R_CUTOFF  = 9.0;     // Angstrom; ignore atoms past this
// Closest physical approach of a water oxygen to a heavy solute atom (~the van
// der Waals contact distance). We clamp r to this floor so the steep 1/r^12 wall
// is capped at a realistic, finite value instead of exploding when a sampled
// water happens to land almost on top of an atom. 2.6 A is a touch below sigma,
// i.e. just inside the repulsive wall -- a sensible "hard contact" for a demo.
constexpr double GIST_R_MIN     = 2.6;     // Angstrom; physical contact floor

GIST_HD inline double gist_water_solute_energy(double wx, double wy, double wz,
                                               const float* atoms, int natoms) {
    double e = 0.0;                                  // running energy (kcal/mol)
    const double sig6_coef = GIST_LJ_SIGMA * GIST_LJ_SIGMA * GIST_LJ_SIGMA;
    const double sigma6 = sig6_coef * sig6_coef;     // sigma^6, precomputed once
    for (int a = 0; a < natoms; ++a) {
        // Each solute atom is 4 floats: position (x,y,z) then partial charge q.
        const double ax = atoms[4 * a + 0];
        const double ay = atoms[4 * a + 1];
        const double az = atoms[4 * a + 2];
        const double aq = atoms[4 * a + 3];
        const double dx = wx - ax, dy = wy - ay, dz = wz - az;
        double r2 = dx * dx + dy * dy + dz * dz;     // squared distance (A^2)
        const double r = std::sqrt(r2);
        if (r > GIST_R_CUTOFF) continue;             // outside cutoff: negligible
        // Soften the core: a water never sits ON an atom, so clamp r to R_MIN to
        // keep the 1/r^12 term finite and the demo numerically stable.
        const double rc = (r < GIST_R_MIN) ? GIST_R_MIN : r;
        const double rc2 = rc * rc;
        const double inv_r6 = 1.0 / (rc2 * rc2 * rc2);   // 1/r^6
        const double sr6 = sigma6 * inv_r6;              // (sigma/r)^6
        // Lennard-Jones 12-6:  4*eps*[ (s/r)^12 - (s/r)^6 ].
        e += 4.0 * GIST_LJ_EPS * (sr6 * sr6 - sr6);
        // Coulomb:  k * q_water * q_atom / r.
        e += GIST_COULOMB_K * GIST_Q_WATER * aq / rc;
    }
    return e;
}

// ---------------------------------------------------------------------------
// PER-VOXEL THERMODYNAMICS  (turn tallies into the GIST quantities)
//   Given a voxel's occupancy count and fixed-point energy sum over `nframes`
//   frames, derive density g, mean dE, entropy term, and free energy dG. This is
//   the "reduce -> physics" step; it runs once per voxel and is shared by CPU and
//   GPU so the final numbers (and the ranking) are identical.
// ---------------------------------------------------------------------------
struct VoxelResult {
    int    index;     // flat voxel index (for back-mapping to (ix,iy,iz))
    double g;         // number density relative to bulk (dimensionless; 1 = bulk)
    double dE;        // mean water-solute energy minus bulk reference (kcal/mol)
    double mTdS;      // -T * dS_trans, the entropic penalty (kcal/mol, >=0)
    double dG;        // dE + (-T dS): GIST free energy (kcal/mol); high = displaceable
    unsigned int count; // raw occupancy (waters summed into this voxel)
};

// Minimum observations for a voxel's GIST statistics to be trustworthy. A voxel
// visited only once or twice over the whole trajectory has a meaningless mean
// energy and density -- one stray diffuse water can give it a deceptively high dG.
// Real GIST analyses likewise discard under-sampled voxels. We require a voxel to
// be occupied in at least this fraction of frames before it counts as a hydration
// SITE; derive_voxels() enforces it. (5% is a permissive teaching threshold.)
constexpr double GIST_MIN_OCCUPANCY_FRACTION = 0.05;

// Translational entropy from the leading density term of the IFST/GIST
// expansion. For an inhomogeneous fluid the first-order translational entropy
// density is  s_trans = -k_B * rho * ln(rho/rho_bulk) = -k_B * rho * ln(g).
// We report the PER-WATER penalty  -T*dS = +k_B*T * ln(g): an ORDERED water
// (g>1, enriched) has ln(g)>0, so -T*dS>0 -- an entropic COST to keep it there,
// which is exactly why displacing it can help binding. (g<=1 waters are bulk-like
// and clamped to zero penalty; production GIST also keeps higher-order
// orientational/six-integral terms -- see THEORY.md "real world".)
GIST_HD inline double gist_translational_penalty(double g) {
    if (g <= 1.0) return 0.0;                 // bulk-or-below: no ordering penalty
    return GIST_KB_KCAL * GIST_TEMPERATURE_K * std::log(g);
}

// Assemble a voxel's full result from its raw tallies.
//   count      : waters summed into this voxel over all frames
//   esum_fixed : fixed-point sum of water-solute energies for those waters
//   nframes    : number of MD frames streamed (for the density normalization)
//   grid       : geometry (we need the voxel volume for the density)
GIST_HD inline VoxelResult gist_voxel_result(int index, unsigned int count,
                                             gist_fixed_t esum_fixed,
                                             int nframes, const GistGrid& grid) {
    VoxelResult r;
    r.index = index;
    r.count = count;
    // Expected waters per voxel in neat bulk = rho_bulk * V * nframes. The voxel's
    // density g is its actual occupancy divided by that expectation. g>1 means the
    // pocket holds this water MORE tightly than bulk -- a structured hydration site.
    const double expected = GIST_RHO_BULK * grid.voxel_volume() *
                            static_cast<double>(nframes);
    r.g = (expected > 0.0) ? (static_cast<double>(count) / expected) : 0.0;
    // Mean water-solute energy of the waters seen here, minus the bulk reference.
    // A very negative dE means the water is glued to the protein (enthalpically
    // hard to remove); near-zero means it is bulk-like.
    if (count > 0) {
        const double mean_e = gist_from_fixed(esum_fixed) / static_cast<double>(count);
        r.dE = mean_e - GIST_E_BULK;
    } else {
        r.dE = 0.0;
    }
    r.mTdS = gist_translational_penalty(r.g);
    // GIST free energy of this water site. dG > 0 (energy + entropy both push to
    // remove the water) flags a favorable displacement target for a ligand.
    r.dG = r.dE + r.mTdS;
    return r;
}
