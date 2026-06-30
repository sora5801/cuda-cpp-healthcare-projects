// ===========================================================================
// src/reference_cpu.cpp  --  Loader, synthetic builder, serial MDFF, metrics
// ---------------------------------------------------------------------------
// Project 2.12 : Flexible Fitting / MDFF
//
// Compiled by the host compiler only (no CUDA). The per-atom physics and the
// trilinear density sampler live in mdff.h and are shared verbatim with the GPU
// kernel, so this serial reference and the GPU path compute identical math.
//
// CONTENTS
//   build_density()      -- rasterise a Gaussian-blob density from atom centres
//   make_synthetic()     -- the deterministic built-in problem (answer embedded)
//   load_problem()       -- parse a problem from a text file (data/README.md fmt)
//   fit_cpu()            -- the serial steepest-descent fitting loop (baseline)
//   rmsd / cross_correlation -- quality metrics reported by main.cu
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // exp, sqrt
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// build_density : turn a set of atom centres into a 3-D density map by summing
//   an isotropic Gaussian blob around each atom -- the standard way a cryo-EM
//   map is *simulated* from a model (e.g. VMD's "mdff sim"). Each voxel (ix,iy,iz)
//   at world position (ix*vox, iy*vox, iz*vox) accumulates
//       rho += exp( -|r_voxel - r_atom|^2 / (2 sigma^2) )
//   summed over atoms. The result is a smooth field whose ridges sit on the
//   atoms -- exactly what the fitting force should climb.
//
//   We rasterise on the host once; both the CPU and GPU solvers then only READ
//   this map (it never changes during fitting), which is why the density can be
//   shared read-only across all GPU threads.
// ---------------------------------------------------------------------------
static std::vector<double> build_density(const std::vector<Vec3>& centres,
                                         const MdffParams& P, double sigma) {
    std::vector<double> rho(static_cast<std::size_t>(P.nx) * P.ny * P.nz, 0.0);
    const double inv2s2 = 1.0 / (2.0 * sigma * sigma);   // 1/(2 sigma^2)

    // For each voxel, sum the Gaussian contribution of every atom. This is an
    // O(voxels * atoms) rasterisation -- fine for a teaching-sized map. (Real
    // tools restrict each atom to a local cutoff box; see THEORY.)
    for (int iz = 0; iz < P.nz; ++iz)
        for (int iy = 0; iy < P.ny; ++iy)
            for (int ix = 0; ix < P.nx; ++ix) {
                const Vec3 r = {ix * P.vox, iy * P.vox, iz * P.vox};
                double acc = 0.0;
                for (const Vec3& c : centres) {
                    const Vec3 d = r - c;
                    acc += std::exp(-dot(d, d) * inv2s2);
                }
                rho[grid_index(ix, iy, iz, P.nx, P.ny)] = acc;
            }
    return rho;
}

// ---------------------------------------------------------------------------
// make_synthetic : the deterministic built-in problem with the answer baked in.
//
//   Geometry: a 3x3x3 lattice of 27 "atoms" (a tiny stand-in for a domain),
//   centred in a 32^3 voxel map at 1.0 unit/voxel. The TARGET positions are the
//   lattice nodes; the density is the Gaussian sum of those targets. We then
//   DISPLACE every atom by a fixed, deterministic offset to create a misfitted
//   start. The fitting force pulls each atom back toward its target's density
//   ridge.
//
//   IMPORTANT design choice (well-separated basins): we space the lattice at
//   L = 6 units with narrow blobs (sigma = 1.2). With overlapping blobs the
//   density gradient points at the COLLECTIVE centre of mass and every atom
//   collapses inward (a real MDFF over-fitting failure mode!). Separating the
//   basins makes each atom's local gradient point at its OWN target, so the fit
//   actually recovers the structure -- the clean teaching story (RMSD drops to
//   ~0, cross-correlation rises). The fuller, harder overlapping-basin case is
//   discussed as a limitation in THEORY/README.
//
//   Everything here is a fixed formula (no RNG), so the produced numbers --
//   and therefore demo/expected_output.txt -- are byte-stable across machines.
// ---------------------------------------------------------------------------
MdffProblem make_synthetic() {
    MdffProblem prob;
    MdffParams& P = prob.params;
    P.nx = P.ny = P.nz = 32;     // 32^3 density grid (room for separated basins)
    P.vox     = 1.0;             // 1 world-unit per voxel
    P.w_dens  = 6.0;             // fitting-force weight (climbs the density)
    P.k_rest  = 0.05;            // restraint stiffness (gently anchors to start)
    P.step    = 0.05;            // overdamped steepest-descent step size
    P.iters   = 200;             // fitting iterations

    const double sigma = 1.2;    // blob width (world units); narrow -> separated
    const double L     = 6.0;    // lattice spacing between atoms (world units)

    // Centre a 3x3x3 lattice in the middle of the map.
    const Vec3 mapc = {0.5 * (P.nx - 1) * P.vox,
                       0.5 * (P.ny - 1) * P.vox,
                       0.5 * (P.nz - 1) * P.vox};

    // Deterministic per-atom displacement pattern (no randomness): we push each
    // atom away from its target by a small fixed function of its lattice index,
    // ~0.9 unit per component -- enough to misfit it but inside the blob's basin.
    int a = 0;
    for (int kz = -1; kz <= 1; ++kz)
        for (int ky = -1; ky <= 1; ++ky)
            for (int kx = -1; kx <= 1; ++kx, ++a) {
                const Vec3 tgt = {mapc.x + kx * L, mapc.y + ky * L, mapc.z + kz * L};
                prob.x_target.push_back(tgt);
                // Fixed displacement that varies per atom but is fully
                // reproducible (a deterministic "swirl"), magnitude 1.0/axis so
                // each atom starts ~1.4 units (3-D) off its target but well
                // inside its own density basin.
                const Vec3 disp = {1.0 * ((a % 3) - 1),
                                   1.0 * (((a / 3) % 3) - 1),
                                   1.0 * (((a / 9) % 3) - 1)};
                prob.x0.push_back(tgt + disp);
            }
    P.natoms = static_cast<int>(prob.x0.size());

    // Anchor the restraint at the (misfitted) start; density does the deforming.
    prob.x_ref = prob.x0;

    // Build the density from the TARGETS (the structure we want to recover).
    prob.rho = build_density(prob.x_target, P, sigma);
    return prob;
}

// ---------------------------------------------------------------------------
// load_problem : parse a problem from a whitespace-separated text file.
//   Layout (see data/README.md), all on whitespace:
//       nx ny nz vox w_dens k_rest step iters natoms sigma
//       x0_x x0_y x0_z   (natoms lines: starting positions)
//       tx ty tz         (natoms lines: target positions)
//   The density is rebuilt from the targets with the given sigma, so the file
//   stays small (we store atoms, not a full nx*ny*nz grid).
// ---------------------------------------------------------------------------
MdffProblem load_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open MDFF problem file: " + path);

    MdffProblem prob;
    MdffParams& P = prob.params;
    double sigma = 1.5;
    if (!(in >> P.nx >> P.ny >> P.nz >> P.vox >> P.w_dens >> P.k_rest
             >> P.step >> P.iters >> P.natoms >> sigma))
        throw std::runtime_error("bad header (expected 'nx ny nz vox w_dens "
            "k_rest step iters natoms sigma') in " + path);
    if (P.nx <= 1 || P.ny <= 1 || P.nz <= 1 || P.natoms <= 0 || P.iters < 0
        || P.vox <= 0.0 || sigma <= 0.0)
        throw std::runtime_error("invalid MDFF parameters in " + path);

    // Read the starting positions, then the target positions.
    prob.x0.resize(P.natoms);
    prob.x_target.resize(P.natoms);
    for (int i = 0; i < P.natoms; ++i)
        if (!(in >> prob.x0[i].x >> prob.x0[i].y >> prob.x0[i].z))
            throw std::runtime_error("not enough start positions in " + path);
    for (int i = 0; i < P.natoms; ++i)
        if (!(in >> prob.x_target[i].x >> prob.x_target[i].y >> prob.x_target[i].z))
            throw std::runtime_error("not enough target positions in " + path);

    prob.x_ref = prob.x0;                          // restraint anchored at start
    prob.rho   = build_density(prob.x_target, P, sigma);
    return prob;
}

// ---------------------------------------------------------------------------
// fit_cpu : the serial reference fitting loop.
//   Every iteration advances ALL atoms by one steepest-descent step. We use
//   DOUBLE BUFFERING (read x_old, write x_new, then swap) so that within an
//   iteration each atom's update depends only on the previous iteration's
//   positions -- mirroring the GPU's Jacobi update exactly, which is what makes
//   the two results agree. (mdff_step_atom only reads x[i] and x_ref[i], so even
//   an in-place loop would work, but double-buffering keeps CPU/GPU semantics
//   identical and is the honest analogue of the kernel.)
// ---------------------------------------------------------------------------
std::vector<Vec3> fit_cpu(const MdffProblem& prob) {
    const MdffParams& P = prob.params;
    std::vector<Vec3> a = prob.x0;          // current positions (read this iter)
    std::vector<Vec3> b(prob.x0.size());    // next positions    (write this iter)

    for (int it = 0; it < P.iters; ++it) {
        for (int i = 0; i < P.natoms; ++i)
            b[i] = mdff_step_atom(prob.rho.data(), a[i], prob.x_ref[i], P);
        a.swap(b);                          // O(1) ping-pong; b becomes scratch
    }
    return a;                               // final fitted positions
}

// ---------------------------------------------------------------------------
// rmsd : root-mean-square distance between two atom sets (same indexing).
//   sqrt( mean_i |a_i - b_i|^2 ). Length mismatch -> +inf so a shape bug cannot
//   masquerade as a perfect fit.
// ---------------------------------------------------------------------------
double rmsd(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
    if (a.size() != b.size() || a.empty()) return INFINITY;
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const Vec3 d = a[i] - b[i];
        s += dot(d, d);
    }
    return std::sqrt(s / a.size());
}

// ---------------------------------------------------------------------------
// cross_correlation : mean interpolated density at the atom positions.
//   This is our proxy for the MDFF cross-correlation score the fit maximises:
//   the higher it is, the better the atoms sit on the density ridges. Computed
//   through the SAME sample_density used by the force, so a rising CC is exactly
//   the descent doing its job.
// ---------------------------------------------------------------------------
double cross_correlation(const std::vector<Vec3>& x, const MdffProblem& prob) {
    if (x.empty()) return 0.0;
    double s = 0.0;
    for (const Vec3& p : x)
        s += sample_density(prob.rho.data(), p, prob.params);
    return s / x.size();
}
