// ===========================================================================
// src/reference_cpu.cpp  --  System loader + serial velocity-Verlet MD driver
// ---------------------------------------------------------------------------
// Project 1.1 : Molecular Dynamics Engine  (reduced-scope teaching version)
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- plain serial loops, no parallelism, no cleverness
//   -- so that when the GPU and CPU agree we believe the GPU. The per-pair physics
//   (LJ force, minimum image) and the energy helpers all come from md.h, the SAME
//   header the GPU kernel uses; that is why the two trajectories match to round-off
//   (PATTERNS.md §2). Compiled by the host C++ compiler only (no CUDA syntax here).
//
// THE ALGORITHM IMPLEMENTED HERE: velocity-Verlet integration of an LJ fluid.
//   For each timestep dt, given positions x, velocities v, forces F (= a*m):
//     1. v(t+dt/2) = v(t)    + (dt/2) * F(t)/m     "half-kick"
//     2. x(t+dt)   = x(t)    + dt * v(t+dt/2)      "drift"
//     3. recompute F(t+dt) from the new positions  "force eval" (the O(N^2) cost)
//     4. v(t+dt)   = v(t+dt/2) + (dt/2) * F(t+dt)/m "half-kick"
//   Velocity-Verlet is the standard MD integrator: it is time-reversible and
//   SYMPLECTIC, so it conserves energy extremely well over long runs (the heart
//   of why MD trajectories are trustworthy). See ../THEORY.md.
//
// READ THIS AFTER: reference_cpu.h, md.h. Compare with kernels.cu (the GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::fabs
#include <cstdint>     // std::uint32_t (deterministic synthetic RNG)
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// compute_forces_cpu: fill F[i] with the total LJ force on atom i from ALL other
//   atoms, and return the system POTENTIAL energy. This is the all-pairs (direct)
//   N-body sum: O(N^2) interactions. Each unordered pair {i,j} is visited once
//   (j from i+1) so its energy is counted once; Newton's third law gives the
//   force on j for free (F_ji = -F_ij), which also halves the force work.
//   The GPU kernel does the SAME sum but with one thread per atom i looping over
//   all j (it recomputes each pair twice -- simpler, still correct; see THEORY).
// ---------------------------------------------------------------------------
static double compute_forces_cpu(const SimParams& p,
                                 const std::vector<Vec3>& pos,
                                 std::vector<Vec3>& F) {
    const int n = p.n;
    for (int i = 0; i < n; ++i) F[static_cast<std::size_t>(i)] = vec3_zero();

    double potential = 0.0;  // accumulates U over every unordered pair, once

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            // Displacement r_i - r_j, wrapped to the nearest periodic image so
            // atoms interact across box faces correctly (minimum-image, md.h).
            Vec3 rij;
            rij.x = minimum_image(pos[i].x - pos[j].x, p.box);
            rij.y = minimum_image(pos[i].y - pos[j].y, p.box);
            rij.z = minimum_image(pos[i].z - pos[j].z, p.box);

            // Force on i from j, and add this pair's U into `potential` once.
            Vec3 fij = lj_pair_force(rij, p, &potential);

            // Apply to i; Newton's third law gives the opposite force to j.
            F[static_cast<std::size_t>(i)].x += fij.x;
            F[static_cast<std::size_t>(i)].y += fij.y;
            F[static_cast<std::size_t>(i)].z += fij.z;
            F[static_cast<std::size_t>(j)].x -= fij.x;
            F[static_cast<std::size_t>(j)].y -= fij.y;
            F[static_cast<std::size_t>(j)].z -= fij.z;
        }
    }
    return potential;
}

// ---------------------------------------------------------------------------
// total_energy_cpu: kinetic (sum 1/2 m v^2) + potential (all-pairs LJ). Declared
//   in the header; uses a throwaway force buffer just to reuse compute_forces_cpu
//   for the potential term (clear over fast for a reference).
// ---------------------------------------------------------------------------
double total_energy_cpu(const SimParams& p,
                        const std::vector<Vec3>& pos,
                        const std::vector<Vec3>& vel) {
    std::vector<Vec3> F(static_cast<std::size_t>(p.n));
    const double potential = compute_forces_cpu(p, pos, F);
    double kinetic = 0.0;
    for (int i = 0; i < p.n; ++i) kinetic += kinetic_energy_one(vel[i], p.mass);
    return kinetic + potential;
}

// ---------------------------------------------------------------------------
// integrate_cpu: the reference velocity-Verlet loop. Copies the initial state so
//   the caller's MdSystem is untouched (the GPU path reuses the same start). Each
//   step does the four Verlet sub-steps above; we track the energy-drift metric
//   and finish by summarizing the trajectory into an MdResult.
// ---------------------------------------------------------------------------
MdResult integrate_cpu(const MdSystem& sys) {
    const SimParams p = sys.params;
    const int n = p.n;

    // Working copies of the state we advance in time.
    std::vector<Vec3> pos = sys.pos;
    std::vector<Vec3> vel = sys.vel;
    std::vector<Vec3> F(static_cast<std::size_t>(n));   // forces F(t)

    // Initial forces and total energy (our conserved-quantity reference E0).
    double potential = compute_forces_cpu(p, pos, F);
    double kinetic = 0.0;
    for (int i = 0; i < n; ++i) kinetic += kinetic_energy_one(vel[i], p.mass);

    MdResult r;
    r.E0 = kinetic + potential;
    r.max_drift = 0.0;

    const double half_dt_over_m = 0.5 * p.dt / p.mass;  // (dt/2)/m used in kicks

    for (int step = 0; step < p.steps; ++step) {
        // (1) half-kick: v += (dt/2) * F/m  using the CURRENT forces F(t).
        for (int i = 0; i < n; ++i) {
            vel[i].x += half_dt_over_m * F[i].x;
            vel[i].y += half_dt_over_m * F[i].y;
            vel[i].z += half_dt_over_m * F[i].z;
        }
        // (2) drift: x += dt * v(t+dt/2), then wrap back into the periodic box.
        for (int i = 0; i < n; ++i) {
            pos[i].x = wrap_into_box(pos[i].x + p.dt * vel[i].x, p.box);
            pos[i].y = wrap_into_box(pos[i].y + p.dt * vel[i].y, p.box);
            pos[i].z = wrap_into_box(pos[i].z + p.dt * vel[i].z, p.box);
        }
        // (3) recompute forces at the new positions F(t+dt) (the O(N^2) work).
        potential = compute_forces_cpu(p, pos, F);
        // (4) second half-kick with the NEW forces -> completes v(t+dt).
        kinetic = 0.0;
        for (int i = 0; i < n; ++i) {
            vel[i].x += half_dt_over_m * F[i].x;
            vel[i].y += half_dt_over_m * F[i].y;
            vel[i].z += half_dt_over_m * F[i].z;
            kinetic += kinetic_energy_one(vel[i], p.mass);
        }
        // Energy-drift diagnostic: how far total energy has wandered from E0.
        const double E = kinetic + potential;
        const double drift = std::fabs(E - r.E0);
        if (drift > r.max_drift) r.max_drift = drift;
    }

    // Final observables. T from equipartition: KE = (3N/2) k_B T, k_B=1 reduced.
    r.E_final = kinetic + potential;
    r.T_final = (n > 0) ? (2.0 * kinetic) / (3.0 * n) : 0.0;
    r.pos_checksum = 0.0;
    for (int i = 0; i < n; ++i) r.pos_checksum += pos[i].x + pos[i].y + pos[i].z;
    return r;
}

// ---------------------------------------------------------------------------
// load_system: parse the sample text file (layout in data/README.md). The header
//   line gives the parameters; each subsequent line is one atom's x y z vx vy vz.
// ---------------------------------------------------------------------------
MdSystem load_system(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open system file: " + path);

    MdSystem sys;
    SimParams& p = sys.params;
    if (!(in >> p.n >> p.box >> p.dt >> p.steps >> p.eps >> p.sigma >> p.rcut >> p.mass))
        throw std::runtime_error(
            "bad header (expected 'n box dt steps eps sigma rcut mass') in " + path);
    if (p.n <= 0 || p.box <= 0.0 || p.dt <= 0.0 || p.steps <= 0 || p.mass <= 0.0)
        throw std::runtime_error("invalid simulation parameters in " + path);

    sys.pos.resize(static_cast<std::size_t>(p.n));
    sys.vel.resize(static_cast<std::size_t>(p.n));
    for (int i = 0; i < p.n; ++i) {
        if (!(in >> sys.pos[i].x >> sys.pos[i].y >> sys.pos[i].z
                 >> sys.vel[i].x >> sys.vel[i].y >> sys.vel[i].z))
            throw std::runtime_error("ran out of atom data at atom " +
                                     std::to_string(i) + " in " + path);
    }
    return sys;
}

// ---------------------------------------------------------------------------
// make_default_system: a deterministic built-in fallback so the program runs even
//   with no input file. We place a small simple-cubic lattice of atoms (well
//   spaced so no overlapping cores) and give them tiny deterministic velocities
//   from a fixed-seed integer hash (NOT std::rand, which varies by platform) so
//   the result is byte-identical everywhere. This mirrors the committed sample.
// ---------------------------------------------------------------------------
MdSystem make_default_system() {
    MdSystem sys;
    SimParams& p = sys.params;
    const int side = 3;            // 3x3x3 = 27 atoms on a cubic lattice
    p.n     = side * side * side;
    p.sigma = 1.0;
    p.eps   = 1.0;
    p.mass  = 1.0;
    p.dt    = 0.004;               // small timestep -> good energy conservation
    p.steps = 50;
    const double spacing = 1.2;    // > sigma so atoms start in the repulsive-soft region
    p.box   = side * spacing;      // box exactly tiles the lattice (period = side*spacing)
    p.rcut  = 2.5 * p.sigma;       // standard LJ cutoff (2.5 sigma)

    sys.pos.resize(static_cast<std::size_t>(p.n));
    sys.vel.resize(static_cast<std::size_t>(p.n));
    int idx = 0;
    for (int a = 0; a < side; ++a)
      for (int b = 0; b < side; ++b)
        for (int c = 0; c < side; ++c) {
            sys.pos[idx] = Vec3{ a * spacing, b * spacing, c * spacing };
            // Deterministic small velocity from a splitmix-style integer hash of
            // the atom index. Pure integer math -> identical on every platform.
            std::uint32_t h = static_cast<std::uint32_t>(idx) * 2654435761u + 1013904223u;
            auto unit = [](std::uint32_t v) {
                // map to [-0.5, 0.5): deterministic, no floating-point RNG state
                return (static_cast<double>(v & 0xFFFFu) / 65536.0) - 0.5;
            };
            sys.vel[idx] = Vec3{ 0.1 * unit(h),
                                 0.1 * unit(h * 2246822519u + 1u),
                                 0.1 * unit(h * 3266489917u + 7u) };
            ++idx;
        }
    return sys;
}
