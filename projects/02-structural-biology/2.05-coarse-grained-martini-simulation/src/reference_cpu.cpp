// ===========================================================================
// src/reference_cpu.cpp  --  Loader, CPU velocity-Verlet MD, diagnostics
// ---------------------------------------------------------------------------
// Project 2.5 : Coarse-Grained / MARTINI Simulation
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. The integrator is a
//   single readable serial loop with no parallelism and no cleverness, so when
//   the GPU and CPU agree we believe the GPU. Compiled by the host C++ compiler
//   only (no CUDA here); the per-pair physics is shared with the GPU via
//   martini.h, which is the trick that makes the two paths produce identical
//   floating-point results.
//
// READ THIS AFTER: reference_cpu.h and martini.h. Compare with kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>
#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_system: parse the committed text file into a System.
//   Format (see data/README.md):
//     line 1 : n box dt steps rcut mass sigma epsCC epsCP epsPP
//     n lines: x y z vx vy vz type
//   We validate aggressively and throw on any problem so main.cu can report a
//   clean error instead of computing on garbage.
// ---------------------------------------------------------------------------
System load_system(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open system file: " + path);

    System sys;
    MdParams& P = sys.P;
    // The MARTINI-style interaction matrix is symmetric (eps_ab == eps_ba), so
    // the file gives the three independent entries: CC, CP, PP.
    double epsCC, epsCP, epsPP;
    if (!(in >> P.n >> P.box >> P.dt >> P.steps >> P.rcut >> P.mass >> P.sigma
             >> epsCC >> epsCP >> epsPP))
        throw std::runtime_error("bad header (expected "
            "'n box dt steps rcut mass sigma epsCC epsCP epsPP') in " + path);
    if (P.n <= 0 || P.box <= 0 || P.dt <= 0 || P.steps < 0 || P.rcut <= 0
        || P.mass <= 0 || P.sigma <= 0)
        throw std::runtime_error("invalid simulation parameters in " + path);

    // Fill the flattened 2x2 epsilon matrix (row-major [type_i*NT + type_j]).
    P.eps[0 * MD_NTYPES + 0] = epsCC;   // C-C : strong (apolar likes apolar)
    P.eps[0 * MD_NTYPES + 1] = epsCP;   // C-P : weak   (oil/water mismatch)
    P.eps[1 * MD_NTYPES + 0] = epsCP;   // P-C : symmetric
    P.eps[1 * MD_NTYPES + 1] = epsPP;   // P-P : strong (polar likes polar)

    // Read the n bead records.
    sys.pos.resize(P.n);
    sys.vel.resize(P.n);
    sys.type.resize(P.n);
    for (int i = 0; i < P.n; ++i) {
        if (!(in >> sys.pos[i].x >> sys.pos[i].y >> sys.pos[i].z
                 >> sys.vel[i].x >> sys.vel[i].y >> sys.vel[i].z
                 >> sys.type[i]))
            throw std::runtime_error("truncated bead list in " + path);
        if (sys.type[i] < 0 || sys.type[i] >= MD_NTYPES)
            throw std::runtime_error("bead type out of range in " + path);
    }
    return sys;
}

// ---------------------------------------------------------------------------
// simulate_cpu: the serial velocity-Verlet time loop (trusted reference).
//   One step is:
//     1. half-kick + drift every bead using the CURRENT forces  (verlet_kick_drift)
//     2. recompute every force at the NEW positions             (compute_force_on)
//     3. second half-kick every bead using the new forces       (verlet_kick)
//   Forces for the whole system are computed into a buffer BEFORE any velocity
//   is updated, exactly like the GPU does, so the two stay in lockstep.
//
//   Complexity: O(steps * n^2) -- the n^2 is the all-pairs force sum, which is
//   the cost the GPU parallelises (THEORY section 3).
// ---------------------------------------------------------------------------
void simulate_cpu(System& sys) {
    const MdParams& P = sys.P;
    const int n = P.n;
    std::vector<Vec3> force(n);

    // Initial forces at the starting configuration (needed for step 1's kick).
    for (int i = 0; i < n; ++i)
        force[i] = compute_force_on(i, sys.pos.data(), sys.type.data(), P);

    for (int step = 0; step < P.steps; ++step) {
        // (1) First half-kick + drift: advances v by half a step using the OLD
        //     force, then moves x at the updated velocity. Positions are wrapped
        //     back into the periodic box inside the helper.
        for (int i = 0; i < n; ++i)
            verlet_kick_drift(sys.pos[i], sys.vel[i], force[i], P);

        // (2) Recompute all forces at the new positions. We compute the WHOLE
        //     force array before touching velocities so every bead sees the same
        //     position snapshot -- the GPU does the same with a barrier between
        //     the drift kernel and the force kernel.
        for (int i = 0; i < n; ++i)
            force[i] = compute_force_on(i, sys.pos.data(), sys.type.data(), P);

        // (3) Second half-kick: completes the velocity update with the NEW force.
        for (int i = 0; i < n; ++i)
            verlet_kick(sys.vel[i], force[i], P);
    }
}

// ---------------------------------------------------------------------------
// total_energy: kinetic + Lennard-Jones potential of the whole system.
//   Kinetic   = sum_i 0.5 * m * |v_i|^2.
//   Potential = sum over UNORDERED pairs (i<j) of lj_pair_energy(r_ij).
//   We loop i<j (not all ordered pairs) so each pair is counted once. This is a
//   diagnostic only; it never feeds the integrator.
// ---------------------------------------------------------------------------
double total_energy(const System& sys) {
    const MdParams& P = sys.P;
    const double rcut2 = P.rcut * P.rcut;
    double ke = 0.0, pe = 0.0;
    for (int i = 0; i < P.n; ++i)
        ke += 0.5 * P.mass * dot(sys.vel[i], sys.vel[i]);
    for (int i = 0; i < P.n; ++i)
        for (int j = i + 1; j < P.n; ++j) {
            Vec3 d;
            d.x = min_image(sys.pos[i].x - sys.pos[j].x, P.box);
            d.y = min_image(sys.pos[i].y - sys.pos[j].y, P.box);
            d.z = min_image(sys.pos[i].z - sys.pos[j].z, P.box);
            const double r2  = dot(d, d);
            const double eps = P.eps[sys.type[i] * MD_NTYPES + sys.type[j]];
            pe += lj_pair_energy(r2, eps, P.sigma, rcut2);
        }
    return ke + pe;
}

// ---------------------------------------------------------------------------
// cp_separation: distance between the centroid of all C beads (type 0) and the
//   centroid of all P beads (type 1). A blunt but reproducible order parameter
//   for demixing: it starts near zero (well-mixed) and grows as the C beads
//   coalesce away from the P beads. We unwrap the centroids relative to bead 0
//   of each type via minimum image so periodic wrapping does not corrupt the
//   average (a small but real subtlety with periodic boundaries).
// ---------------------------------------------------------------------------
double cp_separation(const System& sys) {
    const MdParams& P = sys.P;
    Vec3 cC = {0, 0, 0}, cP = {0, 0, 0};
    int nC = 0, nP = 0;
    // Reference anchors: the first bead of each type, to unwrap around.
    Vec3 aC{0, 0, 0}, aP{0, 0, 0};
    for (int i = 0; i < P.n; ++i) {
        if (sys.type[i] == 0 && nC == 0) aC = sys.pos[i];
        if (sys.type[i] == 1 && nP == 0) aP = sys.pos[i];
        if (sys.type[i] == 0) ++nC; else ++nP;
    }
    nC = nP = 0;
    for (int i = 0; i < P.n; ++i) {
        // Bring each bead to the nearest image of its type's anchor before averaging.
        if (sys.type[i] == 0) {
            cC.x += aC.x + min_image(sys.pos[i].x - aC.x, P.box);
            cC.y += aC.y + min_image(sys.pos[i].y - aC.y, P.box);
            cC.z += aC.z + min_image(sys.pos[i].z - aC.z, P.box);
            ++nC;
        } else {
            cP.x += aP.x + min_image(sys.pos[i].x - aP.x, P.box);
            cP.y += aP.y + min_image(sys.pos[i].y - aP.y, P.box);
            cP.z += aP.z + min_image(sys.pos[i].z - aP.z, P.box);
            ++nP;
        }
    }
    if (nC == 0 || nP == 0) return 0.0;     // one species absent: undefined
    cC = cC * (1.0 / nC);
    cP = cP * (1.0 / nP);
    Vec3 d;
    d.x = min_image(cC.x - cP.x, P.box);
    d.y = min_image(cC.y - cP.y, P.box);
    d.z = min_image(cC.z - cP.z, P.box);
    return std::sqrt(dot(d, d));
}
