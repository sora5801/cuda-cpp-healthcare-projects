// ===========================================================================
// src/reference_cpu.cpp  --  Loader, system builder, serial MD reference
// ---------------------------------------------------------------------------
// Project 2.19 : Membrane Protein Simulation   (reduced-scope teaching version)
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is written to be
//   OBVIOUSLY correct -- one readable loop per phase, no parallelism, no
//   cleverness -- so that when the GPU and CPU agree, we believe the GPU.
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
//   The per-pair / per-bead PHYSICS is NOT duplicated here: it lives in
//   membrane.h and is called by BOTH this file and kernels.cu, so the two
//   implementations are byte-for-byte identical math.
//
// READ THIS AFTER: membrane.h, reference_cpu.h. Compare with kernels.cu (twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// strip_comments: read the whole file, drop any '#...' to end-of-line, and
// return the remaining tokens as one stream. This lets the committed sample
// carry human-readable "# ..." comment lines (which a learner can edit) while
// the loader still parses cleanly with >>. We do the stripping ourselves
// because operator>> has no notion of comments.
// ---------------------------------------------------------------------------
static std::istringstream strip_comments(std::ifstream& in) {
    std::string all, line;
    while (std::getline(in, line)) {
        const std::size_t h = line.find('#');           // '#' starts a comment
        if (h != std::string::npos) line.erase(h);      // chop it off
        all += line;
        all += '\n';
    }
    return std::istringstream(all);
}

// ---------------------------------------------------------------------------
// load_params: parse the whitespace format (see data/README.md). Comment lines
// beginning with '#' (or trailing '# ...') are ignored. The numeric record is:
//   n_lipids n_prot box_x box_y sigma rcut k_bond r_bond dt steps temperature gamma seed
//   eHH eHT eHP eTT eTP ePP        (the 6 unique LJ well depths, symmetric)
// We keep the format flat and human-editable so a learner can poke at it.
// ---------------------------------------------------------------------------
SimParams load_params(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open parameter file: " + path);
    std::istringstream in = strip_comments(file);

    SimParams P{};
    // Read the scalar block first.
    if (!(in >> P.n_lipids >> P.n_prot >> P.box_x >> P.box_y >> P.sigma >> P.rcut
             >> P.k_bond >> P.r_bond >> P.dt >> P.steps >> P.temperature
             >> P.gamma >> P.seed))
        throw std::runtime_error("bad scalar parameters in " + path);

    // Then the 6 unique entries of the symmetric 3x3 epsilon matrix.
    double eHH, eHT, eHP, eTT, eTP, ePP;
    if (!(in >> eHH >> eHT >> eHP >> eTT >> eTP >> ePP))
        throw std::runtime_error("bad epsilon matrix (need 6 values) in " + path);

    // Fill the symmetric matrix: eps[i][j] == eps[j][i].
    P.eps[BEAD_HEAD][BEAD_HEAD] = eHH;
    P.eps[BEAD_HEAD][BEAD_TAIL] = P.eps[BEAD_TAIL][BEAD_HEAD] = eHT;
    P.eps[BEAD_HEAD][BEAD_PROT] = P.eps[BEAD_PROT][BEAD_HEAD] = eHP;
    P.eps[BEAD_TAIL][BEAD_TAIL] = eTT;
    P.eps[BEAD_TAIL][BEAD_PROT] = P.eps[BEAD_PROT][BEAD_TAIL] = eTP;
    P.eps[BEAD_PROT][BEAD_PROT] = ePP;

    // Derived count: 3 beads per lipid + the protein beads.
    P.n_beads = 3 * P.n_lipids + P.n_prot;

    // Validate so the demo fails loudly rather than simulating nonsense.
    if (P.n_lipids <= 0 || P.n_prot < 0 || P.box_x <= 0 || P.box_y <= 0 ||
        P.sigma <= 0 || P.rcut <= 0 || P.dt <= 0 || P.steps < 0)
        throw std::runtime_error("invalid (non-positive) parameter in " + path);
    return P;
}

// ---------------------------------------------------------------------------
// build_system: deterministically lay out a flat bilayer + a protein column.
//
//   Geometry (z is the bilayer normal, the membrane plane is x-y):
//     * The lipids tile a grid of `side x side` columns where side = ceil(sqrt
//       (n_lipids/2)): each grid cell hosts ONE upper-leaflet lipid and ONE
//       lower-leaflet lipid (so n_lipids must be even-ish; we just fill in
//       index order and stop at n_lipids).
//     * Upper leaflet: HEAD at z = +zhead, two TAIL beads stepping DOWN toward
//       the midplane (z = +zhead - bond, -2*bond). Lower leaflet mirrors it.
//       The two leaflets' tails meet near z = 0 -> a bilayer.
//     * The protein beads form a short vertical column near the box centre,
//       threaded through the membrane core (a minimal "embedded protein").
//
//   Velocities: tiny seeded pseudo-random kicks (deterministic via SplitMix64)
//   so the thermostat has something to act on but the run still reproduces.
// ---------------------------------------------------------------------------
void build_system(const SimParams& P, System& sys) {
    const int N = P.n_beads;
    sys.pos.assign(N, Vec3{0, 0, 0});
    sys.vel.assign(N, Vec3{0, 0, 0});
    sys.inv_mass.assign(N, 0.0);
    sys.mass.assign(N, 0.0);
    sys.type.assign(N, BEAD_HEAD);
    sys.bond_i.clear();
    sys.bond_j.clear();

    const double bond = P.r_bond;        // vertical spacing between bonded beads
    // Head height above the midplane. With 3 beads stepping DOWN by `bond`, the
    // lowest tail (t2) sits at z = zhead - 2*bond. We choose zhead = 2.5*bond so
    // t2 lands at +0.5*bond: the upper leaflet stays entirely in z>0 and the
    // lower entirely in z<0, so a column's beads never coincide with the mirror
    // column's beads (which would make r=0 and blow up the LJ force). The two
    // leaflets' t2 tips end up 1*bond apart -> a clean bilayer core.
    const double zhead = 2.5 * bond;     // head height above the midplane
    const double lipid_mass = 1.0;       // reduced mass of a lipid bead
    const double prot_mass = 2.0;        // protein beads are heavier inclusions

    // How many columns to lay the lipids in. We want roughly a square patch.
    const int pairs = (P.n_lipids + 1) / 2;            // upper+lower share a cell
    int side = 1;
    while (side * side < pairs) ++side;                 // smallest side with side^2 >= pairs
    const double dx = P.box_x / side;                   // column spacing in x
    const double dy = P.box_y / side;                   // column spacing in y

    int b = 0;                                          // running bead index
    // Helper lambda to emit one lipid (3 beads + 2 bonds) at grid (gx,gy),
    // leaflet sign s = +1 (upper) or -1 (lower).
    auto emit_lipid = [&](int gx, int gy, int s) {
        const double x = (gx + 0.5) * dx;
        const double y = (gy + 0.5) * dy;
        // HEAD bead: farthest from the midplane (toward the water).
        const int head = b;
        sys.pos[head] = Vec3{x, y, s * zhead};
        sys.type[head] = BEAD_HEAD;
        // First TAIL bead: one bond toward the midplane.
        const int t1 = b + 1;
        sys.pos[t1] = Vec3{x, y, s * (zhead - bond)};
        sys.type[t1] = BEAD_TAIL;
        // Second TAIL bead: another bond toward the midplane (tips meet at z~0).
        const int t2 = b + 2;
        sys.pos[t2] = Vec3{x, y, s * (zhead - 2 * bond)};
        sys.type[t2] = BEAD_TAIL;
        for (int k = 0; k < 3; ++k) {
            sys.mass[b + k] = lipid_mass;
            sys.inv_mass[b + k] = 1.0 / lipid_mass;
        }
        // Bonds: head-tail1 and tail1-tail2 (a 3-bead rod).
        sys.bond_i.push_back(head); sys.bond_j.push_back(t1);
        sys.bond_i.push_back(t1);   sys.bond_j.push_back(t2);
        b += 3;
    };

    // Fill grid cells in row-major order, alternating which leaflet each lipid
    // goes to, until we have placed all n_lipids.
    int placed = 0;
    for (int gy = 0; gy < side && placed < P.n_lipids; ++gy) {
        for (int gx = 0; gx < side && placed < P.n_lipids; ++gx) {
            emit_lipid(gx, gy, +1); ++placed;          // upper-leaflet lipid
            if (placed < P.n_lipids) { emit_lipid(gx, gy, -1); ++placed; }  // lower
        }
    }

    // PROTEIN column: a stack of beads through the membrane, spanning the core.
    // We place it at a CELL CORNER (offset by a quarter box) rather than a cell
    // centre, so the column threads BETWEEN lipid columns instead of landing on
    // top of one (which would overlap beads and spike the LJ energy).
    const double cx = P.box_x * 0.25, cy = P.box_y * 0.25;
    for (int p = 0; p < P.n_prot; ++p) {
        // Centre the column on the midplane: z runs from -(n_prot-1)/2*bond up.
        const double z = (p - (P.n_prot - 1) * 0.5) * bond;
        sys.pos[b] = Vec3{cx, cy, z};
        sys.type[b] = BEAD_PROT;
        sys.mass[b] = prot_mass;
        sys.inv_mass[b] = 1.0 / prot_mass;
        // Bond consecutive protein beads into a chain so it stays a column.
        if (p > 0) { sys.bond_i.push_back(b - 1); sys.bond_j.push_back(b); }
        ++b;
    }

    // Seed tiny initial velocities so the thermostat has dynamics to damp.
    // Deterministic: hashed from (seed, bead, axis) via the shared PRNG, scaled
    // small. Identical on CPU and GPU because both read these positions/vels.
    const double v0 = 0.05;   // initial speed scale (reduced units)
    for (int i = 0; i < N; ++i) {
        sys.vel[i].x = v0 * normal01(P.seed ^ 0x11, rng_key(0, i, 0));
        sys.vel[i].y = v0 * normal01(P.seed ^ 0x11, rng_key(0, i, 1));
        sys.vel[i].z = v0 * normal01(P.seed ^ 0x11, rng_key(0, i, 2));
    }
}

// ---------------------------------------------------------------------------
// compute_forces (CPU): the all-pairs LJ + bonded force evaluation, written in
// the SAME order the GPU kernel uses so the floating-point sums match.
//   For each bead i:
//     * loop j = 0..N-1, j != i: add the truncated LJ force from j (minimum
//       image in x,y), looking up eps by the (type_i, type_j) pair.
//     * loop over all bonds; if bead i is an endpoint, add the spring force.
//   Returns the per-bead force array `f`. Energy is summed separately by
//   total_potential_energy(); keeping force and energy in separate passes keeps
//   each loop single-purpose and easy to read.
// This mirrors compute_force_on_bead() in kernels.cu EXACTLY.
// ---------------------------------------------------------------------------
static void compute_forces(const SimParams& P, const System& sys, std::vector<Vec3>& f) {
    const int N = P.n_beads;
    f.assign(N, Vec3{0, 0, 0});
    double u_unused;   // lj_force/bond_force write energy; here we ignore it.

    for (int i = 0; i < N; ++i) {
        Vec3 fi{0, 0, 0};
        const Vec3 ri = sys.pos[i];
        const int  ti = sys.type[i];
        // --- non-bonded LJ: every other bead within the cutoff ---
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            const Vec3 dij = min_image_delta(ri, sys.pos[j], P.box_x, P.box_y);
            const double e = eps_of(P, ti, sys.type[j]);
            fi = fi + lj_force(dij, e, P.sigma, P.rcut, &u_unused);
        }
        f[i] = fi;
    }
    // --- bonded springs: walk the bond list once, add to BOTH endpoints ---
    // (Equal and opposite by Newton's 3rd law; doing it here avoids re-scanning
    // the whole bond list inside the per-bead loop.)
    for (std::size_t bnd = 0; bnd < sys.bond_i.size(); ++bnd) {
        const int i = sys.bond_i[bnd], j = sys.bond_j[bnd];
        const Vec3 dij = min_image_delta(sys.pos[i], sys.pos[j], P.box_x, P.box_y);
        const Vec3 fij = bond_force(dij, P.k_bond, P.r_bond, &u_unused);
        f[i] = f[i] + fij;
        f[j] = f[j] - fij;
    }
}

// ---------------------------------------------------------------------------
// simulate_cpu: the serial velocity-Verlet + Langevin loop (the trusted run).
//   Per step:
//     0) thermostat force on each bead (friction + deterministic random kick),
//        added to the previous conservative force -> total force for half-kick A
//     A) v += f/m * dt/2 ;  x += v*dt           (kick + drift, verlet_kick_drift)
//     B) recompute conservative forces at the new x
//     C) re-add the thermostat force at x_new, v_half, then v += f/m * dt/2
//
//   We follow the standard "BAOAB-lite" splitting where the Langevin force is
//   applied alongside the conservative force in BOTH half-kicks. Order of float
//   operations here is fixed -> the GPU, doing the identical order, matches.
// ---------------------------------------------------------------------------
void simulate_cpu(const SimParams& P, System& sys) {
    const int N = P.n_beads;
    std::vector<Vec3> f;
    compute_forces(P, sys, f);   // initial conservative forces

    for (int step = 0; step < P.steps; ++step) {
        // (A) half-kick + drift using (conservative + Langevin) force.
        for (int i = 0; i < N; ++i) {
            const Vec3 fl = langevin_force(sys.vel[i], sys.mass[i], P.gamma,
                                           P.temperature, P.dt, P.seed, step, i);
            const Vec3 ftot = f[i] + fl;
            verlet_kick_drift(sys.pos[i], sys.vel[i], ftot, sys.inv_mass[i], P.dt);
        }
        // (B) recompute conservative forces at the new positions.
        compute_forces(P, sys, f);
        // (C) final half-kick using (new conservative + Langevin) force.
        for (int i = 0; i < N; ++i) {
            const Vec3 fl = langevin_force(sys.vel[i], sys.mass[i], P.gamma,
                                           P.temperature, P.dt, P.seed, step, i);
            const Vec3 ftot = f[i] + fl;
            verlet_kick(sys.vel[i], ftot, sys.inv_mass[i], P.dt);
        }
    }
}

// ---------------------------------------------------------------------------
// bilayer_thickness: mean z of upper-leaflet HEAD beads minus mean z of
// lower-leaflet HEAD beads. We identify a head as belonging to the upper
// leaflet by the sign of its z (the build places upper heads at +z, lower at
// -z; during a short run they stay on their side). This is the headline
// "membrane intact?" observable.
// ---------------------------------------------------------------------------
double bilayer_thickness(const SimParams& P, const System& sys) {
    double zu = 0, zl = 0;
    int nu = 0, nl = 0;
    for (int i = 0; i < P.n_beads; ++i) {
        if (sys.type[i] != BEAD_HEAD) continue;
        if (sys.pos[i].z >= 0) { zu += sys.pos[i].z; ++nu; }
        else                   { zl += sys.pos[i].z; ++nl; }
    }
    const double mu = nu ? zu / nu : 0.0;
    const double ml = nl ? zl / nl : 0.0;
    return mu - ml;
}

// total_potential_energy: re-walk pairs + bonds summing U (not forces). Used as
// a second physics diagnostic in the report. O(N^2); fine for the tiny sample.
double total_potential_energy(const SimParams& P, const System& sys) {
    const int N = P.n_beads;
    double U = 0.0, u;
    // Non-bonded LJ: each unordered pair ONCE (j > i) to avoid double counting.
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) {
            const Vec3 dij = min_image_delta(sys.pos[i], sys.pos[j], P.box_x, P.box_y);
            const double e = eps_of(P, sys.type[i], sys.type[j]);
            lj_force(dij, e, P.sigma, P.rcut, &u);   // we only want u here
            U += u;
        }
    // Bonds.
    for (std::size_t bnd = 0; bnd < sys.bond_i.size(); ++bnd) {
        const int i = sys.bond_i[bnd], j = sys.bond_j[bnd];
        const Vec3 dij = min_image_delta(sys.pos[i], sys.pos[j], P.box_x, P.box_y);
        bond_force(dij, P.k_bond, P.r_bond, &u);
        U += u;
    }
    return U;
}
