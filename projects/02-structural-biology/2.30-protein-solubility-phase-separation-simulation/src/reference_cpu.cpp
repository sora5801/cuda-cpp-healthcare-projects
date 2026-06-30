// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial velocity-Verlet HPS reference
// ---------------------------------------------------------------------------
// Project 2.30 : Protein Solubility & Phase Separation Simulation
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU is checked against. It is written to be OBVIOUSLY
//   correct: a single readable loop over beads, no parallelism, no cleverness --
//   so that when GPU and CPU agree we believe the GPU. It calls the SAME shared
//   force routine (bead_force in hps_model.h) the kernel does, in the SAME fixed
//   pair order, so the two FP64 trajectories track each other to ~1e-9.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, hps_model.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// load_system: parse the committed sample.
//   File format (whitespace/newlines, comments with '#'; see data/README.md):
//     line 1 (header): n_beads n_chains box sigma epsilon r_cut k_bond r0 mass dt n_steps
//     then n_beads rows:  x y z vx vy vz lambda
//   chain_id is derived from row order: chain c owns rows [c*chain_len, ...).
//   We validate shapes so a malformed file fails loudly rather than simulating
//   garbage.
// ---------------------------------------------------------------------------
System load_system(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open system file: " + path);

    // Read the file token by token, skipping lines that begin with '#'. We do
    // this by reading whole lines and stripping comments, then re-tokenizing --
    // simple and robust for a tiny teaching sample.
    std::stringstream clean;
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t h = line.find('#');           // strip trailing comments
        if (h != std::string::npos) line = line.substr(0, h);
        clean << line << ' ';
    }

    System s;
    SimParams& p = s.p;
    if (!(clean >> p.n_beads >> p.n_chains >> p.box >> p.sigma >> p.epsilon
                >> p.r_cut >> p.k_bond >> p.r0 >> p.mass >> p.dt >> p.n_steps))
        throw std::runtime_error("bad header in " + path +
            " (expected 'n_beads n_chains box sigma epsilon r_cut k_bond r0 mass dt n_steps')");

    if (p.n_beads <= 0 || p.n_chains <= 0 || p.n_beads % p.n_chains != 0)
        throw std::runtime_error("n_beads must be a positive multiple of n_chains");
    p.chain_len = p.n_beads / p.n_chains;
    if (p.box <= 0 || p.sigma <= 0 || p.r_cut <= 0 || p.dt <= 0 || p.n_steps < 0)
        throw std::runtime_error("invalid simulation parameters in " + path);

    const int N = p.n_beads;
    s.x.resize(N); s.y.resize(N); s.z.resize(N);
    s.vx.resize(N); s.vy.resize(N); s.vz.resize(N);
    s.lambda.resize(N); s.chain_id.resize(N);

    for (int i = 0; i < N; ++i) {
        if (!(clean >> s.x[i] >> s.y[i] >> s.z[i]
                    >> s.vx[i] >> s.vy[i] >> s.vz[i] >> s.lambda[i]))
            throw std::runtime_error("ran out of bead rows in " + path +
                " (need " + std::to_string(N) + " rows of 'x y z vx vy vz lambda')");
        s.chain_id[i] = i / p.chain_len;   // consecutive rows form a chain
    }
    return s;
}

// ---------------------------------------------------------------------------
// compute_forces (file-local helper): fill fx,fy,fz for every bead and return
//   the TOTAL potential energy. Just loops the shared bead_force() over i in
//   index order -- the exact serial analogue of "one thread per bead" on the GPU.
// ---------------------------------------------------------------------------
static double compute_forces(const System& s,
                             std::vector<double>& fx,
                             std::vector<double>& fy,
                             std::vector<double>& fz) {
    const int N = s.n();
    double pe = 0.0;
    for (int i = 0; i < N; ++i) {
        double Fx, Fy, Fz, u_half;
        bead_force(i, N, s.x.data(), s.y.data(), s.z.data(),
                   s.lambda.data(), s.chain_id.data(), s.p,
                   &Fx, &Fy, &Fz, &u_half);
        fx[i] = Fx; fy[i] = Fy; fz[i] = Fz;
        pe += u_half;          // each bead contributed half of every pair it is in
    }
    return pe;
}

// ---------------------------------------------------------------------------
// run_cpu: serial velocity-Verlet (NVE) integration.
//   Velocity-Verlet, the workhorse MD integrator, advances one step as:
//     1.  v(t + dt/2) = v(t) + (dt/2) a(t)            [half kick]
//     2.  r(t + dt)   = r(t) + dt  v(t + dt/2)         [drift]
//     3.  recompute a(t + dt) from the new positions   [new forces]
//     4.  v(t + dt)   = v(t + dt/2) + (dt/2) a(t + dt) [half kick]
//   It is time-reversible and (nearly) energy-conserving, which is why it is the
//   standard choice. a = F/m. We keep velocities for the kinetic-energy report
//   but use NO random thermostat force, so the run is fully deterministic.
// ---------------------------------------------------------------------------
void run_cpu(System s, SimSummary& out) {
    const int N = s.n();
    const double dt = s.p.dt, m = s.p.mass;
    const double half_dt_over_m = 0.5 * dt / m;   // converts force to a half-step velocity change

    std::vector<double> fx(N), fy(N), fz(N);
    double pe = compute_forces(s, fx, fy, fz);    // a(0): forces at the start

    for (int step = 0; step < s.p.n_steps; ++step) {
        // (1)+(2): half-kick velocities, then drift positions with them.
        for (int i = 0; i < N; ++i) {
            s.vx[i] += half_dt_over_m * fx[i];
            s.vy[i] += half_dt_over_m * fy[i];
            s.vz[i] += half_dt_over_m * fz[i];
            s.x[i]  += dt * s.vx[i];
            s.y[i]  += dt * s.vy[i];
            s.z[i]  += dt * s.vz[i];
            // Wrap positions back into [0, box) so the trajectory fingerprint and
            // the periodic analysis stay in-box. (Forces already use min-image,
            // so wrapping is purely bookkeeping and does not change dynamics.)
            s.x[i] -= s.p.box * std::floor(s.x[i] / s.p.box);
            s.y[i] -= s.p.box * std::floor(s.y[i] / s.p.box);
            s.z[i] -= s.p.box * std::floor(s.z[i] / s.p.box);
        }
        // (3): forces at the new positions.
        pe = compute_forces(s, fx, fy, fz);
        // (4): second half-kick using the new forces.
        for (int i = 0; i < N; ++i) {
            s.vx[i] += half_dt_over_m * fx[i];
            s.vy[i] += half_dt_over_m * fy[i];
            s.vz[i] += half_dt_over_m * fz[i];
        }
    }

    // ---- final-state summary (the numbers main.cu compares + prints) --------
    double ke = 0.0, checksum = 0.0;
    for (int i = 0; i < N; ++i) {
        ke += 0.5 * m * (s.vx[i]*s.vx[i] + s.vy[i]*s.vy[i] + s.vz[i]*s.vz[i]);
        checksum += s.x[i] + s.y[i] + s.z[i];   // a cheap whole-trajectory fingerprint
    }
    out.potential = pe;
    out.kinetic   = ke;
    out.pos_checksum = checksum;
    order_params(s.p, s.x, s.y, s.z,
                 out.max_local_density, out.mean_local_density, out.n_condensed);
}

// ---------------------------------------------------------------------------
// order_params: the deterministic phase-separation diagnostics.
//   For each bead, count how many OTHER beads lie within r_cut (its local
//   density). A condensate = a cluster of high-local-density beads; the dilute
//   phase = the low-density background. We report the max and mean local
//   density and how many beads exceed a fixed threshold (here >= 4 neighbours).
//   Counts are integers summed in index order, so the result is bit-reproducible
//   and identical whether fed CPU or GPU final positions.
// ---------------------------------------------------------------------------
void order_params(const SimParams& p,
                  const std::vector<double>& x,
                  const std::vector<double>& y,
                  const std::vector<double>& z,
                  double& max_local_density,
                  double& mean_local_density,
                  int& n_condensed) {
    const int N = static_cast<int>(x.size());
    const double rc2 = p.r_cut * p.r_cut;
    const int CONDENSED_THRESHOLD = 4;   // >= 4 neighbours within r_cut == "in a droplet"

    long long sum_counts = 0;
    int max_count = 0;
    n_condensed = 0;
    for (int i = 0; i < N; ++i) {
        int cnt = 0;
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            double dx = minimum_image(x[i] - x[j], p.box);
            double dy = minimum_image(y[i] - y[j], p.box);
            double dz = minimum_image(z[i] - z[j], p.box);
            if (dx*dx + dy*dy + dz*dz < rc2) ++cnt;
        }
        sum_counts += cnt;
        if (cnt > max_count) max_count = cnt;
        if (cnt >= CONDENSED_THRESHOLD) ++n_condensed;
    }
    max_local_density  = static_cast<double>(max_count);
    mean_local_density = N ? static_cast<double>(sum_counts) / N : 0.0;
}
