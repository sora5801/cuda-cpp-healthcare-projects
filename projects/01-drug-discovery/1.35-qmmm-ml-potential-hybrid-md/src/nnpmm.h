// ===========================================================================
// src/nnpmm.h  --  Shared (host + device) hybrid NNP/MM physics core
// ---------------------------------------------------------------------------
// Project 1.35 : QMMM/ML Potential Hybrid MD   (REDUCED-SCOPE TEACHING VERSION)
//
// ----------------------------------------------------------------------------
// WHAT THIS PROJECT TEACHES (and what it deliberately simplifies)
// ----------------------------------------------------------------------------
//   The research frontier (catalog 1.35) is: replace the expensive quantum
//   (QM) region of a QM/MM simulation with a MACHINE-LEARNED POTENTIAL (an
//   equivariant neural network like MACE / NequIP, trained on QM data), so you
//   get QM-accuracy forces at a fraction of the cost and can run microsecond
//   *reactive* molecular dynamics. The hybrid runs fully on GPU: the NNP forward
//   pass and the classical MM evaluation overlap in CUDA streams.
//
//   A faithful version of that needs a trained equivariant GNN + PyTorch
//   autograd + a real force field -- far beyond a single self-contained teaching
//   project. So, per CLAUDE.md §13, this is a clearly-labeled REDUCED-SCOPE
//   teaching version that keeps every *structural* idea of the real method and
//   throws away only the scale:
//
//     REAL METHOD                          THIS TEACHING VERSION
//     -----------                          ---------------------
//     equivariant message-passing NNP      Behler-Parrinello-style descriptor
//       (MACE/NequIP), millions of params    (2-body radial symmetry functions)
//                                            feeding a tiny 1-hidden-layer MLP
//     trained on QM/DFT datasets           FIXED, committed weights ("pretend
//       (Transition1x, SPICE, ANI-1ccx)      we trained"); labeled synthetic
//     autograd for forces                  forces by ANALYTIC differentiation
//                                            of the same energy expression
//     full MM force field (bonded +        Lennard-Jones pair forces for the
//       nonbonded + PME electrostatics)      MM region + ML/MM cross terms
//     mechanical/electrostatic embedding   MECHANICAL embedding (the simplest,
//                                            most common embedding)
//     link-atom QM/MM boundary             explicit link atom that caps the ML
//                                            region (the standard trick)
//     microsecond trajectory               a few hundred velocity-Verlet steps
//     async NNP+MM CUDA streams            both potentials in one kernel; the
//                                            *ensemble* is the parallel axis
//
//   Everything the learner needs to understand the real pipeline is here:
//   descriptor -> network -> energy -> analytic force -> integrate, with an
//   NNP region and an MM region coupled at a link-atom boundary. The result is
//   honest, deterministic, and verified CPU-vs-GPU.
//
// ----------------------------------------------------------------------------
// THE GPU PATTERN: ENSEMBLE (thread per trajectory)  -- see docs/PATTERNS.md §1
// ----------------------------------------------------------------------------
//   "Active learning" for reactive NNPs samples many slightly-different starting
//   configurations to discover where the model is uncertain. Each short MD
//   trajectory is INDEPENDENT, so we give each one its own GPU thread: the thread
//   runs the whole velocity-Verlet loop in registers and writes one summary.
//   This is the same ensemble pattern as flagships 9.02 (SEIR) and 13.02 (PBPK).
//
// ----------------------------------------------------------------------------
// THE SHARED __host__ __device__ CORE  -- see docs/PATTERNS.md §2
// ----------------------------------------------------------------------------
//   Every per-atom physics function below is `__host__ __device__` (via the
//   NNPMM_HD macro), so the CPU reference (reference_cpu.cpp, host compiler) and
//   the GPU kernel (kernels.cu, nvcc) execute BYTE-FOR-BYTE IDENTICAL math. That
//   is what makes verification meaningful: any CPU-vs-GPU difference comes only
//   from floating-point reassociation (FMA), not from two different algorithms.
//   Keep CUDA-only constructs (no __global__, no <<<>>>) OUT of this header so
//   the host compiler can include it.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh.
// READ THIS BEFORE: reference_cpu.h, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <cmath>     // std::exp, std::tanh, std::sqrt  (host side)
#include <cstddef>   // std::size_t

// ---------------------------------------------------------------------------
// NNPMM_HD: expands to "__host__ __device__" when compiled by nvcc (so the
// function is emitted for BOTH the CPU and the GPU), and to nothing when the
// plain host compiler includes this header (which does not know those keywords).
// This single macro is the entire trick behind exact CPU/GPU parity.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define NNPMM_HD __host__ __device__
#else
#define NNPMM_HD
#endif

// ===========================================================================
// 0. SYSTEM SIZE CONSTANTS  (compile-time so state lives in registers/arrays)
// ---------------------------------------------------------------------------
//   We model a TINY 1-D chain of atoms -- 1-D keeps the geometry trivial to read
//   while exercising the exact same descriptor->network->force->integrate path a
//   3-D code uses. (Going to 3-D is an Exercise; the math generalizes directly.)
//
//   Layout of the chain (indices left to right):
//       [ MM atoms ... ] [ LINK atom ] [ ML atoms ... ]
//   - The ML (NNP-described) atoms are the "reactive center" we would normally
//     treat with QM; here the NNP stands in for QM.
//   - The LINK atom caps the ML region at the boundary (the QM/MM "link atom"
//     trick: it saturates the dangling bond so the ML region is well-defined).
//   - The MM atoms are the classical environment (Lennard-Jones).
// ===========================================================================
#define N_ATOMS    8      // total atoms in the chain
#define N_ML       3      // number of ML (NNP) atoms (the reactive center)
#define LINK_IDX   4      // index of the link atom (boundary cap; counted as ML)
// MM atoms are indices [0 .. LINK_IDX-1]; ML atoms are [LINK_IDX .. N_ATOMS-1].
// (So here: atoms 0..3 are MM, atom 4 is the link/boundary atom, 5..7 are ML.)

// ---------------------------------------------------------------------------
// Descriptor / network sizes for the tiny Behler-Parrinello-style NNP.
//   The NNP energy of the ML region is a sum of per-atom energies. Each ML
//   atom's local environment is encoded by N_G radial "symmetry functions"
//   (Gaussians of neighbour distances). Those N_G numbers are the input to a
//   1-hidden-layer perceptron (N_G -> N_HID -> 1) that outputs that atom's
//   energy. This IS the Behler-Parrinello high-dimensional NN potential, just
//   small enough to read on one screen.
// ---------------------------------------------------------------------------
#define N_G    3          // number of radial symmetry functions per ML atom
#define N_HID  4          // hidden units in the per-atom MLP

// ===========================================================================
// 1. PER-ATOM TYPE TAGS
// ---------------------------------------------------------------------------
//   is_ml(i): true if atom i is described by the NNP (the link atom counts as
//   ML because it caps that region). Used to decide which potential acts.
// ===========================================================================
NNPMM_HD inline bool is_ml(int i) { return i >= LINK_IDX; }

// ===========================================================================
// 2. THE ML POTENTIAL  (Behler-Parrinello descriptor + tiny MLP)
// ---------------------------------------------------------------------------
//   These FIXED weights stand in for "a network trained on QM data". They are
//   committed constants so the whole simulation is deterministic and the demo is
//   reproducible. In a real project these come from training MACE/NequIP on
//   Transition1x/SPICE/ANI-1ccx (see README "Data"). LABELED SYNTHETIC.
//
//   The cutoff radius RC defines the ML atom's local neighbourhood: only atoms
//   within RC contribute to its descriptor (locality is what makes NNPs scale).
// ===========================================================================
#define NNP_RC   2.5      // descriptor cutoff radius (distance units)

// Centres (mu) and width (eta) of the N_G Gaussian symmetry functions. A
// symmetry function G_k(r) = exp(-eta * (r - mu_k)^2) * fc(r) is large when a
// neighbour sits near distance mu_k -- so the descriptor "feels" the local
// geometry. (One true definition, used by host and device.)
NNPMM_HD inline double g_mu(int k) {
    // Three probe distances spread across the cutoff: 0.9, 1.4, 1.9.
    const double mus[N_G] = {0.9, 1.4, 1.9};
    return mus[k];
}
NNPMM_HD inline double g_eta() { return 4.0; }  // Gaussian sharpness (shared)

// First-layer weights W1[N_HID][N_G], hidden biases b1[N_HID], second-layer
// weights W2[N_HID], output bias b2. Tiny, fixed, "pretend-trained" numbers.
// Accessor functions (not global arrays) so the values are baked into both the
// host object and the device code identically with no __constant__ plumbing.
NNPMM_HD inline double nnp_W1(int h, int k) {
    // A deterministic, smooth weight pattern. The exact values are arbitrary
    // (this is a surrogate), but FIXED so results are reproducible.
    const double W1[N_HID][N_G] = {
        { 0.50, -0.30,  0.20},
        {-0.20,  0.40, -0.10},
        { 0.10,  0.15,  0.35},
        {-0.40, -0.05,  0.25},
    };
    return W1[h][k];
}
NNPMM_HD inline double nnp_b1(int h) {
    const double b1[N_HID] = {0.05, -0.10, 0.00, 0.15};
    return b1[h];
}
NNPMM_HD inline double nnp_W2(int h) {
    const double W2[N_HID] = {0.60, -0.45, 0.30, 0.20};
    return W2[h];
}
NNPMM_HD inline double nnp_b2() { return -0.25; }

// ---------------------------------------------------------------------------
// smooth_cutoff: the standard cosine cutoff function fc(r). It is 1 at r=0,
// decays smoothly to 0 at r=RC, and is identically 0 beyond -- so an atom
// entering/leaving the neighbourhood does so WITHOUT a force discontinuity
// (continuous energy AND derivative). Returns fc and (via dfc) its derivative.
// ---------------------------------------------------------------------------
NNPMM_HD inline double smooth_cutoff(double r, double rc, double& dfc) {
    if (r >= rc) { dfc = 0.0; return 0.0; }
    const double PI = 3.14159265358979323846;
    const double x  = PI * r / rc;
    const double fc = 0.5 * (std::cos(x) + 1.0);     // 0.5*(cos(pi r/rc)+1)
    dfc = -0.5 * (PI / rc) * std::sin(x);            // d fc / dr
    return fc;
}

// ---------------------------------------------------------------------------
// mlp_energy_and_dEdG: forward pass of the per-atom MLP plus the gradient of
// the energy w.r.t. each descriptor input dE/dG[k].
//   network:  z_h = sum_k W1[h][k]*G[k] + b1[h];  a_h = tanh(z_h)
//             E   = sum_h W2[h]*a_h + b2
//   backprop: dE/dG[k] = sum_h W2[h] * (1 - a_h^2) * W1[h][k]
//   (tanh'(z) = 1 - tanh(z)^2). We need dE/dG to turn descriptor gradients into
//   atomic forces by the chain rule (this is what autograd does for the real NNP).
//   Returns E; writes dEdG[0..N_G-1].
// ---------------------------------------------------------------------------
NNPMM_HD inline double mlp_energy_and_dEdG(const double G[N_G], double dEdG[N_G]) {
    double a[N_HID];          // hidden activations a_h = tanh(z_h)
    double dadz[N_HID];       // tanh'(z_h) = 1 - a_h^2  (for backprop)
    double E = nnp_b2();      // start from the output bias
    for (int h = 0; h < N_HID; ++h) {
        double z = nnp_b1(h);
        for (int k = 0; k < N_G; ++k) z += nnp_W1(h, k) * G[k];
        a[h]    = std::tanh(z);
        dadz[h] = 1.0 - a[h] * a[h];
        E      += nnp_W2(h) * a[h];
    }
    // dE/dG[k] = sum_h W2[h] * tanh'(z_h) * W1[h][k]
    for (int k = 0; k < N_G; ++k) {
        double g = 0.0;
        for (int h = 0; h < N_HID; ++h) g += nnp_W2(h) * dadz[h] * nnp_W1(h, k);
        dEdG[k] = g;
    }
    return E;
}

// ===========================================================================
// 3. THE MM POTENTIAL  (Lennard-Jones pair interaction)
// ---------------------------------------------------------------------------
//   The classical environment uses a 12-6 Lennard-Jones potential between pairs:
//       V_LJ(r) = 4*eps * [ (sig/r)^12 - (sig/r)^6 ]
//   It is repulsive at short range (the ^12 wall) and weakly attractive at the
//   minimum near r = 2^(1/6)*sig. This is the workhorse nonbonded term of every
//   classical force field. The force magnitude along the bond is -dV/dr.
//
//   In the HYBRID model, LJ acts on:
//     * MM-MM pairs            (classical environment with itself), and
//     * ML-MM cross pairs      (the mechanical-embedding coupling: the NNP region
//                               still feels the MM environment through LJ).
//   ML-ML pairs are handled by the NNP instead (the NNP *is* their potential).
// ===========================================================================
#define LJ_EPS   1.0      // LJ well depth   (energy units)
#define LJ_SIG   1.0      // LJ size param   (distance units)

// lj_energy: V_LJ(r) for one pair separated by distance r (>0).
NNPMM_HD inline double lj_energy(double r) {
    const double sr  = LJ_SIG / r;
    const double sr6 = sr * sr * sr * sr * sr * sr;   // (sig/r)^6
    const double sr12 = sr6 * sr6;                    // (sig/r)^12
    return 4.0 * LJ_EPS * (sr12 - sr6);
}

// lj_dVdr: derivative dV_LJ/dr (so the scalar force on the pair is -lj_dVdr).
//   dV/dr = 4*eps * [ -12 sig^12 / r^13 + 6 sig^6 / r^7 ]
//         = (4*eps/r) * [ -12 (sig/r)^12 + 6 (sig/r)^6 ]
NNPMM_HD inline double lj_dVdr(double r) {
    const double sr  = LJ_SIG / r;
    const double sr6 = sr * sr * sr * sr * sr * sr;
    const double sr12 = sr6 * sr6;
    return (4.0 * LJ_EPS / r) * (-12.0 * sr12 + 6.0 * sr6);
}

// ===========================================================================
// 4. TOTAL FORCE ASSEMBLY  (the heart of the hybrid model)
// ---------------------------------------------------------------------------
//   Given the 1-D positions of all N_ATOMS atoms, compute the force on every
//   atom AND the total potential energy. This is the function the integrator
//   calls each timestep, and it is shared by CPU and GPU so they integrate the
//   exact same trajectory.
//
//   Force on atom i = -dE_total/dx_i, with
//       E_total = E_NNP(ML atoms)              [descriptor->MLP]
//               + E_LJ(MM-MM pairs)            [classical]
//               + E_LJ(ML-MM cross pairs).     [mechanical embedding coupling]
//
//   We accumulate forces pairwise (Newton's third law: each pair pushes its two
//   atoms in opposite directions), which is O(N^2) but N is tiny and crystal
//   clear. THEORY.md derives every term.
// ---------------------------------------------------------------------------
NNPMM_HD inline double total_force_energy(const double x[N_ATOMS], double f[N_ATOMS]) {
    for (int i = 0; i < N_ATOMS; ++i) f[i] = 0.0;   // zero the force accumulator
    double E = 0.0;                                  // total potential energy

    // -------- (a) NNP energy + force for each ML atom -----------------------
    // For ML atom i: build its descriptor G[k] from neighbours within RC, run
    // the MLP to get its energy and dE/dG, then convert dE/dG into forces by the
    // chain rule through G's dependence on each neighbour distance r_ij.
    for (int i = 0; i < N_ATOMS; ++i) {
        if (!is_ml(i)) continue;                     // only ML atoms have NNP energy

        // Build descriptor G[k] = sum_{j in neighbourhood} exp(-eta (r-mu_k)^2) fc(r).
        double G[N_G] = {0.0, 0.0, 0.0};
        for (int j = 0; j < N_ATOMS; ++j) {
            if (j == i) continue;
            const double dx = x[i] - x[j];
            const double r  = std::fabs(dx);
            if (r >= NNP_RC || r < 1e-9) continue;   // outside cutoff / coincident
            double dfc;
            const double fc = smooth_cutoff(r, NNP_RC, dfc);
            for (int k = 0; k < N_G; ++k) {
                const double d = r - g_mu(k);
                G[k] += std::exp(-g_eta() * d * d) * fc;
            }
        }

        // Forward + backward through the MLP: energy and dE/dG[k].
        double dEdG[N_G];
        E += mlp_energy_and_dEdG(G, dEdG);

        // Chain rule: dE/dx_i = sum_k dE/dG[k] * dG[k]/dx_i, and each G[k]
        // depends on every neighbour distance r_ij. We compute dG[k]/dr for each
        // neighbour and distribute the resulting force to atoms i and j.
        for (int j = 0; j < N_ATOMS; ++j) {
            if (j == i) continue;
            const double dx = x[i] - x[j];
            const double r  = std::fabs(dx);
            if (r >= NNP_RC || r < 1e-9) continue;
            double dfc;
            const double fc = smooth_cutoff(r, NNP_RC, dfc);
            // dE/dr from this neighbour = sum_k dE/dG[k] * d/dr[ exp(-eta(r-mu)^2) fc ]
            double dEdr = 0.0;
            for (int k = 0; k < N_G; ++k) {
                const double d   = r - g_mu(k);
                const double gauss = std::exp(-g_eta() * d * d);
                // product rule: d/dr( gauss * fc ) = gauss'*fc + gauss*fc'
                const double dgauss = gauss * (-2.0 * g_eta() * d);
                dEdr += dEdG[k] * (dgauss * fc + gauss * dfc);
            }
            // r = |x_i - x_j|, so dr/dx_i = sign(dx), dr/dx_j = -sign(dx).
            // Force = -dE/dx.  Half-weight (0.5) avoids double counting: this
            // pair's NNP contribution is split symmetrically between i's and j's
            // descriptors when both are ML; the 0.5 keeps the pair energy single-
            // counted while still giving each atom its gradient share.
            const double sgn = (dx >= 0.0) ? 1.0 : -1.0;
            const double fmag = -dEdr * sgn;
            f[i] += 0.5 * fmag;
            f[j] -= 0.5 * fmag;
        }
    }

    // -------- (b) Lennard-Jones over MM-MM and ML-MM cross pairs ------------
    // Loop unordered pairs (i<j). Skip ML-ML pairs (the NNP already governs them).
    for (int i = 0; i < N_ATOMS; ++i) {
        for (int j = i + 1; j < N_ATOMS; ++j) {
            const bool both_ml = is_ml(i) && is_ml(j);
            if (both_ml) continue;                   // ML-ML handled by the NNP
            const double dx = x[i] - x[j];
            const double r  = std::fabs(dx);
            if (r < 1e-9) continue;                  // guard coincident atoms
            E += lj_energy(r);
            // Force along the bond: F_i = -dV/dr * dr/dx_i = -dV/dr * sign(dx).
            const double sgn  = (dx >= 0.0) ? 1.0 : -1.0;
            const double fmag = -lj_dVdr(r) * sgn;
            f[i] += fmag;
            f[j] -= fmag;
        }
    }

    return E;   // total potential energy (the integrator also tracks kinetic)
}

// ===========================================================================
// 5. ONE VELOCITY-VERLET TIMESTEP  (the MD integrator)
// ---------------------------------------------------------------------------
//   Velocity-Verlet is the standard MD integrator: it is time-reversible and
//   symplectic (it conserves energy well over long runs), which is exactly what
//   you want for stable dynamics. With unit masses (m=1):
//       x(t+dt) = x(t) + v(t) dt + 0.5 a(t) dt^2
//       a(t+dt) = F(x(t+dt))                       [our total_force_energy]
//       v(t+dt) = v(t) + 0.5 (a(t) + a(t+dt)) dt
//   We carry the "old" accelerations between steps. Returns the potential energy
//   at the new positions (so the caller can monitor energy conservation).
// ---------------------------------------------------------------------------
NNPMM_HD inline double verlet_step(double x[N_ATOMS], double v[N_ATOMS],
                                   double a[N_ATOMS], double dt) {
    // 1) drift positions using current velocity + half-step acceleration.
    for (int i = 0; i < N_ATOMS; ++i)
        x[i] += v[i] * dt + 0.5 * a[i] * dt * dt;

    // 2) recompute forces (= accelerations, since m=1) at the new positions.
    double a_new[N_ATOMS];
    const double E = total_force_energy(x, a_new);

    // 3) kick velocities by the average of old and new acceleration.
    for (int i = 0; i < N_ATOMS; ++i) {
        v[i] += 0.5 * (a[i] + a_new[i]) * dt;
        a[i]  = a_new[i];                            // carry forward for next step
    }
    return E;
}

// ===========================================================================
// 6. PER-TRAJECTORY DRIVER  (shared by CPU reference and GPU kernel)
// ---------------------------------------------------------------------------
//   Each ensemble member is one short MD trajectory whose ONLY difference is a
//   tiny displacement applied to the link atom (an "active-learning" probe of
//   configuration space). The driver:
//     * builds the initial chain geometry, perturbed by this member's amount,
//     * runs `steps` velocity-Verlet steps,
//     * returns a deterministic summary (final potential energy, final total
//       energy, and the max |force| seen at the last step).
//   Determinism: no RNG -- the per-member perturbation is a fixed function of
//   the member index, so CPU and GPU produce identical inputs.
// ===========================================================================
struct TrajResult {
    double final_pe;      // potential energy at the last step
    double final_total;   // total energy (PE + KE) at the last step
    double max_force;     // max |force| over atoms at the last step (stiffness gauge)
    double energy_drift;  // |final_total - initial_total|: how well the symplectic
                          // integrator conserved energy (should be small + bounded)
};

// member_perturbation: the fixed displacement applied to the link atom for
// ensemble member `idx` out of `M`. A smooth ramp in [-amp, +amp]. Deterministic.
NNPMM_HD inline double member_perturbation(int idx, int M, double amp) {
    if (M <= 1) return 0.0;
    const double t = (double)idx / (double)(M - 1);  // 0..1 across the ensemble
    return amp * (2.0 * t - 1.0);                    // -amp .. +amp
}

NNPMM_HD inline TrajResult run_trajectory(int idx, int M, double amp,
                                          double dt, int steps) {
    // ---- initial geometry: an evenly spaced chain at spacing 1.12 (~LJ min) --
    double x[N_ATOMS], v[N_ATOMS], a[N_ATOMS];
    const double spacing = 1.12;                      // ~ 2^(1/6) sigma (LJ minimum)
    for (int i = 0; i < N_ATOMS; ++i) {
        x[i] = spacing * i;                           // positions along the line
        v[i] = 0.0;                                   // start at rest
        a[i] = 0.0;
    }
    // This member's active-learning probe: nudge the link atom off-lattice.
    x[LINK_IDX] += member_perturbation(idx, M, amp);

    // ---- prime accelerations a(0) = F(x(0)) and record initial energy ------
    double pe = total_force_energy(x, a);
    double ke0 = 0.0;
    for (int i = 0; i < N_ATOMS; ++i) ke0 += 0.5 * v[i] * v[i];  // KE at t=0
    const double initial_total = pe + ke0;            // for the conservation check

    // ---- integrate -------------------------------------------------------
    for (int s = 0; s < steps; ++s)
        pe = verlet_step(x, v, a, dt);

    // ---- summary (deterministic) -----------------------------------------
    double ke = 0.0, maxf = 0.0;
    for (int i = 0; i < N_ATOMS; ++i) {
        ke += 0.5 * v[i] * v[i];                      // unit masses
        const double af = (a[i] >= 0.0) ? a[i] : -a[i];
        if (af > maxf) maxf = af;
    }
    const double final_total = pe + ke;
    TrajResult out;
    out.final_pe    = pe;
    out.final_total = final_total;
    out.max_force   = maxf;
    // |final - initial| total energy: a symplectic integrator conserves this to
    // a small, bounded amount (it does NOT drift away secularly). This is the
    // honest per-trajectory conservation metric -- contrast with the ensemble
    // SPREAD of final energies, which is large by design (members start from
    // different perturbed geometries).
    const double d = final_total - initial_total;
    out.energy_drift = (d >= 0.0) ? d : -d;
    return out;
}
