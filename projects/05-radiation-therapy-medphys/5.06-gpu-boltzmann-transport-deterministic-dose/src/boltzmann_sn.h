// ===========================================================================
// src/boltzmann_sn.h  --  Shared (host + device) discrete-ordinates (S_N) core
// ---------------------------------------------------------------------------
// Project 5.6 : GPU Boltzmann Transport (Deterministic Dose)
//
// THE MODEL (reduced-scope teaching version)
//   The full linear Boltzmann transport equation (LBTE) tracks particle fluence
//   over a 6-DoF phase space (x,y,z, theta,phi, E) -- ~10^9..10^10 unknowns at
//   clinical resolution. That is a research code (Acuros XB, Denovo). To TEACH
//   the exact algorithm the catalog names -- discrete ordinates (S_N), source
//   iteration (SI), a transport SWEEP with upwind/diamond differencing -- we
//   solve the smallest problem that still contains all of those ideas:
//
//     * 1-D SLAB geometry (a stack of homogeneous-per-cell layers along x),
//     * MONO-ENERGETIC particles (one energy group; no photon->electron cascade),
//     * ISOTROPIC scattering (Legendre order 0; scattering source is flat in angle).
//
//   The steady-state 1-D transport equation for the angular flux psi(x, mu) is
//
//       mu d/dx psi(x,mu) + Sigma_t(x) psi(x,mu)
//                         = (Sigma_s(x)/2) * phi(x) + q(x)/2,          (1)
//
//   where mu = cos(theta) in [-1,1] is the direction cosine, Sigma_t the total
//   macroscopic cross-section (1/cm), Sigma_s the scattering cross-section,
//   q(x) a fixed isotropic source (particles/cm^3/s), and the SCALAR flux
//
//       phi(x) = integral_{-1}^{1} psi(x,mu) dmu.                       (2)
//
//   The right-hand side of (1) couples all directions through phi -- that
//   coupling is why we ITERATE (source iteration): guess phi, solve (1) for
//   every direction (a "sweep"), recompute phi from (2), repeat until phi stops
//   changing.
//
// THE DISCRETE ORDINATES (S_N) IDEA
//   Replace the continuous angle integral (2) by an N-point Gauss-Legendre
//   quadrature: pick N direction cosines mu_n (the "ordinates") and weights w_n
//   with sum(w_n) = 2 (so a flat psi integrates to phi = 2*psi). Then
//
//       phi(x) ~= sum_n w_n psi_n(x),        psi_n(x) := psi(x, mu_n).   (3)
//
//   For each fixed ordinate mu_n, equation (1) is a simple 1-D ODE in x that we
//   integrate cell by cell (the "sweep") in the direction of travel:
//     * mu_n > 0  : particles move +x, sweep left  -> right (upwind = left edge),
//     * mu_n < 0  : particles move -x, sweep right -> left  (upwind = right edge).
//
// DIAMOND DIFFERENCE (the spatial discretization used here)
//   Over a cell of width h with total cross-section Sigma_t and a known cell
//   source Q (the RHS of (1), constant in the cell), integrating (1) and using
//   the diamond-difference closure psi_center = (psi_in + psi_out)/2 gives a
//   closed-form update for the outgoing edge flux:
//
//       psi_out = ( (|mu|/h - Sigma_t/2) psi_in + Q ) / ( |mu|/h + Sigma_t/2 ). (4)
//
//   The cell-average flux (what feeds the scalar flux) is the diamond average
//   psi_avg = (psi_in + psi_out)/2. Equation (4) is THE per-cell kernel of a
//   deterministic transport code; everything else is bookkeeping around it.
//
//   This whole per-cell update lives here as ONE __host__ __device__ function so
//   the CPU reference and the GPU kernel run byte-for-byte identical math (the
//   key to exact verification). SN_HD expands to __host__ __device__ under nvcc
//   and to nothing under the host compiler (the HD-macro idiom, PATTERNS.md §2).
//
// READ THIS AFTER: reference_cpu.h (the SlabProblem struct + loader).
// READ THIS BEFORE: kernels.cu (the GPU sweep), reference_cpu.cpp (the CPU sweep).
// ===========================================================================
#pragma once

#include <cstddef>   // std::size_t

// HD-macro idiom: decorate shared functions so nvcc emits BOTH a host and a
// device version; under the plain host compiler the decorators do not exist.
#ifdef __CUDACC__
#define SN_HD __host__ __device__
#else
#define SN_HD
#endif

// ---------------------------------------------------------------------------
// sn_diamond_out: the diamond-difference outgoing edge flux, equation (4).
//   Given the incoming edge angular flux and a (constant-in-cell) source Q,
//   return the outgoing edge angular flux for one ordinate crossing one cell.
//
//   Parameters (all doubles; physical units in brackets):
//     psi_in    incoming edge angular flux   [particles / cm^2 / s / steradian-ish]
//     abs_mu    |mu_n|, the ordinate's speed along x, in (0,1]           [dimensionless]
//     h         cell width                                              [cm]
//     sigma_t   total macroscopic cross-section in this cell            [1/cm]
//     Q         cell source = (Sigma_s*phi + q)/2 for this cell         [same as psi/cm]
//   Returns: psi_out, the outgoing edge angular flux.
//
//   WHY a closed form (not a matrix solve): in 1-D each ordinate's sweep is a
//   short forward recurrence -- edge k+1 depends only on edge k -- so we never
//   assemble a linear system. In multi-D this same closure becomes cuSPARSE's
//   upwind triangular solve; here it is one FMA-friendly division.
// ---------------------------------------------------------------------------
SN_HD inline double sn_diamond_out(double psi_in, double abs_mu, double h,
                                   double sigma_t, double Q) {
    const double a = abs_mu / h;          // transport "conductance" across the cell
    const double half_s = 0.5 * sigma_t;  // half the removal term (diamond closure)
    // Numerator: the incoming flux weighted by (a - half_s) plus the cell source.
    // Denominator: (a + half_s) > 0 always (a>0, sigma_t>=0) -> no divide-by-zero.
    return ((a - half_s) * psi_in + Q) / (a + half_s);
}

// ---------------------------------------------------------------------------
// sn_cell_avg: the diamond cell-average angular flux from the two edge values.
//   psi_avg = (psi_in + psi_out)/2. This average (integrated over angle with the
//   quadrature weights) is what forms the scalar flux phi that closes the
//   iteration -- see equation (3). Kept as a named function so the CPU and GPU
//   compute the average identically.
// ---------------------------------------------------------------------------
SN_HD inline double sn_cell_avg(double psi_in, double psi_out) {
    return 0.5 * (psi_in + psi_out);
}

// ---------------------------------------------------------------------------
// sn_cell_source: the (constant-in-cell) right-hand side Q of equation (1).
//   Q = (Sigma_s * phi + q) / 2. The /2 is because the isotropic emission is
//   spread uniformly over the mu-interval [-1,1] whose measure is 2, so each
//   direction receives half the (per-unit-mu) emission density. phi here is the
//   scalar flux from the PREVIOUS source iteration (that lag is the essence of
//   source iteration; see THEORY §algorithm).
//     sigma_s   scattering cross-section in this cell   [1/cm]
//     phi_old   scalar flux in this cell, previous iterate [particles/cm^2/s]
//     q         fixed external source in this cell       [particles/cm^3/s]
// ---------------------------------------------------------------------------
SN_HD inline double sn_cell_source(double sigma_s, double phi_old, double q) {
    return 0.5 * (sigma_s * phi_old + q);
}

// ---------------------------------------------------------------------------
// sn_sweep_one_ordinate: integrate equation (1) across the WHOLE slab for a
//   SINGLE ordinate n, accumulating this ordinate's weighted contribution into
//   a private scalar-flux tally. This is the per-thread work on the GPU (one
//   thread owns one ordinate) and the inner loop body on the CPU.
//
//   Thread/loop ownership: the caller fixes the ordinate (mu_n, w_n); this
//   function performs the spatial recurrence, which is INHERENTLY SEQUENTIAL in
//   x (edge k+1 needs edge k). The parallelism is ACROSS ordinates, not within a
//   sweep -- an honest statement of where the GPU speed-up comes from.
//
//   Parameters:
//     mu           the signed ordinate cosine mu_n in [-1,1] (sign picks sweep dir)
//     w            the quadrature weight w_n (>0), used to weight this contribution
//     ncell        number of spatial cells
//     h            uniform cell width [cm]
//     sigma_t      [ncell] total cross-section per cell            (device/host ptr)
//     sigma_s      [ncell] scattering cross-section per cell       (device/host ptr)
//     q            [ncell] fixed source per cell                   (device/host ptr)
//     phi_old      [ncell] scalar flux from the previous iterate   (device/host ptr)
//     psi_left_bc  incoming angular flux at the LEFT  boundary (for mu>0 sweeps)
//     psi_right_bc incoming angular flux at the RIGHT boundary (for mu<0 sweeps)
//     phi_contrib  [ncell] OUT: += w * psi_avg per cell (this ordinate's share)
//
//   Side effect: ADDS into phi_contrib[i]. On the GPU each thread writes to its
//   OWN private per-ordinate row (no sharing) and a separate reduction sums the
//   rows in a fixed order -> deterministic, no atomics (PATTERNS.md §3). On the
//   CPU we accumulate straight into the shared phi_new in ordinate order.
// ---------------------------------------------------------------------------
SN_HD inline void sn_sweep_one_ordinate(double mu, double w, int ncell, double h,
                                        const double* sigma_t, const double* sigma_s,
                                        const double* q, const double* phi_old,
                                        double psi_left_bc, double psi_right_bc,
                                        double* phi_contrib) {
    const double abs_mu = (mu < 0.0) ? -mu : mu;   // |mu_n|, the along-x speed

    if (mu > 0.0) {
        // ----- Forward sweep: particles travel +x, so integrate left -> right.
        // The upwind (known) edge is the LEFT edge of each cell; we carry the
        // outgoing edge of cell i in as the incoming edge of cell i+1.
        double psi_in = psi_left_bc;                       // flux entering cell 0's left face
        for (int i = 0; i < ncell; ++i) {
            const double Q      = sn_cell_source(sigma_s[i], phi_old[i], q[i]);
            const double psi_out = sn_diamond_out(psi_in, abs_mu, h, sigma_t[i], Q);
            const double avg    = sn_cell_avg(psi_in, psi_out);
            phi_contrib[i] += w * avg;                     // this ordinate's weighted share
            psi_in = psi_out;                              // hand off to the next cell
        }
    } else {
        // ----- Backward sweep: particles travel -x, so integrate right -> left.
        // The upwind edge is now the RIGHT edge of each cell; we walk cells in
        // decreasing index. mu = 0 cannot occur (Gauss-Legendre nodes are never 0
        // for even N), so this branch handles exactly the mu < 0 ordinates.
        double psi_in = psi_right_bc;                      // flux entering cell (ncell-1)'s right face
        for (int i = ncell - 1; i >= 0; --i) {
            const double Q      = sn_cell_source(sigma_s[i], phi_old[i], q[i]);
            const double psi_out = sn_diamond_out(psi_in, abs_mu, h, sigma_t[i], Q);
            const double avg    = sn_cell_avg(psi_in, psi_out);
            phi_contrib[i] += w * avg;
            psi_in = psi_out;                              // hand off to the cell to the left
        }
    }
}
