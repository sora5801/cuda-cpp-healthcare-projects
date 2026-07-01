// ===========================================================================
// src/reference_cpu.h  --  Slab problem definition, S_N quadrature, CPU solver
// ---------------------------------------------------------------------------
// Project 5.6 : GPU Boltzmann Transport (Deterministic Dose)
//
// Pure C++ (no CUDA), so kernels.cu can reuse SlabProblem / the quadrature and
// the plain host compiler can build reference_cpu.cpp. The actual per-cell
// transport physics lives in the shared boltzmann_sn.h (host+device identical).
//
// READ THIS AFTER: boltzmann_sn.h (the physics).  BEFORE: reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "boltzmann_sn.h"   // shared per-cell S_N update (host+device)

// ---------------------------------------------------------------------------
// SlabProblem: one deterministic-transport job -- a 1-D slab of `ncell` equal
// cells, each with its own cross-sections and fixed source, plus the S_N order
// and iteration controls. All materials are given PER CELL so we can model
// heterogeneous slabs (e.g. a low-density "lung" layer between "tissue"), which
// is exactly where deterministic LBTE shines over pencil-beam superposition.
// ---------------------------------------------------------------------------
struct SlabProblem {
    int    ncell = 0;      // number of spatial cells along x
    int    nord  = 0;      // S_N order = number of discrete ordinates (even, >=2)
    double width = 0.0;    // total slab thickness [cm]  (cell width h = width/ncell)
    int    max_iter = 0;   // maximum source iterations (safety cap)
    double tol = 0.0;      // convergence tolerance on the scalar flux (relative L-inf)
    double psi_left_bc  = 0.0;  // incoming angular flux at the left  face (mu>0 dirs)
    double psi_right_bc = 0.0;  // incoming angular flux at the right face (mu<0 dirs)

    // Per-cell material arrays (all length ncell). Units: cross-sections in 1/cm,
    // source in particles/cm^3/s. sigma_a (absorption) = sigma_t - sigma_s is
    // implied, not stored, so the data cannot become internally inconsistent.
    std::vector<double> sigma_t;   // total macroscopic cross-section per cell
    std::vector<double> sigma_s;   // scattering macroscopic cross-section per cell
    std::vector<double> q;         // fixed (external) isotropic source per cell

    double h() const { return width / ncell; }   // uniform cell width [cm]
};

// ---------------------------------------------------------------------------
// SnQuadrature: the Gauss-Legendre ordinate set {mu_n, w_n} on [-1,1].
//   Gauss-Legendre is the standard S_N angular quadrature: it integrates the
//   scalar-flux integral (2) exactly for angular polynomials up to degree
//   2*nord-1, and its weights satisfy sum(w_n) = 2 (so a flat psi=1 gives
//   phi = 2, matching integral_{-1}^{1} 1 dmu = 2). Nodes come in +/- pairs, so
//   half sweep forward (mu>0) and half backward (mu<0).
// ---------------------------------------------------------------------------
struct SnQuadrature {
    std::vector<double> mu;   // [nord] direction cosines (signed, in (-1,1))
    std::vector<double> w;    // [nord] positive weights, sum(w) = 2
};

// Build the nord-point Gauss-Legendre quadrature (nord even). Nodes/weights are
// computed by Newton iteration on the Legendre polynomial (see the .cpp), so the
// same values feed both the CPU and GPU paths.
SnQuadrature make_gauss_legendre(int nord);

// Load a SlabProblem from the text format documented in data/README.md:
//   line 1: ncell nord width max_iter tol psi_left_bc psi_right_bc
//   then ncell lines: sigma_t sigma_s q       (one cell per line)
SlabProblem load_slab(const std::string& path);

// ---------------------------------------------------------------------------
// solve_sn_cpu: the trusted CPU reference. Runs source iteration to convergence
//   and returns the converged scalar flux phi (length ncell) plus the number of
//   iterations actually taken. This is (a) the teaching baseline that makes the
//   GPU speed-up legible and (b) the ground truth the GPU flux is checked
//   against within tolerance. Physics is the shared boltzmann_sn.h, so CPU and
//   GPU differ only by floating-point summation order.
//     p     the problem
//     quad  the ordinate set
//     phi   OUT: converged scalar flux per cell
//     iters OUT: iterations taken (for reporting; run-varying only if not
//               converged, which we treat as a failure)
// ---------------------------------------------------------------------------
void solve_sn_cpu(const SlabProblem& p, const SnQuadrature& quad,
                  std::vector<double>& phi, int& iters);

// Absorbed-dose proxy per cell. True absorbed dose is (Sigma_a * phi) * E / rho;
// for a fixed particle energy and unit density that is proportional to the
// ENERGY-DEPOSITION RATE density we report here, dep = Sigma_a * phi
// [particles absorbed / cm^3 / s], with Sigma_a = Sigma_t - Sigma_s. Shared so
// the CPU and GPU "dose" match exactly.
void deposition_field(const SlabProblem& p, const std::vector<double>& phi,
                      std::vector<double>& dep);
