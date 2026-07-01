// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial fractional-step NSE reference
// ---------------------------------------------------------------------------
// Project 6.3 : Hemodynamics / Blood-Flow CFD   (reduced-scope teaching version)
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is written to be
//   OBVIOUSLY correct: plain nested loops over cells, no parallelism, no tricks,
//   calling the SAME per-cell physics (nse_channel.h) the GPU kernels call. When
//   the GPU and this CPU reference agree to machine precision, we trust the GPU.
//
//   Compiled by the host C++ compiler only (no CUDA syntax here). The one place
//   both paths share is nse_channel.h, whose NSE_HD functions compile on both.
//
// THE ALGORITHM (Chorin's fractional-step / projection method), per time step:
//   1. predictor : provisional velocity u* (advection + diffusion + body force)
//   2. divergence: build the pressure-Poisson right-hand side (rho/dt) div(u*)
//   3. pressure  : `p_iters` Jacobi sweeps solving laplacian(p) = rhs
//   4. corrector : u = u* - (dt/rho) grad(p)  -> divergence-free velocity
//   Each sub-step is a nearest-neighbour stencil over all cells, so it maps 1:1
//   onto the GPU kernels in kernels.cu. See ../THEORY.md for the derivation.
//
// READ THIS AFTER: reference_cpu.h, nse_channel.h. Compare with kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>   // std::fill
#include <cmath>       // std::sqrt (pulled in by nse_channel.h)
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error
#include <vector>

// ---------------------------------------------------------------------------
// load_channel: parse the one-line sample format (see data/README.md):
//     nx ny steps p_iters h dt rho gx nu0 nu_inf lambda n_cy a_cy
//   All whitespace-separated. We validate the essential invariants so a bad
//   file fails loudly (a demo running on garbage teaches nothing).
// ---------------------------------------------------------------------------
ChannelParams load_channel(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open channel parameter file: " + path);
    ChannelParams p;
    if (!(in >> p.nx >> p.ny >> p.steps >> p.p_iters >> p.h >> p.dt
             >> p.rho >> p.gx >> p.nu0 >> p.nu_inf >> p.lambda >> p.n_cy >> p.a_cy))
        throw std::runtime_error(
            "bad parameters (expected 'nx ny steps p_iters h dt rho gx nu0 "
            "nu_inf lambda n_cy a_cy') in " + path);
    if (p.nx < 3 || p.ny < 3 || p.steps < 0 || p.p_iters < 0)
        throw std::runtime_error("invalid grid/iteration counts in " + path);
    if (p.h <= 0.0 || p.dt <= 0.0 || p.rho <= 0.0 || p.nu0 <= 0.0)
        throw std::runtime_error("invalid physical parameters (h,dt,rho,nu0>0) in " + path);
    return p;
}

// ---------------------------------------------------------------------------
// effective_nu: the constant viscosity used by the analytic Poiseuille check.
//   For the demo (Newtonian, nu0==nu_inf) this is simply nu0. If shear thinning
//   is enabled the analytic parabola no longer applies exactly; THEORY.md notes
//   this and the demo runs in Newtonian mode so the check is meaningful.
// ---------------------------------------------------------------------------
double effective_nu(const ChannelParams& p) {
    return p.nu0;
}

// ---------------------------------------------------------------------------
// poiseuille_umax: analytic steady centreline velocity for Newtonian channel
//   flow driven by a uniform body force gx between walls a distance H apart:
//        u(y) = gx/(2 nu) * ( (H/2)^2 - (y - H/2)^2 ),  u_max at the centreline.
//   With no-slip walls at the first/last grid rows, the fluid gap is
//   H = (ny-1)*h. This is the science-level target the simulation converges to.
// ---------------------------------------------------------------------------
double poiseuille_umax(const ChannelParams& p) {
    const double H = (p.ny - 1) * p.h;          // wall-to-wall distance
    const double half = 0.5 * H;
    return p.gx * half * half / (2.0 * effective_nu(p));
}

// ---------------------------------------------------------------------------
// nse_cpu: the serial reference solver. Allocates the fields, runs the time
//   loop, and returns the final u,v. Uses ping-pong (double) buffers for
//   velocity and for the Jacobi pressure solve so every sweep reads a frozen
//   snapshot and writes a fresh one (matching the GPU exactly).
// ---------------------------------------------------------------------------
void nse_cpu(const ChannelParams& p,
             std::vector<double>& u_final,
             std::vector<double>& v_final) {
    const std::size_t N = static_cast<std::size_t>(p.nx) * p.ny;

    // Velocity buffers (u,v) plus their predictor (us,vs). Start from REST.
    std::vector<double> u(N, 0.0), v(N, 0.0);
    std::vector<double> us(N, 0.0), vs(N, 0.0);
    // Pressure ping-pong buffers and the Poisson right-hand side.
    std::vector<double> pa(N, 0.0), pb(N, 0.0), rhs(N, 0.0);

    for (int s = 0; s < p.steps; ++s) {
        // --- STEP 1: predictor u* over every cell (walls -> zero) --------------
        for (int y = 0; y < p.ny; ++y)
            for (int x = 0; x < p.nx; ++x)
                predictor_cell(x, y, p.nx, p.ny, p.h, p.dt, p.gx,
                               p.nu0, p.nu_inf, p.lambda, p.n_cy, p.a_cy,
                               u.data(), v.data(), us.data(), vs.data());

        // --- STEP 2: build the Poisson RHS = (rho/dt) div(u*) -----------------
        const double scale = p.rho / p.dt;
        for (int y = 0; y < p.ny; ++y)
            for (int x = 0; x < p.nx; ++x)
                rhs[idx(x, y, p.nx)] =
                    scale * divergence_cell(x, y, p.nx, p.ny, p.h, us.data(), vs.data());

        // --- STEP 3: Jacobi solve laplacian(p)=rhs (ping-pong pa<->pb) --------
        std::fill(pa.begin(), pa.end(), 0.0);  // fresh initial guess each step
        double* p_src = pa.data();
        double* p_dst = pb.data();
        for (int it = 0; it < p.p_iters; ++it) {
            for (int y = 0; y < p.ny; ++y)
                for (int x = 0; x < p.nx; ++x)
                    p_dst[idx(x, y, p.nx)] =
                        pressure_jacobi_cell(x, y, p.nx, p.ny, p.h, p_src,
                                             rhs[idx(x, y, p.nx)]);
            double* tmp = p_src; p_src = p_dst; p_dst = tmp;   // swap buffers
        }
        // `p_src` now holds the latest pressure after the final swap.

        // --- STEP 4: corrector u = u* - (dt/rho) grad(p) ----------------------
        for (int y = 0; y < p.ny; ++y)
            for (int x = 0; x < p.nx; ++x)
                corrector_cell(x, y, p.nx, p.ny, p.h, p.dt, p.rho,
                               us.data(), vs.data(), p_src,
                               u.data(), v.data());
    }

    u_final = u;
    v_final = v;
}
