// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial TERMA ray-trace + CCC superposition
// ---------------------------------------------------------------------------
// Project 5.4 : Collapsed-Cone / Superposition-Convolution Dose  (2-D teaching model)
// Compiled by the host C++ compiler only. It calls the SAME physics functions
// (ccc_physics.h) that the GPU kernels call, so its dose grid must match the GPU's
// integer dose grid EXACTLY. See reference_cpu.h for the stage contracts.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// load_dose_problem: parse the tiny text sample.
//   Sample layout (whitespace-separated; see data/README.md):
//     line 1 : nx ny voxel_cm mu_over_rho psi0 n_cones kernel_a dose_scale beam_x0 beam_x1
//     then   : nx*ny density values rho (row-major, y=0 is the top/entry row)
//   We validate aggressively: a dose engine fed a malformed grid should stop, not
//   silently produce a wrong plan.
// ---------------------------------------------------------------------------
DoseProblem load_dose_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open density/parameter file: " + path);

    DoseProblem prob;
    CccParams& P = prob.P;
    if (!(in >> P.nx >> P.ny >> P.voxel_cm >> P.mu_over_rho >> P.psi0
             >> P.n_cones >> P.kernel_a >> P.dose_scale >> P.beam_x0 >> P.beam_x1)) {
        throw std::runtime_error(
            "bad header (expected 'nx ny voxel_cm mu_over_rho psi0 n_cones "
            "kernel_a dose_scale beam_x0 beam_x1') in " + path);
    }
    // Sanity-check every parameter; each bound below would otherwise cause an
    // out-of-range access or a physically meaningless computation.
    if (P.nx <= 0 || P.ny <= 0)
        throw std::runtime_error("nx, ny must be positive in " + path);
    if (P.voxel_cm <= 0.0 || P.mu_over_rho <= 0.0 || P.kernel_a <= 0.0 || P.dose_scale <= 0.0)
        throw std::runtime_error("voxel_cm, mu_over_rho, kernel_a, dose_scale must be > 0 in " + path);
    if (P.n_cones < 1 || P.n_cones > 8)
        throw std::runtime_error("n_cones must be in [1,8] (this teaching model) in " + path);
    if (P.beam_x0 < 0 || P.beam_x1 >= P.nx || P.beam_x0 > P.beam_x1)
        throw std::runtime_error("beam columns [beam_x0,beam_x1] out of range in " + path);

    const size_t n = static_cast<size_t>(P.nx) * static_cast<size_t>(P.ny);
    prob.rho.resize(n);
    for (size_t i = 0; i < n; ++i) {
        if (!(in >> prob.rho[i]))
            throw std::runtime_error("density map has fewer than nx*ny values in " + path);
        if (prob.rho[i] < 0.0f)
            throw std::runtime_error("negative density in " + path);
    }
    return prob;
}

// ---------------------------------------------------------------------------
// terma_cpu: STAGE 1 -- ray-trace TERMA down each irradiated beam column.
//   For a column x, we march from the top row (y=0) downward. `rad_above` is the
//   radiological path length (integral of rho * dl) from the top SURFACE to the
//   TOP of the current voxel; adding half the current cell gives the depth to the
//   voxel CENTER (the standard "dose-to-voxel-center" convention, and it matches
//   the GPU exactly). TERMA outside the beam columns stays 0.
//
//   This is the 2-D, axis-aligned specialization of Siddon's ray-voxel tracer:
//   because the ray is vertical, the intersection length in every voxel is just
//   voxel_cm, so the general Siddon parametric intersection collapses to a clean
//   loop. THEORY.md derives the full oblique-ray version.
// ---------------------------------------------------------------------------
void terma_cpu(const DoseProblem& prob, std::vector<double>& terma) {
    const CccParams& P = prob.P;
    terma.assign(static_cast<size_t>(P.nx) * P.ny, 0.0);

    for (int x = P.beam_x0; x <= P.beam_x1; ++x) {
        double rad_above = 0.0;                 // radiological depth to TOP of current voxel
        for (int y = 0; y < P.ny; ++y) {
            const double rho_here = prob.rho[static_cast<size_t>(y) * P.nx + x];
            // Depth to this voxel's CENTER = density above + half of this cell.
            const double rad_center = rad_above + 0.5 * rho_here * P.voxel_cm;
            terma[static_cast<size_t>(y) * P.nx + x] = terma_at(P, rad_center);
            // Advance the running depth by the FULL cell for the next voxel down.
            rad_above += rho_here * P.voxel_cm;
        }
    }
}

// ---------------------------------------------------------------------------
// spread_one_source: the heart of the collapsed-cone superposition, shared in
//   spirit with the GPU (kernels.cu implements the identical recurrence per
//   thread). For ONE source voxel s with TERMA T_s, deposit its scatter dose
//   along all n_cones cone rays and accumulate integer dose-units.
//
//   Along cone c we walk outward k = 1,2,... voxels. `carry` is the dose still
//   travelling in the ray (starts as T_s's share of this cone). At each step of
//   radiological length d_rad, a fraction (1 - transmit) of `carry` is DEPOSITED
//   in the current voxel and the rest continues:
//       deposit = carry * (1 - transmit)
//       carry  -= deposit             (== carry * transmit)
//   This is the exact discrete form of the analytic cone kernel a*exp(-a r): the
//   deposited fractions telescope to (1 - transmit_total), so the cone conserves
//   the energy it was handed (up to the ray leaving the grid). We stop when the
//   ray exits the grid or `carry` has decayed below the quantization floor
//   (nothing left worth depositing).
//
//   Local to this translation unit (`static`) because only dose_cpu uses it.
// ---------------------------------------------------------------------------
static void spread_one_source(const DoseProblem& prob, int sx, int sy, double T_s,
                              std::vector<long long>& dose_units) {
    const CccParams& P = prob.P;
    const double w = cone_weight(P);            // this cone's share of T_s (1/n_cones)

    for (int c = 0; c < P.n_cones; ++c) {
        const int dx = ccc_cone_dx(c);
        const int dy = ccc_cone_dy(c);
        const double step_cm = ccc_step_cm(c, P.voxel_cm);
        double carry = T_s * w;                 // dose entering the ray from source s

        int x = sx, y = sy;
        // March outward until the ray leaves the grid or the carry is exhausted.
        for (int k = 0; k < P.nx + P.ny; ++k) {  // bounded by the longest grid ray
            x += dx; y += dy;
            if (x < 0 || x >= P.nx || y < 0 || y >= P.ny) break;   // left the grid
            const double rho_here = prob.rho[static_cast<size_t>(y) * P.nx + x];
            const double transmit = cone_transmit(P, step_cm, rho_here);
            const double deposit  = carry * (1.0 - transmit);      // deposited this step
            carry -= deposit;                                       // == carry * transmit
            // Quantize + accumulate (integer => order-independent, GPU-matching).
            dose_units[static_cast<size_t>(y) * P.nx + x] += dose_to_units(P, deposit);
            if (carry * P.dose_scale < 0.5) break;  // < half a unit left: negligible
        }
    }
}

// ---------------------------------------------------------------------------
// dose_cpu: STAGE 2 -- superpose every source voxel's collapsed-cone spread.
//   Serial over all source voxels; the GPU parallelizes this exact loop with one
//   thread per source voxel (see kernels.cu). Because every deposit is quantized
//   to an integer and added, the final grid is deterministic and equals the GPU's.
// ---------------------------------------------------------------------------
void dose_cpu(const DoseProblem& prob, const std::vector<double>& terma,
              std::vector<long long>& dose_units) {
    const CccParams& P = prob.P;
    dose_units.assign(static_cast<size_t>(P.nx) * P.ny, 0LL);

    for (int sy = 0; sy < P.ny; ++sy) {
        for (int sx = 0; sx < P.nx; ++sx) {
            const double T_s = terma[static_cast<size_t>(sy) * P.nx + sx];
            if (T_s <= 0.0) continue;           // no energy released here -> nothing to spread
            spread_one_source(prob, sx, sy, T_s, dose_units);
        }
    }
}
