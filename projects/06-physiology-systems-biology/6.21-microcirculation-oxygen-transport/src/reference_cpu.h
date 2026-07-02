// ===========================================================================
// src/reference_cpu.h  --  Problem containers + CPU reference declarations
// ---------------------------------------------------------------------------
// Project 6.21 : Microcirculation & Oxygen Transport
//
// This header defines the DATA the solver works on and the SHARED per-point
// evaluator used by both the CPU reference and the GPU kernel:
//
//   * OxySource   -- one O2-releasing capillary segment (a point source).
//   * TissueGrid  -- the regular 3-D lattice of tissue points where we want PO2,
//                    plus the physiological parameters of the problem.
//   * solve_point() -- the __host__ __device__ core: compute PO2 at ONE grid
//                    point by summing over all sources. The CPU loops it over
//                    every point; the GPU runs one thread per point. Sharing this
//                    function is what makes CPU==GPU exact (PATTERNS.md section 2).
//
// It is pure C++ (no CUDA types) so kernels.cu can #include it and reuse the same
// structs and the same math. The actual physics kernels live in oxygen.h.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "oxygen.h"   // OXY_HD, green_function, dist3, hill_saturation, mm_consumption, clamp_po2

// ---------------------------------------------------------------------------
// OxySource: one capillary segment modelled as a point source of oxygen.
//   x,y,z   -- position of the segment centre (um).
//   q       -- source strength: how much PO2 this segment injects per unit
//              Green's-function weight. In the loader we DERIVE q from the
//              segment's blood PO2 via the Hill saturation (see load_problem),
//              so a well-oxygenated segment is a stronger source.
// A plain POD struct so it copies trivially to the device.
// ---------------------------------------------------------------------------
struct OxySource {
    double x, y, z;   // position (um)
    double q;         // effective source strength (mmHg * um, lumped)
};

// ---------------------------------------------------------------------------
// TissueGrid: the regular lattice of field points + the problem's physiology.
//   The tissue block is nx*ny*nz points spaced `spacing` um apart, with the
//   lattice origin at (0,0,0). We flatten the 3-D index (ix,iy,iz) to a linear
//   index  idx = (iz*ny + iy)*nx + ix  (x fastest) so one GPU thread owns one idx.
//
//   Physiology carried alongside so solve_point() is self-contained:
//     po2_inflow -- baseline arterial PO2 (mmHg) the tissue starts from.
//     m0, km     -- Michaelis-Menten consumption parameters (see oxygen.h).
//   (p50, n and the per-source blood PO2 are consumed at load time to set q, so
//    they don't need to travel to the device.)
// ---------------------------------------------------------------------------
struct TissueGrid {
    int    nx = 0, ny = 0, nz = 0;   // lattice dimensions (points per axis)
    double spacing = 0.0;            // grid spacing (um)
    double po2_inflow = 0.0;         // baseline / arterial PO2 (mmHg)
    double m0 = 0.0;                 // max Michaelis-Menten consumption
    double km = 0.0;                 // Michaelis constant (mmHg)
};

// Total number of tissue grid points (one output PO2 per point).
OXY_HD inline int grid_size(const TissueGrid& g) { return g.nx * g.ny * g.nz; }

// Physical coordinates (um) of grid point with linear index idx. The inverse of
// the flattening idx = (iz*ny+iy)*nx + ix. Shared so CPU and GPU place points
// identically.
OXY_HD inline void grid_point_coords(const TissueGrid& g, int idx,
                                     double& x, double& y, double& z) {
    const int ix = idx % g.nx;              // x varies fastest
    const int iy = (idx / g.nx) % g.ny;     // then y
    const int iz = idx / (g.nx * g.ny);     // then z (slowest)
    x = ix * g.spacing;
    y = iy * g.spacing;
    z = iz * g.spacing;
}

// ---------------------------------------------------------------------------
// solve_point: the HEART of the project. Compute the steady-state PO2 at ONE
//   tissue grid point by superposing the Green's-function contribution of every
//   source and subtracting the local consumption.
//
//   PO2_i = clamp( po2_inflow
//                  + sum_j  src[j].q * G(|x_i - x_j|)
//                  - M(po2_inflow) )
//
//   Parameters:
//     g          -- the tissue grid + physiology (by value; it is tiny/POD).
//     src        -- pointer to the N_src source array (device or host memory).
//     n_src      -- number of sources.
//     idx        -- which grid point to evaluate (0 .. grid_size(g)-1).
//   Returns: PO2 (mmHg) at that grid point.
//
//   Determinism note: the source loop runs in a FIXED index order (0..n_src-1),
//   so the double-precision partial sums accumulate in the same order on CPU and
//   GPU -> the results match to round-off. (See PATTERNS.md section 3.)
//   Complexity: O(n_src) per point -> O(N_grid * n_src) total. That O(N^2) blow-up
//   is exactly what the real Secomb solver replaces with a fast multipole method;
//   THEORY.md discusses that. Here we do the honest direct sum so the learner sees
//   the baseline the FMM accelerates.
// ---------------------------------------------------------------------------
OXY_HD inline double solve_point(const TissueGrid& g,
                                 const OxySource* src, int n_src, int idx) {
    double x, y, z;
    grid_point_coords(g, idx, x, y, z);

    // Superpose every source's Green's-function contribution.
    double po2 = g.po2_inflow;
    for (int j = 0; j < n_src; ++j) {
        const double r = dist3(x, y, z, src[j].x, src[j].y, src[j].z);
        po2 += src[j].q * green_function(r);
    }

    // Subtract the tissue's O2 demand (evaluated at the inflow PO2 -- a fixed,
    // non-iterative background sink; see oxygen.h mm_consumption for why this is
    // a teaching simplification of the fully-coupled Secomb solve).
    po2 -= mm_consumption(g.po2_inflow, g.m0, g.km);

    return clamp_po2(po2);   // PO2 >= 0
}

// ---------------------------------------------------------------------------
// Problem I/O + the CPU reference. Implemented in reference_cpu.cpp (host only).
// ---------------------------------------------------------------------------

// A fully-loaded problem: the grid/physiology + the derived source list.
struct OxyProblem {
    TissueGrid grid;
    std::vector<OxySource> sources;   // one entry per capillary segment
};

// Load a problem from the sample text format (see data/README.md). The file
// lists the grid, physiology, and one line per capillary segment (position +
// blood PO2); the loader converts each segment's blood PO2 into a source
// strength q via the Hill saturation curve.
OxyProblem load_problem(const std::string& path);

// CPU reference: evaluate solve_point() at every grid point, serially. This is
// the trusted baseline the GPU field is checked against. Fills `po2` (sized to
// grid_size(problem.grid)).
void solve_cpu(const OxyProblem& problem, std::vector<double>& po2);
