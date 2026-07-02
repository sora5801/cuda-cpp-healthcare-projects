// ===========================================================================
// src/abm_core.h  --  Shared (host + device) agent-based tissue/immune model
// ---------------------------------------------------------------------------
// Project 6.9 : Agent-Based Tissue / Immune Simulation   (see ../THEORY.md)
//
// THE MODEL (a reduced-scope teaching version of PhysiCell-style ABM)
//   Tissue is a POPULATION OF AGENTS (cells). Each cell is a soft sphere with a
//   position (x,y), a radius, and a TYPE:
//       TUMOR  cells sit in the tissue and SECRETE a chemokine into a substrate
//              field (a scalar concentration on a Cartesian grid).
//       IMMUNE cells CHEMOTAX -- they crawl UP the chemokine gradient toward the
//              tumor -- while pushing off any neighbour they overlap.
//   Every timestep does three coupled things (the three GPU kernels):
//     (1) SECRETE  : each tumor cell adds chemokine to the grid cell it sits in.
//     (2) DIFFUSE  : the chemokine field relaxes by an explicit reaction-diffusion
//                    stencil (Fick diffusion + linear decay) -- a grid PDE.
//     (3) MOVE     : each cell feels soft-sphere REPULSION from overlapping
//                    neighbours (found by SPATIAL BINNING, O(N) not O(N^2)) plus,
//                    for immune cells, a CHEMOTAXIS drift along grad(chemokine).
//                    Positions integrate by forward Euler (overdamped mechanics).
//
// WHY A GPU, AND WHICH PATTERNS
//   Two independent parallel bottlenecks, each a canonical pattern:
//     * The substrate PDE is a nearest-neighbour STENCIL over the grid
//       (like flagship 6.04 lattice-Boltzmann / 14.02 reaction-diffusion):
//       one thread per grid cell, ping-pong two buffers.
//     * Cell-cell mechanics is a pairwise neighbour search. Naive all-pairs is
//       O(N^2); SPATIAL BINNING (hash each cell into a grid bin, only test cells
//       in the 3x3 neighbouring bins) drops it to O(N). One thread per cell.
//   Secretion is a SCATTER-REDUCTION (many cells -> few grid cells) done with
//   atomicAdd -- and, to stay DETERMINISTIC and CPU-exact, we accumulate in
//   FIXED-POINT INTEGERS (integer adds commute; see PATTERNS.md §3 and 11.09/5.01).
//
//   The whole per-element physics lives HERE as __host__ __device__ inline
//   functions (ABM_HD idiom, PATTERNS.md §2), so the CPU reference and the GPU
//   kernels run byte-for-byte identical math -> exact verification.
//
// READ THIS AFTER: util/cuda_check.cuh; BEFORE: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <cstddef>   // std::size_t
#include <cstdint>   // fixed-width integer types
#include <vector>    // std::vector (host-side Cells storage; device uses raw ptrs)

// ABM_HD expands to __host__ __device__ under nvcc (so the functions run on both
// the CPU reference and inside GPU kernels) and to NOTHING under the plain host
// compiler (which does not know those decorators). Keep this header free of any
// __global__ / CUDA-only types so reference_cpu.cpp can include it too.
#ifdef __CUDACC__
#define ABM_HD __host__ __device__
#else
#define ABM_HD
#endif

// ---------------------------------------------------------------------------
// Cell types. A cell is TUMOR (secretes chemokine, otherwise passive) or IMMUNE
// (chemotaxes up the gradient). Stored as a small int so it copies trivially to
// the device and prints deterministically.
// ---------------------------------------------------------------------------
enum CellType : int { CELL_TUMOR = 0, CELL_IMMUNE = 1 };

// ---------------------------------------------------------------------------
// Structure-of-Arrays (SoA) cell state. We use SoA rather than an
// array-of-structs so that, when one thread per cell reads e.g. all the x's,
// consecutive threads touch consecutive addresses -> COALESCED global loads on
// the GPU (the memory-bandwidth win that makes the whole thing worthwhile).
// Positions are `double` for determinism/precision; the domain is [0,W]x[0,H].
// ---------------------------------------------------------------------------
struct Cells {
    int n = 0;                 // number of cells (agents)
    std::vector<double> x;     // [n] x position (micrometres, domain units)
    std::vector<double> y;     // [n] y position
    std::vector<int>    type;  // [n] CELL_TUMOR or CELL_IMMUNE
};

// ---------------------------------------------------------------------------
// Simulation parameters (loaded from the one-line sample file; see data/README).
// All lengths are in the same arbitrary "domain unit" (think micrometres); the
// grid spacing dx ties the agent world to the substrate grid.
// ---------------------------------------------------------------------------
struct AbmParams {
    // Domain / grid ---------------------------------------------------------
    int    gx = 0;             // substrate grid columns (x)
    int    gy = 0;             // substrate grid rows    (y)
    double dx = 1.0;           // grid spacing (domain units per grid cell)
    // Time ------------------------------------------------------------------
    int    steps = 0;          // number of timesteps
    double dt = 0.0;           // timestep (arbitrary units)
    // Substrate PDE ---------------------------------------------------------
    double D = 0.0;            // chemokine diffusion coefficient
    double decay = 0.0;        // first-order chemokine decay rate
    double secretion = 0.0;    // chemokine added per tumor cell per unit time
    // Cell mechanics --------------------------------------------------------
    double radius = 0.0;       // cell radius (repulsion acts within 2*radius)
    double k_rep = 0.0;        // soft-sphere repulsion stiffness
    double chemotaxis = 0.0;   // immune chemotactic speed per unit gradient
    // RNG seed for the synthetic initial placement (kept in the file so the
    // whole run is reproducible from the sample alone).
    unsigned int seed = 0;

    // Derived: number of grid cells in the substrate field.
    ABM_HD int grid_cells() const { return gx * gy; }
    // Derived: the domain extent (agents live in [0,width] x [0,height]).
    ABM_HD double width()  const { return gx * dx; }
    ABM_HD double height() const { return gy * dx; }
};

// ---------------------------------------------------------------------------
// FIXED-POINT accumulation for the secretion scatter.
//   Why: many tumor cells secrete into the SAME grid cell, so the adds collide
//   and must be atomic. A FLOAT atomicAdd is order-dependent (non-associative)
//   => non-reproducible AND not equal to the CPU's serial sum. Instead we
//   quantize the secreted amount to an integer number of "chemokine quanta" and
//   atomicAdd those (integer adds commute). Result: deterministic and exactly
//   equal to the CPU reference. Same trick as 5.01 (energy quanta) and 11.09.
//
//   ABM_QUANTA_PER_UNIT sets the resolution: 1e6 quanta per unit concentration
//   gives ~6 significant digits, and a grid cell's total stays far inside the
//   range of unsigned long long even for thousands of steps.
// ---------------------------------------------------------------------------
// constexpr (not just const): a compile-time constant is usable inside DEVICE
// code, whereas a namespace-scope `const double` would have host-only storage
// and nvcc rejects reading it from a kernel.
constexpr double ABM_QUANTA_PER_UNIT = 1.0e6;

// Convert a (non-negative) amount of chemokine to an integer number of quanta.
// We round to nearest so CPU and GPU quantize identically. Callers guarantee
// amount >= 0 (secretion is a source term), so no sign handling is needed.
ABM_HD inline unsigned long long abm_to_quanta(double amount) {
    return static_cast<unsigned long long>(amount * ABM_QUANTA_PER_UNIT + 0.5);
}

// Convert accumulated quanta back to a concentration (used after the scatter).
ABM_HD inline double abm_from_quanta(unsigned long long q) {
    return static_cast<double>(q) / ABM_QUANTA_PER_UNIT;
}

// ---------------------------------------------------------------------------
// Grid helpers. The substrate field is a gx*gy array in ROW-MAJOR order:
// index = row*gx + col, so consecutive columns are contiguous -> coalesced.
// ---------------------------------------------------------------------------
ABM_HD inline int abm_grid_idx(int col, int row, int gx) { return row * gx + col; }

// Which grid column/row does a domain position fall in? Clamp into [0,g-1] so a
// cell exactly on the far boundary still lands in a valid cell (defensive; the
// mechanics keep cells inside the domain anyway).
ABM_HD inline int abm_col_of(double x, double dx, int gx) {
    int c = static_cast<int>(x / dx);
    if (c < 0) c = 0; else if (c >= gx) c = gx - 1;
    return c;
}
ABM_HD inline int abm_row_of(double y, double dx, int gy) {
    int r = static_cast<int>(y / dx);
    if (r < 0) r = 0; else if (r >= gy) r = gy - 1;
    return r;
}

// ---------------------------------------------------------------------------
// (2) DIFFUSE: one explicit reaction-diffusion update of grid cell (col,row).
//   PDE:  dc/dt = D * laplacian(c) - decay * c
//   Discretized with the standard 5-point Laplacian on spacing dx and forward
//   Euler in time. ZERO-FLUX (Neumann) boundaries: a neighbour outside the grid
//   is replaced by the centre value, so no chemokine leaks out of the domain.
//   Reads c_old, writes c_new (ping-pong) -> no races, no atomics. Identical on
//   CPU and GPU. Stability needs D*dt/dx^2 <= 1/4 in 2-D (documented in THEORY).
// ---------------------------------------------------------------------------
ABM_HD inline void abm_diffuse_cell(int col, int row, const AbmParams& p,
                                    const double* c_old, double* c_new) {
    const int gx = p.gx, gy = p.gy;
    const int i = abm_grid_idx(col, row, gx);
    const double c = c_old[i];
    // Neumann (zero-flux) edges: clamp neighbour index to the boundary so an
    // out-of-domain neighbour equals the centre (gradient across the wall = 0).
    const double cl = c_old[abm_grid_idx(col > 0      ? col - 1 : col, row, gx)];
    const double cr = c_old[abm_grid_idx(col < gx - 1 ? col + 1 : col, row, gx)];
    const double cu = c_old[abm_grid_idx(col, row > 0      ? row - 1 : row, gx)];
    const double cd = c_old[abm_grid_idx(col, row < gy - 1 ? row + 1 : row, gx)];
    const double lap = (cl + cr + cu + cd - 4.0 * c) / (p.dx * p.dx);
    // Forward Euler: c_new = c + dt*(D*lap - decay*c). Clamp tiny negatives that
    // rounding could produce so the field stays a physical concentration.
    double next = c + p.dt * (p.D * lap - p.decay * c);
    if (next < 0.0) next = 0.0;
    c_new[i] = next;
}

// ---------------------------------------------------------------------------
// Chemokine gradient at a grid cell via central differences (zero-flux edges).
// Returns (gx_out, gy_out) = grad(c). Immune cells follow this uphill.
// ---------------------------------------------------------------------------
ABM_HD inline void abm_gradient(int col, int row, const AbmParams& p,
                                const double* c, double* gx_out, double* gy_out) {
    const int gx = p.gx, gy = p.gy;
    const double cl = c[abm_grid_idx(col > 0      ? col - 1 : col, row, gx)];
    const double cr = c[abm_grid_idx(col < gx - 1 ? col + 1 : col, row, gx)];
    const double cu = c[abm_grid_idx(col, row > 0      ? row - 1 : row, gx)];
    const double cd = c[abm_grid_idx(col, row < gy - 1 ? row + 1 : row, gx)];
    // Central difference; at a clamped edge the two samples coincide -> 0 there.
    *gx_out = (cr - cl) / (2.0 * p.dx);
    *gy_out = (cd - cu) / (2.0 * p.dx);
}

// ---------------------------------------------------------------------------
// (3) MOVE: compute cell `i`'s new position for one step, given the substrate
//   field and a SPATIAL-BINNING neighbour list. Overdamped mechanics: velocity
//   is proportional to force (Stokes drag), so x <- x + dt * (F_repulsion +
//   v_chemotaxis). We keep the reduction over neighbours in a FIXED ORDER (bins
//   scanned in index order, cells within a bin in stored order) so the CPU and
//   GPU sum the forces identically -> exact match, no atomics on the force.
//
//   Neighbour access is abstracted through the bin arrays so the SAME function
//   serves the CPU loop and the GPU kernel:
//     bin_start[b], bin_count[b]   : cells in bin b occupy sorted[start..start+count)
//     sorted[]                     : cell indices sorted by bin
//   We only scan the 3x3 block of bins around cell i's bin -> O(1) neighbours on
//   average -> O(N) total. See THEORY.md "GPU mapping".
// ---------------------------------------------------------------------------
ABM_HD inline void abm_move_cell(int i, const AbmParams& p,
                                 const double* cx, const double* cy, const int* ctype,
                                 const double* field,
                                 int bins_x, int bins_y, double bin_size,
                                 const int* bin_start, const int* bin_count,
                                 const int* sorted,
                                 double* new_x, double* new_y) {
    const double xi = cx[i];
    const double yi = cy[i];

    // --- Soft-sphere repulsion from overlapping neighbours -----------------
    // Force accumulates in a fixed neighbour order (deterministic). Two cells
    // repel when their centre distance < contact = 2*radius; the force is linear
    // in the overlap (a Hookean spring) directed along the centre-to-centre line.
    double fx = 0.0, fy = 0.0;
    const double contact = 2.0 * p.radius;
    const int bxi = static_cast<int>(xi / bin_size);   // this cell's bin column
    const int byi = static_cast<int>(yi / bin_size);   // this cell's bin row
    for (int bb = -1; bb <= 1; ++bb) {                 // 3x3 neighbouring bins
        const int by = byi + bb;
        if (by < 0 || by >= bins_y) continue;
        for (int aa = -1; aa <= 1; ++aa) {
            const int bx = bxi + aa;
            if (bx < 0 || bx >= bins_x) continue;
            const int b = by * bins_x + bx;             // flat bin index
            const int start = bin_start[b];
            const int cnt   = bin_count[b];
            for (int s = 0; s < cnt; ++s) {
                const int j = sorted[start + s];        // candidate neighbour
                if (j == i) continue;                   // skip self
                const double dxv = xi - cx[j];
                const double dyv = yi - cy[j];
                const double r2  = dxv * dxv + dyv * dyv;
                if (r2 >= contact * contact || r2 == 0.0) continue; // no overlap
                const double r = sqrt(r2);
                const double overlap = contact - r;     // >0 when overlapping
                const double inv = 1.0 / r;             // unit vector normalizer
                fx += p.k_rep * overlap * (dxv * inv);  // push apart along the line
                fy += p.k_rep * overlap * (dyv * inv);
            }
        }
    }

    // --- Chemotaxis: immune cells drift up the chemokine gradient ----------
    // Tumor cells are mechanically passive (they only secrete + get pushed).
    double vx = 0.0, vy = 0.0;
    if (ctype[i] == CELL_IMMUNE) {
        const int col = abm_col_of(xi, p.dx, p.gx);
        const int row = abm_row_of(yi, p.dx, p.gy);
        double gcx = 0.0, gcy = 0.0;
        abm_gradient(col, row, p, field, &gcx, &gcy);
        vx = p.chemotaxis * gcx;                        // speed along grad(c)
        vy = p.chemotaxis * gcy;
    }

    // --- Overdamped forward-Euler integration ------------------------------
    // Combine mechanical force (as a velocity, drag=1) and chemotactic velocity.
    double nx = xi + p.dt * (fx + vx);
    double ny = yi + p.dt * (fy + vy);
    // Keep cells inside the domain (reflectionless clamp to the walls).
    const double W = p.width(), H = p.height();
    if (nx < 0.0) nx = 0.0; else if (nx > W) nx = W;
    if (ny < 0.0) ny = 0.0; else if (ny > H) ny = H;
    new_x[i] = nx;
    new_y[i] = ny;
}
