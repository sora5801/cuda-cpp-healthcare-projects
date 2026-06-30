// ===========================================================================
// src/pbe.h  --  Shared (host + device) finite-difference Poisson-Boltzmann core
// ---------------------------------------------------------------------------
// Project 2.9 : Solvent-Accessible Surface & Poisson-Boltzmann Electrostatics
//               (reduced-scope teaching version; see ../THEORY.md and the
//               catalog deep-dive for the full picture)
//
// WHAT THIS PROJECT COMPUTES
//   The electrostatic potential phi(r) of a protein in salty water, by solving
//   the *linearized Poisson-Boltzmann equation* (LPBE) on a 3-D Cartesian grid.
//   In a medium with position-dependent dielectric eps(r) and a Debye screening
//   that switches on only in the solvent, the LPBE is the elliptic PDE
//
//        div( eps(r) grad phi(r) )  -  eps_w * kappa^2 * phi(r)  =  -rho(r)/eps0
//
//   where rho is the protein's fixed atomic charge density (point charges
//   spread onto the grid), kappa is the inverse Debye length (ionic strength),
//   and eps(r) is ~2-4 inside the protein, ~80 in water. From phi you can read
//   off pKa shifts, binding electrostatics, and surface (zeta) potentials.
//
//   This is *continuum electrostatics*: water and ions are not simulated atom
//   by atom but as a structureless dielectric continuum -- the same model APBS
//   and DelPhi use. We solve it the way GPU PB solvers do: finite differences
//   on a grid, iterated with RED-BLACK GAUSS-SEIDEL (see kernels.cu).
//
// WHY THIS HEADER EXISTS  (the host/device parity idiom -- PATTERNS.md sec.2)
//   The *per-cell relaxation update* -- the one true formula that takes a grid
//   cell's six neighbours and its source term and produces its next potential --
//   lives here as a __host__ __device__ inline function. The CPU reference loops
//   it; the GPU kernel calls it from one thread per cell. Because both sides run
//   byte-for-byte identical arithmetic, verification is exact-ish (see THEORY
//   "How we verify"). PBE_HD expands to __host__ __device__ under nvcc and to
//   nothing under the plain host compiler (which has never heard of those
//   keywords), so reference_cpu.cpp can include this same header.
//
//   Keep this header free of CUDA-only constructs (no __global__, no <<<>>>),
//   so the host compiler can include it. Only the per-cell math lives here.
//
// READ THIS AFTER: nothing (start here), then reference_cpu.h, then kernels.cuh.
// ===========================================================================
#pragma once

#include <cmath>   // std::sqrt, std::exp -- pulled in for host + device use

// ---------------------------------------------------------------------------
// PBE_HD: the host/device decoration switch.
//   * Compiled by nvcc (__CUDACC__ defined): mark functions __host__ __device__
//     so the SAME function is emitted for the CPU and the GPU.
//   * Compiled by cl.exe / g++: expand to nothing -- the decorators don't exist
//     for a non-CUDA compiler, and reference_cpu.cpp must still compile.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define PBE_HD __host__ __device__
#else
#define PBE_HD
#endif

// ---------------------------------------------------------------------------
// GridParams: everything that defines one PBE problem instance.
//   The grid is n x n x n cells (a cube, for teaching simplicity), with spacing
//   `h` angstroms between cells. All physical constants are folded into a few
//   reduced numbers so the arithmetic on the grid is simple and unit-consistent
//   (THEORY "The math" derives these). Sizes are kept small (n ~ 48) so the demo
//   runs in well under a second on a laptop GPU.
// ---------------------------------------------------------------------------
struct GridParams {
    int    n;          // cells per side: the grid is n*n*n (cube)
    double h;          // grid spacing in angstrom (distance between neighbours)
    double eps_in;     // dielectric constant inside the protein (~2-4)
    double eps_out;    // dielectric constant of water (~80)
    double kappa2;     // squared inverse Debye length kappa^2 (1/angstrom^2);
                       //   proportional to ionic strength; 0 => pure Poisson
    double charge_to_phi; // unit-folding factor: converts a unit point charge
                       //   (in e) and the grid into potential units (kT/e).
                       //   Derived once in reference_cpu.cpp so CPU == GPU.
    int    iters;      // number of red-black Gauss-Seidel sweeps to run
};

// flatten a 3-D index (x,y,z) into the linear array offset, x fastest.
// We isolate this so EVERY access (CPU + GPU) uses the identical layout; an
// index bug here would silently desynchronize the two solvers.
PBE_HD inline int pbe_idx(int x, int y, int z, int n) {
    return (z * n + y) * n + x;
}

// ---------------------------------------------------------------------------
// pbe_diag: the centre coefficient of the 7-point LPBE stencil for one cell.
//   The finite-difference Laplacian on a uniform grid is
//       (sum of 6 neighbour phi  -  6*phi_center) / h^2.
//   Moving the Debye term over, the discretized LPBE for an interior cell is
//       (eps/h^2) * (sum_neighbours phi  -  6 phi_c)  -  eps_w kappa^2 phi_c
//                                                                = -rho_c.
//   Solving the centre equation for phi_c (Gauss-Seidel) divides by the
//   coefficient of phi_c, which is this diagonal:
//       diag = 6*eps/h^2 + eps_w*kappa2_here.
//   We treat eps as locally uniform per cell (a "harmonic-mean dielectric" is
//   the production refinement -- see THEORY "real world"); kappa2 is nonzero
//   only in solvent cells (passed in as kappa2_here).
//
//   Returning the diagonal as its own function keeps the relaxation formula
//   below readable and makes the algebra auditable against THEORY.
// ---------------------------------------------------------------------------
PBE_HD inline double pbe_diag(double eps, double h, double eps_w, double kappa2_here) {
    const double inv_h2 = 1.0 / (h * h);
    return 6.0 * eps * inv_h2 + eps_w * kappa2_here;
}

// ---------------------------------------------------------------------------
// pbe_relax_cell: ONE Gauss-Seidel update of a single interior grid cell.
//   This is THE shared formula -- the only place the PDE is discretized, used
//   verbatim by the CPU reference loop and by the GPU kernel (one thread/cell).
//
//   Inputs
//     x,y,z          : the cell's grid coordinates (must be interior: 1..n-2)
//     P              : grid parameters (n, h, dielectrics, kappa^2, units)
//     phi            : the current potential field [n^3], read for neighbours
//     rho            : the source term (gridded charge) for this cell's RHS [n^3]
//     eps_grid       : per-cell dielectric eps(r) [n^3] (low in protein, high in water)
//     kappa2_grid    : per-cell kappa^2 [n^3] (0 in protein, P.kappa2 in solvent)
//   Returns
//     the NEW potential value for cell (x,y,z).
//
//   Method (Gauss-Seidel pointwise solve): rearrange the centre stencil equation
//       diag * phi_c  =  (eps/h^2) * sum_neighbours phi  +  rho_c
//   and read off phi_c. Each neighbour contributes (eps/h^2)*phi_neighbour; we
//   use this cell's eps as the face dielectric (uniform-per-cell approximation).
//   Gauss-Seidel uses the LATEST available neighbour values, which is why the
//   sweep ORDER matters and why we colour the grid red/black on the GPU so that
//   same-colour cells never read each other mid-sweep (THEORY "GPU mapping").
//
//   Determinism note: this is a fixed, short sequence of double-precision adds
//   and one divide. Given the same neighbour values it returns the same bits on
//   CPU and GPU, so red-black GS in a fixed colour order is reproducible.
// ---------------------------------------------------------------------------
PBE_HD inline double pbe_relax_cell(int x, int y, int z, const GridParams& P,
                                    const double* phi, const double* rho,
                                    const double* eps_grid, const double* kappa2_grid) {
    const int n = P.n;
    const int c = pbe_idx(x, y, z, n);          // this cell's linear index

    const double eps = eps_grid[c];             // local dielectric (eps(r))
    const double inv_h2 = 1.0 / (P.h * P.h);
    const double face = eps * inv_h2;           // coupling per neighbour face

    // Sum the six axis-neighbours' potentials (the 7-point Laplacian arms).
    // Interior-only guarantee (caller enforces 1..n-2) means all six exist;
    // the outermost shell is held at the boundary condition (see apply BCs).
    const double neigh =
          phi[pbe_idx(x - 1, y, z, n)] + phi[pbe_idx(x + 1, y, z, n)]
        + phi[pbe_idx(x, y - 1, z, n)] + phi[pbe_idx(x, y + 1, z, n)]
        + phi[pbe_idx(x, y, z - 1, n)] + phi[pbe_idx(x, y, z + 1, n)];

    // Diagonal (centre coefficient): 6*eps/h^2 + eps_w*kappa^2 here.
    const double diag = pbe_diag(eps, P.h, P.eps_out, kappa2_grid[c]);

    // Gauss-Seidel pointwise solve: phi_c = (face*sum_neigh + rho_c) / diag.
    // rho already carries the unit-folding factor (charge_to_phi) so the result
    // is directly in reduced potential units (kT/e). diag > 0 always (eps>0),
    // so no divide-by-zero guard is needed for a well-posed grid.
    return (face * neigh + rho[c]) / diag;
}
