// ===========================================================================
// src/reference_cpu.h  --  PBE problem setup + serial Gauss-Seidel reference
// ---------------------------------------------------------------------------
// Project 2.9 : Solvent-Accessible Surface & Poisson-Boltzmann Electrostatics
//
// Pure C++ (no CUDA). It declares:
//   * the Atom record and the .pqr-style loader,
//   * the grid-building step that turns atoms into the eps / kappa^2 / rho maps,
//   * the trusted serial red-black Gauss-Seidel reference solver,
//   * a small SASA (solvent-accessible surface area) estimate -- the "S" in the
//     project title -- computed from the same atomic radii.
//
// The per-cell relaxation math is in pbe.h and is shared with the GPU kernel,
// so the CPU reference and the GPU solver produce the same field (THEORY "How
// we verify"). main.cu compares them.
//
// READ THIS AFTER: pbe.h. READ BEFORE: reference_cpu.cpp, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "pbe.h"   // GridParams, pbe_relax_cell, pbe_idx

// ---------------------------------------------------------------------------
// Atom: one protein atom as continuum electrostatics needs it -- a point charge
// at a position, with a radius that defines where the low-dielectric protein
// interior is. This is exactly the information a .pqr file carries (PDB + per-
// atom Partial charge and Radius), which is what PDB2PQR produces for APBS.
//   x,y,z  : coordinates in angstrom
//   q      : partial charge in units of the elementary charge e (can be < 0)
//   radius : van der Waals / atomic radius in angstrom
// ---------------------------------------------------------------------------
struct Atom {
    double x, y, z;   // position (angstrom)
    double q;         // partial charge (e)
    double radius;    // atomic radius (angstrom)
};

// ---------------------------------------------------------------------------
// PbeProblem: a fully-built grid problem ready to solve.
//   `P`            : numeric grid parameters (size, spacing, dielectrics...).
//   `eps`          : per-cell dielectric eps(r), [n^3].
//   `kappa2`       : per-cell screening kappa^2(r), [n^3] (0 inside protein).
//   `rho`          : per-cell source term (gridded charge * unit factor), [n^3].
//   `origin_*`     : world coordinate of grid cell (0,0,0), so we can map atoms.
// All fields are filled by build_problem() from the atom list + the GridParams.
// ---------------------------------------------------------------------------
struct PbeProblem {
    GridParams P;
    std::vector<double> eps;
    std::vector<double> kappa2;
    std::vector<double> rho;
    double origin_x, origin_y, origin_z;   // world coords of cell (0,0,0)
};

// Load atoms from a tiny whitespace text format (data/README.md describes it):
//   first line:  natoms  n  h  eps_in  eps_out  kappa2  iters
//   then natoms lines:  x y z q radius
// Fills P_out with the numeric grid parameters from the header line.
// Throws std::runtime_error on a bad/missing file so demos fail loudly.
std::vector<Atom> load_atoms(const std::string& path, GridParams& P_out);

// Build the eps / kappa^2 / rho grids from the atoms.
//   * A cell is "inside the protein" (eps = eps_in, kappa^2 = 0) if it lies
//     within any atom's radius; otherwise it is solvent (eps_out, P.kappa2).
//   * Each atom's charge is deposited onto its nearest grid cell (nearest-grid-
//     point assignment) and scaled by P.charge_to_phi into the rho source term.
//   The grid is auto-sized/centred on the atoms' bounding box (see the .cpp).
PbeProblem build_problem(const std::vector<Atom>& atoms, const GridParams& P);

// Serial reference solver: run P.iters red-black Gauss-Seidel sweeps over the
// interior cells, holding the boundary at phi = 0 (a grounded box -- THEORY
// discusses the Debye-Huckel boundary used in production). `phi` comes in zero-
// initialized (size n^3) and is updated in place to the converged potential.
// This is the trusted baseline the GPU kernel is checked against.
void solve_cpu(const PbeProblem& prob, std::vector<double>& phi);

// Solvent-accessible surface area (SASA) of the molecule, in angstrom^2, by the
// Shrake-Rupley sphere-sampling method on each atom (probe radius 1.4 A = water).
// This is the geometric "surface" companion to the electrostatics; reported by
// main.cu as a deterministic scalar. Implemented on the CPU only (it is cheap);
// the GPU work is the PBE solve.
double compute_sasa(const std::vector<Atom>& atoms, double probe_radius, int sphere_points);
