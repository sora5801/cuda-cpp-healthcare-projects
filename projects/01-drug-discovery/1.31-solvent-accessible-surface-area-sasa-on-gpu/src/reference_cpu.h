// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for Shrake-Rupley SASA
// ---------------------------------------------------------------------------
// Project 1.31 : Solvent-Accessible Surface Area (SASA) on GPU
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (the Molecule container, the file
//   loader) and the CPU reference prototypes live here. kernels.cuh also
//   includes this header to reuse the Molecule type and the Atom struct -- so
//   nothing CUDA-specific leaks into the host path, and nothing host-only leaks
//   into the device path. The per-atom MATH lives one level deeper, in
//   sasa_core.h (shared __host__ __device__), so CPU and GPU run identical math.
//
// THE PROBLEM (full derivation in ../THEORY.md)
//   Solvent-Accessible Surface Area = the area of a protein/ligand surface that
//   a rolling water probe (radius 1.4 A) can touch. Shrake & Rupley (1973)
//   estimate it by sprinkling test points over each atom's probe-inflated sphere
//   and counting how many are NOT buried inside a neighbor. Each atom is an
//   INDEPENDENT job -> one GPU thread per atom (in kernels.cu).
//
// READ THIS AFTER: sasa_core.h. READ THIS BEFORE: reference_cpu.cpp, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sasa_core.h"   // Atom, PROBE_RADIUS, N_SPHERE_POINTS, the shared math

// A loaded molecule: n atoms, each with coordinates + a van der Waals radius.
//   atoms : length-n array of Atom (POD), ready to memcpy to the device in one
//           contiguous block (host and device layouts are identical -> no
//           repacking on upload).
struct Molecule {
    int               n = 0;     // number of atoms
    std::vector<Atom> atoms;     // [n] centers (Angstrom) + vdW radii (Angstrom)
};

// Map a one-letter element symbol to its van der Waals radius in Angstrom
// (Bondi 1964 values, the set FreeSASA and most tools default to). Defined in
// reference_cpu.cpp. Unknown elements fall back to carbon's radius so a stray
// atom never crashes the demo (the substitution is logged to stderr).
double vdw_radius(char element);

// Load a molecule from the simple whitespace text format in data/README.md:
//   line 1 : "<n>"                       (atom count)
//   next n : "<element> <x> <y> <z>"     (element letter + coords in Angstrom)
// Lines beginning with '#' are comments and blank lines are skipped. Throws
// std::runtime_error on a missing file or a malformed/short body so demos fail
// loudly, never on empty input.
Molecule load_molecule(const std::string& path);

// CPU reference (the trusted baseline the GPU is verified against). Fills:
//   exposed[i] : integer count of solvent-accessible test points for atom i
//                (0..N_SPHERE_POINTS) -- the EXACT quantity both sides compare.
//   sasa[i]    : atom i's SASA in Angstrom^2, derived from exposed[i].
// Both are resized to mol.n; the total SASA is the sum of sasa[]. This loops the
// shared count_exposed_points()/atom_sasa() from sasa_core.h -- the very same
// functions the kernel calls -- which is what makes verification exact.
void sasa_cpu(const Molecule& mol,
              std::vector<int>& exposed,
              std::vector<double>& sasa);
