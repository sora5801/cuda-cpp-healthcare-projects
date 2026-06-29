// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for 3D shape screening
// ---------------------------------------------------------------------------
// Project 1.13 : Pharmacophore & 3D Shape Screening
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (the on-disk format, the ConformerSet
//   container, the file loader) and the CPU reference prototype live here. The
//   GPU side (kernels.cuh) includes this header too, to reuse the exact same
//   ConformerSet type -- nothing CUDA-specific leaks in either direction. The
//   actual physics is in shape_overlap.h (shared by both, see PATTERNS.md sec 2).
//
// THE PROBLEM (full derivation in ../THEORY.md)
//   We have ONE 3D query molecule and N library conformers, each a list of
//   heavy atoms (x,y,z + van der Waals radius). For every conformer we compute
//   the Gaussian-volume Shape Tanimoto against the query (shape_overlap.h) and
//   rank the library by similarity. Every conformer is INDEPENDENT -> perfect
//   data parallelism (one GPU thread per conformer, in kernels.cu). This is the
//   shape-based pre-filter that precedes docking in a virtual-screening pipeline.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh. The physics is shape_overlap.h.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "shape_overlap.h"   // Atom, Molecule, MAX_ATOMS, the overlap functions

// ---------------------------------------------------------------------------
// ConformerSet: a loaded screening problem = one query molecule + a flat array
// of N library conformers.
//   query : the reference shape every conformer is scored against.
//   lib   : N library molecules, stored back-to-back (lib[k] is conformer k).
//           A std::vector<Molecule> is contiguous, so &lib[0] is one block of
//           N*sizeof(Molecule) bytes -- copyable to the GPU in a single memcpy.
//   name  : a human-readable label per conformer (for the report); NOT uploaded
//           to the GPU (it is host-only bookkeeping).
// ---------------------------------------------------------------------------
struct ConformerSet {
    Molecule              query;   // the reference shape
    std::vector<Molecule> lib;     // [n] library conformers (contiguous)
    std::vector<std::string> name; // [n] labels, host-only
    int n = 0;                     // number of library conformers (== lib.size())
};

// ---------------------------------------------------------------------------
// load_conformers: parse the tiny text dataset documented in data/README.md.
//   Format (whitespace-separated; '#' starts a comment line):
//     N                                # number of library conformers
//     then a QUERY block, then N library blocks; each block is:
//       M                              # number of atoms in this molecule
//       label                          # one token, no spaces (e.g. QUERY, lib_07)
//       x y z radius                   # one line per atom (angstrom, angstrom)
//   The radius is converted to a Gaussian alpha via atom_alpha() at load time,
//   so the hot loop never touches a radius again.
//   Throws std::runtime_error on a missing file, a bad count, or an atom
//   overflow (M > MAX_ATOMS).
// ---------------------------------------------------------------------------
ConformerSet load_conformers(const std::string& path);

// ---------------------------------------------------------------------------
// shape_tanimoto_cpu: the trusted serial baseline. Fills out[k] with the Shape
// Tanimoto of the query against library conformer k, for every k. This is the
// obviously-correct reference the GPU kernel is checked against (and the timing
// baseline that makes the speed-up legible). out is resized to set.n.
//
//   It precomputes the query self-overlap O_AA exactly once (it is identical
//   for every conformer) -- the same optimization the GPU wrapper uses, so the
//   two code paths stay structurally parallel.
// ---------------------------------------------------------------------------
void shape_tanimoto_cpu(const ConformerSet& set, std::vector<double>& out);
