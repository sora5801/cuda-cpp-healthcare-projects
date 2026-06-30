// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for the ΔΔG scan
// ---------------------------------------------------------------------------
// Project 2.16 : ΔΔG Stability Prediction (reduced-scope teaching version)
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (the Protein container, the file
//   loader, the amino-acid code map) and the CPU-reference prototypes live here.
//   kernels.cuh also includes this header to reuse the Protein type and the scan
//   geometry -- nothing CUDA-specific leaks in either direction. The actual
//   per-mutation MATH lives in ddg_model.h (shared host+device); this file is
//   about LAYOUT and ORCHESTRATION.
//
// THE PROBLEM IN ONE PICTURE  (full derivation in ../THEORY.md)
//   A protein of L residues. "Saturation mutagenesis" asks: for EVERY position
//   p in 0..L-1, and EVERY one of the 20 amino acids a, what is the predicted
//   ΔΔG of mutating residue p to amino acid a? That is an L × 20 grid of
//   independent scores -- a "deep mutational scan" heatmap. Each cell is one
//   call to ddg_predict(); the cells are mutually independent, which is exactly
//   why the GPU maps so cleanly (one thread per cell -- see kernels.cu).
//
//   Row-major layout of the score grid:  score[p * NUM_AA + a]  is the ΔΔG of
//   mutating position p to amino acid a (kcal/mol). The diagonal cell where
//   a == wild-type(p) is the self-mutation and is exactly 0.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh. The math is in ddg_model.h.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "ddg_model.h"   // NUM_AA, amino-acid tables, ddg_predict (pure-C++ safe)

// ---------------------------------------------------------------------------
// Protein : the loaded structural "feature" set for one chain.
//   L          : number of residues (sequence length).
//   wt_code    : [L] wild-type amino-acid index (0..NUM_AA-1) at each position.
//   buried     : [L] burial fraction in [0,1] per position (1 = core). This is
//                the single scalar structural feature our reduced model consumes
//                in place of a learned per-residue embedding (THEORY.md explains
//                what a real GNN would put here).
//   name       : a label for the report (e.g. a synthetic PDB-like id).
//
//   Both arrays are length L and are indexed by residue position p.
// ---------------------------------------------------------------------------
struct Protein {
    int                L = 0;        // number of residues
    std::vector<int>   wt_code;      // [L] wild-type AA index per position
    std::vector<float> buried;       // [L] burial fraction per position
    std::string        name;         // label for reporting
};

// Map a one-letter amino-acid code ('A','R',...) to its canonical index 0..19,
// or -1 if the character is not one of the 20 standard residues. Defined in
// reference_cpu.cpp; declared here because both the loader and tests use it.
int aa_index(char one_letter);

// Load a Protein from the tiny text format documented in data/README.md:
//   line 1:  "<name>"                         (a label, no spaces)
//   line 2:  "<L>"                             (residue count)
//   next L:  "<one_letter_AA> <buried>"        (wild-type residue + burial frac)
// Throws std::runtime_error on a missing file, a bad code, or a length mismatch.
Protein load_protein(const std::string& path);

// CPU reference: fill out[p*NUM_AA + a] with ddg_predict(wt_p, a, buried_p) for
// every position p and amino acid a. This is the obviously-correct serial
// baseline the GPU kernel is verified against (and the timing baseline that
// makes the speed-up legible). `out` is resized to L*NUM_AA. Row-major, as above.
void ddg_scan_cpu(const Protein& prot, std::vector<float>& out);
