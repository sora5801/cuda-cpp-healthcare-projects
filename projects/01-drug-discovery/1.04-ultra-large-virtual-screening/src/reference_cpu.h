// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for virtual screening
// ---------------------------------------------------------------------------
// Project 1.4 : Ultra-Large Virtual Screening
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must never see a
//   __global__ or other CUDA token, so the shared DATA MODEL (the loaded library
//   container, the file loader) and the CPU reference prototype live here. The
//   GPU side (kernels.cuh) ALSO includes this header to reuse the same library
//   container and the same Ligand/Target structs -- nothing CUDA-specific leaks.
//   The per-ligand math itself is in screen_core.h (the __host__ __device__ core
//   shared by both sides); this header only adds host-side loading + the loop.
//
// THE PROBLEM (see ../THEORY.md for the full derivation)
//   We have ONE binding-site TARGET and N library LIGANDS. For every ligand we
//   run a cheap drug-likeness FILTER CASCADE; survivors get a SURROGATE DOCKING
//   SCORE; we then report how many survived and the TOP-K best-scoring hits.
//   Every ligand is independent -> perfect data parallelism (one GPU thread per
//   ligand in kernels.cu). This is the reduced-scope teaching shape of a real
//   billion-compound campaign (AutoDock-GPU on Summit) -- honestly labelled.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh. screen_core.h first of all.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "screen_core.h"   // Ligand, Target, score_ligand (pure structs + math)

// ---------------------------------------------------------------------------
// LigandLibrary: a loaded screening problem -- the target plus N ligands.
//   target  : the binding-site wish list we screen against (one per file).
//   ligands : N library compounds, stored contiguously so the whole vector
//             uploads to the GPU as one cudaMemcpy (kernels.cu).
// ---------------------------------------------------------------------------
struct LigandLibrary {
    Target              target;    // the screening target (binding-site profile)
    std::vector<Ligand> ligands;   // N library compounds
    int n() const { return static_cast<int>(ligands.size()); }  // convenience
};

// ---------------------------------------------------------------------------
// load_library: parse the tiny text dataset documented in data/README.md.
//   Format (whitespace-separated, '#'-prefixed comment lines ignored):
//     line: n                                  (number of ligands)
//     line: TARGET mw_opt logp_opt_x100 psa_opt feat_required_hex
//     n lines: mw logp_x100 hbd hba rotb psa feat_hex   (one per ligand)
//   Throws std::runtime_error on a missing file or a malformed record so demos
//   fail loudly rather than silently screening empty input.
// ---------------------------------------------------------------------------
LigandLibrary load_library(const std::string& path);

// ---------------------------------------------------------------------------
// screen_cpu: the trusted serial baseline. Fills score[i] with score_ligand()
// for ligand i (the surrogate score, or REJECTED if it failed the cascade). This
// is the obviously-correct reference the GPU kernel is verified against, AND the
// timing baseline that makes the GPU speed-up legible (CLAUDE.md sec 5). score is
// resized to lib.n(). It calls the SAME score_ligand() the kernel calls, so the
// two agree bit-for-bit (integer arithmetic -> exact; see screen_core.h).
// ---------------------------------------------------------------------------
void screen_cpu(const LigandLibrary& lib, std::vector<int>& score);
