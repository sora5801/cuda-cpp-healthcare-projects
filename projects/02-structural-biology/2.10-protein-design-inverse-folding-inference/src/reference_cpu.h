// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for inverse-folding design
// ---------------------------------------------------------------------------
// Project 2.10 : Protein Design / Inverse Folding Inference
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (the Backbone container, the file
//   loader) and the CPU reference prototypes live here. The GPU side
//   (kernels.cuh) also includes this header to reuse the Backbone type, and BOTH
//   sides include inverse_folding.h for the shared per-residue scoring core.
//   Nothing CUDA-specific leaks in either direction.
//
// THE PROBLEM (full derivation in ../THEORY.md)
//   Input : a protein BACKBONE = L residues, each with a Calpha (x,y,z) position,
//           plus the NATIVE sequence (the amino acid nature actually used at each
//           position -- our synthetic "ground truth" to score recovery against).
//   Step 1: for each residue i, count how many other residues' Calpha atoms lie
//           within CONTACT_RADIUS -> a per-residue BURIAL (neighbor count). This
//           is an all-pairs O(L^2) computation -- the analog of message-passing
//           over the protein graph in a real GNN.
//   Step 2: for each residue i, score all 20 amino acids with the shared
//           score_aa_at_residue() and pick the best (argmax) -> the DESIGNED
//           amino acid at i. Doing this independently per residue is the
//           "autoregressive decode at temperature 0" of this teaching model.
//   Output: the designed sequence, its per-position scores, and the fraction of
//           positions where the design matches the native sequence (RECOVERY).
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "inverse_folding.h"   // BackboneResidue, NUM_AA, score_aa_at_residue

// One-letter amino-acid codes in canonical index order (see inverse_folding.h).
//   AA_CODES[k] is the letter for amino-acid index k. Host-only (used for
//   printing the designed/native sequences and for parsing the sample file);
//   device code never needs the letters, only the count NUM_AA.
//   Defined once in reference_cpu.cpp.
extern const char AA_CODES[NUM_AA + 1];   // 20 letters + a terminating '\0'

// ---------------------------------------------------------------------------
// Backbone: a loaded protein backbone problem.
//   res    : L residues, each a Calpha coordinate (BackboneResidue). The "soa
//            vs aos" choice here is array-of-structs for readability; the kernel
//            reads x,y,z together per residue so locality is fine either way.
//   native : L amino-acid indices (0..NUM_AA-1), the synthetic ground-truth
//            sequence we measure design recovery against.
//   L (== res.size() == native.size()) is the number of residues.
// ---------------------------------------------------------------------------
struct Backbone {
    std::vector<BackboneResidue> res;     // [L] Calpha coordinates (angstrom)
    std::vector<int>             native;  // [L] native amino-acid indices 0..19
    int size() const { return static_cast<int>(res.size()); }  // residue count L
};

// ---------------------------------------------------------------------------
// DesignResult: what an inverse-folding pass produces. Both the CPU reference
// and the GPU path fill one of these; main.cu compares them field by field.
//   neighbors : [L] per-residue contact count (the burial signal, step 1)
//   designed  : [L] chosen amino-acid index per residue (the argmax, step 2)
//   score     : [L] the integer score of the chosen amino acid (its argmax value)
// All three are EXACT integers, so CPU and GPU must agree bit-for-bit.
// ---------------------------------------------------------------------------
struct DesignResult {
    std::vector<int> neighbors;   // [L] neighbor counts
    std::vector<int> designed;    // [L] designed amino-acid indices
    std::vector<int> score;       // [L] best per-residue score
};

// Load a backbone problem from the text format documented in data/README.md:
//   line 1 : "<L>"                       (number of residues)
//   next L : "<x> <y> <z> <native_letter>" per residue
// Throws std::runtime_error on a missing file or a malformed line.
Backbone load_backbone(const std::string& path);

// design_cpu: the trusted serial baseline. Computes neighbor counts (O(L^2))
// then the per-residue argmax (O(L*NUM_AA)), filling `out`. This is the
// obviously-correct reference the GPU result is verified against, and the timing
// baseline that makes the GPU speed-up legible. See reference_cpu.cpp.
void design_cpu(const Backbone& bb, DesignResult& out);

// recovery_percent: fraction (in percent) of positions where designed == native,
//   rounded to an integer so it prints deterministically. This is the headline
//   "did we recover the native sequence?" metric (ProteinMPNN reports ~50%).
//   Pure integer arithmetic -> identical on any machine.
int recovery_percent(const Backbone& bb, const DesignResult& d);

// sequence_string: turn a vector of amino-acid indices into its one-letter
//   string (e.g. {9,10,19} -> "ILV") for human-readable, deterministic output.
std::string sequence_string(const std::vector<int>& aa_indices);
