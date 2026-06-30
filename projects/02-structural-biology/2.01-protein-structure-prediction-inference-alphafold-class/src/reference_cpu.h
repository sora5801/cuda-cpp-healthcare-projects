// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for one attention head
// ---------------------------------------------------------------------------
// Project 2.1 : Protein Structure Prediction Inference (AlphaFold-class)
//               REDUCED-SCOPE TEACHING VERSION.
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (the attention tensors, the file
//   loader) and the CPU-reference prototypes live here. kernels.cuh also
//   includes this header (it is pure C++, safe inside a .cu) to reuse the same
//   AttentionProblem type -- nothing CUDA-specific leaks in either direction.
//
// THE PROBLEM, CONCRETELY  (full derivation in ../THEORY.md)
//   A protein of L residues is represented by an [L x d] matrix of feature
//   vectors. A self-attention layer first projects that matrix into three
//   [L x d] matrices -- Query Q, Key K, Value V -- and then mixes information
//   across residues:
//       Out[i] = sum_j softmax_j( (Q[i].K[j]) / sqrt(d) ) * V[j]
//   We load Q, K, V from a tiny text sample, compute Out on the CPU (here) and
//   on the GPU (kernels.cu), and check they agree. The per-element math is
//   shared via attention_core.h so the two implementations are numerically twins.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh. READ attention_core.h FIRST.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "attention_core.h"   // D_MODEL, dot_d, scaled_score, stable_exp (pure C++)

// ---------------------------------------------------------------------------
// AttentionProblem : one self-attention head over one protein's residues.
//   All three matrices are ROW-MAJOR and [L x D_MODEL]: residue i's query
//   vector occupies q[i*D_MODEL .. i*D_MODEL + D_MODEL - 1], and likewise for k
//   and v. Storing them flat (not vector<vector>) keeps each row contiguous in
//   memory, which is exactly what both the CPU cache and the GPU's coalesced
//   loads want.
//
//   `out` has the same [L x D_MODEL] shape: the context-mixed representation,
//   the thing every Evoformer block produces and feeds to the next block.
// ---------------------------------------------------------------------------
struct AttentionProblem {
    int L = 0;                  // number of residues (sequence length)
    int d = D_MODEL;            // feature width per residue (== D_MODEL)
    std::vector<float> q;       // [L * D_MODEL] queries, row-major
    std::vector<float> k;       // [L * D_MODEL] keys,    row-major
    std::vector<float> v;       // [L * D_MODEL] values,  row-major
};

// Load an AttentionProblem from the text format documented in data/README.md:
//   line 1 : "<L> <d>"  (d must equal D_MODEL or the loader throws)
//   then    : L rows of d floats for Q, then L rows for K, then L rows for V.
// Throws std::runtime_error on a missing file or a width/shape mismatch.
AttentionProblem load_attention(const std::string& path);

// CPU reference: fill `out` ([L * D_MODEL], row-major) with one head of
// scaled dot-product self-attention over the problem. This is the trusted,
// obviously-correct serial baseline the GPU kernel is verified against (and the
// timing baseline that makes the speed-up legible). `out` is resized to
// L * D_MODEL. Uses the shared primitives in attention_core.h so its arithmetic
// matches the kernel's.
void attention_cpu(const AttentionProblem& prob, std::vector<float>& out);
