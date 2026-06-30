// ===========================================================================
// src/reference_cpu.h  --  Protein model + the serial self-attention reference
// ---------------------------------------------------------------------------
// Project 3.18 : Protein Language Model Inference
//
// WHAT THIS PROJECT COMPUTES
//   The forward pass of ONE multi-head self-attention block -- the core compute
//   unit of a protein language model such as Meta's ESM-2. Given a protein
//   sequence of L residues:
//     1. Each residue's amino acid becomes a token id, then a d_model embedding
//        vector X[i]  (a synthetic stand-in for ESM-2's learned embeddings).
//     2. Three linear projections produce per-residue Query, Key, Value vectors,
//        split across H heads of width d_head = d_model/H.
//     3. For each head, the attention weights are
//          A = softmax( Q Kᵀ / sqrt(d_head) )            (L x L per head)
//        and the head output is  A V  (each residue = a weighted blend of all
//        residues' values).
//     4. Heads are concatenated and passed through an output projection Wo to
//        give the block's per-residue output embeddings  Y  (L x d_model).
//
// WHY A GPU
//   Self-attention is O(L²·d) per layer and is dominated by dense matrix
//   products (QKᵀ, AV) and per-row softmaxes -- exactly the GEMM + reduction
//   workloads GPUs excel at. ESMFold ran ESM-2 over millions of proteins on GPU
//   clusters because this kernel, repeated across 33 layers and 20 heads, is the
//   whole cost. We implement ONE block at teaching scale; the pattern is the one
//   that production stacks (FlashAttention, cuBLAS) optimize.
//
//   This header is PURE C++ (no CUDA) so it is safe to include from kernels.cu.
//
// READ THIS AFTER: attention_math.h (the shared per-element math).
// READ THIS BEFORE: kernels.cuh (the GPU twin of attention_cpu()).
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "attention_math.h"   // AttnConfig + the shared __host__ __device__ math

// ---------------------------------------------------------------------------
// ProteinInput: a loaded inference problem.
//   `sequence` is the raw amino-acid string (e.g. "MKTAYIAK...").
//   `tokens`   is its per-residue token id in [0,19] (filled by load_protein).
//   `cfg`      holds L, d_model, H, d_head (L derived from the sequence length).
// ---------------------------------------------------------------------------
struct ProteinInput {
    std::string sequence;        // the protein, one char per residue
    std::vector<int> tokens;     // [L] token ids (index into AA_ALPHABET)
    AttnConfig cfg;              // shapes of the attention block
};

// ---------------------------------------------------------------------------
// AttnResult: everything the demo reports + everything we verify.
//   `out`      : [L*d_model] the block's output embeddings Y, row-major.
//   `attn`     : [L*L] the attention map of head 0 (row i = where residue i
//                looks), the most interpretable single tensor.
//   `out_norm` : [L] L2 norm of each residue's output embedding (a compact,
//                deterministic per-residue summary we print).
//   `top_attn` : [L] argmax over keys of head-0 attention for each query residue
//                (which other residue it attends to most -- a "contact"-like
//                readout that recovers our planted long-range pair).
// ---------------------------------------------------------------------------
struct AttnResult {
    std::vector<float> out;        // [L * d_model]
    std::vector<float> attn;       // [L * L]   (head 0)
    std::vector<float> out_norm;   // [L]
    std::vector<int>   top_attn;   // [L]
};

// Load a protein from the sample file (format documented in data/README.md):
//   header: "d_model n_heads"      then a line with the amino-acid sequence.
// Derives L from the sequence, validates d_model % n_heads == 0, fills tokens.
// Throws std::runtime_error on a bad file so demos fail loudly.
ProteinInput load_protein(const std::string& path);

// Build the [L*d_model] input embedding matrix X from the tokens, using the
// shared embed_value() generator (so the GPU builds the identical matrix). This
// is the model's input; both reference and kernel start from it.
std::vector<float> build_embeddings(const ProteinInput& p);

// CPU REFERENCE: the full single-block multi-head self-attention forward pass.
//   Input  : the embeddings X (from build_embeddings) and the config.
//   Output : fills `r` (out, attn, out_norm, top_attn).
// This is the trusted serial baseline the GPU kernel is checked against; it uses
// the SAME attention_math.h primitives the kernel does. See reference_cpu.cpp.
void attention_cpu(const std::vector<float>& X, const AttnConfig& cfg, AttnResult& r);
