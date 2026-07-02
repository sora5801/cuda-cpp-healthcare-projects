// ===========================================================================
// src/reference_cpu.h  --  Trusted serial baseline: types + declarations
// ---------------------------------------------------------------------------
// Project 7.3 : Clinical NLP over Notes & Records
//
// ROLE
//   Declares the plain-C++ data structures and the CPU reference functions that
//   main.cu uses as the GROUND TRUTH for verifying the GPU. The reference runs
//   the SAME computation the GPU does -- one transformer self-attention encoder
//   block over a batch of tokenized notes -- but with simple, readable serial
//   loops. Because both sides call the SAME per-element math in attn_core.h, the
//   GPU output must match this to (near) machine precision.
//
//   This header is included by reference_cpu.cpp (the implementation), main.cu
//   (the caller), and kernels.cu (which reuses NoteBatch). It contains no CUDA
//   constructs, so the host compiler is happy.
//
// READ THIS AFTER: attn_core.h.  READ THIS BEFORE: reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "attn_core.h"   // attn:: shared per-element formulas + special-token ids

// ---------------------------------------------------------------------------
// NoteBatch -- the whole problem, loaded from the sample file.
//   A batch of B clinical notes, each a sequence of S token ids (short notes are
//   padded with [PAD]=1). We also carry the vocabulary strings (for readable
//   output) and the token embedding table.
//
//   Memory layout (all row-major, all std::vector so ownership is obvious):
//     token_ids : [B*S] int   -- token_ids[b*S + s] is position s of note b
//     valid_len : [B]   int   -- real (non-pad) length of each note (<= S)
//     embed     : [V*D] double-- embed[t*D + d] is dim d of token t's embedding
//                                (the file stores it dim-per-row; the loader
//                                transposes it into this token-per-row layout)
//   D, H, S, V, B are the model dims. dh = D/H is the per-head width.
// ---------------------------------------------------------------------------
struct NoteBatch {
    int V = 0;   // vocabulary size
    int D = 0;   // model / embedding dimension
    int H = 0;   // number of attention heads (D % H == 0)
    int S = 0;   // sequence length (all notes padded to this)
    int B = 0;   // number of notes in the batch

    std::vector<int>         token_ids;  // [B*S] row-major token ids
    std::vector<int>         valid_len;  // [B]   non-pad length per note
    std::vector<std::string> vocab;      // [V]   human-readable token strings
    std::vector<double>      embed;      // [V*D] row-per-token embedding table

    int dh() const { return D / H; }     // per-head dimension

    // Convenience: token id at (note b, position s).
    int tok(int b, int s) const {
        return token_ids[static_cast<std::size_t>(b) * S + s];
    }
};

// ---------------------------------------------------------------------------
// AttnResult -- everything the attention block produces for the whole batch,
//   laid out so the GPU can memcpy into the identical shapes for verification.
//     out     : [B*S*D]     contextualized output embedding O per token
//     weights : [B*H*S*S]   attention probabilities A (per note, head, query row)
//               index: ((b*H + h)*S + qi)*S + kj  = P(token qi attends to kj)
//   These two arrays are what main.cu compares CPU-vs-GPU entrywise.
// ---------------------------------------------------------------------------
struct AttnResult {
    std::vector<double> out;       // [B*S*D]
    std::vector<double> weights;   // [B*H*S*S]

    void allocate(int B, int H, int S, int D) {
        out.assign(static_cast<std::size_t>(B) * S * D, 0.0);
        weights.assign(static_cast<std::size_t>(B) * H * S * S, 0.0);
    }
};

// load_notes: parse the committed sample file into a NoteBatch.
//   File format documented in data/README.md and scripts/make_synthetic.py
//   (header "V D H S B", vocab lines, note lines, an embedding table, a proj
//   recipe tag). Throws std::runtime_error on a malformed/missing file so demos
//   fail loudly instead of silently running on empty input.
NoteBatch load_notes(const std::string& path);

// build_projection: materialize one [D x D] projection matrix (Wq/Wk/Wv) from
//   the shared attn::proj_entry recipe. kind: 0=Wq, 1=Wk, 2=Wv. Output W is
//   [D*D] row-major. Both CPU and GPU rebuild the SAME matrices this way.
void build_projection(int D, int kind, std::vector<double>& W);

// attention_reference: run ONE self-attention encoder block over the whole
//   batch on the CPU with obvious serial loops. Fills `res` (out + weights).
//   This is the ground truth the GPU is checked against. O(B*(S*S*D + S*S*dh*H)).
void attention_reference(const NoteBatch& nb, AttnResult& res);
