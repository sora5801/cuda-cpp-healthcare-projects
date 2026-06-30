// ===========================================================================
// src/reference_cpu.cpp  --  Loader, embeddings, serial self-attention baseline
// ---------------------------------------------------------------------------
// Project 3.18 : Protein Language Model Inference
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is written to be
//   OBVIOUSLY correct -- one residue at a time, no parallelism -- so that when
//   the GPU and CPU agree we believe the GPU. It calls the SAME per-element
//   helpers (attention_math.h) the kernel does, so agreement is exact-ish.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, attention_math.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_protein: parse "d_model n_heads" then a sequence line into a problem.
//   The sequence length L is whatever the (cleaned) amino-acid line contains.
//   We validate the head split (d_model % n_heads == 0) up front because every
//   downstream loop assumes d_head = d_model / n_heads divides evenly.
// ---------------------------------------------------------------------------
ProteinInput load_protein(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open protein file: " + path);

    ProteinInput p;
    int d_model = 0, n_heads = 0;
    if (!(in >> d_model >> n_heads) || d_model <= 0 || n_heads <= 0)
        throw std::runtime_error("bad header (expected 'd_model n_heads') in " + path);
    if (d_model % n_heads != 0)
        throw std::runtime_error("d_model must be divisible by n_heads in " + path);

    // Read the remaining non-whitespace characters as the sequence. We skip any
    // stray spaces/newlines so the file can wrap the sequence across lines.
    std::string seq, tok;
    while (in >> tok) seq += tok;
    if (seq.empty()) throw std::runtime_error("empty sequence in " + path);

    p.sequence = seq;
    p.cfg.seq_len = static_cast<int>(seq.size());
    p.cfg.d_model = d_model;
    p.cfg.n_heads = n_heads;
    p.cfg.d_head  = d_model / n_heads;

    // Tokenize: every residue must be one of the 20 canonical amino acids.
    p.tokens.resize(seq.size());
    for (std::size_t i = 0; i < seq.size(); ++i) {
        const int t = token_of(seq[i]);
        if (t < 0) {
            std::ostringstream os;
            os << "unknown residue '" << seq[i] << "' at position " << i << " in " << path;
            throw std::runtime_error(os.str());
        }
        p.tokens[i] = t;
    }
    return p;
}

// ---------------------------------------------------------------------------
// build_embeddings: X[i, f] = embed_value(token_i, f).  [L * d_model] row-major.
//   This is the model's input matrix. Because embed_value() is a pure integer
//   hash of (token, feature), the GPU reconstructs the IDENTICAL matrix from the
//   tokens alone -- no float data crosses the PCIe bus, and there is no RNG-order
//   ambiguity to make CPU and GPU diverge.
// ---------------------------------------------------------------------------
std::vector<float> build_embeddings(const ProteinInput& p) {
    const int L = p.cfg.seq_len, D = p.cfg.d_model;
    std::vector<float> X(static_cast<std::size_t>(L) * D);
    for (int i = 0; i < L; ++i)
        for (int f = 0; f < D; ++f)
            X[static_cast<std::size_t>(i) * D + f] = embed_value(p.tokens[i], f, SALT_EMBED);
    return X;
}

// ---------------------------------------------------------------------------
// attention_cpu: the full single-block multi-head self-attention forward pass.
//
//   Notation (all matrices row-major):
//     X : [L x D]            input embeddings        (D = d_model)
//     Q,K,V : [L x D]        projections of X        (Q = X Wq, etc.)
//     For head h, columns [h*d_head .. h*d_head+d_head) of Q/K/V are that head's
//     d_head-wide vectors. Per head:
//       S = Q_h K_hᵀ / sqrt(d_head)     [L x L] logits
//       A = softmax(S, over keys)       [L x L] attention weights (rows sum to 1)
//       H_h = A V_h                     [L x d_head] head output
//     The heads' outputs are concatenated back into a [L x D] matrix Z, then
//       Y = Z Wo                        [L x D] the block output.
//
//   We compute Q,K,V on the fly via proj_one() (no separate matrices stored),
//   matching exactly what the kernel does. Complexity: O(L²·D) -- the L² is the
//   attention map, the D the projection/blend width.
// ---------------------------------------------------------------------------
void attention_cpu(const std::vector<float>& X, const AttnConfig& cfg, AttnResult& r) {
    const int L = cfg.seq_len, D = cfg.d_model, H = cfg.n_heads, dh = cfg.d_head;

    // Project the whole sequence into Q, K, V once (each [L x D]). Storing them
    // makes the per-head loops below readable; the kernel recomputes instead to
    // save memory, but the arithmetic is identical (proj_one in both).
    std::vector<float> Q(static_cast<std::size_t>(L) * D);
    std::vector<float> K(static_cast<std::size_t>(L) * D);
    std::vector<float> V(static_cast<std::size_t>(L) * D);
    for (int i = 0; i < L; ++i) {
        const float* xi = &X[static_cast<std::size_t>(i) * D];
        for (int j = 0; j < D; ++j) {
            Q[static_cast<std::size_t>(i) * D + j] = proj_one(xi, D, j, SALT_WQ);
            K[static_cast<std::size_t>(i) * D + j] = proj_one(xi, D, j, SALT_WK);
            V[static_cast<std::size_t>(i) * D + j] = proj_one(xi, D, j, SALT_WV);
        }
    }

    // Z = concatenated per-head outputs, [L x D]. Filled head by head.
    std::vector<float> Z(static_cast<std::size_t>(L) * D, 0.0f);
    r.attn.assign(static_cast<std::size_t>(L) * L, 0.0f);  // head-0 map we report

    std::vector<float> srow(L);  // reusable scratch for one query's L logits

    for (int h = 0; h < H; ++h) {
        const int off = h * dh;  // column offset of this head inside Q/K/V/Z
        for (int i = 0; i < L; ++i) {              // query residue i
            const float* qi = &Q[static_cast<std::size_t>(i) * D + off];
            // (1) logits row S[i,*] = q_i . k_j / sqrt(d_head) for all keys j.
            for (int j = 0; j < L; ++j) {
                const float* kj = &K[static_cast<std::size_t>(j) * D + off];
                srow[j] = scaled_score(qi, kj, dh);
            }
            // (2) softmax over keys -> attention weights (probabilities).
            softmax_inplace(srow.data(), L);
            // Keep head 0's full attention map for reporting/verification.
            if (h == 0)
                for (int j = 0; j < L; ++j)
                    r.attn[static_cast<std::size_t>(i) * L + j] = srow[j];
            // (3) head output row = sum_j A[i,j] * V_h[j]   (the value blend).
            for (int t = 0; t < dh; ++t) {
                double acc = 0.0;
                for (int j = 0; j < L; ++j)
                    acc += static_cast<double>(srow[j])
                         * V[static_cast<std::size_t>(j) * D + off + t];
                Z[static_cast<std::size_t>(i) * D + off + t] = static_cast<float>(acc);
            }
        }
    }

    // (4) Output projection  Y = Z Wo, [L x D], + per-residue L2 norms.
    r.out.assign(static_cast<std::size_t>(L) * D, 0.0f);
    r.out_norm.assign(L, 0.0f);
    for (int i = 0; i < L; ++i) {
        const float* zi = &Z[static_cast<std::size_t>(i) * D];
        double n2 = 0.0;
        for (int j = 0; j < D; ++j) {
            const float y = proj_one(zi, D, j, SALT_WO);
            r.out[static_cast<std::size_t>(i) * D + j] = y;
            n2 += static_cast<double>(y) * y;
        }
        r.out_norm[i] = static_cast<float>(std::sqrt(n2));
    }

    // top_attn[i] = the key residue head 0 attends to most for query i (argmax,
    // ties -> lowest index). This is the "contact"-like readout the demo reports.
    r.top_attn.assign(L, 0);
    for (int i = 0; i < L; ++i) {
        int best = 0;
        float bv = r.attn[static_cast<std::size_t>(i) * L + 0];
        for (int j = 1; j < L; ++j) {
            const float v = r.attn[static_cast<std::size_t>(i) * L + j];
            if (v > bv) { bv = v; best = j; }
        }
        r.top_attn[i] = best;
    }
}
