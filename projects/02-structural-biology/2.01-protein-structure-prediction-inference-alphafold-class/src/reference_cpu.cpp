// ===========================================================================
// src/reference_cpu.cpp  --  Trusted serial self-attention + data loader
// ---------------------------------------------------------------------------
// Project 2.1 : Protein Structure Prediction Inference (AlphaFold-class)
//               REDUCED-SCOPE TEACHING VERSION.
//
// ROLE
//   (1) load_attention(): parse the tiny text dataset (format in data/README.md).
//   (2) attention_cpu(): the obviously-correct serial computation the GPU kernel
//       is verified against. No cleverness on purpose -- it walks the textbook
//       three-line definition of scaled dot-product attention so that, if the
//       GPU agrees with it, we trust the GPU.
//
//   Compiled by the host C++ compiler only (no CUDA). The per-element math is
//   the SAME code the kernel runs, via attention_core.h (PATTERNS.md sec 2).
//
// READ THIS AFTER: reference_cpu.h and attention_core.h. Compare line-for-line
// against kernels.cu (the GPU twin) -- the two compute the identical formula.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// read_token: pull the next whitespace-separated token from `in`, SKIPPING any
// line whose first non-space character is '#'. This lets the committed sample
// carry teaching comments (e.g. "# Q matrix ...") inline without breaking the
// parser. Returns false at end-of-data. We tokenize line-by-line so a '#' only
// comments out the REST of its line, mirroring the make_synthetic.py writer.
// ---------------------------------------------------------------------------
static bool read_token(std::istream& in, std::string& tok) {
    // We keep a static-free design by buffering the current line in the stream's
    // own state: read whitespace-delimited tokens, but if we hit a '#' token,
    // discard the remainder of that line and continue.
    while (in >> tok) {
        if (!tok.empty() && tok[0] == '#') {
            std::string rest;
            std::getline(in, rest);    // drop the rest of the comment line
            continue;
        }
        return true;
    }
    return false;
}

// read_int / read_float: typed wrappers around read_token that convert and throw
// a clear error if the token is not a valid number.
static int read_int(std::istream& in, const std::string& what) {
    std::string tok;
    if (!read_token(in, tok)) throw std::runtime_error("missing " + what);
    return std::stoi(tok);
}
static float read_float(std::istream& in, const std::string& what) {
    std::string tok;
    if (!read_token(in, tok)) throw std::runtime_error("missing " + what);
    return std::stof(tok);
}

// ---------------------------------------------------------------------------
// load_attention: read "<L> <d>" then 3*L rows of d floats (Q, then K, then V).
//   We validate d == D_MODEL because the kernel's shared-memory sizing and the
//   attention_core primitives are all written for the compile-time D_MODEL; a
//   mismatched file would silently misalign rows. Failing loudly here saves the
//   learner a baffling wrong-answer hunt later. '#' lines are skipped (comments).
// ---------------------------------------------------------------------------
AttentionProblem load_attention(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open attention file: " + path);

    const int L = read_int(in, "header field L");
    const int d = read_int(in, "header field d");
    if (d != D_MODEL)
        throw std::runtime_error("feature-width mismatch: file has d=" +
                                 std::to_string(d) + " but this build expects D_MODEL=" +
                                 std::to_string(D_MODEL) + " (rebuild with matching D_MODEL)");
    if (L <= 0) throw std::runtime_error("non-positive sequence length in " + path);

    AttentionProblem prob;
    prob.L = L;
    prob.d = d;
    const std::size_t mat = static_cast<std::size_t>(L) * D_MODEL;  // floats per matrix
    prob.q.resize(mat);
    prob.k.resize(mat);
    prob.v.resize(mat);

    // Helper: read exactly `mat` floats into a matrix, or throw if data runs out.
    // Uses read_float so inline '#' comment lines between matrices are skipped.
    auto read_matrix = [&](std::vector<float>& m, const char* name) {
        for (std::size_t e = 0; e < m.size(); ++e) {
            m[e] = read_float(in, std::string(name) + " matrix element");
        }
    };
    read_matrix(prob.q, "Q");   // rows 0..L-1
    read_matrix(prob.k, "K");   // rows L..2L-1
    read_matrix(prob.v, "V");   // rows 2L..3L-1
    return prob;
}

// ---------------------------------------------------------------------------
// attention_cpu: one head of scaled dot-product self-attention, residue by
// residue. For each query residue i we do the three textbook steps:
//   (1) scores[j] = scaled_score(Q[i], K[j])     for all j           [O(L*d)]
//   (2) w[j]      = softmax(scores)               (stable, max-subtract)
//   (3) out[i]    = sum_j w[j] * V[j]             (weighted value avg) [O(L*d)]
// Total cost per row is O(L*d); over all L rows it is O(L^2 * d) -- the
// quadratic-in-sequence-length cost that makes long proteins expensive and is
// exactly why this step is pushed onto the GPU (THEORY sec 3).
// ---------------------------------------------------------------------------
void attention_cpu(const AttentionProblem& prob, std::vector<float>& out) {
    const int L = prob.L;
    const int d = prob.d;
    out.assign(static_cast<std::size_t>(L) * d, 0.0f);

    // A scratch buffer for one row's scores; reused across i to avoid reallocations.
    std::vector<double> scores(static_cast<std::size_t>(L));

    for (int i = 0; i < L; ++i) {
        const float* q_i = &prob.q[static_cast<std::size_t>(i) * d];   // query row i

        // (1) raw scaled affinities of residue i against every residue j, and
        //     track the row maximum for the numerically-stable softmax shift.
        double row_max = -1.0e308;   // effectively -infinity for double
        for (int j = 0; j < L; ++j) {
            const float* k_j = &prob.k[static_cast<std::size_t>(j) * d];
            const double s = scaled_score(q_i, k_j, d);   // (Q[i].K[j]) / sqrt(d)
            scores[static_cast<std::size_t>(j)] = s;
            if (s > row_max) row_max = s;
        }

        // (2) exponentiate the SHIFTED scores and accumulate the normaliser.
        //     stable_exp(s, row_max) = exp(s - row_max) is in (0,1] -> no overflow.
        double denom = 0.0;
        for (int j = 0; j < L; ++j) {
            const double e = stable_exp(scores[static_cast<std::size_t>(j)], row_max);
            scores[static_cast<std::size_t>(j)] = e;   // overwrite score with its exp
            denom += e;
        }
        const double inv_denom = 1.0 / denom;          // 1 / sum_j exp(...)

        // (3) out[i][:] = sum_j w[j] * V[j][:], where w[j] = exp(...) / denom.
        //     We accumulate each output channel in double, then store as float.
        float* out_i = &out[static_cast<std::size_t>(i) * d];
        for (int j = 0; j < L; ++j) {
            const double w   = scores[static_cast<std::size_t>(j)] * inv_denom;  // softmax weight
            const float* v_j = &prob.v[static_cast<std::size_t>(j) * d];
            for (int c = 0; c < d; ++c) {
                // read-modify-write each channel; promote to double to match the
                // GPU accumulation order/precision (THEORY sec 5 "determinism").
                out_i[c] = static_cast<float>(static_cast<double>(out_i[c]) +
                                              w * static_cast<double>(v_j[c]));
            }
        }
    }
}
