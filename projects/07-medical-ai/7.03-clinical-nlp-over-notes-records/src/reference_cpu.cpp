// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ baseline we trust
// ---------------------------------------------------------------------------
// Project 7.3 : Clinical NLP over Notes & Records
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. Every routine here is
//   written to be OBVIOUSLY correct -- straight serial loops, no parallelism, no
//   cleverness -- so that when the GPU and CPU agree, we believe the GPU. All the
//   per-element arithmetic (projection weights, softmax numerics) is delegated to
//   attn_core.h, the SAME header the kernels use, so agreement is near-exact.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h
//   for the data types and the function contracts.
//
//   Pipeline mirrored from main.cu / kernels.cu:
//     load_notes -> build embeddings X per note
//        -> project X into Q, K, V with Wq/Wk/Wv
//        -> per head: scores = QKᵀ/sqrt(dh), mask pads, softmax rows, O = A V
//        -> concatenate heads into the output O [S x D]
//
// READ THIS AFTER: reference_cpu.h, attn_core.h. Compare with kernels.cu (twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// load_notes -- parse the committed text sample into a NoteBatch.
//
//   File grammar (see data/sample/notes_sample.txt and scripts/make_synthetic.py):
//     header line   : "<V> <D> <H> <S> <B>"
//     V lines       : "tok: <id> <string>"                 (vocabulary)
//     B lines       : "note: <len> <id0> <id1> ... "       (one per note)
//     D lines       : "emb: <V doubles>"   (row d = embedding dim d for all toks)
//     1 line        : "proj: <tag>"        (recipe id; weights rebuilt in code)
//   Lines beginning with '#' are comments and are skipped, so the sample file
//   can carry an explanatory header. We parse defensively and throw on anything
//   malformed -- a demo that silently runs on half a batch is worse than one that
//   stops and says why.
// ---------------------------------------------------------------------------
NoteBatch load_notes(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open notes file: " + path);

    NoteBatch nb;
    std::string line;

    // Read the next non-comment, non-blank line into `line`; false at EOF.
    auto next_line = [&](void) -> bool {
        while (std::getline(in, line)) {
            std::size_t s = line.find_first_not_of(" \t\r\n");
            if (s == std::string::npos) continue;   // blank
            if (line[s] == '#') continue;           // comment
            return true;
        }
        return false;
    };

    // ---- header: V D H S B -------------------------------------------------
    if (!next_line()) throw std::runtime_error("empty notes file: " + path);
    {
        std::istringstream hs(line);
        if (!(hs >> nb.V >> nb.D >> nb.H >> nb.S >> nb.B) ||
            nb.V <= 0 || nb.D <= 0 || nb.H <= 0 || nb.S <= 0 || nb.B <= 0)
            throw std::runtime_error("bad header (expected 'V D H S B'): " + line);
        if (nb.D % nb.H != 0)
            throw std::runtime_error("D must be divisible by H");
    }

    // ---- V vocabulary lines: "tok: <id> <string>" --------------------------
    nb.vocab.assign(static_cast<std::size_t>(nb.V), std::string());
    for (int t = 0; t < nb.V; ++t) {
        if (!next_line()) throw std::runtime_error("missing vocab line " + std::to_string(t));
        std::istringstream vs(line);
        std::string tag, str;
        int id = -1;
        vs >> tag >> id >> str;                   // "tok: 2 patient"
        if (id < 0 || id >= nb.V)
            throw std::runtime_error("vocab id out of range: " + line);
        nb.vocab[static_cast<std::size_t>(id)] = str;
    }

    // ---- B note lines: "note: <len> <ids...>" ------------------------------
    nb.token_ids.assign(static_cast<std::size_t>(nb.B) * nb.S, attn::TOK_PAD);
    nb.valid_len.assign(static_cast<std::size_t>(nb.B), 0);
    for (int b = 0; b < nb.B; ++b) {
        if (!next_line()) throw std::runtime_error("missing note line " + std::to_string(b));
        std::istringstream ns(line);
        std::string tag;
        int len = 0;
        ns >> tag >> len;                          // "note: 8 0 2 4 ..."
        if (len <= 0 || len > nb.S)
            throw std::runtime_error("note length out of [1,S]: " + line);
        nb.valid_len[static_cast<std::size_t>(b)] = len;
        for (int s = 0; s < len; ++s) {
            int id;
            if (!(ns >> id) || id < 0 || id >= nb.V)
                throw std::runtime_error("bad token id in note " + std::to_string(b));
            nb.token_ids[static_cast<std::size_t>(b) * nb.S + s] = id;
        }
        // positions [len, S) stay [PAD] from the assign() above.
    }

    // ---- D embedding rows: "emb: <V doubles>" ------------------------------
    // The file stores the table DIM-per-row (row d holds dim d for every token),
    // which is compact to read. We TRANSPOSE it into token-per-row `embed`
    // (embed[t*D + d]) because the attention math walks a token's whole vector.
    nb.embed.assign(static_cast<std::size_t>(nb.V) * nb.D, 0.0);
    for (int d = 0; d < nb.D; ++d) {
        if (!next_line()) throw std::runtime_error("missing emb row " + std::to_string(d));
        std::istringstream es(line);
        std::string tag;
        es >> tag;                                 // consume "emb:"
        for (int t = 0; t < nb.V; ++t) {
            double v;
            if (!(es >> v))
                throw std::runtime_error("emb row " + std::to_string(d)
                                         + " has fewer than V values");
            nb.embed[static_cast<std::size_t>(t) * nb.D + d] = v;   // transpose
        }
    }

    // ---- proj recipe tag (informational; weights rebuilt in build_projection)
    if (next_line()) {
        // Presence is enough; we do not branch on the tag value in this teaching
        // version (there is one recipe). A real loader would select a variant.
    }

    return nb;
}

// ---------------------------------------------------------------------------
// build_projection -- materialize one [D x D] projection matrix from the shared
//   attn::proj_entry recipe. Same code path as the GPU (kernels.cu builds these
//   with a kernel calling the SAME attn::proj_entry), so Wq/Wk/Wv are identical.
// ---------------------------------------------------------------------------
void build_projection(int D, int kind, std::vector<double>& W) {
    W.assign(static_cast<std::size_t>(D) * D, 0.0);
    for (int i = 0; i < D; ++i)
        for (int j = 0; j < D; ++j)
            W[static_cast<std::size_t>(i) * D + j] = attn::proj_entry(i, j, kind);
}

// ---------------------------------------------------------------------------
// project_note -- Y = X * W  for one note.
//   X is [S x D] (the note's token embeddings), W is [D x D], Y is [S x D].
//   This is the small dense matmul the GPU hands to cuBLAS; here it is the
//   textbook triple loop so the reference is unmistakably correct. O(S*D*D).
// ---------------------------------------------------------------------------
static void project_note(const std::vector<double>& X, const std::vector<double>& W,
                         int S, int D, std::vector<double>& Y) {
    Y.assign(static_cast<std::size_t>(S) * D, 0.0);
    for (int s = 0; s < S; ++s) {
        for (int j = 0; j < D; ++j) {
            double acc = 0.0;
            for (int k = 0; k < D; ++k)
                acc += X[static_cast<std::size_t>(s) * D + k]
                     * W[static_cast<std::size_t>(k) * D + j];
            Y[static_cast<std::size_t>(s) * D + j] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// attention_reference -- ONE self-attention encoder block over the whole batch.
//
//   For each note b:
//     1. Gather X [S x D] from the embedding table (X[s] = embed[token_ids[b,s]]).
//     2. Project: Q = X Wq, K = X Wk, V = X Wv  (each [S x D]).
//     3. For each head h (columns [h*dh, (h+1)*dh) of Q/K/V):
//          a. scores[qi][kj] = (Q_h[qi] · K_h[kj]) * 1/sqrt(dh)   ([S x S])
//             -- mask: if key kj is a [PAD] position, set score = -1e30 so it
//                gets ~0 probability (padding must never receive attention).
//          b. A_h[qi] = softmax(scores[qi])  (stable, max-subtracted; attn_core)
//          c. O_h[qi] = Σ_kj A_h[qi][kj] * V_h[kj]   ([S x dh])
//        Write A_h into res.weights and O_h into the head's slice of res.out.
//
//   Complexity per note: O(S*D*D) projections + O(H*S*S*dh) attention. Serial,
//   readable, and the ground truth for the GPU. Query rows for PAD positions are
//   still computed (harmless) but are ignored by the report in main.cu.
// ---------------------------------------------------------------------------
void attention_reference(const NoteBatch& nb, AttnResult& res) {
    const int S = nb.S, D = nb.D, H = nb.H, dh = nb.dh();
    res.allocate(nb.B, H, S, D);

    // Projection matrices (shared recipe) -- built once, reused for every note.
    std::vector<double> Wq, Wk, Wv;
    build_projection(D, 0, Wq);
    build_projection(D, 1, Wk);
    build_projection(D, 2, Wv);

    const double scale = attn::attn_scale(dh);

    std::vector<double> X, Q, K, Vv;   // per-note scratch, reused across notes
    std::vector<double> scores(static_cast<std::size_t>(S));  // one softmax row

    for (int b = 0; b < nb.B; ++b) {
        // ---- 1. gather this note's token embeddings into X [S x D] ---------
        X.assign(static_cast<std::size_t>(S) * D, 0.0);
        for (int s = 0; s < S; ++s) {
            int t = nb.tok(b, s);
            for (int d = 0; d < D; ++d)
                X[static_cast<std::size_t>(s) * D + d] =
                    nb.embed[static_cast<std::size_t>(t) * D + d];
        }

        // ---- 2. project into Q, K, V --------------------------------------
        project_note(X, Wq, S, D, Q);
        project_note(X, Wk, S, D, K);
        project_note(X, Wv, S, D, Vv);

        // ---- 3. per-head scaled dot-product attention ---------------------
        for (int h = 0; h < H; ++h) {
            const int col0 = h * dh;              // this head's column offset
            for (int qi = 0; qi < S; ++qi) {
                // (a) raw scores for query qi against every key kj, masked.
                for (int kj = 0; kj < S; ++kj) {
                    int ktok = nb.tok(b, kj);
                    if (ktok == attn::TOK_PAD) {
                        scores[static_cast<std::size_t>(kj)] = -1.0e30;  // mask
                        continue;
                    }
                    double dot = 0.0;
                    for (int c = 0; c < dh; ++c)
                        dot += Q[static_cast<std::size_t>(qi) * D + col0 + c]
                             * K[static_cast<std::size_t>(kj) * D + col0 + c];
                    scores[static_cast<std::size_t>(kj)] = dot * scale;
                }
                // (b) stable softmax over the S keys (attn_core numerics).
                double m = attn::row_max(scores.data(), S);
                double denom = 0.0;
                for (int kj = 0; kj < S; ++kj) {
                    double e = attn::softmax_exp(scores[static_cast<std::size_t>(kj)], m);
                    scores[static_cast<std::size_t>(kj)] = e;   // reuse as exp buffer
                    denom += e;
                }
                std::size_t wbase =
                    ((static_cast<std::size_t>(b) * H + h) * S + qi) * S;
                for (int kj = 0; kj < S; ++kj)
                    res.weights[wbase + kj] =
                        scores[static_cast<std::size_t>(kj)] / denom;   // probs

                // (c) context vector O_h[qi] = Σ_kj A[qi][kj] * V_h[kj].
                for (int c = 0; c < dh; ++c) {
                    double acc = 0.0;
                    for (int kj = 0; kj < S; ++kj)
                        acc += res.weights[wbase + kj]
                             * Vv[static_cast<std::size_t>(kj) * D + col0 + c];
                    res.out[(static_cast<std::size_t>(b) * S + qi) * D + col0 + c] = acc;
                }
            }
        }
    }
}
