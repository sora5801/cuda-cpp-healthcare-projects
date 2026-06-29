// ===========================================================================
// src/reference_cpu.cpp  --  The trusted serial Tanimoto baseline + data loader
// ---------------------------------------------------------------------------
// Project 1.12 : Molecular Fingerprint Similarity Search
//
// ROLE
//   (1) load_fingerprints(): parse the tiny text dataset (data/README.md format).
//   (2) tanimoto_cpu(): the obviously-correct serial computation the GPU kernel
//       is verified against. No cleverness on purpose -- if CPU and GPU agree,
//       we trust the GPU.
//
//   Compiled by the host C++ compiler only (no CUDA). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h. Compare against kernels.cu (the GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// popcount64: count the set bits in a 64-bit word (Brian Kernighan's method:
// `x &= x - 1` clears the lowest set bit, so the loop runs once per set bit).
// We use this plain version for the REFERENCE so it is dependency-free and
// obviously correct. (The GPU uses the single-instruction __popcll intrinsic;
// modern CPUs have a POPCNT instruction too -- see THEORY "Numerical notes".)
static inline int popcount64(uint64_t x) {
    int count = 0;
    while (x) { x &= (x - 1); ++count; }
    return count;
}

FingerprintSet load_fingerprints(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open fingerprint file: " + path);

    int n = 0, words = 0;
    if (!(in >> n >> words))
        throw std::runtime_error("bad header (expected '<n> <FP_WORDS>') in " + path);
    if (words != FP_WORDS)
        throw std::runtime_error("fingerprint width mismatch: file has " +
                                 std::to_string(words) + " words but this build expects " +
                                 std::to_string(FP_WORDS) + " (rebuild with matching FP_WORDS)");
    if (n <= 0) throw std::runtime_error("non-positive library size in " + path);

    FingerprintSet fps;
    fps.n = n;
    fps.query.resize(FP_WORDS);
    fps.lib.resize(static_cast<std::size_t>(n) * FP_WORDS);

    // Each word is a 16-hex-digit (64-bit) value. std::stoull(.,.,16) parses it.
    auto read_word = [&](uint64_t& w) {
        std::string tok;
        if (!(in >> tok)) throw std::runtime_error("unexpected end of data in " + path);
        w = std::stoull(tok, nullptr, 16);
    };
    for (int j = 0; j < FP_WORDS; ++j) read_word(fps.query[j]);
    for (std::size_t k = 0; k < fps.lib.size(); ++k) read_word(fps.lib[k]);
    return fps;
}

void tanimoto_cpu(const FingerprintSet& fps, std::vector<float>& out) {
    out.assign(static_cast<std::size_t>(fps.n), 0.0f);
    for (int i = 0; i < fps.n; ++i) {
        // Walk the i-th library fingerprint word by word, accumulating the
        // popcounts of the intersection (A & B) and the union (A | B).
        const uint64_t* b = &fps.lib[static_cast<std::size_t>(i) * FP_WORDS];
        int inter = 0, uni = 0;
        for (int w = 0; w < FP_WORDS; ++w) {
            const uint64_t a = fps.query[w];
            inter += popcount64(a & b[w]);   // bits set in BOTH  -> intersection
            uni   += popcount64(a | b[w]);   // bits set in EITHER -> union
        }
        // Tanimoto = |A ^ B| / |A v B|. Guard the (degenerate) all-zero case so
        // two empty fingerprints score 0 rather than NaN. inter,uni <= 2048 are
        // exactly representable in float, so this division is identical on the
        // GPU -> the two implementations agree bit-for-bit (THEORY "verify").
        out[static_cast<std::size_t>(i)] = uni ? static_cast<float>(inter) / static_cast<float>(uni)
                                               : 0.0f;
    }
}
