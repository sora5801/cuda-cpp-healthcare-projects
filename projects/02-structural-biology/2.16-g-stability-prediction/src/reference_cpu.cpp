// ===========================================================================
// src/reference_cpu.cpp  --  Trusted serial ΔΔG scan + protein loader
// ---------------------------------------------------------------------------
// Project 2.16 : ΔΔG Stability Prediction (reduced-scope teaching version)
//
// ROLE
//   (1) aa_index()    : one-letter amino-acid code -> canonical index 0..19.
//   (2) load_protein(): parse the tiny text dataset (format in data/README.md).
//   (3) ddg_scan_cpu(): the obviously-correct serial saturation-mutagenesis scan
//       the GPU kernel is verified against. No cleverness on purpose -- if CPU
//       and GPU agree to tolerance, we trust the GPU.
//
//   Compiled by the host C++ compiler ONLY (no CUDA). The per-mutation math is
//   the SHARED ddg_predict() from ddg_model.h, so this serial loop and the GPU
//   kernel evaluate the identical function (PATTERNS.md §2).
//
// READ THIS AFTER: reference_cpu.h, ddg_model.h. Compare with kernels.cu (twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cctype>      // std::toupper
#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream
#include <stdexcept>   // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// aa_index: map a one-letter code to its 0..19 index by scanning the canonical
//   AA_ONE_LETTER table from ddg_model.h. We uppercase first so 'a' and 'A' both
//   work. Returns -1 for any non-standard character (e.g. 'X', 'B', a gap '-').
//   Linear scan over 20 entries -- trivially cheap and keeps the single source
//   of truth (AA_ONE_LETTER) authoritative.
// ---------------------------------------------------------------------------
int aa_index(char one_letter) {
    const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(one_letter)));
    for (int i = 0; i < NUM_AA; ++i) {
        if (AA_ONE_LETTER[i] == c) return i;
    }
    return -1;   // not one of the 20 standard amino acids
}

// ---------------------------------------------------------------------------
// load_protein: parse the committed sample format (see data/README.md):
//     line 1 : <name>            (single token, used only for the report)
//     line 2 : <L>               (residue count, a positive integer)
//     next L : <AA> <buried>     (one-letter wild-type residue + burial fraction)
//   We validate aggressively and throw with a precise message on any problem so
//   a malformed file fails loudly instead of silently scanning garbage.
// ---------------------------------------------------------------------------
Protein load_protein(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open protein file: " + path);

    Protein prot;

    // --- line 1: the name label ------------------------------------------
    if (!(in >> prot.name))
        throw std::runtime_error("missing <name> on line 1 of " + path);

    // --- line 2: the residue count ---------------------------------------
    int L = 0;
    if (!(in >> L))
        throw std::runtime_error("missing <L> (residue count) in " + path);
    if (L <= 0)
        throw std::runtime_error("non-positive residue count in " + path);
    prot.L = L;
    prot.wt_code.resize(static_cast<std::size_t>(L));
    prot.buried.resize(static_cast<std::size_t>(L));

    // --- next L lines: <one-letter AA> <burial fraction> -----------------
    for (int p = 0; p < L; ++p) {
        std::string aa_tok;
        float buried = 0.0f;
        if (!(in >> aa_tok >> buried))
            throw std::runtime_error("unexpected end of residue data at position " +
                                     std::to_string(p) + " in " + path);
        if (aa_tok.size() != 1)
            throw std::runtime_error("residue " + std::to_string(p) +
                                     ": expected a single one-letter code, got '" + aa_tok + "'");
        const int code = aa_index(aa_tok[0]);
        if (code < 0)
            throw std::runtime_error("residue " + std::to_string(p) +
                                     ": '" + aa_tok + "' is not a standard amino acid");
        // Clamp burial into [0,1]; out-of-range values are almost certainly data
        // errors, but clamping keeps the model well-defined and the demo robust.
        if (buried < 0.0f) buried = 0.0f;
        if (buried > 1.0f) buried = 1.0f;
        prot.wt_code[static_cast<std::size_t>(p)] = code;
        prot.buried[static_cast<std::size_t>(p)]  = buried;
    }
    return prot;
}

// ---------------------------------------------------------------------------
// ddg_scan_cpu: the serial saturation-mutagenesis scan.
//   For every position p (outer loop) and every amino acid a (inner loop), call
//   the SHARED ddg_predict() and store the result row-major in out[p*NUM_AA + a].
//   Complexity: O(L * NUM_AA) calls, each O(1) -> O(L) work overall (NUM_AA is a
//   constant 20). This is the baseline the GPU parallelises across L*20 threads.
//
//   Because ddg_predict() is the *same* inline function the kernel calls, the
//   only sources of CPU/GPU disagreement are floating-point library rounding and
//   FMA contraction inside tanhf -- bounded well below the demo's 1e-3 tolerance.
// ---------------------------------------------------------------------------
void ddg_scan_cpu(const Protein& prot, std::vector<float>& out) {
    out.assign(static_cast<std::size_t>(prot.L) * NUM_AA, 0.0f);
    for (int p = 0; p < prot.L; ++p) {
        const int   wt  = prot.wt_code[static_cast<std::size_t>(p)];   // wild-type AA at p
        const float bur = prot.buried[static_cast<std::size_t>(p)];    // burial fraction at p
        for (int a = 0; a < NUM_AA; ++a) {
            // ddg_predict(wt, a, bur): predicted ΔΔG (kcal/mol) of p -> a.
            // The a == wt cell is exactly 0 by construction (self-mutation).
            out[static_cast<std::size_t>(p) * NUM_AA + a] = ddg_predict(wt, a, bur);
        }
    }
}
