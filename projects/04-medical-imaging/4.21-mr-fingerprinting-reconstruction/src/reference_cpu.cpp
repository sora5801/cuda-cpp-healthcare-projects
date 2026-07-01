// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ baseline MRF reconstruction we trust
// ---------------------------------------------------------------------------
// Project 4.21 : MR Fingerprinting Reconstruction
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- simple serial loops, no parallelism, no cleverness
//   -- so that when the GPU and CPU agree, we believe the GPU. Every scalar it
//   computes comes from the shared mrf_core.h, so the two sides are guaranteed
//   to build the SAME dictionary and the SAME normalized signals; only the ORDER
//   in which the big inner-product sum is accumulated differs (serial here vs.
//   cuBLAS SGEMM on the GPU), which is exactly the floating-point subtlety
//   main.cu's tolerance accounts for.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h
//   for the data model and mrf_core.h for the shared per-element math.
//
//   The MRF reconstruction pipeline, serially:
//     1. build_dictionary_cpu    : simulate + normalize every atom  (D x T)
//     2. normalize_signals_cpu   : normalize every voxel signal      (V x T)
//     3. match_cpu               : per voxel, argmax cosine over the D atoms
//
// READ THIS AFTER: reference_cpu.h, mrf_core.h. Compare against kernels.cu
// (the GPU twin that swaps step 3's per-voxel loop for one cuBLAS SGEMM).
// ===========================================================================
#include "reference_cpu.h"
#include "mrf_core.h"          // mrf::simulate_atom, normalize_inplace, dot, l2_norm

#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// load_problem: parse the committed sample file into an MrfProblem.
//
//   File format (whitespace-separated tokens; comments start with '#').
//   Header keywords make the file self-describing and order-independent per
//   block. scripts/make_synthetic.py writes exactly this and data/README.md
//   documents each field. The parser is intentionally strict: any missing
//   block throws, so a truncated file fails loudly instead of silently
//   reconstructing garbage.
//
//     T <int>                       # number of frames
//     D <int>                       # number of dictionary atoms
//     V <int>                       # number of voxels
//     ALPHA <T doubles>             # flip angles (radians)
//     TR    <T doubles>             # repetition times (ms)
//     TE    <T doubles>             # echo times (ms)
//     DICT  <D lines of: T1 T2>     # atom relaxation-time grid (ms)
//     SIGNAL<V lines of: truth_atom  T floats>   # voxel signals + ground truth
// ---------------------------------------------------------------------------
MrfProblem load_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open input file: " + path);

    MrfProblem p;

    // Small helper: read the next non-comment token as a string. Skips lines
    // beginning with '#' and blank whitespace. Throws at EOF so a truncated
    // file is reported rather than silently accepted.
    auto next_token = [&](std::string& tok) -> bool {
        while (in >> tok) {
            if (!tok.empty() && tok[0] == '#') {
                std::string skip;
                std::getline(in, skip);       // drop the rest of the comment line
                continue;
            }
            return true;
        }
        return false;
    };
    auto expect = [&](const char* kw) {
        std::string tok;
        if (!next_token(tok) || tok != kw)
            throw std::runtime_error(std::string("expected keyword '") + kw +
                                     "' but got '" + tok + "' in " + path);
    };
    auto read_int = [&]() -> int {
        std::string tok;
        if (!next_token(tok)) throw std::runtime_error("unexpected EOF (int) in " + path);
        return std::stoi(tok);
    };
    auto read_double = [&]() -> double {
        std::string tok;
        if (!next_token(tok)) throw std::runtime_error("unexpected EOF (double) in " + path);
        return std::stod(tok);
    };

    // ---- scalar sizes ------------------------------------------------------
    expect("T"); p.T = read_int();
    expect("D"); p.D = read_int();
    expect("V"); p.V = read_int();
    if (p.T <= 0 || p.D <= 0 || p.V <= 0)
        throw std::runtime_error("T, D, V must all be positive in " + path);

    // ---- acquisition schedule (three length-T blocks) ---------------------
    expect("ALPHA"); p.alpha.resize(p.T);
    for (int t = 0; t < p.T; ++t) p.alpha[t] = read_double();
    expect("TR");    p.tr.resize(p.T);
    for (int t = 0; t < p.T; ++t) p.tr[t] = read_double();
    expect("TE");    p.te.resize(p.T);
    for (int t = 0; t < p.T; ++t) p.te[t] = read_double();

    // ---- dictionary grid (D pairs of T1 T2) -------------------------------
    expect("DICT");
    p.dict_t1.resize(p.D);
    p.dict_t2.resize(p.D);
    for (int d = 0; d < p.D; ++d) {
        p.dict_t1[d] = read_double();
        p.dict_t2[d] = read_double();
    }

    // ---- voxel signals (V rows of: truth_atom followed by T floats) -------
    expect("SIGNAL");
    p.signal.assign(static_cast<std::size_t>(p.V) * p.T, 0.0f);
    p.truth_atom.resize(p.V);
    for (int v = 0; v < p.V; ++v) {
        p.truth_atom[v] = read_int();                  // ground-truth atom index
        for (int t = 0; t < p.T; ++t)
            p.signal[static_cast<std::size_t>(v) * p.T + t] =
                static_cast<float>(read_double());
    }
    return p;
}

// ---------------------------------------------------------------------------
// build_dictionary_cpu: simulate + L2-normalize every atom into dict_norm.
//   dict_norm is row-major D*T. For each atom d we (1) run mrf::simulate_atom
//   with that atom's (T1, T2) over the shared schedule, then (2) normalize the
//   resulting length-T fingerprint to unit energy. This is the serial twin of
//   the build_dictionary kernel (kernels.cu), one atom per GPU thread there.
// ---------------------------------------------------------------------------
void build_dictionary_cpu(const MrfProblem& p, std::vector<float>& dict_norm) {
    dict_norm.assign(static_cast<std::size_t>(p.D) * p.T, 0.0f);
    for (int d = 0; d < p.D; ++d) {
        float* atom = &dict_norm[static_cast<std::size_t>(d) * p.T];
        // The ONE TRUE simulator, shared with the GPU (mrf_core.h §B).
        mrf::simulate_atom(p.alpha.data(), p.tr.data(), p.te.data(), p.T,
                           p.dict_t1[d], p.dict_t2[d], atom);
        // Normalize so matching is a pure cosine (mrf_core.h §C).
        mrf::normalize_inplace(atom, p.T);
    }
}

// ---------------------------------------------------------------------------
// normalize_signals_cpu: L2-normalize each voxel signal into sig_norm (V*T),
//   recording each voxel's original norm in sig_scale[v] (used for the proton-
//   density map). We copy first, then normalize the copy, so the original
//   measured signal in p.signal is left untouched for any later inspection.
// ---------------------------------------------------------------------------
void normalize_signals_cpu(const MrfProblem& p, std::vector<float>& sig_norm,
                           std::vector<float>& sig_scale) {
    sig_norm = p.signal;                               // copy V*T floats
    sig_scale.assign(static_cast<std::size_t>(p.V), 0.0f);
    for (int v = 0; v < p.V; ++v) {
        float* sig = &sig_norm[static_cast<std::size_t>(v) * p.T];
        sig_scale[v] = mrf::l2_norm(sig, p.T);         // remember the scale (PD)
        mrf::normalize_inplace(sig, p.T);              // unit-energy shape
    }
}

// ---------------------------------------------------------------------------
// match_cpu: the reference matcher. For each voxel, scan all D atoms, compute
//   the cosine (an inner product of two unit vectors via mrf::dot), and keep
//   the argmax. Ties broken by SMALLEST atom index so the answer is unique.
//   Complexity O(V*D*T) -- the same ~10^11 inner-product work the catalog notes;
//   here it is a triple loop, which is exactly what the GPU replaces with one
//   cuBLAS SGEMM (V x T times T x D) plus a per-row argmax.
// ---------------------------------------------------------------------------
void match_cpu(const MrfProblem& p, const std::vector<float>& dict_norm,
               const std::vector<float>& sig_norm,
               const std::vector<float>& sig_scale,
               std::vector<MatchResult>& out) {
    out.assign(static_cast<std::size_t>(p.V), MatchResult{});
    for (int v = 0; v < p.V; ++v) {
        const float* sig = &sig_norm[static_cast<std::size_t>(v) * p.T];
        int   best_atom = 0;
        float best_cos  = -2.0f;                       // below the [-1,1] range
        for (int d = 0; d < p.D; ++d) {
            const float* atom = &dict_norm[static_cast<std::size_t>(d) * p.T];
            float c = mrf::dot(sig, atom, p.T);        // cosine (both unit norm)
            // Strict '>' keeps the FIRST (smallest-index) atom on ties -> the
            // deterministic tie-break that lets us verify GPU==CPU exactly.
            if (c > best_cos) { best_cos = c; best_atom = d; }
        }
        MatchResult r;
        r.best_atom = best_atom;
        r.score     = best_cos;
        r.t1_ms     = p.dict_t1[best_atom];
        r.t2_ms     = p.dict_t2[best_atom];
        r.pd        = sig_scale[v] * best_cos;         // recovered proton density
        out[v] = r;
    }
}
