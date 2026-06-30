// ===========================================================================
// src/reference_cpu.cpp  --  Serial CPU reference: load + normalize + KNN graph
// ---------------------------------------------------------------------------
// Project 3.12 : Single-Cell RNA-seq Analysis  (reduced-scope teaching version)
//
// This is the READABLE BASELINE the GPU is checked against (CLAUDE.md section 5).
// It calls the SAME shared math as the GPU kernels (scrna.h: sc_normalize_entry,
// sc_knn_one_cell), just in a plain serial loop, so the two results agree
// exactly on the integer neighbour indices and to the last bit on the floats.
//
// Compiled by the HOST C++ compiler (no CUDA here). main.cu times this against
// the GPU and prints the speed-up as a teaching artifact.
//
// READ THIS AFTER: scrna.h (the per-element math) and reference_cpu.h (types).
// ===========================================================================
#include "reference_cpu.h"
#include "scrna.h"          // sc_normalize_entry, sc_knn_one_cell (shared math)

#include <cctype>           // std::isspace
#include <cmath>            // std::sqrt
#include <cstddef>          // std::size_t
#include <fstream>          // std::ifstream
#include <sstream>          // std::istringstream
#include <stdexcept>        // std::runtime_error
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// load_dataset: parse the tiny text sample.
//
//   FILE FORMAT (whitespace/newline separated; '#' starts a comment line):
//     line 1 header : N  G  k  target_sum
//     then N rows   : <label> c0 c1 ... c(G-1)
//                     where <label> is the ground-truth type id (-1 if unknown)
//                     followed by G integer counts for that cell.
//
//   We parse line-by-line so we can skip comment lines and give precise errors.
//   The whole point of the loud throws is that a demo on a truncated file fails
//   immediately and visibly rather than silently producing a wrong graph.
// ---------------------------------------------------------------------------
Dataset load_dataset(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open dataset: " + path);

    // Helper: read the next NON-comment, non-blank line into `line`.
    auto next_line = [&](std::string& line) -> bool {
        while (std::getline(in, line)) {
            // Trim a leading '#': anything from '#' onward is a comment.
            std::size_t h = line.find('#');
            if (h != std::string::npos) line = line.substr(0, h);
            // Keep the line only if it has a non-space character.
            for (char ch : line) {
                if (!std::isspace(static_cast<unsigned char>(ch))) return true;
            }
        }
        return false;
    };

    Dataset d;
    std::string line;

    // ---- header: N G k target_sum ----------------------------------------
    if (!next_line(line)) throw std::runtime_error("empty dataset: " + path);
    {
        std::istringstream hs(line);
        if (!(hs >> d.N >> d.G >> d.k >> d.target_sum))
            throw std::runtime_error("bad header (need: N G k target_sum) in " + path);
    }
    if (d.N <= 0 || d.G <= 0 || d.k <= 0)
        throw std::runtime_error("N, G, k must all be positive in " + path);
    if (d.G > SC_MAX_GENES)
        throw std::runtime_error("G exceeds SC_MAX_GENES (this teaching build caps the gene panel)");
    if (d.k > SC_MAX_K)
        throw std::runtime_error("k exceeds SC_MAX_K (each thread keeps a fixed-size top-k list)");
    if (d.k >= d.N)
        throw std::runtime_error("k must be < N (a cell cannot have N or more distinct neighbours)");
    if (d.target_sum <= 0.0)
        throw std::runtime_error("target_sum must be positive in " + path);

    // ---- N data rows: <label> followed by G counts -----------------------
    d.counts.resize(static_cast<std::size_t>(d.N) * d.G);
    d.labels.resize(d.N);
    for (int c = 0; c < d.N; ++c) {
        if (!next_line(line))
            throw std::runtime_error("dataset ended early: expected " + std::to_string(d.N) + " cells");
        std::istringstream rs(line);
        if (!(rs >> d.labels[c]))
            throw std::runtime_error("missing label for cell " + std::to_string(c));
        for (int g = 0; g < d.G; ++g) {
            float v;
            if (!(rs >> v))
                throw std::runtime_error("cell " + std::to_string(c) + " has fewer than G counts");
            d.counts[static_cast<std::size_t>(c) * d.G + g] = v;
        }
    }
    return d;
}

// ---------------------------------------------------------------------------
// cpu_normalize: fill out.normalized[N*G] from the raw counts.
//   Step 1: per cell, sum its G counts -> the library size (total counts).
//   Step 2: per entry, apply sc_normalize_entry (counts-per-target + log1p).
//   We store float (the GPU does too); the cast happens identically on both
//   sides so the normalized matrices are bit-identical.
// ---------------------------------------------------------------------------
static void cpu_normalize(const Dataset& d, KnnGraph& out) {
    out.normalized.resize(static_cast<std::size_t>(d.N) * d.G);
    for (int c = 0; c < d.N; ++c) {
        const float* row = d.counts.data() + static_cast<std::size_t>(c) * d.G;

        // Library size = total counts captured in this cell.
        double cell_total = 0.0;
        for (int g = 0; g < d.G; ++g) cell_total += static_cast<double>(row[g]);

        // Normalize each entry with the shared formula.
        for (int g = 0; g < d.G; ++g) {
            const double val = sc_normalize_entry(static_cast<double>(row[g]),
                                                  cell_total, d.target_sum);
            out.normalized[static_cast<std::size_t>(c) * d.G + g] = static_cast<float>(val);
        }
    }
}

// ---------------------------------------------------------------------------
// cpu_knn: fill out.nbr_idx[N*k] / out.nbr_dist[N*k] by brute force.
//   For each query cell q, sc_knn_one_cell scans all other cells and returns the
//   k nearest (by squared distance). We then take sqrt for the human-readable
//   Euclidean distance. The ranking itself used squared distance (monotonic), so
//   taking sqrt here does not change any ordering.
// ---------------------------------------------------------------------------
static void cpu_knn(const Dataset& d, KnnGraph& out) {
    out.nbr_idx.resize(static_cast<std::size_t>(d.N) * d.k);
    out.nbr_dist.resize(static_cast<std::size_t>(d.N) * d.k);

    int    idx[SC_MAX_K];     // scratch: this query's neighbour indices
    double sq [SC_MAX_K];     // scratch: this query's squared distances

    for (int q = 0; q < d.N; ++q) {
        sc_knn_one_cell(out.normalized.data(), d.N, d.G, q, d.k, idx, sq);  // shared KNN
        for (int j = 0; j < d.k; ++j) {
            out.nbr_idx [static_cast<std::size_t>(q) * d.k + j] = idx[j];
            out.nbr_dist[static_cast<std::size_t>(q) * d.k + j] =
                static_cast<float>(std::sqrt(sq[j]));   // report true Euclidean distance
        }
    }
}

// ---------------------------------------------------------------------------
// run_cpu: the two-step pipeline, in order. Exposed via reference_cpu.h.
// ---------------------------------------------------------------------------
void run_cpu(const Dataset& d, KnnGraph& out) {
    cpu_normalize(d, out);   // step 1: counts-per-target + log1p
    cpu_knn(d, out);         // step 2: exact brute-force KNN graph
}
