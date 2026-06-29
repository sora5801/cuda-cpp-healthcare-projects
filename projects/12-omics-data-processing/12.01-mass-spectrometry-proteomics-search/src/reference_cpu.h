// ===========================================================================
// src/reference_cpu.h  --  Spectral data model + CPU cosine-search reference
// ---------------------------------------------------------------------------
// Project 12.01 : Mass-Spectrometry Proteomics Search
//
// WHAT THIS PROJECT COMPUTES
//   Database peptide search: score ONE observed MS/MS spectrum (the "query")
//   against a LIBRARY of theoretical peptide spectra, and return the best
//   matches. Each spectrum is binned to a fixed-length intensity vector; the
//   match score is the NORMALIZED DOT PRODUCT (cosine similarity / spectral
//   contrast angle) between the query and each library spectrum.
//
// WHY A GPU
//   A real search scores ~10^5 observed spectra against ~10^6 theoretical
//   peptides -> 10^11 comparisons, the most time-consuming step in proteomics.
//   Each query-vs-library score is independent: one GPU thread per library
//   spectrum, the query broadcast from constant memory (cf. project 1.12).
//
//   Pure C++ header (no CUDA). kernels.cu reuses SpectralData.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// A loaded search problem: one query spectrum + N library spectra, each binned
// to `bins` intensity values. `target` is the library index the synthetic query
// was derived from (for reporting; -1 if unknown).
struct SpectralData {
    int N = 0;                  // number of library spectra
    int bins = 0;               // intensity bins per spectrum
    int target = -1;            // known best match (synthetic ground truth), or -1
    std::vector<float> query;   // [bins]
    std::vector<float> lib;     // [N * bins], row-major
};

// Load from the text format (data/README.md):
//   header: "N bins target"  then 1 query row of `bins` floats, then N library rows.
SpectralData load_spectra(const std::string& path);

// L2 norms (in double) of the query and of every library spectrum -- precomputed
// once so cosine scoring is a single dot product per comparison.
void compute_norms(const SpectralData& s, double& qnorm, std::vector<double>& libnorm);

// CPU reference: cosine similarity of the query against each library spectrum.
//   scores[i] = dot(query, lib_i) / (||query|| * ||lib_i||)   in [0,1] for >=0 data.
// The trusted baseline the GPU kernel is checked against.
void cosine_cpu(const SpectralData& s, double qnorm, const std::vector<double>& libnorm,
                std::vector<float>& scores);
