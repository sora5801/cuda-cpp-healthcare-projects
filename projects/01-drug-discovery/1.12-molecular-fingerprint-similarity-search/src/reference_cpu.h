// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for Tanimoto search
// ---------------------------------------------------------------------------
// Project 1.12 : Molecular Fingerprint Similarity Search
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (fingerprint width, the FingerprintSet
//   container, the file loader) and the CPU reference prototype live here. The
//   GPU side (kernels.cuh) also includes this header to reuse FP_WORDS and the
//   FingerprintSet type -- nothing CUDA-specific leaks in either direction.
//
// THE PROBLEM (see ../THEORY.md for the full derivation)
//   A molecular fingerprint is a fixed-length BIT STRING (here 2048 bits): bit k
//   is 1 if chemical substructure k is present. The Tanimoto (a.k.a. Jaccard)
//   similarity of two fingerprints A, B is
//       T(A,B) = popcount(A & B) / popcount(A | B)
//   i.e. (shared bits) / (bits present in either). We compare ONE query against
//   N library fingerprints; every comparison is independent -> perfect data
//   parallelism (one GPU thread per library molecule, in kernels.cu).
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Fingerprint width. 2048 bits = 32 x 64-bit words is the standard ECFP4 size
// used by RDKit/ChEMBL. It is a COMPILE-TIME constant so the inner loop can be
// fully unrolled and the query can live in fixed-size GPU constant memory.
constexpr int FP_WORDS = 32;             // 64-bit words per fingerprint
constexpr int FP_BITS  = FP_WORDS * 64;  // = 2048 bits

// A loaded dataset: one query fingerprint + n library fingerprints.
//   query : FP_WORDS words.
//   lib   : n * FP_WORDS words, ROW-MAJOR (molecule i occupies
//           lib[i*FP_WORDS .. i*FP_WORDS + FP_WORDS-1]).
struct FingerprintSet {
    int n = 0;                       // number of library fingerprints
    std::vector<uint64_t> query;     // [FP_WORDS]
    std::vector<uint64_t> lib;       // [n * FP_WORDS], row-major
};

// Load a dataset from the text format documented in data/README.md:
//   line 1:  "<n> <FP_WORDS>"
//   line 2:  the query as FP_WORDS space-separated 16-hex-digit words
//   next n:  each library fingerprint, same encoding
// Throws std::runtime_error on a missing file or a width mismatch.
FingerprintSet load_fingerprints(const std::string& path);

// CPU reference: fill out[i] with the Tanimoto similarity of the query against
// library molecule i. This is the trusted, obviously-correct baseline the GPU
// result is checked against (and the timing baseline that makes the speed-up
// legible). out is resized to fps.n.
void tanimoto_cpu(const FingerprintSet& fps, std::vector<float>& out);
