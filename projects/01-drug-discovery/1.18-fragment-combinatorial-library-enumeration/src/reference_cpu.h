// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for combinatorial enum.
// ---------------------------------------------------------------------------
// Project 1.18 : Fragment / Combinatorial Library Enumeration
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (how building-block synthons are
//   stored, the file loader) and the CPU-reference prototypes live here. The GPU
//   side (kernels.cuh) also includes this header to reuse the SynthonLibrary
//   type and the slot sizes -- nothing CUDA-specific leaks in either direction.
//   The per-product MATH is one level down in product_core.h (shared HD core).
//
// THE PROBLEM (see ../THEORY.md for the full derivation)
//   We are given, for each of N_SLOTS reactant slots, a list of building blocks
//   ("synthons"). Each synthon carries N_DESC additive descriptor contributions
//   (MW, cLogP, TPSA, HBD, HBA -- see product_core.h). The combinatorial library
//   is the Cartesian product of the slots: choosing one synthon per slot yields
//   one product. There are
//       N = sizes[0] * sizes[1] * ... * sizes[N_SLOTS-1]
//   products. For every product we sum its synthons' descriptors and test the
//   Lipinski + Veber drug-likeness filter. We report (1) how many products PASS
//   and (2) the first few passing product indices -- both fully deterministic.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh. Read product_core.h first
// for the per-product math this file orchestrates.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "product_core.h"   // N_SLOTS, N_DESC, descriptor math (pure C++ safe)

// ---------------------------------------------------------------------------
// SynthonLibrary: the loaded building-block catalog, one block of synthons per
// reactant slot.
//   sizes[k]   : number of synthons available in slot k.
//   desc[k]    : a FLAT row-major array of sizes[k] * N_DESC doubles; synthon j
//                of slot k occupies desc[k][j*N_DESC .. +N_DESC).
//   name[k]    : human labels for the synthons (used only for reporting).
//   We keep one std::vector per slot (rather than one big jagged buffer) because
//   slots have different sizes; the GPU side flattens them into a single device
//   buffer with per-slot offsets (see kernels.cu).
// ---------------------------------------------------------------------------
struct SynthonLibrary {
    int sizes[N_SLOTS] = {0, 0, 0};            // synthons per slot
    std::vector<double> desc[N_SLOTS];          // [k] = sizes[k]*N_DESC, row-major
    std::vector<std::string> name[N_SLOTS];     // [k] = sizes[k] labels

    // Total number of products N = product of the slot sizes. Returned as int64
    // because even a few hundred synthons per slot can exceed 2^31 products --
    // a concrete reminder of why this space is "billions" in the real world.
    int64_t num_products() const {
        int64_t n = 1;
        for (int k = 0; k < N_SLOTS; ++k) n *= static_cast<int64_t>(sizes[k]);
        return n;
    }
};

// How many passing product indices to report (a small, deterministic preview).
constexpr int FIRST_K = 8;

// Fixed-point scale for the MW reduction: store summed MW in MILLI-g/mol (x1000)
// as an integer so the sum is EXACT and order-independent -- integer addition is
// associative, so the GPU's atomic accumulation matches the CPU bit-for-bit
// (PATTERNS.md sec.3 rule 2). 1000 keeps full mg precision and stays far inside
// int64 for any teaching-scale library.
constexpr int64_t MW_FIXED_SCALE = 1000;

// The deterministic outcome of an enumeration, returned by both CPU and GPU so
// main.cu can compare them field by field.
//   n_pass            : how many of the N products pass the drug-likeness filter.
//   first_pass        : the flat indices of the first FIRST_K passing products,
//                       in ascending product-index order (the demo prints these).
//   sum_mw_pass_milli : SUM of MW over passing products, in milli-g/mol (fixed
//                       point) so the reduction is exactly reproducible.
struct EnumResult {
    int64_t n_pass = 0;                    // count of passing products
    std::vector<int64_t> first_pass;       // up to FIRST_K passing product indices
    int64_t sum_mw_pass_milli = 0;         // sum of MW(passing) in milli-g/mol
};

// Load a synthon library from the text format documented in data/README.md.
//   Throws std::runtime_error on a missing file or a malformed record.
SynthonLibrary load_synthons(const std::string& path);

// CPU reference: enumerate ALL products, apply the filter, and fill `out`.
//   This is the trusted serial baseline the GPU result is verified against (and
//   the timing baseline that makes the speed-up legible). It visits products in
//   flat-index order 0,1,2,...,N-1 so its "first K passing" list is canonical.
void enumerate_cpu(const SynthonLibrary& lib, EnumResult& out);

// Decode a flat product index into a human-readable "A.i + B.j + C.k" label
// using the synthon names. Pure host; used only for the stdout report.
std::string product_label(const SynthonLibrary& lib, int64_t p);
