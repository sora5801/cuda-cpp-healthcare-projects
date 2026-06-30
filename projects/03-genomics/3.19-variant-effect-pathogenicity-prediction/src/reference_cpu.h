// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for variant-effect scoring
// ---------------------------------------------------------------------------
// Project 3.19 : Variant Effect / Pathogenicity Prediction
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (the variant set, the file loader, the
//   fixed-model initialiser) and the CPU reference prototype live here. The GPU
//   side (kernels.cuh) also includes this header to reuse the VariantSet type --
//   nothing CUDA-specific leaks in either direction. The per-variant *math* is
//   in vep_model.h (the __host__ __device__ core), included by both sides.
//
// THE PROBLEM  (see ../THEORY.md for the full derivation)
//   A genetic variant (SNV) flips one DNA base at one position. To predict
//   whether it is pathogenic, modern tools score the surrounding sequence
//   CONTEXT with a deep network and take the difference between the alternate
//   and reference alleles -- "in-silico mutagenesis". We model exactly that
//   structure with a tiny fixed CNN (vep_model.h): for each variant we hold a
//   reference window and an alternate window (identical except at the centre
//   base) and the predicted effect is score(alt) - score(ref).
//
//   Each variant is INDEPENDENT, so this is N independent forward-pass pairs ->
//   perfect data parallelism (one GPU thread per variant, in kernels.cu). This
//   is the same "score one model against N items" pattern as project 1.12.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh. Read vep_model.h first.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vep_model.h"   // VEP_WINDOW, VepModel (pure C++/HD, safe in .cpp & .cu)

// ---------------------------------------------------------------------------
// VariantSet: a batch of variants to score.
//   n        : number of variants.
//   ref      : [n * VEP_WINDOW] base codes (A=0,C=1,G=2,T=3), ROW-MAJOR, the
//              REFERENCE context window for each variant.
//   alt      : [n * VEP_WINDOW] base codes, the ALTERNATE window. It is identical
//              to ref except at the centre position VEP_CENTER (the variant base).
//   ref_base : [n] the reference base code at the centre (for the report).
//   alt_base : [n] the alternate base code at the centre (for the report).
//   pos      : [n] a synthetic 1-based genomic coordinate, purely for labelling
//              the output rows (so the demo prints a stable, readable id).
// We store base CODES (int8) rather than a materialised 4xL one-hot matrix: the
// one-hot column has a single 1, so the conv inner loop just indexes the weight
// by the hot channel (see vep_score_window). This is 4x less memory to ship to
// the GPU and exactly the layout the kernel consumes.
// ---------------------------------------------------------------------------
struct VariantSet {
    int n = 0;
    std::vector<int8_t> ref;       // [n * VEP_WINDOW]
    std::vector<int8_t> alt;       // [n * VEP_WINDOW]
    std::vector<int8_t> ref_base;  // [n]
    std::vector<int8_t> alt_base;  // [n]
    std::vector<int>    pos;       // [n]  (synthetic coordinate, for labels)
};

// Map an ASCII nucleotide letter to its base code, or -1 if not A/C/G/T (case
// insensitive). Defined here (inline) because both the loader and the synthetic
// fallback want it. 'N' or any other symbol returns -1 so the loader can reject.
inline int8_t base_code(char ch) {
    switch (ch) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:            return -1;
    }
}

// Inverse map: base code [0,3] -> uppercase letter (for printing). Out-of-range
// returns 'N' so a corrupt code is visible rather than silently wrong.
inline char base_char(int8_t code) {
    static const char* L = "ACGT";
    return (code >= 0 && code < 4) ? L[code] : 'N';
}

// ---------------------------------------------------------------------------
// load_variants: parse the tiny text dataset (format in data/README.md):
//   line 1 : "<n> <VEP_WINDOW>"
//   next n : one variant per line, as
//              <pos> <ref_letter> <alt_letter> <window_of_VEP_WINDOW_letters>
//            where the window is the REFERENCE context (the variant base sits at
//            the centre, index VEP_WINDOW/2, and must equal <ref_letter>). The
//            loader builds the ALT window by copying ref and flipping the centre.
// Throws std::runtime_error on a missing file, a width mismatch, a bad letter,
// or a centre base that disagrees with the stated reference allele.
// ---------------------------------------------------------------------------
VariantSet load_variants(const std::string& path);

// ---------------------------------------------------------------------------
// init_model: fill a VepModel with the FIXED, SYNTHETIC weights this project
// ships. The weights are deterministic (seeded, not random at runtime) and are
// hand-engineered so the toy network reacts to a couple of planted "deleterious"
// 5-mer motifs -- enough to make the demo's ranking meaningful and reproducible.
// They are NOT trained on real genomics and carry NO clinical meaning (sec 8).
// Defined in reference_cpu.cpp so CPU and GPU upload byte-identical weights.
// ---------------------------------------------------------------------------
void init_model(VepModel& m);

// ---------------------------------------------------------------------------
// score_variants_cpu: the trusted serial baseline. For each variant i, fill
// effect[i] with vep_variant_effect(model, ref_i, alt_i) -- the delta score the
// GPU kernel is verified against (and the timing baseline that makes the
// speed-up legible). effect is resized to vs.n.
// ---------------------------------------------------------------------------
void score_variants_cpu(const VepModel& m, const VariantSet& vs,
                        std::vector<double>& effect);
