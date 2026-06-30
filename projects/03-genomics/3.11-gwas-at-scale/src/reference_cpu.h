// ===========================================================================
// src/reference_cpu.h  --  Trusted serial baseline: types + declarations
// ---------------------------------------------------------------------------
// Project 3.11 : GWAS at Scale
//
// ROLE
//   Declares the plain-C++ data structures and the CPU reference functions that
//   main.cu uses as the GROUND TRUTH for verifying the GPU. The reference does
//   the same two computations the GPU does -- build the genetic relatedness
//   matrix (GRM) and run the per-SNP association scan -- but with simple,
//   readable serial loops. Because both sides call the SAME per-element math in
//   gwas_core.h, the GPU result must match this to (near) machine precision.
//
//   This header is included by BOTH reference_cpu.cpp (the implementation),
//   main.cu (the caller), and kernels.cu (which reuses GenotypeData). It
//   contains no CUDA constructs, so the host compiler is happy.
//
// READ THIS AFTER: gwas_core.h.  READ THIS BEFORE: reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "gwas_core.h"   // gwas::AssocResult and the shared per-element formulas

// ---------------------------------------------------------------------------
// GenotypeData -- the whole problem, loaded from the sample file.
//   The genotype matrix G is N individuals (rows) x M SNPs (columns), stored
//   ROW-MAJOR in `geno` as 8-bit dosages in {0,1,2}. We use signed char (int8)
//   because a dosage needs only 2 bits; for N=500k, M=800k that matrix is
//   ~400 GB, so the compact type matters at scale (here the sample is tiny, but
//   we teach the real layout). `pheno` is the length-N phenotype (e.g. a
//   standardized quantitative trait). `snp_id`/`causal` are demo bookkeeping.
// ---------------------------------------------------------------------------
struct GenotypeData {
    int N = 0;                       // number of individuals (matrix rows)
    int M = 0;                       // number of SNPs        (matrix columns)
    std::vector<signed char> geno;   // [N*M] row-major dosages, each in {0,1,2}
    std::vector<double> pheno;       // [N]   quantitative phenotype y
    std::vector<std::string> snp_id; // [M]   human-readable SNP names (rs-like)
    std::vector<int> causal;         // [M]   1 if this SNP was injected as causal

    // Convenience accessor: dosage of individual i at SNP j (row-major index).
    signed char g(int i, int j) const {
        return geno[static_cast<std::size_t>(i) * M + j];
    }
};

// load_genotypes: parse the committed sample file into a GenotypeData.
//   File format is documented in data/README.md (header "N M", a phenotype
//   row, one row per individual of M dosages, then SNP metadata lines). Throws
//   std::runtime_error on a malformed/missing file so demos fail loudly.
GenotypeData load_genotypes(const std::string& path);

// center_phenotype: subtract the mean from y so the regression intercept drops
//   out (Section B of gwas_core.h assumes a centered y). Returns the centered
//   copy; the original is left untouched.
std::vector<double> center_phenotype(const std::vector<double>& y);

// standardize_columns: turn raw G into the standardized matrix Z (Section A of
//   gwas_core.h), column by column. Output `Z` is [N*M] row-major doubles; also
//   returns per-SNP allele frequency `freq` and HWE scale `sd` for reporting.
//   This is the CPU half of the GRM pipeline; the GPU builds the same Z on the
//   device with a kernel.
void standardize_columns(const GenotypeData& d, std::vector<double>& Z,
                         std::vector<double>& freq, std::vector<double>& sd);

// grm_reference: build the NxN genetic relatedness matrix GRM = (1/M) Z Zᵀ on
//   the CPU with a triple loop. `Z` is the standardized [N*M] matrix from
//   standardize_columns; `grm` is filled [N*N] row-major. O(N^2 M) -- the
//   expensive step the GPU offloads to cuBLAS DGEMM. Used to verify the GPU GRM
//   entry by entry.
void grm_reference(const std::vector<double>& Z, int N, int M,
                   std::vector<double>& grm);

// assoc_reference: run the single-marker regression scan on the CPU.
//   For each SNP j it accumulates the sufficient statistics (Σx², Σxy, Σy²)
//   over individuals and calls gwas::assoc_from_sufficient_stats. `Z` is the
//   standardized genotype matrix; `y_centered` is the mean-centered phenotype.
//   Fills `out` with one gwas::AssocResult per SNP. O(N*M).
void assoc_reference(const std::vector<double>& Z, const std::vector<double>& y_centered,
                     int N, int M, std::vector<gwas::AssocResult>& out);
