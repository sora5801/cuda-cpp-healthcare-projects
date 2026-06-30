// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ baseline we trust
// ---------------------------------------------------------------------------
// Project 3.11 : GWAS at Scale
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. Every routine
//   here is written to be OBVIOUSLY correct -- straight serial loops, no
//   parallelism, no cleverness -- so that when the GPU and CPU agree, we believe
//   the GPU. All the per-element arithmetic is delegated to gwas_core.h, the
//   SAME header the kernels use, so agreement can be near-exact.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h
//   for the data types and the function contracts.
//
//   Pipeline mirrored from main.cu:
//     load_genotypes -> center_phenotype -> standardize_columns
//        -> grm_reference        (the O(N^2 M) GRM = (1/M) Z Zᵀ)
//        -> assoc_reference      (the O(N M) per-SNP regression scan)
//
// READ THIS AFTER: reference_cpu.h, gwas_core.h. Compare with kernels.cu (twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cctype>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// load_genotypes -- parse the committed text sample into a GenotypeData.
//
//   File grammar (see data/sample/gwas_sample.txt and data/README.md):
//     line 1            : "<N> <M>"                       (two integers)
//     line 2            : "pheno: y_0 y_1 ... y_{N-1}"    (N phenotype values)
//     next N lines      : "<dosage_0> ... <dosage_{M-1}>" (one individual/row)
//     next M lines      : "snp: <id> <causal_flag>"       (per-SNP metadata)
//   Lines beginning with '#' are comments and are skipped, so the sample file
//   can carry an explanatory header. We parse defensively and throw on anything
//   malformed -- a demo that silently runs on half a matrix is worse than one
//   that stops and tells you why.
// ---------------------------------------------------------------------------
GenotypeData load_genotypes(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open genotype file: " + path);

    GenotypeData d;
    std::string line;

    // Helper lambda: read the next non-comment, non-blank line into `line`.
    // Returns false at end-of-file. Keeps the parser below readable.
    auto next_line = [&](void) -> bool {
        while (std::getline(in, line)) {
            // Trim a leading run of whitespace to test for comments/blank.
            std::size_t s = line.find_first_not_of(" \t\r\n");
            if (s == std::string::npos) continue;            // blank line
            if (line[s] == '#') continue;                    // comment line
            return true;
        }
        return false;
    };

    // ---- header: N and M --------------------------------------------------
    if (!next_line()) throw std::runtime_error("empty genotype file: " + path);
    {
        std::istringstream hs(line);
        if (!(hs >> d.N >> d.M) || d.N <= 0 || d.M <= 0)
            throw std::runtime_error("bad header (expected 'N M'): " + line);
    }

    // ---- phenotype row ----------------------------------------------------
    if (!next_line()) throw std::runtime_error("missing phenotype row");
    {
        std::istringstream ps(line);
        std::string tag;
        ps >> tag;                                    // consume the "pheno:" tag
        d.pheno.resize(static_cast<std::size_t>(d.N));
        for (int i = 0; i < d.N; ++i) {
            if (!(ps >> d.pheno[static_cast<std::size_t>(i)]))
                throw std::runtime_error("phenotype row has fewer than N values");
        }
    }

    // ---- N individual rows of M dosages -----------------------------------
    d.geno.resize(static_cast<std::size_t>(d.N) * d.M);
    for (int i = 0; i < d.N; ++i) {
        if (!next_line())
            throw std::runtime_error("missing genotype row " + std::to_string(i));
        std::istringstream gs(line);
        for (int j = 0; j < d.M; ++j) {
            int val;
            if (!(gs >> val))
                throw std::runtime_error("genotype row " + std::to_string(i)
                                         + " has fewer than M dosages");
            if (val < 0 || val > 2)
                throw std::runtime_error("dosage out of {0,1,2} at row "
                                         + std::to_string(i));
            d.geno[static_cast<std::size_t>(i) * d.M + j] =
                static_cast<signed char>(val);
        }
    }

    // ---- M SNP metadata lines (optional; default names if absent) ---------
    d.snp_id.assign(static_cast<std::size_t>(d.M), std::string());
    d.causal.assign(static_cast<std::size_t>(d.M), 0);
    for (int j = 0; j < d.M; ++j) {
        if (!next_line()) {                           // metadata is optional
            d.snp_id[static_cast<std::size_t>(j)] = "snp" + std::to_string(j);
            continue;
        }
        std::istringstream ms(line);
        std::string tag, id;
        int flag = 0;
        ms >> tag >> id >> flag;                      // "snp: rsXXXX 1"
        d.snp_id[static_cast<std::size_t>(j)] =
            id.empty() ? ("snp" + std::to_string(j)) : id;
        d.causal[static_cast<std::size_t>(j)] = flag ? 1 : 0;
    }

    return d;
}

// ---------------------------------------------------------------------------
// center_phenotype -- subtract the mean so the regression intercept vanishes.
//   With a centered y, the OLS slope reduces to Σxy/Σx² (Section B of
//   gwas_core.h). O(N).
// ---------------------------------------------------------------------------
std::vector<double> center_phenotype(const std::vector<double>& y) {
    double mean = 0.0;
    for (double v : y) mean += v;
    mean /= static_cast<double>(y.size());
    std::vector<double> out(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) out[i] = y[i] - mean;
    return out;
}

// ---------------------------------------------------------------------------
// standardize_columns -- raw G -> standardized Z, one SNP column at a time.
//   For each SNP j:
//     1. sum its dosages over individuals -> allele frequency p (gwas::allele_freq)
//     2. HWE scale sd = sqrt(2 p (1-p))            (gwas::hwe_sd)
//     3. z_ij = (g_ij - 2p) / sd                   (gwas::standardize)
//   Output Z is [N*M] row-major doubles; freq[j], sd[j] are returned for the
//   report. O(N*M). This is exactly what the GPU standardize kernel computes.
// ---------------------------------------------------------------------------
void standardize_columns(const GenotypeData& d, std::vector<double>& Z,
                         std::vector<double>& freq, std::vector<double>& sd) {
    Z.assign(static_cast<std::size_t>(d.N) * d.M, 0.0);
    freq.assign(static_cast<std::size_t>(d.M), 0.0);
    sd.assign(static_cast<std::size_t>(d.M), 0.0);

    for (int j = 0; j < d.M; ++j) {
        // Pass 1: column sum of dosages -> minor-allele frequency p.
        double sum = 0.0;
        for (int i = 0; i < d.N; ++i) sum += static_cast<double>(d.g(i, j));
        double p   = gwas::allele_freq(sum, d.N);
        double sdj = gwas::hwe_sd(p);
        freq[static_cast<std::size_t>(j)] = p;
        sd[static_cast<std::size_t>(j)]   = sdj;

        // Pass 2: write the standardized z-score for every individual.
        for (int i = 0; i < d.N; ++i) {
            Z[static_cast<std::size_t>(i) * d.M + j] =
                gwas::standardize(static_cast<int>(d.g(i, j)), p, sdj);
        }
    }
}

// ---------------------------------------------------------------------------
// grm_reference -- the genetic relatedness matrix GRM = (1/M) Z Zᵀ.
//   GRM[a][b] = (1/M) Σ_j Z[a][j] * Z[b][j]  -- the standardized-genotype dot
//   product of individuals a and b, averaged over SNPs. Diagonal ~ 1 (self
//   relatedness), off-diagonal ~ 0 for unrelated people, larger for relatives.
//   Triple loop -> O(N^2 M); this is the cost cuBLAS DGEMM erases on the GPU.
//   We exploit symmetry (GRM is symmetric) by computing b <= a and mirroring.
// ---------------------------------------------------------------------------
void grm_reference(const std::vector<double>& Z, int N, int M,
                   std::vector<double>& grm) {
    grm.assign(static_cast<std::size_t>(N) * N, 0.0);
    const double inv_m = 1.0 / static_cast<double>(M);
    for (int a = 0; a < N; ++a) {
        for (int b = 0; b <= a; ++b) {
            double acc = 0.0;
            // Dot product of rows a and b of Z (each row is one individual).
            for (int j = 0; j < M; ++j) {
                acc += Z[static_cast<std::size_t>(a) * M + j]
                     * Z[static_cast<std::size_t>(b) * M + j];
            }
            acc *= inv_m;
            grm[static_cast<std::size_t>(a) * N + b] = acc;   // lower triangle
            grm[static_cast<std::size_t>(b) * N + a] = acc;   // mirror (upper)
        }
    }
}

// ---------------------------------------------------------------------------
// assoc_reference -- the single-marker regression scan, one SNP at a time.
//   For SNP j, accumulate over individuals:
//       sxx = Σ z_ij^2 ,  sxy = Σ z_ij y_i ,  syy = Σ y_i^2
//   then hand (sxx, sxy, syy, N) to gwas::assoc_from_sufficient_stats, which
//   produces beta, se, t, chi2, -log10(p). syy is the same for every SNP (it
//   depends only on y), but we recompute it per SNP for clarity and to mirror
//   the kernel's per-thread structure exactly. O(N*M).
// ---------------------------------------------------------------------------
void assoc_reference(const std::vector<double>& Z, const std::vector<double>& y_centered,
                     int N, int M, std::vector<gwas::AssocResult>& out) {
    out.assign(static_cast<std::size_t>(M), gwas::AssocResult{});
    for (int j = 0; j < M; ++j) {
        double sxx = 0.0, sxy = 0.0, syy = 0.0;
        for (int i = 0; i < N; ++i) {
            double x = Z[static_cast<std::size_t>(i) * M + j];   // standardized genotype
            double y = y_centered[static_cast<std::size_t>(i)];  // centered phenotype
            sxx += x * x;
            sxy += x * y;
            syy += y * y;
        }
        out[static_cast<std::size_t>(j)] =
            gwas::assoc_from_sufficient_stats(sxx, sxy, syy, N);
    }
}
