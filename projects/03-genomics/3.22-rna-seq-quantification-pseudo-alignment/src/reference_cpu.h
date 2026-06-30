// ===========================================================================
// src/reference_cpu.h  --  EcDataset + shared EM helpers + CPU reference
// ---------------------------------------------------------------------------
// Project 3.22 : RNA-seq Quantification / Pseudo-alignment
//
// Pure C++ (no CUDA). The per-ec E-step math lives in pseudoalign.h. The pieces
// declared here are reused by BOTH the CPU reference (reference_cpu.cpp) and the
// GPU wrapper (kernels.cu):
//   * EcDataset            -- the parsed equivalence-class problem.
//   * load_dataset         -- read the text format in data/README.md.
//   * init_rho_uniform     -- the deterministic EM starting point.
//   * counts_to_rho        -- the M-step's renormalise (fixed-point sums -> rho).
//   * em_cpu               -- the trusted serial EM reference.
//   * tpm_from_rho         -- convert abundances to the reported TPM unit.
// Sharing counts_to_rho/init means the CPU and GPU take identical update steps,
// so their final abundances agree exactly.
//
// READ THIS AFTER: pseudoalign.h. READ THIS BEFORE: kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pseudoalign.h"   // PSA_* helpers + PSA_MAX_EC_SIZE

// ---------------------------------------------------------------------------
// EcDataset: a pseudo-alignment problem in the "equivalence class" form.
//
//   T  = number of transcripts in the reference.
//   M  = number of equivalence classes (distinct read-compatibility patterns).
//   eff_len[t]  = effective length of transcript t (length units).
//   ec_count[e] = number of reads that fell into ec e.
//   members are stored in CSR (compressed sparse row) layout so an ec's member
//   list is contiguous and the GPU can index it without a 2-D ragged array:
//       ec_offset has size M+1; ec e's members are
//           ec_members[ ec_offset[e] .. ec_offset[e+1] ).
//   total_reads = sum of ec_count (the conserved quantity the EM redistributes).
// ---------------------------------------------------------------------------
struct EcDataset {
    int T = 0;                              // number of transcripts
    int M = 0;                              // number of equivalence classes
    std::vector<double>       eff_len;      // [T] effective lengths
    std::vector<double>       ec_count;     // [M] reads per ec
    std::vector<std::int32_t> ec_offset;    // [M+1] CSR row offsets into ec_members
    std::vector<std::int32_t> ec_members;   // [nnz] flattened member transcript ids
    double total_reads = 0.0;               // sum of ec_count

    // Optional ground-truth abundances (fraction of reads per transcript), parsed
    // from the sample so the demo can report HOW WELL the EM recovered the truth.
    // Empty if the file provides none.
    std::vector<double> truth_rho;          // [T] or empty
};

// Parse the text format documented in data/README.md. Throws std::runtime_error
// on a malformed file or an ec larger than PSA_MAX_EC_SIZE.
EcDataset load_dataset(const std::string& path);

// Deterministic EM start: every transcript equally abundant, rho[t] = 1/T.
void init_rho_uniform(const EcDataset& d, std::vector<double>& rho);

// M-step finish, shared by CPU and GPU: given per-transcript fixed-point read
// sums (the M-step output) and the dataset, produce the next rho by normalising
// the read counts to sum to 1. Writing this once guarantees the CPU and GPU
// normalise identically.
void counts_to_rho(const EcDataset& d, const std::vector<unsigned long long>& fixed_counts,
                   std::vector<double>& rho);

// Convert abundances rho (fraction of reads) to TPM (transcripts per million),
// the standard reported unit: TPM_t proportional to rho[t]/eff_len[t], scaled so
// the TPMs sum to 1e6. TPM is comparable across samples; rho is not.
void tpm_from_rho(const EcDataset& d, const std::vector<double>& rho,
                  std::vector<double>& tpm);

// CPU reference: run `iters` EM iterations from the uniform start. Fills `rho`
// (final abundances, length T) and `est_counts` (final expected read counts per
// transcript, length T). Returns the final per-iteration change (L1 distance
// between the last two rho vectors) as a convergence witness. The trusted baseline.
double em_cpu(const EcDataset& d, int iters,
              std::vector<double>& rho, std::vector<double>& est_counts);
