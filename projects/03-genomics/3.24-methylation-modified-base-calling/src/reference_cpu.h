// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU-reference API for methylation calling
// ---------------------------------------------------------------------------
// Project 3.24 : Methylation / Modified-Base Calling
//
// ROLE IN THE PROJECT
//   Declares (a) the plain-C++ data structures that hold one demo instance -- a
//   reference sequence, a canonical + a methylated pore model, and a batch of
//   nanopore reads, each carrying observed events for the CpG sites it covers --
//   and (b) the CPU reference functions that base-call methylation by adaptive
//   banded event-alignment DP + log-likelihood-ratio (LLR) scoring.
//
//   This header is pure C++ (no CUDA): it is included by reference_cpu.cpp (host
//   compiler) AND by kernels.cuh / main.cu (nvcc, which also accepts plain C++).
//   The per-event physics lives in meth_core.h, shared by CPU and GPU verbatim.
//
// THE COMPUTATION IN ONE BREATH
//   For every (read, CpG-site) "job": take the events that read observed over the
//   site's local reference window; run a banded DP that threads those events onto
//   the reference k-mers TWICE -- once with the CANONICAL pore model, once with
//   the METHYLATED model -- and take the difference of the two best-path
//   log-likelihoods. That difference is the per-job LLR. Average the LLRs of all
//   reads covering a site to get the site's methylation evidence; LLR > 0 => call
//   5mC. Each job is independent -> perfect GPU batch (see kernels.cuh).
//
// READ THIS AFTER: meth_core.h.   READ THIS BEFORE: reference_cpu.cpp, kernels.cuh.
// ===========================================================================
#pragma once

#include <string>
#include <vector>
#include <stdexcept>

#include "meth_core.h"   // PoreModelEntry, KMER_K, BAND_WIDTH, emission physics

// ---------------------------------------------------------------------------
// Per-job geometry. Each (read, site) job aligns EVENTS_PER_JOB observed event
// currents against a reference window of WINDOW_KMERS k-mers centered on the CpG.
// We FIX these sizes so the band geometry is uniform across jobs -- this makes the
// GPU kernel a clean "one thread per job, fixed-size local DP" (no ragged loops),
// and keeps the committed sample small and readable. THEORY.md "real world"
// explains how production tools handle variable-length events with an adaptive
// band; here a fixed band is the simplest correct teaching version (CLAUDE.md §13).
// ---------------------------------------------------------------------------
static constexpr int WINDOW_BASES   = 12;                        // reference bases per window
static constexpr int WINDOW_KMERS   = WINDOW_BASES - KMER_K + 1; // k-mers in the window (= 10)
static constexpr int EVENTS_PER_JOB = WINDOW_KMERS;              // observed events per job (1 per k-mer)

// One methylation-calling job: a single read's events over a single CpG site.
struct Job {
    int read_id;     // which read this came from (for reporting)
    int site_id;     // which CpG site this covers (index into MethData::site_pos)
    // The reference window's base codes (2-bit A/C/G/T), WINDOW_BASES of them,
    // centered so the CpG 'C' sits at a known offset. Stored as int for kmer_code.
    int   ref_codes[WINDOW_BASES];
    // The observed event mean currents for this read over the window.
    float events[EVENTS_PER_JOB];
};

// One full demo instance, parsed from data/sample/*.txt (see data/README.md for
// the exact on-disk format). All arrays are host-side std::vectors.
struct MethData {
    int num_sites = 0;                  // number of CpG sites in the reference
    int num_reads = 0;                  // number of reads
    int coverage  = 0;                  // reads per site (uniform in the synthetic set)
    std::vector<int> site_pos;          // [num_sites] reference coordinate of each CpG
    std::vector<int> truth;             // [num_sites] synthetic ground truth: 1=5mC, 0=canonical
    std::vector<PoreModelEntry> canon;  // [NUM_KMERS] canonical (unmodified C) pore model
    std::vector<PoreModelEntry> meth;   // [NUM_KMERS] methylated (5mC) pore model
    std::vector<Job> jobs;              // [num_jobs] one per (read, site); num_jobs = num_sites*coverage
};

// ---------------------------------------------------------------------------
// load_meth_data: parse a sample file into a MethData. Throws std::runtime_error
//   on any malformed input so demos fail loudly. Format is documented in
//   data/README.md and produced by scripts/make_synthetic.py.
// ---------------------------------------------------------------------------
MethData load_meth_data(const std::string& path);

// ---------------------------------------------------------------------------
// banded_align_logL: the CPU adaptive-banded event-alignment DP for ONE job
//   under ONE pore model. Returns the log-likelihood of the best alignment path
//   that threads the job's EVENTS_PER_JOB events onto the WINDOW_KMERS reference
//   k-mers, allowing match/insert/delete moves within a band of half-width
//   BAND_WIDTH. This is the exact function the GPU kernel mirrors (same recurrence,
//   same emission physics from meth_core.h) so CPU and GPU agree to tolerance.
//     job   : the (read, site) event block
//     model : NUM_KMERS pore-model rows (canonical OR methylated)
//   Complexity: O(EVENTS_PER_JOB * (2*BAND_WIDTH+1)) per call.
// ---------------------------------------------------------------------------
double banded_align_logL(const Job& job, const PoreModelEntry* model);

// ---------------------------------------------------------------------------
// score_jobs_cpu: run banded_align_logL twice per job (methylated minus
//   canonical) to fill llr[j] = LLR of job j. The headline per-element result we
//   verify the GPU against.
//     d        : the loaded instance
//     llr      : OUT [num_jobs] per-job log-likelihood ratios
// ---------------------------------------------------------------------------
void score_jobs_cpu(const MethData& d, std::vector<float>& llr);

// ---------------------------------------------------------------------------
// call_sites: aggregate per-job LLRs into a per-site decision. mean_llr[s] is the
//   average LLR over the `coverage` reads at site s; call[s] = 1 if mean_llr[s] > 0
//   (more methylated-model evidence than canonical), else 0. Deterministic integer
//   decision, identical on any machine. Shared by CPU and GPU result reporting.
//     d        : the instance (for num_sites, coverage)
//     llr      : [num_jobs] per-job LLRs (CPU or GPU; they agree)
//     mean_llr : OUT [num_sites] mean LLR per site
//     call     : OUT [num_sites] 0/1 methylation call per site
// ---------------------------------------------------------------------------
void call_sites(const MethData& d, const std::vector<float>& llr,
                std::vector<float>& mean_llr, std::vector<int>& call);
