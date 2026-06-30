// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ baseline we trust
// ---------------------------------------------------------------------------
// Project 3.24 : Methylation / Modified-Base Calling
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is written to be
//   OBVIOUSLY correct -- straight loops, no parallelism, no cleverness -- so that
//   when the GPU and CPU agree we believe the GPU. The actual per-job alignment
//   math lives in meth_core.h (banded_align_core), shared verbatim with the GPU,
//   so "agree" here means agree to floating-point tolerance, not by luck.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, meth_core.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cstdio>
#include <fstream>
#include <sstream>

// ---------------------------------------------------------------------------
// load_meth_data: parse the whitespace/line text format produced by
// scripts/make_synthetic.py. The format (see data/README.md) is:
//
//   num_sites num_reads coverage
//   <NUM_KMERS lines>  canon_mean canon_stdv   (canonical pore model, k-mer order)
//   <NUM_KMERS lines>  meth_mean  meth_stdv    (methylated pore model)
//   <num_sites lines>  site_pos truth          (CpG coordinate + 0/1 ground truth)
//   <num_jobs lines>   read_id site_id  b0..b11  e0..e9
//                       (12 reference base codes, then 10 event currents)
//
//   We validate counts so a malformed file aborts the demo loudly rather than
//   silently scoring garbage.
// ---------------------------------------------------------------------------
MethData load_meth_data(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open input file: " + path);

    MethData d;
    if (!(in >> d.num_sites >> d.num_reads >> d.coverage))
        throw std::runtime_error("bad header (expected: num_sites num_reads coverage)");
    if (d.num_sites <= 0 || d.coverage <= 0)
        throw std::runtime_error("num_sites and coverage must be positive");

    // --- canonical pore model: NUM_KMERS rows of (mean, stdv) ---------------
    d.canon.resize(NUM_KMERS);
    for (int k = 0; k < NUM_KMERS; ++k) {
        if (!(in >> d.canon[k].level_mean >> d.canon[k].level_stdv))
            throw std::runtime_error("truncated canonical pore model");
        if (d.canon[k].level_stdv <= 0.0f)
            throw std::runtime_error("canonical stdv must be > 0");
    }
    // --- methylated pore model ---------------------------------------------
    d.meth.resize(NUM_KMERS);
    for (int k = 0; k < NUM_KMERS; ++k) {
        if (!(in >> d.meth[k].level_mean >> d.meth[k].level_stdv))
            throw std::runtime_error("truncated methylated pore model");
        if (d.meth[k].level_stdv <= 0.0f)
            throw std::runtime_error("methylated stdv must be > 0");
    }
    // --- site table: position + ground-truth label -------------------------
    d.site_pos.resize(d.num_sites);
    d.truth.resize(d.num_sites);
    for (int s = 0; s < d.num_sites; ++s) {
        if (!(in >> d.site_pos[s] >> d.truth[s]))
            throw std::runtime_error("truncated site table");
    }
    // --- jobs: one per (read, site) ----------------------------------------
    const int num_jobs = d.num_sites * d.coverage;
    d.jobs.resize(num_jobs);
    for (int j = 0; j < num_jobs; ++j) {
        Job& job = d.jobs[j];
        if (!(in >> job.read_id >> job.site_id))
            throw std::runtime_error("truncated job header");
        for (int b = 0; b < WINDOW_BASES; ++b)
            if (!(in >> job.ref_codes[b]))
                throw std::runtime_error("truncated job reference window");
        for (int e = 0; e < EVENTS_PER_JOB; ++e)
            if (!(in >> job.events[e]))
                throw std::runtime_error("truncated job events");
        if (job.site_id < 0 || job.site_id >= d.num_sites)
            throw std::runtime_error("job site_id out of range");
    }
    return d;
}

// ---------------------------------------------------------------------------
// banded_align_logL: derive the WINDOW_KMERS reference k-mer codes from the job's
//   base-code window, then call the shared banded DP core under one pore model.
//   This thin wrapper is what reference_cpu.h promises; the GPU kernel performs
//   the identical two steps in-thread (kernels.cu), so both return the same logL.
// ---------------------------------------------------------------------------
double banded_align_logL(const Job& job, const PoreModelEntry* model) {
    // Slide a KMER_K window across the WINDOW_BASES base codes to get the
    // WINDOW_KMERS k-mer indices the events will be aligned to.
    int kmer_ids[WINDOW_KMERS];
    for (int j = 0; j < WINDOW_KMERS; ++j)
        kmer_ids[j] = kmer_code(&job.ref_codes[j]);   // 2-bit pack of bases j..j+K-1

    // Run the one true DP (defined in meth_core.h, shared with the GPU).
    return banded_align_core(job.events, kmer_ids, model, EVENTS_PER_JOB, WINDOW_KMERS);
}

// ---------------------------------------------------------------------------
// score_jobs_cpu: per job, score under BOTH models and take the difference.
//   LLR = logL(methylated) - logL(canonical). Positive => the events look more
//   like 5mC; negative => more like canonical C. This is the per-element result
//   the GPU must reproduce. Serial double-precision loop -> deterministic.
// ---------------------------------------------------------------------------
void score_jobs_cpu(const MethData& d, std::vector<float>& llr) {
    const int num_jobs = static_cast<int>(d.jobs.size());
    llr.assign(num_jobs, 0.0f);
    for (int j = 0; j < num_jobs; ++j) {
        const double logL_meth  = banded_align_logL(d.jobs[j], d.meth.data());
        const double logL_canon = banded_align_logL(d.jobs[j], d.canon.data());
        // Store as float: the verification tolerance (main.cu) accounts for the
        // float round-trip; the DECISION (call_sites) uses the float consistently
        // on both sides so the integer call is identical regardless.
        llr[j] = static_cast<float>(logL_meth - logL_canon);
    }
}

// ---------------------------------------------------------------------------
// call_sites: average the LLRs of the `coverage` reads at each site, then make a
//   hard 0/1 call. Jobs are laid out site-major in the synthetic file (all reads
//   of site 0, then site 1, ...), but we DO NOT rely on that: we accumulate by
//   each job's site_id so the aggregation is order-independent and deterministic.
//   The mean is computed in double then stored as float; the call is mean>0.
// ---------------------------------------------------------------------------
void call_sites(const MethData& d, const std::vector<float>& llr,
                std::vector<float>& mean_llr, std::vector<int>& call) {
    mean_llr.assign(d.num_sites, 0.0f);
    call.assign(d.num_sites, 0);

    std::vector<double> sum(d.num_sites, 0.0);   // double accumulator per site
    std::vector<int>    cnt(d.num_sites, 0);     // reads counted per site
    for (int j = 0; j < static_cast<int>(d.jobs.size()); ++j) {
        const int s = d.jobs[j].site_id;
        sum[s] += static_cast<double>(llr[j]);
        cnt[s] += 1;
    }
    for (int s = 0; s < d.num_sites; ++s) {
        const double m = (cnt[s] > 0) ? sum[s] / cnt[s] : 0.0;
        mean_llr[s] = static_cast<float>(m);
        call[s] = (m > 0.0) ? 1 : 0;
    }
}
