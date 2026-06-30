// ===========================================================================
// src/reference_cpu.h  --  Dataset + SV-call types + CPU reference interface
// ---------------------------------------------------------------------------
// Project 3.21 : Structural Variant (SV) Calling
//
// Pure C++ (no CUDA). The per-read math (banded SW, breakpoint refinement,
// binning, genotype) lives in sv.h and is shared with the GPU kernels so both
// sides compute byte-identical results. This header declares:
//   * SvDataset  -- the loaded problem (reference + candidate reads).
//   * SvCall     -- one merged structural-variant call (the output).
//   * load_dataset / encode helpers -- host-only file I/O.
//   * sv_call_cpu -- the trusted serial reference (refine -> histogram -> merge).
//
// kernels.cu reuses SvDataset + SvCall + the histogram-to-calls host helper, so
// the CPU and GPU pipelines differ ONLY in *where* the per-read refinement runs.
//
// READ THIS BEFORE: kernels.cuh, main.cu.   READ AFTER: sv.h.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sv.h"   // SV_FLANK, sv_refine_breakpoint, sv_bin, sv_geno_from_vaf, ...

// ---------------------------------------------------------------------------
// A candidate split read crossing a putative DELETION breakpoint.
//   * raw_guess : the read-aligner's noisy estimate of the LEFT breakpoint
//                 (reference coordinate). The realignment step refines this.
//   * del_len   : the read's estimate of the deleted length (bp). Clustered
//                 alongside the breakpoint so different SVs at nearby positions
//                 stay distinct.
//   * left[]    : the read's SV_FLANK bases ending at its breakpoint (base codes
//                 0..4), the input to banded SW.
// One struct per read; the arrays of these are what we parallelize over.
// ---------------------------------------------------------------------------
struct SvRead {
    int          raw_guess = 0;             // raw left-breakpoint guess (ref coord)
    int          del_len   = 0;             // estimated deletion length (bp)
    signed char  left[SV_FLANK] = {0};      // read flank bases (codes) ending at break
};

// ---------------------------------------------------------------------------
// The whole problem loaded from disk.
//   * ref        : reference sequence as base codes 0..4 (length ref_len).
//   * reads      : the candidate split reads (N of them).
//   * truth_*    : the synthetic GROUND TRUTH (breakpoint + length) baked into the
//                  sample so the demo can report "did we recover the planted SV?".
//                  Present only for synthetic data; -1 means "unknown".
// ---------------------------------------------------------------------------
struct SvDataset {
    std::vector<signed char> ref;           // [ref_len] reference base codes
    int                      ref_len = 0;
    std::vector<SvRead>      reads;          // [N] candidate split reads
    int                      truth_bp  = -1; // planted true breakpoint (or -1)
    int                      truth_len = -1; // planted true deletion length (or -1)
    int N() const { return static_cast<int>(reads.size()); }
};

// ---------------------------------------------------------------------------
// One emitted structural-variant call (a cluster of agreeing reads).
//   * breakpoint : consensus left breakpoint (ref coord) -- the mode of the
//                  refined-breakpoint histogram within the merged window.
//   * del_len    : consensus deleted length (integer vote average).
//   * support    : number of reads supporting this call (the histogram mass).
//   * genotype   : 0=0/0, 1=0/1, 2=1/1 from sv_geno_from_vaf (integer-only).
// Calls are emitted in a DETERMINISTIC order (ascending breakpoint) so stdout is
// reproducible.
// ---------------------------------------------------------------------------
struct SvCall {
    int          breakpoint = 0;
    int          del_len    = 0;
    unsigned int support    = 0;
    int          genotype   = 0;
};

// Base-code helpers (host-only; the kernel receives already-encoded codes).
//   A->0 C->1 G->2 T->3 (a/c/g/t too); anything else -> 4 (N / unknown).
signed char sv_encode_base(char c);
char        sv_decode_base(signed char code);

// Load the sample file (format documented in data/README.md). Throws on error.
SvDataset load_dataset(const std::string& path);

// ---------------------------------------------------------------------------
// histogram_to_calls: shared host post-processing (used by BOTH CPU and GPU).
//   Given the per-bin refined-breakpoint vote histogram (and the parallel
//   deletion-length vote sums), merge votes within +/- SV_MERGE bp into SV calls,
//   compute support, consensus length, and genotype. Deterministic.
//
//   hist        : [ref_len] support count per 1-bp breakpoint bin
//   len_sum     : [ref_len] sum of del_len votes per bin (for consensus length)
//   ref_len     : reference length (histogram size)
//   total_reads : reads spanning the region (denominator for VAF/genotype)
//   min_support : ignore bins below this (noise floor)
//   Returns calls sorted by ascending breakpoint.
// ---------------------------------------------------------------------------
std::vector<SvCall> histogram_to_calls(const std::vector<unsigned int>& hist,
                                       const std::vector<unsigned long long>& len_sum,
                                       int ref_len, unsigned int total_reads,
                                       unsigned int min_support);

// CPU REFERENCE: the trusted serial pipeline. For each read: refine its
// breakpoint by banded SW, vote into the histogram; then merge into calls.
//   d           : the loaded problem
//   min_support : noise floor for emitting a call
//   Fills `hist` (per-bin support; returned for the GPU to compare against) and
//   `len_sum`, and returns the SV calls. The baseline main.cu verifies against.
std::vector<SvCall> sv_call_cpu(const SvDataset& d, unsigned int min_support,
                                std::vector<unsigned int>& hist,
                                std::vector<unsigned long long>& len_sum);
