// ===========================================================================
// src/reference_cpu.cpp  --  FASTA loader, ragged->padded batching, and the
//                            plain-C++ aggregation-scan reference we trust
// ---------------------------------------------------------------------------
// Project 1.34 : Amyloid / Aggregation Propensity Prediction
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is written to be
//   OBVIOUSLY correct -- readable loops, no parallelism, no cleverness -- so
//   that when the GPU and CPU agree we believe the GPU. The per-residue math is
//   NOT duplicated here: scan_dataset_cpu calls propensity_of_code() and
//   windowed_mean() from propensity.h, the exact same functions the kernel uses
//   (PATTERNS.md §2), so any disagreement is a real bug, not a formula drift.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, propensity.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"
#include "propensity.h"     // AA_PROPENSITY, propensity_of_code, windowed_mean

#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// code_of_char: one-letter amino-acid symbol -> index into AA_PROPENSITY.
//   The 20 standard residues map to 0..19 in the AA_ORDER documented in
//   propensity.h; everything else (lowercase, 'X', 'B', 'Z', gaps, digits) maps
//   to 20, the "other" bucket whose propensity is 0. A direct switch is the
//   clearest, fastest mapping and makes the encoding auditable at a glance.
// ---------------------------------------------------------------------------
int code_of_char(char c) {
    switch (c) {
        case 'A': return 0;   case 'R': return 1;   case 'N': return 2;
        case 'D': return 3;   case 'C': return 4;   case 'Q': return 5;
        case 'E': return 6;   case 'G': return 7;   case 'H': return 8;
        case 'I': return 9;   case 'L': return 10;  case 'K': return 11;
        case 'M': return 12;  case 'F': return 13;  case 'P': return 14;
        case 'S': return 15;  case 'T': return 16;  case 'W': return 17;
        case 'Y': return 18;  case 'V': return 19;
        default:  return 20;  // any non-standard symbol -> "other" (propensity 0)
    }
}

// ---------------------------------------------------------------------------
// load_dataset: parse a FASTA-style text file into a Dataset.
//   FASTA rule: a line beginning with '>' starts a new record (the rest of that
//   line is the name); all following non-'>' lines are sequence characters,
//   concatenated. We skip whitespace and ignore blank lines. Each residue char
//   is encoded to an index immediately (code_of_char), so downstream code never
//   re-parses text. Throws if the file cannot be opened or contains no protein.
// ---------------------------------------------------------------------------
Dataset load_dataset(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open dataset file: " + path);

    Dataset ds;
    Protein cur;                 // the protein currently being accumulated
    bool have_cur = false;       // true once we've seen a '>' header

    std::string line;
    while (std::getline(in, line)) {
        // Strip a trailing '\r' so files with Windows CRLF endings parse on Linux.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        if (line[0] == '>') {
            // A new header: flush the protein we were building (if any)...
            if (have_cur) { cur.len = static_cast<int>(cur.codes.size());
                            ds.proteins.push_back(std::move(cur)); }
            cur = Protein{};                 // ...and start a fresh one.
            cur.name = line.substr(1);       // header text after '>'
            have_cur = true;
        } else if (have_cur) {
            // Sequence line: encode each residue character to an index 0..20.
            for (char c : line) {
                if (c == ' ' || c == '\t') continue;   // tolerate stray spaces
                cur.codes.push_back(code_of_char(c));
            }
        }
        // (Lines before the first '>' are ignored: not valid FASTA.)
    }
    // Flush the final protein (no trailing '>' to trigger the flush above).
    if (have_cur) { cur.len = static_cast<int>(cur.codes.size());
                    ds.proteins.push_back(std::move(cur)); }

    if (ds.proteins.empty())
        throw std::runtime_error("no proteins parsed from " + path);

    build_flat_layout(ds);   // pack into the flat padded matrix for the GPU
    return ds;
}

// ---------------------------------------------------------------------------
// build_flat_layout: pack the ragged proteins into a flat [num*stride] matrix.
//   stride = the longest sequence length, so EVERY row fits. Row p holds its
//   real `lengths[p]` codes followed by PAD_CODE padding. This is the layout the
//   kernel reads: one block per row, threads walk a row with coalesced accesses.
//   We do NOT round stride up to a multiple of the block here for simplicity;
//   the kernel guards residue indices against `lengths[p]` instead.
// ---------------------------------------------------------------------------
void build_flat_layout(Dataset& ds) {
    ds.num = static_cast<int>(ds.proteins.size());
    ds.max_len = 0;
    for (const Protein& p : ds.proteins)
        if (p.len > ds.max_len) ds.max_len = p.len;
    ds.stride = ds.max_len;                       // padded row width

    ds.lengths.assign(ds.num, 0);
    ds.flat_codes.assign(static_cast<std::size_t>(ds.num) * ds.stride, PAD_CODE);
    for (int p = 0; p < ds.num; ++p) {
        const Protein& prot = ds.proteins[p];
        ds.lengths[p] = prot.len;
        const std::size_t base = static_cast<std::size_t>(p) * ds.stride;
        for (int i = 0; i < prot.len; ++i)
            ds.flat_codes[base + i] = prot.codes[i];
        // positions [prot.len, stride) remain PAD_CODE from the assign() above.
    }
}

// ---------------------------------------------------------------------------
// scan_dataset_cpu: the serial reference. For each protein:
//   (a) look up the intrinsic propensity of every residue (propensity.h),
//   (b) compute the centered windowed mean at every residue (propensity.h),
//   (c) reduce the smoothed profile to an AggResult (peak, prone count, APR).
//   Steps (a),(b) call the SAME functions the kernel calls, so CPU==GPU to
//   float epsilon. Complexity: O(sum_p len_p * W) -- linear in residues for a
//   fixed window. The deliberate per-protein temporary `prop` mirrors the
//   kernel's shared-memory tile of per-residue propensities.
// ---------------------------------------------------------------------------
void scan_dataset_cpu(const Dataset& ds, int window, float threshold,
                      std::vector<AggResult>& results,
                      std::vector<float>* smoothed) {
    const int half = (window - 1) / 2;            // window W = 2*half + 1
    results.assign(ds.num, AggResult{});
    if (smoothed) smoothed->assign(
        static_cast<std::size_t>(ds.num) * ds.stride, 0.0f);

    std::vector<float> prop;                       // per-residue propensities (reused)
    for (int p = 0; p < ds.num; ++p) {
        const int len = ds.lengths[p];
        const std::size_t base = static_cast<std::size_t>(p) * ds.stride;

        // (a) Materialize this protein's per-residue propensity signal.
        prop.assign(len, 0.0f);
        for (int i = 0; i < len; ++i)
            prop[i] = propensity_of_code(ds.flat_codes[base + i]);

        // (b)+(c) Smooth and reduce in one pass over the residues.
        AggResult r;
        r.peak_score = -1.0f;                      // so the first residue wins
        int run = 0;                               // current contiguous APR length
        for (int i = 0; i < len; ++i) {
            const float s = windowed_mean(prop.data(), len, i, half);
            if (smoothed) (*smoothed)[base + i] = s;
            if (s > r.peak_score) { r.peak_score = s; r.peak_pos = i; }
            if (s >= threshold) {                  // residue i is aggregation-prone
                ++r.prone_count;
                ++run;
                if (run > r.longest_apr) r.longest_apr = run;
            } else {
                run = 0;                           // break the contiguous run
            }
        }
        if (len == 0) r.peak_score = 0.0f;         // empty protein guard
        results[p] = r;
    }
}
