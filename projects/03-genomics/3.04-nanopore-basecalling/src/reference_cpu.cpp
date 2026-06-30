// ===========================================================================
// src/reference_cpu.cpp  --  The trusted serial CTC decoder + data loader
// ---------------------------------------------------------------------------
// Project 3.4 : Nanopore Basecalling  (REDUCED-SCOPE: CTC greedy decode)
//
// ROLE
//   (1) load_reads(): parse the tiny text dataset (format in data/README.md)
//       into a ReadSet (flat posterior buffer + per-read offsets).
//   (2) basecall_cpu(): the obviously-correct serial decode -- a plain loop over
//       reads, each decoded by the SHARED ctc_core.h routine. Because the GPU
//       kernel calls the EXACT SAME ctc_greedy_decode(), CPU and GPU agree
//       bit-for-bit; main.cu asserts that.
//
//   Compiled by the host C++ compiler only (no CUDA). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h and ctc_core.h. Compare against kernels.cu
//   (the GPU twin that runs the same ctc_core decode from one thread per read).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_reads: parse the jagged posterior dataset into a ReadSet.
//   File format (whitespace-tolerant; see data/README.md):
//     <n_reads> <C>
//     <T_0>
//     <C floats>   x T_0      (read 0's posterior rows, one step per line)
//     <T_1>
//     <C floats>   x T_1
//     ...
//   We build `offset` as a running prefix sum so each read's slice of `probs`
//   is contiguous -- the layout the GPU wants for a single bulk upload.
// ---------------------------------------------------------------------------
ReadSet load_reads(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open reads file: " + path);

    int n_reads = 0, C = 0;
    if (!(in >> n_reads >> C))
        throw std::runtime_error("bad header (expected '<n_reads> <C>') in " + path);
    if (C != CTC_NUM_CLASSES)
        throw std::runtime_error("class-count mismatch: file has C=" + std::to_string(C) +
                                 " but this build expects C=" + std::to_string(CTC_NUM_CLASSES) +
                                 " (rebuild with matching CTC_NUM_CLASSES)");
    if (n_reads <= 0) throw std::runtime_error("non-positive read count in " + path);

    ReadSet rs;
    rs.n_reads = n_reads;
    rs.T.resize(n_reads);
    rs.offset.resize(static_cast<std::size_t>(n_reads) + 1);
    rs.offset[0] = 0;          // read 0 starts at step 0
    rs.max_T = 0;

    // We don't know the total step count up front, so we append to `probs` as we
    // go and grow `offset` by each read's T (a running prefix sum, in steps).
    for (int r = 0; r < n_reads; ++r) {
        int T = 0;
        if (!(in >> T)) throw std::runtime_error("unexpected EOF before read " +
                                                 std::to_string(r) + "'s length in " + path);
        if (T <= 0) throw std::runtime_error("non-positive T for read " +
                                             std::to_string(r) + " in " + path);
        rs.T[r] = T;
        if (T > rs.max_T) rs.max_T = T;
        rs.offset[r + 1] = rs.offset[r] + T;   // prefix sum (in time steps)

        // Read this read's T*C probabilities and append them to the flat buffer.
        for (int t = 0; t < T; ++t) {
            for (int c = 0; c < C; ++c) {
                float p;
                if (!(in >> p))
                    throw std::runtime_error("unexpected EOF in read " + std::to_string(r) +
                                             " step " + std::to_string(t) + " of " + path);
                rs.probs.push_back(p);
            }
        }
    }
    return rs;
}

// ---------------------------------------------------------------------------
// basecall_cpu: decode every read serially with the shared ctc_core routine.
//   For each read r: locate its posterior slice via offset[r], run the SAME
//   ctc_greedy_decode() the GPU runs, then record the base string, its length,
//   and the deterministic integer checksum. No cleverness on purpose -- this is
//   the readable baseline that makes the GPU result trustworthy.
// ---------------------------------------------------------------------------
void basecall_cpu(const ReadSet& rs, std::vector<DecodedRead>& out) {
    out.assign(static_cast<std::size_t>(rs.n_reads), DecodedRead{});
    // Scratch buffer big enough for the longest read: at most one base is
    // emitted per time step, so max_T chars always suffice.
    std::vector<char> buf(static_cast<std::size_t>(rs.max_T > 0 ? rs.max_T : 1));

    for (int r = 0; r < rs.n_reads; ++r) {
        const int T = rs.T[r];
        // Pointer to read r's first posterior value. `offset` is in STEPS, so we
        // multiply by C to get the element index into the flat `probs` array.
        const float* p = &rs.probs[static_cast<std::size_t>(rs.offset[r]) * CTC_NUM_CLASSES];

        // THE decode: identical call to the one the GPU kernel makes.
        const int len = ctc_greedy_decode(p, T, buf.data());

        DecodedRead& d = out[static_cast<std::size_t>(r)];
        d.base_seq.assign(buf.data(), static_cast<std::size_t>(len));
        d.length   = len;
        d.checksum = ctc_base_checksum(buf.data(), len);   // exact integer hash
    }
}
