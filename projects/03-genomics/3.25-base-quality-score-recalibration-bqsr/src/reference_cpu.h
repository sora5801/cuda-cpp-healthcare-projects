// ===========================================================================
// src/reference_cpu.h  --  Dataset model + CPU reference API for BQSR
// ---------------------------------------------------------------------------
// Project 3.25 : Base Quality Score Recalibration (BQSR)
//
// Declares:
//   * Dataset           -- the loaded alignment (reference, reads, known sites),
//                          flattened into GPU-friendly arrays.
//   * load_dataset      -- parse the data/sample text format into a Dataset.
//   * build_table_cpu   -- the serial covariate-table accumulation (the baseline
//                          the GPU is verified against).
//   * recalibrate_cpu   -- apply an (obs,err) table to produce new qualities.
//
// The per-base covariate math lives in bqsr.h (shared host+device). This header
// is included by reference_cpu.cpp (host) and main.cu / kernels.cu (nvcc).
// READ THIS AFTER: bqsr.h. READ BEFORE: reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "bqsr.h"   // covariate layout, empirical_q, PHRED helpers

// ---------------------------------------------------------------------------
// Dataset: one aligned "tile" of sequencing, flattened so both the CPU loops and
// the GPU kernels can index it without pointer-chasing.
//
// Layout choice: reads are RAGGED in principle, but for a clean teaching kernel
// we store every read with the SAME fixed length `read_len` (<= MAX_CYCLE). Each
// read i occupies the half-open row [i*read_len, (i+1)*read_len) of `read_bases`
// and `read_quals`, and starts at reference position `read_pos[i]`. This regular
// layout means base global-index `g` decomposes as read = g/read_len,
// cycle = g%read_len -- the same trick 11.09 used for (event, marker).
// ---------------------------------------------------------------------------
struct Dataset {
    // The reference genome substring this tile is aligned to (1 char per base).
    // Read i base c compares against reference[ read_pos[i] + c ].
    std::string reference;

    int num_reads = 0;   // R: number of reads
    int read_len  = 0;   // L: bases per read (fixed; L <= MAX_CYCLE)

    // Flattened read data, row-major [R * L].
    std::vector<char> read_bases;        // the called base letters (A/C/G/T/N)
    std::vector<int>  read_quals;        // the reported PHRED quality per base
    std::vector<int>  read_pos;          // [R] reference start position per read

    // Known-variant mask over the reference: known_site[p] != 0 means reference
    // position p is a known polymorphism (dbSNP/Mills) and bases there are SKIPPED
    // during table building. Sized to reference.size().
    std::vector<unsigned char> known_site;

    // Convenience: total number of bases = num_reads * read_len.
    int total_bases() const { return num_reads * read_len; }
};

// Parse the data/sample text format (documented in data/README.md) into a
// Dataset. Throws std::runtime_error on malformed input so demos fail loudly.
Dataset load_dataset(const std::string& path);

// ---------------------------------------------------------------------------
// build_table_cpu: the serial covariate-table accumulation (the BQSR baseline).
//   Walks every base, skips known-variant sites, bins survivors by
//   (Q, cycle, context), and tallies obs/err as integers. Fills `obs` and `err`
//   (each length NUM_BINS). This is exactly what accumulate_kernel does on the
//   GPU; main.cu checks the two integer tables are identical.
// ---------------------------------------------------------------------------
void build_table_cpu(const Dataset& d,
                     std::vector<unsigned int>& obs,
                     std::vector<unsigned int>& err);

// ---------------------------------------------------------------------------
// recalibrate_cpu: given a finished (obs,err) table, compute the new quality of
//   every base = empirical_q(bin). Bases with no evidence (Q_emp == -1) keep
//   their original reported quality. Output `new_quals` is [R*L], matching the
//   read_quals layout. Shared by CPU and (via the same helper) the GPU path so
//   the recalibrated scores agree exactly.
// ---------------------------------------------------------------------------
void recalibrate_cpu(const Dataset& d,
                     const std::vector<unsigned int>& obs,
                     const std::vector<unsigned int>& err,
                     std::vector<int>& new_quals);
