// ===========================================================================
// src/reference_cpu.h  --  CPU reference API: MSA loading + MI/APC pipeline
// ---------------------------------------------------------------------------
// Project 2.25 : Coevolutionary Contact Prediction & MSA Transformer
//
// ROLE
//   Declares (a) the in-memory MSA representation, (b) a tiny FASTA loader, and
//   (c) the plain-C++ reference computation of the coevolution score matrix.
//   The reference is the teaching baseline AND the correctness oracle: main.cu
//   runs it and the GPU kernels on the same MSA and asserts they agree
//   (CLAUDE.md section 5 "CPU reference path").
//
//   This header is included by BOTH reference_cpu.cpp (host compiler) and
//   main.cu (nvcc), so it stays free of CUDA constructs. The per-pair MATH lives
//   in coevolution.h (shared host/device); this file is the host-only PIPELINE
//   around it (loading, looping over pairs, the APC correction).
//
// READ THIS BEFORE: reference_cpu.cpp, main.cu. READ coevolution.h first.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Msa: a Multiple Sequence Alignment stored as a dense integer token matrix.
//   * N    = number of sequences (rows). Each row is one homolog of the protein.
//   * L    = alignment length (columns). Every row has exactly L tokens (an MSA
//            is rectangular -- gaps pad shorter sequences).
//   * token[r*L + c] = the amino-acid token in [0,CV_Q) at row r, column c,
//            row-major. We keep tokens as uint8_t (a small integer alphabet) so
//            the whole MSA is compact and cheap to upload to the GPU.
// The struct owns its storage (std::vector), so copies are deep and safe.
// ---------------------------------------------------------------------------
struct Msa {
    int N = 0;                         // sequences (rows)
    int L = 0;                         // alignment length (columns)
    std::vector<uint8_t> token;        // [N*L] tokens, row-major, values in [0,CV_Q)
};

// ---------------------------------------------------------------------------
// load_msa: read a FASTA-format alignment file into an Msa.
//   FASTA = repeating ">header\n" then one or more lines of sequence letters.
//   We require every sequence to have the SAME length (a valid alignment); the
//   loader throws std::runtime_error otherwise so a malformed file fails loudly
//   instead of silently producing garbage. Letters are mapped to tokens via
//   cv_token_of_aa (coevolution.h): the 20 amino acids plus gap.
//   Throws std::runtime_error if the file cannot be opened or is not rectangular.
// ---------------------------------------------------------------------------
Msa load_msa(const std::string& path);

// ---------------------------------------------------------------------------
// coevolution_cpu: the full reference pipeline.
//   Given an MSA, compute the L x L coevolution score matrix in two stages:
//     1. RAW MI:  mi[i*L + j] = Mutual Information of columns i and j (nats),
//        from integer co-occurrence counts (cv_mi_from_counts in coevolution.h).
//        The diagonal mi[i,i] is set to 0 (a column is trivially dependent on
//        itself; self-MI is not a contact signal).
//     2. APC:     score[i,j] = mi[i,j] - (MIcol(i)*MIcol(j))/MImean, the Average
//        Product Correction (Dunn 2008) that removes per-column background bias.
//        MIcol(i) is the mean of row i of MI (excluding the diagonal), MImean is
//        the overall off-diagonal mean. This is what separates true contacts
//        from entropic/phylogenetic noise.
//
//   OUTPUTS (caller-allocated via resize inside): both are length L*L, row-major.
//     mi    : the raw MI matrix (returned so we can verify GPU vs CPU on the part
//             the GPU actually computes -- the per-pair MI).
//     score : the APC-corrected coevolution score (the thing we rank for contacts).
//
//   This same arithmetic runs on the GPU side (kernels.cu computes `mi`; the APC
//   reduction is done on the host in main.cu from the GPU's mi, exactly as
//   project 11.09 finishes its reduction on the host so CPU and GPU match).
// ---------------------------------------------------------------------------
void coevolution_cpu(const Msa& msa,
                     std::vector<double>& mi,
                     std::vector<double>& score);

// ---------------------------------------------------------------------------
// apc_correct: apply the Average Product Correction to a raw MI matrix.
//   Pulled out as a free function because BOTH the CPU reference and main.cu's
//   GPU path call it (the GPU computes raw MI; APC is a cheap L^2 host post-step,
//   so we share ONE implementation -> identical corrected scores). In-place is
//   avoided: `mi` is read, `score` (length L*L) is written.
// ---------------------------------------------------------------------------
void apc_correct(const std::vector<double>& mi, int L, std::vector<double>& score);
