// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 3.11 : GWAS at Scale
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls two host wrappers declared
//   here; kernels.cu implements them (the device kernels + the cuBLAS call live
//   there). Included only by .cu translation units (it names __global__ kernels,
//   so the plain host compiler must never see it -- that is why the CPU
//   reference uses a separate pure-C++ header, reference_cpu.h).
//
// THE TWO GPU JOBS (and the pattern each uses -- see ../THEORY.md, PATTERNS.md)
//
//   1. GRM = (1/M) Z Zᵀ  -- the genetic relatedness matrix.
//        * Standardizing G -> Z is one thread per matrix entry (the classic
//          "grid of threads over a 2D array" map).
//        * The big multiply Z Zᵀ is a DENSE matrix multiply -> we hand it to
//          cuBLAS DGEMM (PATTERNS.md §1 "dense linear algebra -> use cuBLAS").
//          DGEMM is the single most optimized GPU routine in existence; writing
//          a competitive one by hand means shared-memory tiling, register
//          blocking, and bank-conflict avoidance -- kernels.cu explains what it
//          would take and why we don't.
//
//   2. Per-SNP association scan -- one regression per SNP, all SNPs at once.
//        * Each SNP's fit is independent -> ONE GPU THREAD PER SNP. Thread j
//          loops over the N individuals, accumulates the sufficient statistics
//          (Σx², Σxy, Σy²), and calls the shared gwas::assoc_from_sufficient_stats
//          (gwas_core.h) so its numbers match the CPU exactly. This is the
//          "independent jobs" pattern (PATTERNS.md §1), like 1.12 Tanimoto.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, gwas_core.h.
// THEN READ kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "gwas_core.h"      // gwas::AssocResult (POD result per SNP)

// ---------------------------------------------------------------------------
// gpu_build_grm: standardize G on the device and form GRM = (1/M) Z Zᵀ.
//   INPUTS (host):
//     geno : [N*M] row-major int8 dosages in {0,1,2}  (the raw genotype matrix)
//     N, M : individuals, SNPs
//   OUTPUT (host):
//     grm  : resized to [N*N] row-major, the relatedness matrix
//   TIMING:
//     *standardize_ms : ms in the standardize kernel
//     *gemm_ms        : ms in the cuBLAS DGEMM (the headline GPU win)
//   The function owns all device memory + the cuBLAS handle; main.cu just sees
//   host vectors in and out.
void gpu_build_grm(const std::vector<signed char>& geno, int N, int M,
                   std::vector<double>& grm,
                   float* standardize_ms, float* gemm_ms);

// ---------------------------------------------------------------------------
// gpu_assoc_scan: run the single-marker regression for every SNP on the GPU.
//   INPUTS (host):
//     geno      : [N*M] row-major int8 dosages (raw genotypes; the kernel
//                 standardizes on the fly so we never materialize Z twice)
//     y_centered: [N] mean-centered phenotype
//     N, M      : individuals, SNPs
//   OUTPUT (host):
//     out : resized to [M], one gwas::AssocResult per SNP
//   TIMING:
//     *kernel_ms : ms spent in the association kernel (CUDA-event measured)
//   One thread per SNP; see kernels.cu for the launch-configuration reasoning.
void gpu_assoc_scan(const std::vector<signed char>& geno,
                    const std::vector<double>& y_centered, int N, int M,
                    std::vector<gwas::AssocResult>& out, float* kernel_ms);
