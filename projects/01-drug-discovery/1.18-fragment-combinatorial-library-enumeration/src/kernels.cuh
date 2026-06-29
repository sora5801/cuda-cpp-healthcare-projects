// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for combinatorial enumeration
// ---------------------------------------------------------------------------
// Project 1.18 : Fragment / Combinatorial Library Enumeration
//
// THE BIG IDEA
//   Enumerating the N = s0*s1*s2 products of a combinatorial library is N
//   INDEPENDENT jobs: product p has no dependence on product q. So we give each
//   product its own GPU thread. Each thread:
//     1. decodes its flat index p into per-slot synthon indices (mixed-radix,
//        the "odometer" in product_core.h),
//     2. sums the chosen synthons' additive descriptors,
//     3. applies the Lipinski + Veber filter, and
//     4. if the product passes, increments a global counter and adds its MW to a
//        global FIXED-POINT sum -- both with INTEGER atomics, so the totals are
//        deterministic and match the CPU bit-for-bit (PATTERNS.md sec.3 rule 2).
//   A grid-stride loop lets a fixed-size grid cover an arbitrarily large library.
//
//   Two CUDA features carry the teaching weight here:
//     * the synthon descriptor tables live in CONSTANT memory -- every thread
//       reads them, none writes them, and they are tiny (a few dozen rows), so
//       the constant cache broadcasts them warp-wide for free; and
//     * INTEGER atomicAdd into device counters gives an order-independent
//       reduction (a FLOAT atomic sum would be irreproducible -- see THEORY).
//
//   This header is included only by .cu units (it declares a __global__). The
//   pure-C++ data model is in reference_cpu.h; the shared math is product_core.h.
//
// READ THIS AFTER: product_core.h, util/cuda_check.cuh, util/timer.cuh.
// Then read kernels.cu. The science/GPU-mapping lives in ../THEORY.md.
// ===========================================================================
#pragma once

#include <cstdint>

#include "reference_cpu.h"   // SynthonLibrary, EnumResult (pure C++, safe in .cu)

// ---- Device kernel (declared for documentation; defined in kernels.cu) ----
// enumerate_kernel: one logical thread per product, via a grid-stride loop.
//   sizes      : the N_SLOTS slot sizes, passed by value in a small struct so
//                each thread holds them in registers (see kernels.cu).
//   N          : total number of products = product of sizes.
//   d_count    : [1] device counter, atomically incremented per passing product.
//   d_sum_mw   : [1] device int64 accumulator of passing-product MW in milli-
//                g/mol (fixed point) -- integer atomics keep it deterministic.
//   The synthon descriptor tables are read from CONSTANT memory (see kernels.cu),
//   not passed as parameters.
//
// (We declare it here for the code tour; the actual __global__ signature with
// its constant-memory dependency is defined and fully commented in kernels.cu.)

// ---- Host wrapper --------------------------------------------------------
// enumerate_gpu: the host-callable "do the whole enumeration on the GPU".
//   Uploads the synthon tables to constant memory, launches the kernel, reduces
//   the global counter + MW sum, recovers the first FIRST_K passing indices, and
//   reports the measured KERNEL time (CUDA events) via *kernel_ms.
//     lib       : the loaded synthon library (input)
//     out       : filled with n_pass, first_pass, sum_mw_pass_milli (output)
//     kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
//   main.cu calls exactly this; all CUDA bookkeeping is hidden here.
void enumerate_gpu(const SynthonLibrary& lib, EnumResult& out, float* kernel_ms);
