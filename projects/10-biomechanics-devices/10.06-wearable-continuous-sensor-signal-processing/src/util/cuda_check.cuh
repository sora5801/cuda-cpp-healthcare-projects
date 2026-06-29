// ===========================================================================
// src/util/cuda_check.cuh  --  Always-on, self-explaining CUDA error checking
// ---------------------------------------------------------------------------
// ROLE IN THE PROJECT
//   Every CUDA runtime call can fail (bad launch config, out-of-memory, an
//   illegal address inside a kernel, ...). Silently ignoring those failures is
//   the #1 source of "my kernel does nothing and I don't know why". This header
//   gives every project the SAME two macros so error checking is visible,
//   uniform, and impossible to forget. CLAUDE.md section 6.1 rule 7 requires it.
//
//   This file is COPIED verbatim into every project's src/util/ (a deliberate,
//   documented duplication so each project stays self-contained and buildable
//   on its own -- see CLAUDE.md section 10).
//
// HOW TO USE
//   #include "util/cuda_check.cuh"
//   CUDA_CHECK(cudaMalloc(&d_ptr, bytes));        // wrap any cudaXxx() call
//   my_kernel<<<grid, block>>>(...);              // launch
//   CUDA_CHECK_LAST("my_kernel");                 // check the launch + run
//
// READ THIS BEFORE: util/timer.cuh, kernels.cu (they rely on these macros).
// ===========================================================================
#pragma once

#include <cuda_runtime.h>   // cudaError_t, cudaGetErrorString, cudaGetLastError
#include <cstdio>           // std::fprintf
#include <cstdlib>          // std::exit, EXIT_FAILURE

// ---------------------------------------------------------------------------
// CUDA_CHECK(call)
//   Evaluate a CUDA runtime API call exactly once, capture its cudaError_t,
//   and if it is not cudaSuccess, print a precise diagnostic (file, line, the
//   literal call text via the #call stringizer, and the human-readable error)
//   then abort. We abort rather than return because in teaching code a failed
//   CUDA call means the result is meaningless -- continuing would only confuse.
//
//   The do/while(0) wrapper is the classic C idiom that lets the macro behave
//   like a single statement (so `if (x) CUDA_CHECK(...); else ...;` is legal).
//   The `err__` name carries a trailing underscore to avoid colliding with a
//   variable named `err` in the caller's scope.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr,                                                \
                "[CUDA_CHECK] %s:%d\n    call : %s\n    error: %s (code %d)\n", \
                __FILE__, __LINE__, #call,                                      \
                cudaGetErrorString(err__), static_cast<int>(err__));           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// CUDA_CHECK_LAST(what)
//   Use immediately AFTER a kernel launch. A launch can fail in two distinct
//   ways and we must catch both:
//     1) ASYNCHRONOUS LAUNCH ERRORS (e.g. too many threads/block, too much
//        shared memory) surface via cudaGetLastError() right after the launch.
//     2) EXECUTION ERRORS inside the kernel (e.g. an out-of-bounds write) are
//        only reported once the device finishes; cudaDeviceSynchronize() blocks
//        until then and returns that error.
//   `what` is a short label (usually the kernel name) for the message.
//
//   NOTE (performance vs. teaching): the cudaDeviceSynchronize() here serializes
//   the CPU and GPU after every launch. That is exactly what we want while
//   LEARNING (errors show up at the line that caused them), but in a throughput-
//   oriented build you would remove the sync from the hot path. We keep it on by
//   default because correctness-you-can-see beats speed-you-can't-explain.
// ---------------------------------------------------------------------------
#define CUDA_CHECK_LAST(what)                                                   \
    do {                                                                        \
        cudaError_t launch__ = cudaGetLastError();      /* (1) launch config */ \
        if (launch__ != cudaSuccess) {                                          \
            std::fprintf(stderr,                                                \
                "[CUDA_CHECK_LAST] launch of '%s' failed at %s:%d -> %s\n",     \
                (what), __FILE__, __LINE__, cudaGetErrorString(launch__));      \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
        cudaError_t run__ = cudaDeviceSynchronize();    /* (2) execution    */  \
        if (run__ != cudaSuccess) {                                             \
            std::fprintf(stderr,                                                \
                "[CUDA_CHECK_LAST] execution of '%s' failed at %s:%d -> %s\n",  \
                (what), __FILE__, __LINE__, cudaGetErrorString(run__));         \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)
