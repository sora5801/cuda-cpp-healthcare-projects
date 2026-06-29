// ===========================================================================
// src/util/timer.cuh  --  GPU timing with CUDA events (the right way)
// ---------------------------------------------------------------------------
// ROLE IN THE PROJECT
//   We want to show the learner a fair, clearly-labeled kernel time so the
//   GPU-vs-CPU comparison is legible. The WRONG way is to wrap a kernel in a
//   host clock: kernel launches are asynchronous, so the CPU clock would mostly
//   measure launch overhead, not the kernel. The RIGHT way is CUDA EVENTS,
//   which are timestamps recorded IN THE GPU's execution stream -- they measure
//   actual on-device elapsed time at ~0.5 microsecond resolution.
//
//   This is a teaching artifact, never a benchmark claim (CLAUDE.md section 12).
//
//   Copied verbatim into every project's src/util/ (documented duplication).
//
// HOW TO USE
//   #include "util/timer.cuh"
//   GpuTimer t;
//   t.start();
//   my_kernel<<<grid, block>>>(...);
//   float ms = t.stop_ms();        // blocks until the kernel finishes
//
// READ THIS AFTER: util/cuda_check.cuh (we use CUDA_CHECK here).
// ===========================================================================
#pragma once

#include <cuda_runtime.h>
#include "cuda_check.cuh"   // CUDA_CHECK

// A tiny RAII wrapper around a pair of cudaEvent_t timestamps.
//   * RAII (constructor creates the events, destructor frees them) guarantees
//     we never leak the two event handles even if an error path returns early.
struct GpuTimer {
    cudaEvent_t start_evt;   // recorded by start()      -> "GPU clock = now"
    cudaEvent_t stop_evt;    // recorded by stop_ms()    -> "GPU clock = now"

    GpuTimer() {
        // cudaEventCreate allocates a lightweight device-side timestamp object.
        CUDA_CHECK(cudaEventCreate(&start_evt));
        CUDA_CHECK(cudaEventCreate(&stop_evt));
    }

    ~GpuTimer() {
        // Destructors must not throw/abort on a stale context, so we do NOT
        // wrap these in CUDA_CHECK -- a failure here is harmless at teardown.
        cudaEventDestroy(start_evt);
        cudaEventDestroy(stop_evt);
    }

    // Place a "start" marker into the (default) stream. Everything launched
    // after this and before stop_ms() is what we are timing.
    void start() { CUDA_CHECK(cudaEventRecord(start_evt)); }

    // Place a "stop" marker, then BLOCK the CPU until the GPU actually reaches
    // it (cudaEventSynchronize). Only then is the elapsed time well-defined.
    // Returns milliseconds between the two markers (float, GPU-measured).
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_evt));
        CUDA_CHECK(cudaEventSynchronize(stop_evt));   // wait for GPU to finish
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_evt, stop_evt));
        return ms;
    }
};
