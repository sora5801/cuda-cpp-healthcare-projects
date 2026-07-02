// ===========================================================================
// src/kernels.cuh  --  GPU interface for the defibrillation DFT sweep
// ---------------------------------------------------------------------------
// Project 6.19 : Defibrillation & High-Voltage Shock Simulation
//
// THE BIG IDEA (the "ensemble of trajectories" pattern -- PATTERNS.md section 1,
// as in flagships 9.02 SEIR and 13.02 PBPK)
//   We must simulate the SAME 1-D cable many times, once per candidate shock
//   amplitude, to find the defibrillation threshold. Each amplitude's simulation
//   is completely independent of the others, so we assign ONE GPU THREAD to ONE
//   shock amplitude. Thread k loops the shared cable_step() (defib.h) over all
//   time steps on its own private cable, then writes a single residual-activity
//   number. No inter-thread communication, no atomics -- pure independent work.
//
//   Within each thread the cable update is itself a 3-point diffusion STENCIL
//   with ping-pong buffers; because a thread owns its whole cable, it keeps the
//   two buffers in per-thread scratch (device global memory here) and swaps
//   pointers each step -- exactly mirroring the CPU reference.
//
//   Why not one thread PER CELL (a 2-D grid of amplitude x cell)? That would be
//   the natural choice for a single huge cable, but here the cables are small
//   (~128 cells) and we have many of them; thread-per-trajectory keeps the time
//   loop and the ping-pong entirely inside one thread, which is simpler to teach
//   and needs no cross-block synchronisation between the (thousands of) steps.
//   THEORY.md "GPU mapping" discusses the crossover and how production
//   whole-heart bidomain codes (one thread per cell + a cuSPARSE CG solve for
//   the elliptic extracellular equation) differ.
//
// READ THIS AFTER: defib.h, reference_cpu.h (for FhnParams / ShockSweep),
//                  util/cuda_check.cuh, util/timer.cuh.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // FhnParams, ShockSweep (pure C++, safe in .cu)

// ---------------------------------------------------------------------------
// sweep_gpu: run the whole DFT sweep on the GPU.
//   p        : shared cable/FHN/shock parameters (passed by value to the kernel)
//   amps     : the shock amplitudes to test (one per GPU thread)
//   residual : filled (resized to amps.size()) with each amplitude's residual
//              activity, in the SAME order as `amps` -> directly comparable to
//              the CPU reference's output.
//   kernel_ms: out-param, GPU time of the sweep kernel (CUDA-event measured).
// The per-amplitude math is the shared cable_step()/activity_metric() from
// defib.h, so residual[k] here equals simulate_one_cpu(p, amps[k]) on the CPU.
// ---------------------------------------------------------------------------
void sweep_gpu(const FhnParams& p, const std::vector<double>& amps,
               std::vector<double>& residual, float* kernel_ms);
