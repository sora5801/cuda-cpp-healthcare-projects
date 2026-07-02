// ===========================================================================
// src/kernels.cuh  --  GPU ABM interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 6.9 : Agent-Based Tissue / Immune Simulation
//
// THE BIG IDEA (a HYBRID of three flagship patterns)
//   An agent-based tissue simulation couples a grid PDE with a particle system.
//   We map each of the three per-step phases to the GPU pattern that fits it:
//
//     (1) SECRETE  -> SCATTER-REDUCTION with atomics (like 11.09 k-means, 5.01
//                     Monte-Carlo tally): one thread per cell atomic-adds a
//                     FIXED-POINT quantum into the grid cell it occupies. Integer
//                     atomics commute => deterministic and CPU-exact.
//     (2) DIFFUSE  -> STENCIL + ping-pong (like 6.04 lattice-Boltzmann, 14.02
//                     reaction-diffusion): one thread per grid cell, read c_old,
//                     write c_new, swap buffers.
//     (3) MOVE     -> per-agent update with SPATIAL BINNING for O(N) neighbour
//                     search (the ABM-specific pattern): one thread per cell,
//                     scanning only the 3x3 neighbouring bins.
//
//   The per-element math is the shared abm_core.h (ABM_HD functions), so the GPU
//   reproduces the CPU reference exactly. The spatial bins are rebuilt on the
//   HOST each step (a counting sort) and uploaded -- a deliberate teaching choice:
//   it keeps the neighbour order identical to the CPU (hence exact verification)
//   and isolates the three GPU patterns cleanly. THEORY.md discusses the fully
//   on-GPU binning (Thrust sort by key) that production codes use.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, abm_core.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // AbmParams, Cells, SpatialBins, AbmResult (pure C++)

// Host wrapper: run the full ABM time loop on the GPU and return the
// deterministic summary. `field_out` receives the final chemokine field.
// `kernel_ms` receives the total GPU time of the loop (a teaching artifact).
AbmResult abm_gpu(const AbmParams& p, const Cells& cells0,
                  std::vector<double>& field_out, float* kernel_ms);
