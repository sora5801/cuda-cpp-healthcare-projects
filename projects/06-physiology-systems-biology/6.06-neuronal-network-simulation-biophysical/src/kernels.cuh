// ===========================================================================
// src/kernels.cuh  --  GPU network-integration interface
// ---------------------------------------------------------------------------
// Project 6.6 : Neuronal Network Simulation (Biophysical)
//
// THE BIG IDEA (pattern: ENSEMBLE / ONE-THREAD-PER-CELL time integration)
//   The network is `ncell` neurons, each an independent multi-compartment ODE
//   system EXCEPT for the synaptic coupling. We map ONE GPU THREAD to ONE NEURON
//   (in production, one thread BLOCK per cell with a warp walking the dendritic
//   tree -- see THEORY -- but one thread per cell is the clearest teaching form
//   and keeps each cell's small state in registers/local memory).
//
//   Coupling is handled by SEPARATING TIME STEPS INTO KERNEL LAUNCHES. Within a
//   step a thread must not read another thread's freshly-written spike (there is
//   no global barrier inside a kernel across all blocks). So we keep TWO spike
//   buffers in global memory and launch the step kernel once per timestep: every
//   thread reads its presynaptic partner's spike from the PREVIOUS buffer and
//   writes its own spike into the CURRENT buffer; the host swaps the two buffers
//   between launches (ping-pong). The kernel boundary IS the grid-wide barrier.
//   This is exactly the double-buffering the CPU reference uses, so GPU and CPU
//   spike counts match EXACTLY.
//
//   The per-neuron physics (HH + Rush-Larsen + Hines/Thomas + synapse) is the
//   SAME neuron.h code the CPU calls -> identical arithmetic. kernels.cu defines
//   the kernel and the host driver.
//
// READ THIS AFTER: neuron.h, reference_cpu.h, util/*.  READ BEFORE: kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // NetworkConfig, CellResult (pure C++, safe in .cu)

// Host driver: integrate the whole network on the GPU (one thread per neuron,
// one kernel launch per timestep with ping-pong spike buffers), copy the
// per-cell spike summaries back, and report the total kernel time.
//   c            : the simulation configuration (sizes, dt, HH params, wiring)
//   results      : filled with one CellResult per neuron (sized to c.ncell)
//   kernel_ms    : total GPU time across all per-step launches (teaching artifact)
void integrate_gpu(const NetworkConfig& c,
                   std::vector<CellResult>& results,
                   float* kernel_ms);
