// ===========================================================================
// src/kernels.cuh  --  GPU simulation interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 6.7 : Spiking Neural Network (Point-Neuron) Simulation
//
// THE BIG IDEA (PATTERNS.md sec.1: per-element state update + atomic scatter)
//   A point-neuron network is two nested parallel loops per timestep:
//     * UPDATE  -- one thread per neuron advances that neuron's LIF state
//                  (embarrassingly parallel, like the 9.02 ensemble ODE).
//     * DELIVER -- when a neuron spikes, it must add its weight to THOUSANDS of
//                  postsynaptic targets; many source threads hit the same target,
//                  so we accumulate with atomicAdd. To keep that sum deterministic
//                  and CPU-matching, we accumulate in INTEGER fixed-point
//                  (PATTERNS.md sec.3; the exact same trick as 5.01 / 11.09).
//   We run these as separate kernels each step, with a one-timestep synaptic
//   delay (spikes from step t are delivered on step t+1) so there is no
//   read-after-write hazard between the two kernels -> the parallel result equals
//   the serial reference exactly.
//
//   All per-neuron physics is shared with the CPU via lif.h, so "GPU matches CPU"
//   holds to the bit. kernels.cu defines the kernels and the host driver.
//
// READ THIS AFTER: lif.h, reference_cpu.h, util/cuda_check.cuh, util/timer.cuh.
// ===========================================================================
#pragma once

#include "reference_cpu.h"   // NetworkConfig, SimResult (pure C++, safe in a .cu)

// ---------------------------------------------------------------------------
// simulate_gpu: run the whole simulation on the device and fill `out` with the
//   SAME deterministic summary the CPU produces (total/per-neuron/per-step spike
//   counts + final voltages). `kernel_ms` receives the total device time spent in
//   the per-step kernels (a teaching timing, measured with CUDA events).
//
//   The host keeps the small per-step spike histogram on the device and copies it
//   back once at the end (one D2H copy of `steps` ints), so the time loop stays on
//   the GPU with no per-step host synchronisation in the timed region.
// ---------------------------------------------------------------------------
void simulate_gpu(const NetworkConfig& c, SimResult& out, float* kernel_ms);
