// ===========================================================================
// src/reference_cpu.h  --  Network loader + CPU reference simulation interface
// ---------------------------------------------------------------------------
// Project 6.7 : Spiking Neural Network (Point-Neuron) Simulation
//
// ROLE
//   Declares (a) how we load a NetworkConfig from the tiny text sample, and
//   (b) the CPU reference simulator that the GPU is verified against. The actual
//   per-neuron physics lives in lif.h (shared with the GPU); this file adds only
//   the plain-C++ host scaffolding (file I/O, the serial time loop, result struct).
//
//   Pure C++ (no CUDA) so it compiles under cl.exe/g++. kernels.cu reuses the
//   NetworkConfig + SimResult types declared here.
//
// READ THIS AFTER: lif.h. READ NEXT: reference_cpu.cpp (implementation), kernels.cuh.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "lif.h"   // NetworkConfig, NeuronState, lif_step, connectivity helpers

// ---------------------------------------------------------------------------
// SimResult: the DETERMINISTIC summary of one simulation run. These are the
//   numbers main.cu prints to stdout and compares CPU-vs-GPU. Everything here is
//   integer or an exactly-reproduced double so the comparison can be exact.
//
//   total_spikes        : total number of spikes emitted by all neurons over all steps.
//   spikes_per_neuron   : [n] per-neuron spike counts (the strongest exact check).
//   spikes_per_step     : [steps] population spike count each timestep (the raster/PSTH).
//   final_v             : [n] each neuron's membrane potential at the last step.
//   mean_rate_hz        : population-mean firing rate (derived; for the human report).
// ---------------------------------------------------------------------------
struct SimResult {
    long long total_spikes = 0;
    std::vector<int>    spikes_per_neuron;   // size n
    std::vector<int>    spikes_per_step;     // size steps
    std::vector<double> final_v;             // size n
    double mean_rate_hz = 0.0;
};

// Load a NetworkConfig from the text format documented in data/README.md:
//   line 1:  n_exc n_inh out_degree
//   line 2:  w_exc w_inh ext_kick ext_every
//   line 3:  steps seed
//   line 4:  v_rest v_reset v_thresh tau_m tau_syn r_m refractory_ms dt
// Throws std::runtime_error if the file is missing or malformed (fail loudly).
NetworkConfig load_network(const std::string& path);

// CPU reference: run the full simulation serially and fill `out`. This is the
// trusted baseline. It uses lif_step() (shared with the GPU) and the SAME
// fixed-point synaptic accumulation, so its results match the GPU exactly.
void simulate_cpu(const NetworkConfig& c, SimResult& out);
