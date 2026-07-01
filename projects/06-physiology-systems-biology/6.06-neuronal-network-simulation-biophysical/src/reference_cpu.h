// ===========================================================================
// src/reference_cpu.h  --  Network config + CPU reference integration
// ---------------------------------------------------------------------------
// Project 6.6 : Neuronal Network Simulation (Biophysical)
//
// The simulation is a NETWORK of `ncell` identical multi-compartment neurons
// wired in a RING: neuron i excites neuron (i+1) mod ncell (one deterministic
// outgoing synapse per cell). A few leading cells are given a startup
// depolarisation, and the excitation then propagates around the ring as a
// travelling wave of spikes.
//
// This header holds:
//   * NetworkConfig            -- all runtime parameters (parsed from the sample)
//   * the (idx -> presynaptic) ring wiring helper
//   * the CPU reference integrator (the trusted baseline the GPU is checked against)
//
// The per-neuron PHYSICS lives in neuron.h (shared host+device). Everything here
// is plain C++ (compiled by cl.exe/g++); kernels.cu reuses NetworkConfig.
//
// READ THIS AFTER: neuron.h.  READ BEFORE: reference_cpu.cpp, kernels.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "neuron.h"   // NN_HD, HHParams, NeuronState, step_neuron, init_neuron

// ---------------------------------------------------------------------------
// NetworkConfig: one simulation job. Fixed HH parameters + network wiring +
// integration settings + which cells receive the external kick that starts the
// wave. All fields are read from the sample file (see data/README.md format).
// ---------------------------------------------------------------------------
struct NetworkConfig {
    int    ncell   = 0;      // number of neurons in the ring
    int    ncomp   = 0;      // compartments per neuron (soma + dendrites)
    double dt      = 0.0;    // integration timestep (ms)
    int    steps   = 0;      // number of timesteps (run length = steps*dt ms)
    double v_rest  = 0.0;    // resting voltage all compartments start at (mV)
    int    n_stim  = 0;      // how many leading cells get the startup kick
    double i_stim  = 0.0;    // startup depolarisation added to stimulated
                             //   cells' soma at t=0 (mV added to v_rest)
    HHParams hh;             // the biophysical constants (see neuron.h)
};

// Presynaptic partner of neuron `i` on the ring: neuron (i-1) excites neuron i,
// i.e. the input to i comes from its "left" ring neighbour. Returns that index.
// (Equivalently: neuron i's spike drives neuron (i+1) mod ncell.)
NN_HD inline int presynaptic_of(int i, int ncell) {
    return (i - 1 + ncell) % ncell;   // +ncell keeps the modulus non-negative
}

// Per-neuron summary the analysis cares about: how many times this cell spiked,
// and the step index of its FIRST spike (a clean, deterministic wave-timing
// signature we can print and verify).
struct CellResult {
    int spike_count;    // total spikes over the whole run
    int first_spike;    // step index of the first spike (-1 if it never fired)
};

// Load a NetworkConfig from the text format documented in data/README.md:
//   "ncell ncomp dt steps v_rest n_stim i_stim gAxial wSyn tauSyn"
// (the remaining HH constants use their standard textbook defaults in neuron.h).
NetworkConfig load_network(const std::string& path);

// CPU reference: integrate the whole network serially and fill one CellResult
// per neuron. This is the trusted baseline; the GPU kernel must reproduce these
// integer spike counts EXACTLY (same double-precision math -> same crossings).
//   `spike_raster` (optional, may be nullptr) receives, per step, the total
//   number of cells that spiked -- used only for the stderr activity trace.
void integrate_cpu(const NetworkConfig& c,
                   std::vector<CellResult>& results,
                   std::vector<int>* spike_raster);
