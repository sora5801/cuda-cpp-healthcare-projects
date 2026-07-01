// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial network integration (CPU baseline)
// ---------------------------------------------------------------------------
// Project 6.6 : Neuronal Network Simulation (Biophysical)
//
// Compiled by the host compiler only. The per-neuron physics lives in neuron.h
// (shared host+device); this file just wires the ring together and steps time.
//
// THE STEP-SYNCHRONOUS SPIKE BUFFER (the one subtle bit)
//   Synapses act with a one-step delay: a spike produced on step t is DELIVERED
//   on step t+1. We therefore keep a "spike[]" buffer describing who fired on
//   the PREVIOUS step, read it to drive each neuron's synaptic input this step,
//   and write a fresh "spike_next[]" for who fires now -- then swap. This
//   double-buffering (a) matches how the GPU kernel must work (no thread can see
//   another thread's same-step result without a barrier), and (b) makes the
//   whole simulation DETERMINISTIC and order-independent: neuron i's update this
//   step depends only on last step's spikes, never on the order cells are
//   processed. That is what lets the CPU loop and the GPU's one-thread-per-cell
//   kernel produce byte-identical spike counts. (docs/PATTERNS.md section 2/3.)
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_network: parse the whitespace-separated sample file. Fields, in order:
//   ncell ncomp dt steps v_rest n_stim i_stim gAxial wSyn tauSyn
// The three trailing values override the corresponding HHParams defaults so a
// learner can retune coupling/synapse strength from the data file alone.
// ---------------------------------------------------------------------------
NetworkConfig load_network(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open network file: " + path);
    NetworkConfig c;
    double gAxial, wSyn, tauSyn;
    if (!(in >> c.ncell >> c.ncomp >> c.dt >> c.steps >> c.v_rest
             >> c.n_stim >> c.i_stim >> gAxial >> wSyn >> tauSyn))
        throw std::runtime_error("bad parameters (expected "
            "'ncell ncomp dt steps v_rest n_stim i_stim gAxial wSyn tauSyn') in " + path);
    // Validate: sizes must be positive and fit the compile-time compartment cap.
    if (c.ncell <= 0 || c.ncomp <= 0 || c.ncomp > NN_MAX_COMP ||
        c.steps <= 0 || c.dt <= 0.0)
        throw std::runtime_error("invalid network parameters (check ncomp <= "
            "NN_MAX_COMP, positive sizes/dt) in " + path);
    c.hh.gAxial = gAxial;
    c.hh.wSyn   = wSyn;
    c.hh.tauSyn = tauSyn;
    return c;
}

// ---------------------------------------------------------------------------
// integrate_cpu: serial reference. Allocate all neuron states, seed the wave on
// the first n_stim cells, then march time with the step-synchronous spike
// buffer described above.
// ---------------------------------------------------------------------------
void integrate_cpu(const NetworkConfig& c,
                   std::vector<CellResult>& results,
                   std::vector<int>* spike_raster) {
    const int NC = c.ncell;
    results.assign(NC, CellResult{0, -1});

    // Per-neuron dynamical state (soma + dendrites, gates, synapse).
    std::vector<NeuronState> net(NC);
    for (int i = 0; i < NC; ++i) init_neuron(net[i], c.ncomp, c.v_rest);

    // Seed the travelling wave: give the first n_stim cells a soma depolarisation
    // so they are already above threshold-ish and fire early, kicking the ring.
    for (int i = 0; i < c.n_stim && i < NC; ++i) net[i].V[0] = c.v_rest + c.i_stim;

    // Double-buffered spike flags: spike_prev = who fired LAST step (drives
    // synapses this step); spike_now = who fires THIS step (built as we go).
    std::vector<int> spike_prev(NC, 0), spike_now(NC, 0);

    if (spike_raster) spike_raster->assign(c.steps, 0);

    for (int t = 0; t < c.steps; ++t) {
        int fired_this_step = 0;
        for (int i = 0; i < NC; ++i) {
            // Synaptic input to neuron i = did its ring predecessor fire last step?
            const int pre = presynaptic_of(i, NC);
            const int spike_in = spike_prev[pre];
            // Advance this neuron one dt; step_neuron returns 1 on a soma spike.
            const int fired = step_neuron(net[i], c.hh, c.dt, spike_in);
            spike_now[i] = fired;
            if (fired) {
                ++results[i].spike_count;
                if (results[i].first_spike < 0) results[i].first_spike = t;
                ++fired_this_step;
            }
        }
        if (spike_raster) (*spike_raster)[t] = fired_this_step;
        // Swap buffers: this step's spikes become next step's synaptic drive.
        spike_prev.swap(spike_now);
        // (spike_now still holds old values; it is fully overwritten next step.)
    }
}
