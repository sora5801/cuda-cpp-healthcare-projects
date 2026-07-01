// ===========================================================================
// src/main.cu  --  Entry point: simulate the network, verify, report
// ---------------------------------------------------------------------------
// Project 6.6 : Neuronal Network Simulation (Biophysical)
//
// 5-step shape (the same skeleton every project in this repo follows):
//   1. Load the network config (ring of multi-compartment HH neurons).
//   2. CPU reference: integrate the whole network serially (reference_cpu.cpp).
//   3. GPU: one thread per neuron, one kernel launch per timestep (kernels.cu).
//   4. VERIFY: per-cell spike counts / first-spike steps match EXACTLY
//      (integer crossing counts of identical double-precision voltages).
//   5. REPORT: a deterministic per-cell + network summary to STDOUT; timing and
//      run-varying detail to STDERR (so the demo can diff stdout reproducibly).
//
// Code tour: start here, then neuron.h (the physics), reference_cpu.cpp (the
// serial baseline + spike-buffer idea), then kernels.cu (the GPU twin).
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // integrate_gpu
#include "reference_cpu.h"    // load_network, integrate_cpu, NetworkConfig, CellResult
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "6.6";
static const char* PROJECT_NAME = "Neuronal Network Simulation (Biophysical)";

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/network.txt";
    NetworkConfig c;
    try {
        c = load_network(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<CellResult> res_cpu;
    std::vector<int> raster;                 // per-step total spikes (stderr trace)
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    integrate_cpu(c, res_cpu, &raster);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU network (kernel-timed) ------------------------------------
    std::vector<CellResult> res_gpu;
    float gpu_kernel_ms = 0.0f;
    integrate_gpu(c, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify: EXACT integer agreement -------------------------------
    // The GPU and CPU run identical double-precision arithmetic (neuron.h), so
    // the soma voltage sequences are bit-identical and the threshold crossings
    // -- and thus spike_count / first_spike -- must match EXACTLY. Any mismatch
    // is a real bug (a race, a wiring error), not floating-point noise, so the
    // tolerance is ZERO. (docs/PATTERNS.md section 4: integer result -> exact.)
    int mismatches = 0;
    for (int i = 0; i < c.ncell; ++i) {
        if (res_cpu[i].spike_count != res_gpu[i].spike_count ||
            res_cpu[i].first_spike != res_gpu[i].first_spike) ++mismatches;
    }
    const bool pass = (mismatches == 0);

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    // Network totals: total spikes, how many cells ever fired, and the mean
    // firing rate (spikes per cell per second of simulated time).
    long total_spikes = 0;
    int  active_cells = 0;
    for (int i = 0; i < c.ncell; ++i) {
        total_spikes += res_gpu[i].spike_count;
        if (res_gpu[i].spike_count > 0) ++active_cells;
    }
    const double sim_seconds = c.steps * c.dt / 1000.0;    // ms -> s
    const double mean_rate_hz =
        (c.ncell > 0 && sim_seconds > 0.0)
            ? static_cast<double>(total_spikes) / (c.ncell * sim_seconds) : 0.0;

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("network: %d neurons x %d compartments, %d steps @ dt=%.3f ms (%.1f ms sim)\n",
                c.ncell, c.ncomp, c.steps, c.dt, c.steps * c.dt);
    std::printf("ring wiring: neuron i excites neuron (i+1); %d leading cells kicked (+%.1f mV)\n",
                c.n_stim, c.i_stim);
    std::printf("per-cell (idx : spikes firstStep):\n");
    // Print up to 8 evenly-spaced cells so the wave signature is visible but the
    // output stays compact and deterministic regardless of ncell.
    const int shown = (c.ncell < 8) ? c.ncell : 8;
    for (int s = 0; s < shown; ++s) {
        const int i = (shown > 1) ? (s * (c.ncell - 1)) / (shown - 1) : 0;
        std::printf("  c%-4d: %3d %6d\n", i, res_gpu[i].spike_count, res_gpu[i].first_spike);
    }
    std::printf("totals: %ld spikes across %d/%d active cells; mean rate = %.3f Hz\n",
                total_spikes, active_cells, c.ncell, mean_rate_hz);
    std::printf("RESULT: %s (GPU spike counts match CPU exactly across %d cells)\n",
                pass ? "PASS" : "FAIL", c.ncell);

    // ---- 5b. Run-varying detail -> STDERR ---------------------------------
    // A tiny activity trace: the step at which the network first shows any spike
    // and the peak simultaneous spike count -- handy for eyeballing the wave.
    int first_active = -1, peak_simul = 0;
    for (int t = 0; t < c.steps; ++t) {
        if (raster[t] > 0 && first_active < 0) first_active = t;
        if (raster[t] > peak_simul) peak_simul = raster[t];
    }
    std::fprintf(stderr, "[data]   source: %s  (%d neurons)\n", path.c_str(), c.ncell);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU: %.3f ms  (%d per-step launches)\n",
                 cpu_ms, gpu_kernel_ms, c.steps);
    std::fprintf(stderr, "[timing] teaching artifact -- one kernel launch per dt is launch-bound "
                         "on tiny nets; the GPU wins as ncell grows into the thousands.\n");
    std::fprintf(stderr, "[trace]  first network activity at step %d; peak simultaneous spikes = %d\n",
                 first_active, peak_simul);
    std::fprintf(stderr, "[verify] per-cell spike mismatches CPU vs GPU = %d (tolerance 0 = exact)\n",
                 mismatches);

    return pass ? 0 : 1;
}
