// ===========================================================================
// src/main.cu  --  Entry point: load network, simulate CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 6.7 : Spiking Neural Network (Point-Neuron) Simulation
//
// 5-step shape (mirrors the flagships):
//   1. Load the network config (sizes, weights, biophysics) from the text sample.
//   2. CPU reference: simulate serially (reference_cpu.cpp).
//   3. GPU: simulate with per-step kernels + atomic fixed-point scatter (kernels.cu).
//   4. VERIFY: total, per-step, and per-neuron spike counts are IDENTICAL (the
//      shared physics + integer accumulation make this exact, tolerance == 0);
//      final membrane potentials agree to ~double round-off.
//   5. REPORT: a deterministic network summary + a sample of the spike raster to
//      stdout; timing + verification detail to stderr.
//
// Code tour: start here, then lif.h (the physics), reference_cpu.cpp (the serial
// algorithm), kernels.cu (the parallel version). See ../README.md "Code tour".
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>   // std::pair, std::make_pair, std::swap
#include <vector>

#include "kernels.cuh"       // simulate_gpu, NetworkConfig, SimResult
#include "reference_cpu.h"   // load_network, simulate_cpu
#include "util/io.hpp"       // util::CpuTimer

static const char* PROJECT_ID   = "6.7";
static const char* PROJECT_NAME = "Spiking Neural Network (Point-Neuron) Simulation";

// Spike counts are integers computed with identical fixed-point arithmetic on
// both sides, so they must match EXACTLY. Final voltages are doubles produced by
// the identical sequence of ops, so they match to ~machine round-off; we allow a
// tiny slack there purely as defensive documentation (PATTERNS.md sec.4).
static constexpr double VOLTAGE_TOL_MV = 1.0e-9;

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
    const int n = net_n(c);

    // ---- 2. CPU reference (timed) -----------------------------------------
    SimResult res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    simulate_cpu(c, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU simulation (kernel-loop timed) ----------------------------
    SimResult res_gpu;
    float gpu_kernel_ms = 0.0f;
    simulate_gpu(c, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    // (a) exact integer checks: total, per-step, per-neuron spike counts.
    bool counts_match = (res_cpu.total_spikes == res_gpu.total_spikes);
    long long per_neuron_mismatches = 0, per_step_mismatches = 0;
    for (int i = 0; i < n; ++i)
        if (res_cpu.spikes_per_neuron[i] != res_gpu.spikes_per_neuron[i]) ++per_neuron_mismatches;
    for (int t = 0; t < c.steps; ++t)
        if (res_cpu.spikes_per_step[t] != res_gpu.spikes_per_step[t]) ++per_step_mismatches;
    counts_match = counts_match && (per_neuron_mismatches == 0) && (per_step_mismatches == 0);

    // (b) voltages: should agree to round-off.
    double worst_v = 0.0;
    for (int i = 0; i < n; ++i)
        worst_v = std::fmax(worst_v, std::fabs(res_cpu.final_v[i] - res_gpu.final_v[i]));

    const bool pass = counts_match && (worst_v <= VOLTAGE_TOL_MV);

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    // A compact, reproducible fingerprint of the network's behaviour: sizes, the
    // total spike count, and a down-sampled spike raster (population count at a
    // handful of evenly-spaced timesteps) plus the most-active neurons.
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("network: %d neurons (%d exc + %d inh), out_degree=%d, %d steps @ dt=%.2f ms (%.1f ms)\n",
                n, c.n_exc, c.n_inh, c.out_degree, c.steps, c.np.dt, c.steps * c.np.dt);
    std::printf("weights: w_exc=%.3f  w_inh=%.3f  ext_kick=%.3f every %d\n",
                c.w_exc, c.w_inh, c.ext_kick, c.ext_every);
    std::printf("total spikes (GPU) = %lld   mean rate = %.3f Hz\n",
                res_gpu.total_spikes, res_gpu.mean_rate_hz);

    // Down-sampled population raster: print the spike count at 8 evenly-spaced
    // steps so the demo shows the network's temporal activity deterministically.
    std::printf("population spike raster (step: count):\n ");
    const int n_ticks = (c.steps < 8) ? c.steps : 8;
    for (int s = 0; s < n_ticks; ++s) {
        const int t = (n_ticks == 1) ? 0 : (s * (c.steps - 1)) / (n_ticks - 1);
        std::printf(" [%3d:%3d]", t, res_gpu.spikes_per_step[t]);
    }
    std::printf("\n");

    // Top-5 most active neurons, ordered by (spike count DESC, then id ASC) so the
    // tie-break is deterministic -- a stable fingerprint of which cells the
    // dynamics drives hardest. Small n, so a partial selection sort is plenty.
    std::printf("most active neurons (id:spikes:type):\n ");
    {
        std::vector<std::pair<int,int>> cd(n);   // (spike count, neuron id)
        for (int i = 0; i < n; ++i) cd[i] = std::make_pair(res_gpu.spikes_per_neuron[i], i);
        const int top = (n < 5) ? n : 5;
        for (int r = 0; r < top; ++r) {
            int bi = r;
            for (int j = r + 1; j < n; ++j) {
                const bool better = (cd[j].first > cd[bi].first) ||
                                    (cd[j].first == cd[bi].first && cd[j].second < cd[bi].second);
                if (better) bi = j;
            }
            std::swap(cd[r], cd[bi]);
            const int id = cd[r].second, sp = cd[r].first;
            std::printf(" [%d:%d:%s]", id, sp, is_excitatory(c, id) ? "E" : "I");
        }
        std::printf("\n");
    }

    std::printf("RESULT: %s (GPU spike counts match CPU exactly; final V within %.0e mV)\n",
                pass ? "PASS" : "FAIL", VOLTAGE_TOL_MV);

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d neurons, %lld synapses)\n",
                 path.c_str(), n, (long long)n * c.out_degree);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernels: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- tiny nets are launch-bound (3 kernels/step); "
                         "the GPU's edge grows toward 10^6-neuron networks.\n");
    std::fprintf(stderr, "[verify] spike-count exact match: %s (per-neuron mismatches=%lld, per-step=%lld)\n",
                 counts_match ? "yes" : "NO", per_neuron_mismatches, per_step_mismatches);
    std::fprintf(stderr, "[verify] worst final-V diff = %.3e mV  (tolerance %.0e)\n", worst_v, VOLTAGE_TOL_MV);

    return pass ? 0 : 1;
}
