// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial ensemble integration (CPU baseline)
// ---------------------------------------------------------------------------
// Project 1.35 : QMMM/ML Potential Hybrid MD   (reduced-scope teaching version)
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- a single readable loop, no parallelism -- so that
//   when the GPU and CPU agree, we believe the GPU. All the physics lives in
//   nnpmm.h as __host__ __device__ inline functions, so this reference and the
//   GPU kernel run identical math; this file just (a) parses the tiny config and
//   (b) loops run_trajectory() over the ensemble -- the embarrassingly-parallel
//   loop the GPU kernel turns into one-thread-per-trajectory.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// load_ensemble: read "M dt steps amp" from a whitespace-separated text file.
// Fails LOUDLY (throws) on a missing/garbled file so the demo never silently
// runs on empty input. The parsed fields are validated for sane ranges.
EnsembleConfig load_ensemble(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open ensemble file: " + path);
    EnsembleConfig c;
    if (!(in >> c.M >> c.dt >> c.steps >> c.amp))
        throw std::runtime_error(
            "bad parameters (expected 'M dt steps amp') in " + path);
    if (c.M <= 0 || c.steps <= 0 || c.dt <= 0.0)
        throw std::runtime_error("invalid ensemble parameters in " + path);
    return c;
}

// integrate_cpu: the serial reference. Each member is independent, so this is a
// plain for-loop here; kernels.cu gives each iteration its own GPU thread.
void integrate_cpu(const EnsembleConfig& c, std::vector<TrajResult>& results) {
    const int M = ensemble_size(c);
    results.assign(M, TrajResult{});
    for (int idx = 0; idx < M; ++idx)
        results[idx] = run_trajectory(idx, M, c.amp, c.dt, c.steps);
}
