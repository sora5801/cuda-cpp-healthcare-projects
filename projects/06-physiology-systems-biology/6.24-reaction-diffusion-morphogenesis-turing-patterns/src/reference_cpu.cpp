// ===========================================================================
// src/reference_cpu.cpp  --  Loader, deterministic seeding, serial RD reference
// ---------------------------------------------------------------------------
// Project 6.24 : Reaction-Diffusion Morphogenesis (Turing Patterns)
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is written to be
//   OBVIOUSLY correct -- plain nested loops, no parallelism, no cleverness -- so
//   that when the GPU and CPU agree we believe the GPU. Compiled by the host C++
//   compiler only (no CUDA here). The per-cell physics is the SHARED tu_update()
//   in turing.h, so this baseline and the kernel do identical arithmetic.
//
// READ THIS AFTER: turing.h and reference_cpu.h. Compare with kernels.cu (twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cstdint>      // std::uint32_t for the reproducible hash
#include <fstream>      // std::ifstream to read the parameter file
#include <stdexcept>    // std::runtime_error for bad input
#include <utility>      // std::swap for the ping-pong buffer exchange

// ---------------------------------------------------------------------------
// load_params  --  parse the one-line whitespace-separated parameter file.
//
// Format (see data/README.md):
//   nx ny Da Dh rho mu_a mu_h rho_a dt steps noise_seed
// We validate the physically-meaningful invariants so a typo fails loudly here
// rather than silently producing garbage or NaNs deep in the time loop.
// ---------------------------------------------------------------------------
TuringParams load_params(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open parameter file: " + path);

    TuringParams P{};
    if (!(in >> P.nx >> P.ny >> P.Da >> P.Dh >> P.rho
             >> P.mu_a >> P.mu_h >> P.rho_a >> P.dt >> P.steps >> P.noise_seed)) {
        throw std::runtime_error(
            "bad parameters (expected 'nx ny Da Dh rho mu_a mu_h rho_a dt "
            "steps noise_seed') in " + path);
    }
    // Sanity checks. nx,ny > 2 so the 5-point stencil has real neighbours; the
    // rates/diffusions must be non-negative; dt > 0; steps >= 0.
    if (P.nx <= 2 || P.ny <= 2 || P.steps < 0 || P.dt <= 0.0)
        throw std::runtime_error("invalid grid/timestep parameters in " + path);
    if (P.Da < 0.0 || P.Dh < 0.0 || P.rho < 0.0 || P.mu_a < 0.0 || P.mu_h < 0.0)
        throw std::runtime_error("negative rate/diffusion parameter in " + path);
    return P;
}

// ---------------------------------------------------------------------------
// tu_perturbation  --  a tiny, reproducible per-cell noise value.
//
// We must break the uniform state to let a pattern nucleate, but we also need
// the SAME initial field on every machine, every run, and on both CPU and GPU
// paths (so their comparison is exact). rand() gives none of that. Instead we
// hash the integer triple (x, y, seed) with a splitmix-style bit mixer and map
// the result to about [-amp, +amp]. Pure function -> perfectly reproducible.
//
//   x, y : cell coordinates
//   seed : the run's noise_seed (lets the learner get a different pattern)
//   amp  : perturbation amplitude (small, e.g. 1e-2 of the baseline)
// Returns a deterministic value in roughly [-amp, +amp].
// ---------------------------------------------------------------------------
double tu_perturbation(int x, int y, int seed, double amp) {
    // Combine the three integers into one 32-bit key, then avalanche the bits so
    // neighbouring cells get uncorrelated noise (a good hash, not a good RNG --
    // but for a static seed that is exactly what we want).
    std::uint32_t k = static_cast<std::uint32_t>(x) * 73856093u
                    ^ static_cast<std::uint32_t>(y) * 19349663u
                    ^ static_cast<std::uint32_t>(seed) * 83492791u;
    k ^= k >> 16; k *= 0x7feb352dU;    // splitmix32 mixing constants
    k ^= k >> 15; k *= 0x846ca68bU;
    k ^= k >> 16;
    // Map the 32-bit hash to [0,1), then to [-amp, +amp]. The 4294967296.0 =
    // 2^32 divisor makes this exact and identical across compilers.
    const double unit = static_cast<double>(k) / 4294967296.0;  // [0,1)
    return amp * (2.0 * unit - 1.0);                            // [-amp, +amp)
}

// ---------------------------------------------------------------------------
// init_fields  --  build the near-uniform, deterministically-perturbed seed.
//
// Both fields start AT the homogeneous steady state (a*, h*): the inhibitor is
// flat at h*, the activator is a* PLUS a tiny hash-noise perturbation (about 1%
// of a*). This is exactly the setup Turing analyzed: a spatially-uniform
// equilibrium that is stable to uniform perturbations but UNSTABLE to a band of
// spatial modes when Dh >> Da. The noise self-amplifies at the fastest-growing
// wavelength (predicted by the dispersion relation in main.cu) and organizes
// into a pattern; without any noise the exact equilibrium would sit forever.
// ---------------------------------------------------------------------------
void init_fields(const TuringParams& P, std::vector<double>& a, std::vector<double>& h) {
    const int N = P.nx * P.ny;
    const double a_star = tu_baseline_activator(P);         // steady activator a*
    const double h_star = tu_baseline_inhibitor(P, a_star); // matching inhibitor h*
    const double amp    = 0.01 * a_star;                    // 1% perturbation amplitude
    a.assign(N, a_star);
    h.assign(N, h_star);
    for (int y = 0; y < P.ny; ++y)
        for (int x = 0; x < P.nx; ++x)
            a[y * P.nx + x] = a_star + tu_perturbation(x, y, P.noise_seed, amp);
}

// ---------------------------------------------------------------------------
// simulate_cpu  --  the serial reference time loop (double-buffered Euler).
//
// Classic ping-pong: read from the "source" buffers, write the next state to the
// "destination" buffers, then swap the roles. Using two buffers means every cell
// reads the FROZEN previous state -- exactly what the PDE's simultaneous update
// requires, and exactly what the GPU kernel does with two device buffers.
//
// After the loop, whichever buffers hold the latest state are copied back into
// the caller's a/h if necessary (odd step counts leave the result in the private
// scratch buffers).
// ---------------------------------------------------------------------------
void simulate_cpu(const TuringParams& P, std::vector<double>& a, std::vector<double>& h) {
    const int N = P.nx * P.ny;
    std::vector<double> ab(N), hb(N);         // private "next-state" scratch buffers
    double* as = a.data();  double* hs = h.data();   // source (current state)
    double* ad = ab.data(); double* hd = hb.data();  // destination (next state)

    for (int s = 0; s < P.steps; ++s) {
        // Sweep every cell once; each writes only its own (x,y) in the dest.
        for (int y = 0; y < P.ny; ++y)
            for (int x = 0; x < P.nx; ++x)
                tu_update(x, y, P, as, hs, ad, hd);
        std::swap(as, ad);   // the just-written dest becomes next step's source
        std::swap(hs, hd);
    }
    // If the final state lives in the scratch buffers (odd `steps`), copy it back
    // so the caller (main.cu) sees the result in the vectors it passed in.
    if (as != a.data()) { a.assign(as, as + N); h.assign(hs, hs + N); }
}
