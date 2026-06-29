// ===========================================================================
// src/umbrella.h  --  Shared (host + device) physics: RNG, potential, biased
//                     Langevin dynamics, and one window's histogram simulation
// ---------------------------------------------------------------------------
// Project 1.24 : Umbrella Sampling / WHAM Free Energy Profiles
//
// WHY THIS HEADER IS SHARED (the single most important idea, PATTERNS.md section 2)
//   Umbrella sampling here is verified two ways. The cheap, exact check is
//   "GPU histogram == CPU histogram, bit-for-bit". That is only possible if the
//   CPU reference and the GPU kernel run the *identical* simulation: same RNG,
//   same Langevin update, same binning. So ALL of that physics lives in ONE
//   header, compiled by the plain host compiler (for reference_cpu.cpp) AND by
//   nvcc (for kernels.cu / main.cu). The US_HD macro expands to
//   `__host__ __device__` under nvcc and to nothing under the host compiler, so
//   the same inline functions compile in both worlds.
//
//   Counting positions into bins is integer work, and integer adds commute, so
//   the per-window histograms are deterministic and CPU-identical regardless of
//   thread scheduling (PATTERNS.md section 3, the "accumulate in integers" rule).
//
// THE SCIENCE WE ARE TEACHING (THEORY.md has the full story)
//   A molecule moving along a reaction coordinate xi (e.g. a ligand's distance
//   from a binding pocket, or an ion's depth in a channel) feels a free-energy
//   landscape U(xi). The quantity we want is the POTENTIAL OF MEAN FORCE (PMF)
//   F(xi) = -kT ln p(xi), where p(xi) is the equilibrium probability of being at
//   xi. The trouble: if U has a tall barrier, plain dynamics almost never crosses
//   it, so p(xi) on top of the barrier is never sampled -> no PMF there.
//
//   UMBRELLA SAMPLING fixes this. We run many independent simulations ("windows"),
//   each with an extra HARMONIC BIAS  w_k(xi) = 1/2 k_spring (xi - x0_k)^2  that
//   tethers the system near a different center x0_k. Window k therefore samples a
//   little neighbourhood around x0_k even if that sits on the barrier. We collect
//   one histogram of visited xi per window. WHAM (Weighted Histogram Analysis
//   Method, see reference_cpu / main) then stitches all the biased histograms
//   back into the single unbiased PMF F(xi).
//
//   In THIS teaching version the underlying landscape is a KNOWN analytic
//   double-well U(xi) (two minima separated by a barrier -- the simplest model of
//   a bound vs. unbound state). Because we know the true U, we can check that the
//   WHAM-reconstructed PMF actually recovers it. Real umbrella sampling runs full
//   molecular dynamics inside each window; we use overdamped Langevin dynamics on
//   a 1-D coordinate, which is the same statistical-mechanics skeleton without the
//   thousands of atoms (THEORY.md section "Where this sits in the real world").
//
//   *** Educational only. Synthetic potential. Not a real molecule, not clinical. ***
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh.
// READ THIS BEFORE: reference_cpu.h, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <cstdint>   // uint64_t  (fixed-width RNG state, identical host/device)
#include <cmath>     // std::sqrt, std::log, std::cos  (device intrinsics under nvcc)

// US_HD: "umbrella-sampling host+device". Under nvcc (__CUDACC__ defined) the
// shared functions get both decorators so one definition serves the CPU loop and
// the GPU thread. Under the host compiler the decorators do not exist, so the
// macro vanishes and the same code is plain C++.
#ifdef __CUDACC__
#define US_HD __host__ __device__
#else
#define US_HD
#endif

// Boltzmann-scale energy unit. We work in REDUCED units where kT = 1, so all
// energies/PMFs are measured in units of kT (the natural unit for free energy).
// This keeps the numbers clean and dimensionless; THEORY.md explains how a real
// run carries kcal/mol or kJ/mol instead. Defining it once here guarantees the
// bias force, the acceptance of moves, and the WHAM solver all agree on kT.
#define US_KT 1.0

// ---------------------------------------------------------------------------
// THE RANDOM NUMBER GENERATOR  (counter-based splitmix64 -> Gaussian via Box-Muller)
//   Langevin dynamics is driven by a random thermal "kick" each step. For the
//   CPU and GPU to produce identical trajectories we need identical random
//   numbers, so we use a small, fully-deterministic, stateless-seedable RNG that
//   compiles the same on both sides (production GPU MD uses cuRAND; we want
//   bit-reproducibility for teaching -- THEORY.md "Numerical considerations").
// ---------------------------------------------------------------------------
struct Rng { uint64_t state; };   // the entire RNG is one 64-bit word

// One splitmix64 step: advance `x` and return a thoroughly mixed 64-bit value.
// splitmix64 is a well-known, high-quality finalizer; one multiply-xor-shift
// chain decorrelates successive outputs. It is the same on host and device.
US_HD inline uint64_t us_splitmix64(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;                       // golden-ratio increment
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Seed an INDEPENDENT stream for window `window` from a global `base` seed, so
// every window's trajectory is uncorrelated yet exactly reproducible from
// (base, window). Mixing the window index into the state (rather than just
// advancing a shared counter) is what lets each GPU thread own a private stream.
US_HD inline Rng rng_seed(uint64_t base, uint64_t window) {
    Rng r;
    r.state = base ^ (window * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL);
    us_splitmix64(r.state);   // warm up so nearby windows start well-separated
    return r;
}

// Uniform double in (0,1] from 53 random bits. We return (0,1] rather than [0,1)
// because Box-Muller below takes log(u): u must never be 0.
US_HD inline double rng_uniform(Rng& r) {
    uint64_t z = us_splitmix64(r.state);
    // (z >> 11) is a 53-bit integer in [0, 2^53); +1 then scale -> (0,1].
    return ((z >> 11) + 1) * (1.0 / 9007199254740993.0);   // 1 / (2^53 + 1)
}

// One standard-normal sample N(0,1) via the Box-Muller transform.
//   Given two uniforms u1,u2 in (0,1],  z = sqrt(-2 ln u1) cos(2 pi u2)  is
//   exactly standard-normal. We use only the cosine branch (discarding the sine
//   partner) for simplicity -- it still consumes a fixed, reproducible number of
//   RNG draws per call, which keeps host and device in lock-step.
US_HD inline double rng_gaussian(Rng& r) {
    const double TWO_PI = 6.283185307179586476925286766559;
    double u1 = rng_uniform(r);
    double u2 = rng_uniform(r);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(TWO_PI * u2);
}

// ---------------------------------------------------------------------------
// THE TRUE (UNBIASED) LANDSCAPE  --  a symmetric double-well
//   U(x) = A (x^2 - b^2)^2 / b^4
//   Two minima at x = -b and x = +b (where U = 0), separated by a barrier of
//   height A at x = 0. This is the canonical 1-D model of a two-state system
//   (e.g. "bound" vs "unbound"): the barrier is exactly what plain dynamics
//   cannot cross, which is *why* umbrella sampling exists. Parameters A,b come
//   from the data file so the demo can change the landscape without recompiling.
// ---------------------------------------------------------------------------
struct Potential {
    double A;   // barrier height at x=0, in units of kT
    double b;   // half-separation of the wells (minima at +/- b)
};

// The potential energy U(x) of the bare landscape (no bias). Used by WHAM's
// verification (we compare the reconstructed PMF to this) and, with the bias
// added, by the Langevin force below.
US_HD inline double potential_U(const Potential& p, double x) {
    const double t = x * x - p.b * p.b;          // (x^2 - b^2)
    return p.A * t * t / (p.b * p.b * p.b * p.b); // A (x^2-b^2)^2 / b^4
}

// The force from the BARE landscape, F = -dU/dx.
//   dU/dx = A * 2(x^2 - b^2) * 2x / b^4 = 4 A x (x^2 - b^2) / b^4
// so the bare force is the negative of that. The bias force is added separately
// in the integrator so the same potential_force() serves both the dynamics and
// any future diagnostic.
US_HD inline double potential_force(const Potential& p, double x) {
    const double t = x * x - p.b * p.b;
    const double dUdx = 4.0 * p.A * x * t / (p.b * p.b * p.b * p.b);
    return -dUdx;
}

// ---------------------------------------------------------------------------
// HISTOGRAM GEOMETRY  --  shared bin layout so every window, the CPU, the GPU,
// and WHAM all agree on "which bin is xi in".
//   The reaction coordinate is gridded into `nbins` equal bins spanning
//   [x_min, x_max). bin_width = (x_max - x_min)/nbins. bin_center(i) is the xi
//   value WHAM associates with bin i.
// ---------------------------------------------------------------------------
struct HistGrid {
    double x_min;   // left edge of bin 0
    double x_max;   // right edge of the last bin
    int    nbins;   // number of bins along the reaction coordinate
};

US_HD inline double grid_bin_width(const HistGrid& g) {
    return (g.x_max - g.x_min) / g.nbins;
}

// The xi value at the CENTER of bin i (where WHAM evaluates the PMF).
US_HD inline double grid_bin_center(const HistGrid& g, int i) {
    return g.x_min + (i + 0.5) * grid_bin_width(g);
}

// Map a coordinate x to its bin index, or -1 if it falls outside [x_min,x_max).
// Out-of-range samples are simply not counted (they carry no PMF information for
// our fixed grid); returning -1 makes the caller's "drop it" branch explicit.
US_HD inline int grid_bin_of(const HistGrid& g, double x) {
    if (x < g.x_min || x >= g.x_max) return -1;
    int i = static_cast<int>((x - g.x_min) / grid_bin_width(g));
    if (i < 0) i = 0;                       // guard against round-off at the edge
    if (i >= g.nbins) i = g.nbins - 1;
    return i;
}

// ---------------------------------------------------------------------------
// ONE WINDOW'S SIMULATION  --  overdamped (Brownian) Langevin dynamics under a
// harmonic umbrella bias, accumulating a histogram of visited bins.
//
//   Overdamped Langevin (the high-friction limit of Newtonian dynamics, valid for
//   a coordinate immersed in solvent) advances the coordinate by:
//
//       x_{n+1} = x_n + (D/kT) * F_total(x_n) * dt + sqrt(2 D dt) * N(0,1)
//
//   where D is the diffusion constant, F_total = F_bare(x) + F_bias(x), and
//   F_bias(x) = -k_spring (x - x0) pulls toward the window center x0. The first
//   term is the deterministic drift down the (biased) gradient; the second is the
//   random thermal kick that, by the fluctuation-dissipation theorem with
//   variance 2 D dt, makes the long-run distribution proportional to
//   exp(-(U(x)+w(x))/kT). That biased Boltzmann distribution is exactly what the
//   histogram estimates, and what WHAM unbiases. (THEORY.md derives this.)
//
//   The function writes the per-bin counts into `hist` (length g.nbins). On the
//   GPU the histogram lives in this thread's slice of global memory and is
//   written with plain integer stores (one window per thread -> no contention);
//   the CPU reference does the same with a plain array. Both increment the same
//   integer counts in the same order -> identical histograms.
//
//   Returns the number of samples that landed inside the grid (for diagnostics).
// ---------------------------------------------------------------------------
struct WindowSpec {
    double x0;         // umbrella restraint center (reaction-coordinate value)
    double k_spring;   // harmonic spring constant of the bias (units of kT / x^2)
};

US_HD inline long long simulate_window(const Potential& pot,
                                       const HistGrid&  grid,
                                       const WindowSpec& win,
                                       double D,            // diffusion constant
                                       double dt,           // Langevin timestep
                                       int    n_equil,      // discarded warm-up steps
                                       int    n_sample,     // recorded steps
                                       uint64_t base_seed,
                                       int    window_index, // for the RNG stream
                                       unsigned int* hist)  // [grid.nbins] counts, zeroed by caller
{
    // Private, reproducible RNG stream for this window.
    Rng rng = rng_seed(base_seed, static_cast<uint64_t>(window_index));

    // Start the walker AT the restraint center: the bias makes this the most
    // probable region, so we begin already near equilibrium for this window.
    double x = win.x0;

    // Precompute the constants of the update so the hot loop is cheap.
    const double drift_coeff = D / US_KT;            // multiplies the force
    const double noise_coeff = std::sqrt(2.0 * D * dt);  // std-dev of the kick

    long long counted = 0;

    // --- Equilibration: run the dynamics but record NOTHING. This lets the
    //     walker forget its artificial starting point and reach the window's
    //     biased equilibrium before we start collecting statistics. ---
    for (int s = 0; s < n_equil; ++s) {
        double f_bare = potential_force(pot, x);
        double f_bias = -win.k_spring * (x - win.x0);    // harmonic restraint force
        x += drift_coeff * (f_bare + f_bias) * dt + noise_coeff * rng_gaussian(rng);
    }

    // --- Sampling: same dynamics, now histogram every step's position. ---
    for (int s = 0; s < n_sample; ++s) {
        double f_bare = potential_force(pot, x);
        double f_bias = -win.k_spring * (x - win.x0);
        x += drift_coeff * (f_bare + f_bias) * dt + noise_coeff * rng_gaussian(rng);

        int bin = grid_bin_of(grid, x);
        if (bin >= 0) {                 // ignore the rare excursion outside the grid
            hist[bin] += 1u;            // integer increment -> deterministic, exact
            ++counted;
        }
    }
    return counted;
}
