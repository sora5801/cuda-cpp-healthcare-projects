// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ baseline we trust
// ---------------------------------------------------------------------------
// Project 4.13 : Photoacoustic Image Reconstruction
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- a single readable double loop over pixels, no
//   parallelism, no cleverness -- so that when the GPU and CPU agree we believe
//   the GPU. The per-pixel math itself is NOT duplicated here: it lives in
//   pa_core.h (pa_pixel_das), which the GPU kernel also calls, so the two match
//   exactly rather than merely approximately.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: pa_core.h, reference_cpu.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>     // std::ifstream for the text loader
#include <stdexcept>   // std::runtime_error on malformed input

// ---------------------------------------------------------------------------
// load_pa: read the PAProblem text format (see data/README.md).
//   The format is deliberately human-readable whitespace-separated numbers so
//   the tiny committed sample is inspectable and make_synthetic.py can emit it.
// ---------------------------------------------------------------------------
PAProblem load_pa(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open PA data file: " + path);

    PAProblem pa;
    // Header line: geometry + acquisition parameters, in the fixed order below.
    if (!(in >> pa.n_sensors >> pa.n_samples >> pa.dt >> pa.c >> pa.img >> pa.world_half))
        throw std::runtime_error(
            "bad header (expected: n_sensors n_samples dt c img world_half) in " + path);
    if (pa.n_sensors <= 0 || pa.n_samples <= 0 || pa.img <= 0)
        throw std::runtime_error("non-positive geometry in " + path);
    if (pa.dt <= 0.0f || pa.c <= 0.0f)
        throw std::runtime_error("dt and c must be positive in " + path);

    // Sensor positions: n_sensors pairs (sx, sy) in metres.
    pa.sx.resize(pa.n_sensors);
    pa.sy.resize(pa.n_sensors);
    for (int s = 0; s < pa.n_sensors; ++s) {
        if (!(in >> pa.sx[s] >> pa.sy[s]))
            throw std::runtime_error("sensor positions truncated in " + path);
    }

    // Pressure traces: n_sensors * n_samples floats, sensor-major (row per sensor).
    pa.sig.resize(static_cast<std::size_t>(pa.n_sensors) * pa.n_samples);
    for (std::size_t k = 0; k < pa.sig.size(); ++k) {
        if (!(in >> pa.sig[k]))
            throw std::runtime_error("pressure traces truncated in " + path);
    }
    return pa;
}

// ---------------------------------------------------------------------------
// reconstruct_cpu: serial delay-and-sum over every pixel.
//   Complexity: O(img^2 * n_sensors) -- one inner DAS loop per pixel. On the GPU
//   the outer img^2 loop becomes the thread grid; the inner sensor loop stays.
//   We precompute the reciprocals 1/c, 1/dt, 1/n_sensors ONCE (identically to
//   the kernel) so pa_pixel_das gets the same operands on both sides -- their
//   results then match to ~1e-5 (the GPU fuses multiply-adds; PATTERNS.md §4).
// ---------------------------------------------------------------------------
void reconstruct_cpu(const PAProblem& pa, std::vector<float>& image) {
    const int   N = pa.img;
    const float W = pa.world_half;
    // Pixel spacing so that pixel 0 sits at -W and pixel N-1 at +W (metres).
    const float pix = (N > 1) ? (2.0f * W / (N - 1)) : 0.0f;
    // Reciprocals computed the SAME way the kernel computes them (see kernels.cu)
    // so both paths feed pa_pixel_das identical float operands.
    const float inv_c  = 1.0f / pa.c;
    const float inv_dt = 1.0f / pa.dt;
    const float inv_ns = 1.0f / static_cast<float>(pa.n_sensors);

    image.assign(static_cast<std::size_t>(N) * N, 0.0f);
    const float* sx  = pa.sx.data();
    const float* sy  = pa.sy.data();
    const float* sig = pa.sig.data();

    for (int py = 0; py < N; ++py) {
        const float wy = -W + py * pix;             // world y of this pixel row
        for (int px = 0; px < N; ++px) {
            const float wx = -W + px * pix;         // world x of this pixel
            // Delegate the actual physics to the shared core so CPU == GPU.
            image[static_cast<std::size_t>(py) * N + px] =
                pa_pixel_das(wx, wy, sx, sy, sig, pa.n_sensors, pa.n_samples,
                             inv_c, inv_dt, inv_ns);
        }
    }
}
