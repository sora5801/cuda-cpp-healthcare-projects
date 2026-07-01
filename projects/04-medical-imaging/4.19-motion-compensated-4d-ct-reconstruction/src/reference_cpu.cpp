// ===========================================================================
// src/reference_cpu.cpp  --  Loader, ramp filter, serial 4D-CT reconstruction
// ---------------------------------------------------------------------------
// Project 4.19 : Motion-Compensated 4D-CT Reconstruction (2-D teaching version)
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. Written to be
//   OBVIOUSLY correct -- a single readable loop over pixels calling the shared
//   mc_pixel() physics from mc4dct.h -- so that when the GPU and CPU agree, we
//   believe the GPU. The per-pixel arithmetic is IDENTICAL to the kernel because
//   both call the same __host__ __device__ functions (PATTERNS.md section 2).
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, mc4dct.h. Compare with kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::cos, std::sin, std::floor
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_4dct: parse the text sinogram in data/README.md format.
//   header: "<img> <n_det> <n_phases> <n_ang_phase> <ds> <world_half> <amp>"
//   then (n_phases*n_ang_phase) rows of n_det floats (phase-major).
//   Throws on any malformed input so demos fail loudly, never on garbage.
// ---------------------------------------------------------------------------
FourDCTProblem load_4dct(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open 4D-CT sinogram file: " + path);

    FourDCTProblem prob;
    Geom& g = prob.geom;
    // Read the seven header fields in order. If ANY fails to parse, the header
    // is malformed and we cannot know the array shapes -> hard error.
    if (!(in >> g.img >> g.n_det >> g.n_phases >> g.n_ang_phase
             >> g.ds >> g.world_half >> g.amp))
        throw std::runtime_error(
            "bad header (want: img n_det n_phases n_ang_phase ds world_half amp) in " + path);

    if (g.img <= 0 || g.n_det <= 0 || g.n_phases <= 0 || g.n_ang_phase <= 0)
        throw std::runtime_error("non-positive geometry in " + path);

    const std::size_t rows  = static_cast<std::size_t>(g.n_phases) * g.n_ang_phase;
    const std::size_t cells = rows * static_cast<std::size_t>(g.n_det);
    prob.sino.resize(cells);
    for (std::size_t i = 0; i < cells; ++i) {
        if (!(in >> prob.sino[i]))
            throw std::runtime_error("sinogram truncated in " + path);
    }
    return prob;
}

// ---------------------------------------------------------------------------
// compute_trig: cos/sin of every GLOBAL projection angle, computed once.
//   Angles fill a uniform half-turn [0, pi) across ALL phases combined:
//     theta_k = k * pi / total_angles,  k = 0 .. total_angles-1.
//   Because the phases are INTERLEAVED (phase-major storage but angles assigned
//   round-robin at generation time), each individual phase samples only a sparse
//   subset of [0, pi) -> that sparsity is the under-sampling MCR overcomes.
//   Done in double then cast, matching mc4dct.h's cosf_portable exactly.
// ---------------------------------------------------------------------------
void compute_trig(const FourDCTProblem& prob,
                  std::vector<float>& cosv, std::vector<float>& sinv) {
    const int total = prob.total_angles();
    cosv.resize(total);
    sinv.resize(total);
    for (int k = 0; k < total; ++k) {
        const double theta = MC_PI * k / total;   // uniform over [0, pi)
        cosv[k] = static_cast<float>(std::cos(theta));
        sinv[k] = static_cast<float>(std::sin(theta));
    }
}

// ---------------------------------------------------------------------------
// ramp_filter: Ram-Lak (spatial-domain) ramp filter on each projection row.
//   FBP theory: backprojection alone reconstructs a 1/r-blurred image; the ramp
//   filter |w| in frequency (the discrete Ram-Lak kernel h in space) undoes that
//   blur. We convolve each row directly (O(n_det^2) per row) because n_det is
//   tiny here; production code filters via FFT. IDENTICAL to flagship 4.01, and
//   applied to BOTH reconstructions so filtering is not the variable under test.
// ---------------------------------------------------------------------------
void ramp_filter(const FourDCTProblem& prob, std::vector<float>& filtered) {
    const int    n  = prob.geom.n_det;
    const double ds = prob.geom.ds;

    // Discrete Ram-Lak kernel h[lag] (spatial domain):
    //   h[0]         =  1 / (4 ds^2)
    //   h[even != 0] =  0
    //   h[odd]       = -1 / (pi^2 lag^2 ds^2)
    auto hker = [ds](int lag) -> double {
        if (lag == 0)     return 1.0 / (4.0 * ds * ds);
        if (lag % 2 == 0) return 0.0;
        return -1.0 / (MC_PI * MC_PI * static_cast<double>(lag) * lag * ds * ds);
    };

    const std::size_t rows = static_cast<std::size_t>(prob.geom.n_phases) * prob.geom.n_ang_phase;
    filtered.assign(prob.sino.size(), 0.0f);
    for (std::size_t r = 0; r < rows; ++r) {
        const float* row = &prob.sino[r * n];
        float*       out = &filtered[r * n];
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            // Direct convolution: sum row[jp] * h[j-jp] over all detector bins.
            for (int jp = 0; jp < n; ++jp)
                acc += static_cast<double>(row[jp]) * hker(j - jp);
            out[j] = static_cast<float>(acc * ds);   // ds = integration measure
        }
    }
}

// ---------------------------------------------------------------------------
// reconstruct_cpu: loop mc_pixel() over every output pixel.
//   The ONLY thing that changes between naive and motion-compensated is the
//   `motion_comp` flag passed straight through to mc_pixel() (which either does
//   or does not displace each pixel by the phase DVF). This function is
//   deliberately trivial -- all the physics is in the shared header.
//   Complexity: O(img^2 * total_angles) -- serial, the baseline the GPU beats.
// ---------------------------------------------------------------------------
void reconstruct_cpu(const FourDCTProblem& prob, const std::vector<float>& filtered,
                     const std::vector<float>& cosv, const std::vector<float>& sinv,
                     int motion_comp, std::vector<float>& image) {
    const Geom& g = prob.geom;
    image.assign(static_cast<std::size_t>(g.img) * g.img, 0.0f);
    for (int py = 0; py < g.img; ++py) {
        for (int px = 0; px < g.img; ++px) {
            image[static_cast<std::size_t>(py) * g.img + px] =
                mc_pixel(px, py, g, cosv.data(), sinv.data(), filtered.data(), motion_comp);
        }
    }
}

// ---------------------------------------------------------------------------
// image_sharpness: mean squared gradient magnitude (a blur-o-meter).
//   For each interior pixel we take forward differences gx, gy and average
//   gx^2 + gy^2 over the image. A crisp, well-registered reconstruction has
//   strong edges -> HIGH value; a motion-blurred one has soft edges -> LOW
//   value. We report this deterministic scalar for both reconstructions so the
//   learner can watch motion compensation raise it. Double accumulation keeps
//   the sum order-independent and reproducible.
// ---------------------------------------------------------------------------
double image_sharpness(const std::vector<float>& image, int img) {
    double sum = 0.0;
    long long count = 0;
    for (int py = 0; py < img - 1; ++py) {
        for (int px = 0; px < img - 1; ++px) {
            const std::size_t i = static_cast<std::size_t>(py) * img + px;
            const double gx = static_cast<double>(image[i + 1]) - image[i];
            const double gy = static_cast<double>(image[i + img]) - image[i];
            sum += gx * gx + gy * gy;
            ++count;
        }
    }
    return (count > 0) ? sum / static_cast<double>(count) : 0.0;
}

// ---------------------------------------------------------------------------
// image_peak: the largest reconstructed pixel value (and where it is).
//   A plain deterministic argmax over the image. The moving nodule is the
//   brightest feature by construction, so this reports how well its true density
//   (1.0) is recovered: motion smears it (naive peak < 1), motion compensation
//   re-focuses it (MCR peak ~= 1). Ties keep the first (lowest-index) pixel so
//   the result is order-independent and reproducible.
// ---------------------------------------------------------------------------
float image_peak(const std::vector<float>& image, int img, int* peak_px, int* peak_py) {
    float vmax = image.empty() ? 0.0f : image[0];
    std::size_t amax = 0;
    for (std::size_t i = 1; i < image.size(); ++i) {
        if (image[i] > vmax) { vmax = image[i]; amax = i; }
    }
    if (peak_px) *peak_px = static_cast<int>(amax % img);
    if (peak_py) *peak_py = static_cast<int>(amax / img);
    return vmax;
}
