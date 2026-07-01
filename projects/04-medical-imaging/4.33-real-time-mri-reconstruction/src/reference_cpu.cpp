// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ radial-MRI gridding baseline we trust
// ---------------------------------------------------------------------------
// Project 4.33 : Real-Time MRI Reconstruction
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written to
//   be OBVIOUSLY correct -- a readable radix-2 FFT and a textbook gridding loop, no
//   parallelism, no cleverness -- so that when the GPU (kernels.cu, using cuFFT and
//   an atomic scatter) agrees with it, we believe the GPU. Every per-sample formula
//   (KB weight, density compensation, deapodization, fixed-point quantization) comes
//   from grid_core.h, the SAME header the kernels use, so the two paths differ only
//   in the FFT implementation (our radix-2 vs cuFFT) and in HOW they walk the samples
//   -- and because the gridding accumulator is FIXED-POINT (integer, associative),
//   even the scatter order cannot change the result.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h and grid_core.h. Compare with kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::sin, std::cos, std::sqrt, std::floor
#include <cstddef>     // std::size_t
#include <cstdint>     // (documentation) fixed-width reasoning; long long used directly
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error
#include <string>
#include <vector>

// M_PI is not guaranteed by the C++ standard; define our own to stay portable.
static const double PI = 3.14159265358979323846;

// ===========================================================================
// SECTION 1 -- Loading the sample
// ===========================================================================

// load_radial: read the text problem file. Format (see data/README.md):
//   line 1 : "<n> <n_spokes> <n_ro> <win> <stride> <n_frames> <kb_w> <has_truth>"
//   n_spokes*n_ro rows: "re im"           (spoke-major, readout-minor)
//   n*n rows (iff has_truth): "truth"      (row-major image order)
RadialData load_radial(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open radial file: " + path);

    RadialData d;
    int has_truth_flag = 0;
    // Header line drives every subsequent allocation.
    if (!(in >> d.n >> d.n_spokes >> d.n_ro >> d.win >> d.stride
             >> d.n_frames >> d.kb_w >> has_truth_flag))
        throw std::runtime_error("bad header in radial file: " + path);
    d.has_truth = (has_truth_flag != 0);

    // Guard: n must be a positive power of two (the radix-2 FFT requires it).
    if (d.n <= 0 || (d.n & (d.n - 1)) != 0)
        throw std::runtime_error("grid side n must be a power of two");
    if (d.n_spokes <= 0 || d.n_ro <= 0 || d.win <= 0 || d.n_frames <= 0)
        throw std::runtime_error("n_spokes, n_ro, win, n_frames must be positive");
    if (d.win > d.n_spokes)
        throw std::runtime_error("window (win) cannot exceed n_spokes");

    // The Kaiser-Bessel shape parameter is a deterministic function of the width, so
    // both the loader and make_synthetic.py agree without storing beta in the file.
    d.kb_beta = kb_beta_for_width(d.kb_w);

    // Read every (spoke, readout) complex sample, spoke-major.
    const std::size_t n_samp = static_cast<std::size_t>(d.n_spokes) * d.n_ro;
    d.samples.resize(n_samp);
    for (std::size_t i = 0; i < n_samp; ++i) {
        float re, im;
        if (!(in >> re >> im))
            throw std::runtime_error("radial file truncated at sample " + std::to_string(i));
        d.samples[i] = c_make(re, im);
    }

    // Optional ground-truth magnitude image (synthetic sample only).
    if (d.has_truth) {
        const std::size_t total = static_cast<std::size_t>(d.n) * d.n;
        d.truth.resize(total);
        for (std::size_t i = 0; i < total; ++i) {
            float t;
            if (!(in >> t))
                throw std::runtime_error("truth column truncated at index " + std::to_string(i));
            d.truth[i] = t;
        }
    }
    return d;
}

// ===========================================================================
// SECTION 2 -- Radial sample geometry (shared with the GPU via this definition)
// ===========================================================================

// sample_kpos: map (spoke s, readout j) -> Cartesian k-space position in grid cells,
//   with the k-space origin at the grid center (n/2). Readout offset ro = j - n_ro/2
//   runs the spoke from -n_ro/2 to +n_ro/2; we scale it so a full spoke spans the
//   grid diameter [-n/2, n/2], then rotate by the spoke's golden angle.
//     kx = center + r * cos(theta),  ky = center + r * sin(theta)
//   The GPU kernel calls THIS SAME geometry (via a device copy of the formula) so
//   both scatter each sample around the identical grid location.
void sample_kpos(int s, int j, int n_ro, int n, float& kx, float& ky) {
    const double center = 0.5 * n;                       // grid center (k-space origin)
    const double ro = static_cast<double>(j) - 0.5 * n_ro;   // signed readout offset
    // Scale the spoke so its FULL length spans the grid: a readout of n_ro samples
    // maps to a diameter of n cells (radius runs [-n/2, +n/2]).
    const double r = ro * (static_cast<double>(n) / static_cast<double>(n_ro));
    const double theta = golden_angle_rad(s);
    kx = static_cast<float>(center + r * std::cos(theta));
    ky = static_cast<float>(center + r * std::sin(theta));
}

// ===========================================================================
// SECTION 3 -- The reference FFT (radix-2 Cooley-Tukey, separable 2D)
// ---------------------------------------------------------------------------
// The SAME transform cuFFT computes, written out by hand -- the "no black box"
// rule (CLAUDE.md section 6). Internals are double precision for accuracy; results
// are stored back as float (Cplx) so the comparison against cuFFT (single precision)
// is apples-to-apples.
// ===========================================================================

// fft1_double: in-place 1D radix-2 FFT of a length-`n` complex signal (n a power of
//   two). sign = -1 is the forward transform (exp(-i..)), +1 the inverse (exp(+i..),
//   un-normalized). Iterative bit-reversal + butterfly form: O(n log n).
static void fft1_double(std::vector<double>& re, std::vector<double>& im, int n, int sign) {
    // --- Bit-reversal permutation so butterflies read contiguous pairs.
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;   // ripple-carry in reversed bit order
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    // --- Butterflies over doubling block lengths len = 2, 4, ..., n.
    for (int len = 2; len <= n; len <<= 1) {
        const double ang = sign * 2.0 * PI / len;              // twiddle angle step
        const double wr = std::cos(ang), wi = std::sin(ang);   // principal root W_len
        for (int i = 0; i < n; i += len) {
            double cr = 1.0, ci = 0.0;                         // running twiddle W_len^k
            for (int k = 0; k < len / 2; ++k) {
                const int a = i + k, b = i + k + len / 2;
                const double ur = re[a],           ui = im[a];
                const double vr = re[b] * cr - im[b] * ci;     // twiddled odd part
                const double vi = re[b] * ci + im[b] * cr;
                re[a] = ur + vr; im[a] = ui + vi;
                re[b] = ur - vr; im[b] = ui - vi;
                const double ncr = cr * wr - ci * wi;          // advance twiddle
                ci = cr * wi + ci * wr; cr = ncr;
            }
        }
    }
}

// fft2_core: separable 2D FFT -- FFT every row, then FFT every column.
static void fft2_core(std::vector<Cplx>& data, int n, int sign) {
    std::vector<double> re(n), im(n);   // scratch for one row/column at a time
    for (int r = 0; r < n; ++r) {       // rows
        for (int c = 0; c < n; ++c) { re[c] = data[(std::size_t)r * n + c].re;
                                      im[c] = data[(std::size_t)r * n + c].im; }
        fft1_double(re, im, n, sign);
        for (int c = 0; c < n; ++c) data[(std::size_t)r * n + c] = c_make((float)re[c], (float)im[c]);
    }
    for (int c = 0; c < n; ++c) {       // columns
        for (int r = 0; r < n; ++r) { re[r] = data[(std::size_t)r * n + c].re;
                                      im[r] = data[(std::size_t)r * n + c].im; }
        fft1_double(re, im, n, sign);
        for (int r = 0; r < n; ++r) data[(std::size_t)r * n + c] = c_make((float)re[r], (float)im[r]);
    }
}

// fft2_cpu: forward 2D FFT, UN-normalized (matches cuFFT's CUFFT_FORWARD).
void fft2_cpu(std::vector<Cplx>& data, int n) { fft2_core(data, n, -1); }

// ifft2_cpu: inverse 2D FFT WITH the 1/(n*n) normalization, so ifft2(fft2(x))==x.
// cuFFT also leaves its inverse un-normalized, so kernels.cu applies the same scale.
void ifft2_cpu(std::vector<Cplx>& data, int n) {
    fft2_core(data, n, +1);
    const float inv = 1.0f / (static_cast<float>(n) * static_cast<float>(n));
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = c_scale(data[i], inv);
}

// ===========================================================================
// SECTION 4 -- Gridding (density comp + Kaiser-Bessel convolution)
// ---------------------------------------------------------------------------
// This is the NON-CARTESIAN part: each radial sample is SPREAD onto the ~W x W
// nearest Cartesian grid cells with the KB kernel, weighted by its density factor.
// We accumulate into a FIXED-POINT integer grid (grid_core.h to_fixed/from_fixed)
// so the result is order-independent -- identical to the GPU's atomic scatter and
// deterministic. The GPU kernel (kernels.cu) is the exact parallel twin of this loop.
// ===========================================================================

// grid_frame_cpu: grid the window [spoke0, spoke0+win) into a Cartesian k-space grid.
void grid_frame_cpu(const RadialData& d, int spoke0, std::vector<Cplx>& grid) {
    const int n = d.n;
    const std::size_t total = static_cast<std::size_t>(n) * n;
    const GriddingParams p = d.params();
    const int half = d.kb_w / 2;                 // KB half-width in cells (W=4 -> 2)

    // Fixed-point accumulators, one 64-bit integer per grid cell per component.
    // Integer sums commute, so scatter order (CPU loop vs GPU atomics) cannot change
    // the result -- the whole point of PATTERNS.md section 3.
    std::vector<long long> acc_re(total, 0), acc_im(total, 0);

    // Walk every sample in the window. spoke index sabs is absolute (0..n_spokes-1).
    for (int sw = 0; sw < d.win; ++sw) {
        const int sabs = spoke0 + sw;
        for (int j = 0; j < d.n_ro; ++j) {
            // 1) Where does this sample sit in the Cartesian grid?
            float kx, ky;
            sample_kpos(sabs, j, d.n_ro, n, kx, ky);

            // 2) Density compensation: multiply by the |k| ramp weight (grid_core.h).
            const float ro = static_cast<float>(j) - 0.5f * d.n_ro;   // signed offset
            const float dcf = radial_dcf(ro * (static_cast<float>(n) / d.n_ro));
            const Cplx  val = c_scale(d.samples[(std::size_t)sabs * d.n_ro + j], dcf);

            // 3) Spread onto the nearby grid cells with the separable KB kernel.
            const int gx0 = static_cast<int>(std::floor(kx)) - half;   // window start x
            const int gy0 = static_cast<int>(std::floor(ky)) - half;   // window start y
            for (int gy = gy0; gy <= gy0 + d.kb_w; ++gy) {
                if (gy < 0 || gy >= n) continue;                       // clip to grid
                const float wy = kb_weight(std::fabs(static_cast<float>(gy) - ky), p);
                if (wy == 0.0f) continue;
                for (int gx = gx0; gx <= gx0 + d.kb_w; ++gx) {
                    if (gx < 0 || gx >= n) continue;
                    const float wx = kb_weight(std::fabs(static_cast<float>(gx) - kx), p);
                    if (wx == 0.0f) continue;
                    const float w = wx * wy;                           // separable weight
                    const std::size_t idx = (std::size_t)gy * n + gx;
                    // Quantize to fixed point and add (integer add == deterministic).
                    acc_re[idx] += to_fixed(val.re * w);
                    acc_im[idx] += to_fixed(val.im * w);
                }
            }
        }
    }

    // Convert the fixed-point grid back to float once, at the end.
    grid.resize(total);
    for (std::size_t i = 0; i < total; ++i)
        grid[i] = c_make(from_fixed(acc_re[i]), from_fixed(acc_im[i]));
}

// ===========================================================================
// SECTION 5 -- Full single-frame reconstruction
// ===========================================================================

// reconstruct_frame_cpu: grid -> ifftshift -> inverse FFT -> fftshift -> deapodize.
//   The gridded data lives in k-space with the origin (DC) at the grid CENTER (n/2),
//   because that is where the density-compensated spokes cross. A straight inverse
//   FFT, however, expects DC at index 0. So we bracket the FFT with the standard
//   pair of circular rolls (n even -> ifftshift == fftshift == roll by n/2):
//       image = FFTSHIFT( IFFT2( IFFTSHIFT( gridded k-space ) ) )
//   The IFFTSHIFT moves DC from the center to index 0 so the FFT reads the right
//   frequency bins; the FFTSHIFT re-centers the resulting image so the anatomy sits
//   in the middle (matching our center-referenced phantom). Both rolls are pure
//   index permutations, trivially identical on CPU and GPU. THEORY "the algorithm".
void reconstruct_frame_cpu(const RadialData& d, int spoke0, std::vector<float>& out_mag) {
    const int n = d.n;
    const std::size_t total = static_cast<std::size_t>(n) * n;
    const GriddingParams p = d.params();
    const int h = n / 2;                 // shift amount (n even -> ifftshift == fftshift)

    // (1) Grid the window's samples onto the Cartesian grid (fixed-point, exact).
    std::vector<Cplx> grid;
    grid_frame_cpu(d, spoke0, grid);

    // (2) IFFTSHIFT: circularly roll the center-origin k-space so DC moves to index 0.
    //     Cell (r,c) carries frequency (r-h, c-h); it belongs at bin ((r+h)%n,(c+h)%n).
    std::vector<Cplx> shifted(total);
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            shifted[(std::size_t)((r + h) % n) * n + ((c + h) % n)] =
                grid[(std::size_t)r * n + c];
    grid.swap(shifted);

    // (3) Inverse FFT to image space (F^{-1}, with the 1/(n*n) normalization).
    ifft2_cpu(grid, n);

    // (4) FFTSHIFT the image (roll by n/2) so the reconstructed anatomy is centered,
    //     then (5) deapodize + magnitude. We fold both into one pass: pixel (r,c) of
    //     the OUTPUT reads the un-shifted FFT pixel ((r+h)%n, (c+h)%n).
    //     Deapodize = divide by the separable KB Fourier transform (grid_core.h) to
    //     correct the shading the gridding convolution imprinted; then |x| gives the
    //     radiologist-facing brightness (phase discarded).
    out_mag.resize(total);
    for (int r = 0; r < n; ++r) {
        const float dr = kb_deapod_1d(r, p);         // 1-D factor along rows (centered)
        for (int c = 0; c < n; ++c) {
            const float dc = kb_deapod_1d(c, p);     // 1-D factor along cols (centered)
            const float deapod = dr * dc;            // separable 2-D correction
            const int sr = (r + h) % n, sc = (c + h) % n;   // fftshift source pixel
            const Cplx px = c_scale(grid[(std::size_t)sr * n + sc], 1.0f / deapod);
            out_mag[(std::size_t)r * n + c] = c_abs(px);
        }
    }
}
