// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ CS-MRI baseline we trust
// ---------------------------------------------------------------------------
// Project 4.3 : MRI Reconstruction with Compressed Sensing
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written to
//   be OBVIOUSLY correct -- readable radix-2 FFTs and a textbook FISTA loop, no
//   parallelism, no cleverness -- so that when the GPU (kernels.cu, using cuFFT)
//   agrees with it, we believe the GPU. Every per-element formula it uses is the
//   SAME one the kernels use, imported from cs_core.h, so the only thing that can
//   differ between the two paths is the FFT implementation (our radix-2 vs cuFFT).
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h and cs_core.h. Compare with kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::sin, std::cos, std::sqrt
#include <cstddef>     // std::size_t
#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream
#include <stdexcept>   // std::runtime_error
#include <string>
#include <vector>

// M_PI is not guaranteed by the C++ standard; define our own to stay portable.
static const double PI = 3.14159265358979323846;

// ===========================================================================
// SECTION 1 -- Loading the sample
// ===========================================================================

// load_kspace: read the text problem file. Format (see data/README.md):
//   line 1 : "<n> <lambda> <iters> <has_truth>"
//   n*n rows: "re im mask [truth]"  (row-major; truth column present iff has_truth)
KSpaceData load_kspace(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open k-space file: " + path);

    KSpaceData d;
    int has_truth_flag = 0;
    // Header line drives every subsequent allocation.
    if (!(in >> d.n >> d.lambda >> d.iters >> has_truth_flag))
        throw std::runtime_error("bad header in k-space file: " + path);
    d.has_truth = (has_truth_flag != 0);

    // Guard: n must be a positive power of two (the radix-2 FFT requires it).
    if (d.n <= 0 || (d.n & (d.n - 1)) != 0)
        throw std::runtime_error("image side n must be a power of two");

    const std::size_t total = static_cast<std::size_t>(d.n) * d.n;
    d.kspace.resize(total);
    d.mask.resize(total);
    if (d.has_truth) d.truth.resize(total);

    // Read the grid row-major. Each acquired sample carries its k-space value and
    // a mask flag; unsampled positions carry (0,0) with mask 0 (already zero-filled).
    for (std::size_t i = 0; i < total; ++i) {
        float re, im;
        int m;
        if (!(in >> re >> im >> m))
            throw std::runtime_error("k-space file truncated at index " + std::to_string(i));
        d.kspace[i] = c_make(re, im);
        d.mask[i]   = m;
        if (d.has_truth) {
            float t;
            if (!(in >> t))
                throw std::runtime_error("truth column truncated at index " + std::to_string(i));
            d.truth[i] = t;
        }
    }
    return d;
}

// ===========================================================================
// SECTION 2 -- The reference FFT (radix-2 Cooley-Tukey, separable 2D)
// ---------------------------------------------------------------------------
// We compute the DFT the classic divide-and-conquer way. It is the SAME transform
// cuFFT computes; writing it out by hand is exactly the "no black box" rule from
// CLAUDE.md section 6 -- the learner can see precisely what cuFFT does for us on
// the GPU. Internals are double precision for accuracy; results are stored back as
// float (Cplx) so the comparison against cuFFT (single precision) is apples-to-apples.
// ===========================================================================

// fft1_double: in-place 1D radix-2 FFT of a length-`n` complex signal (n a power
// of two). `sign = -1` is the forward transform (exp(-i...)), `sign = +1` the
// inverse (exp(+i...), UN-normalized -- the caller scales). Iterative
// bit-reversal + butterfly form: O(n log n), the whole point of the FFT.
static void fft1_double(std::vector<double>& re, std::vector<double>& im, int n, int sign) {
    // --- Bit-reversal permutation: reorder so butterflies read contiguous pairs.
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;   // ripple-carry in reversed bit order
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    // --- Butterflies over doubling block lengths len = 2, 4, 8, ..., n.
    for (int len = 2; len <= n; len <<= 1) {
        const double ang = sign * 2.0 * PI / len;      // twiddle angle step
        const double wr = std::cos(ang), wi = std::sin(ang);  // principal root W_len
        for (int i = 0; i < n; i += len) {
            double cr = 1.0, ci = 0.0;                 // running twiddle W_len^k
            for (int k = 0; k < len / 2; ++k) {
                // Butterfly: combine element k with element k+len/2 of this block.
                const int a = i + k, b = i + k + len / 2;
                const double ur = re[a],           ui = im[a];
                const double vr = re[b] * cr - im[b] * ci;      // twiddled odd part
                const double vi = re[b] * ci + im[b] * cr;
                re[a] = ur + vr; im[a] = ui + vi;
                re[b] = ur - vr; im[b] = ui - vi;
                // Advance the twiddle: (cr,ci) *= (wr,wi).
                const double ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr; cr = ncr;
            }
        }
    }
}

// fft2_core: separable 2D FFT -- FFT every row, then FFT every column. A 2D DFT is
// exactly this product of 1D DFTs. `sign` and normalization handled by the wrappers.
static void fft2_core(std::vector<Cplx>& data, int n, int sign) {
    std::vector<double> re(n), im(n);   // scratch for one row/column at a time
    // --- Rows -------------------------------------------------------------
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) { re[c] = data[(std::size_t)r * n + c].re;
                                      im[c] = data[(std::size_t)r * n + c].im; }
        fft1_double(re, im, n, sign);
        for (int c = 0; c < n; ++c) { data[(std::size_t)r * n + c] = c_make((float)re[c], (float)im[c]); }
    }
    // --- Columns ----------------------------------------------------------
    for (int c = 0; c < n; ++c) {
        for (int r = 0; r < n; ++r) { re[r] = data[(std::size_t)r * n + c].re;
                                      im[r] = data[(std::size_t)r * n + c].im; }
        fft1_double(re, im, n, sign);
        for (int r = 0; r < n; ++r) { data[(std::size_t)r * n + c] = c_make((float)re[r], (float)im[r]); }
    }
}

// fft2_cpu: forward 2D FFT, UN-normalized (matches cuFFT's CUFFT_FORWARD).
void fft2_cpu(std::vector<Cplx>& data, int n) { fft2_core(data, n, -1); }

// ifft2_cpu: inverse 2D FFT WITH the 1/(n*n) normalization, so ifft2(fft2(x))==x.
// cuFFT also leaves its inverse un-normalized, so kernels.cu applies the same
// 1/(n*n) scale -- keeping the two paths identical.
void ifft2_cpu(std::vector<Cplx>& data, int n) {
    fft2_core(data, n, +1);
    const float inv = 1.0f / (static_cast<float>(n) * static_cast<float>(n));
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = c_scale(data[i], inv);
}

// ===========================================================================
// SECTION 3 -- The reconstructions
// ===========================================================================

// zero_filled_magnitude: the naive baseline. Inverse-FFT the zero-filled k-space
// and take the magnitude. No regularization -> aliasing artifacts remain. This is
// the "before" picture that CS improves upon.
void zero_filled_magnitude(const KSpaceData& d, std::vector<float>& out_mag) {
    std::vector<Cplx> img = d.kspace;   // copy measured (zero-filled) k-space
    ifft2_cpu(img, d.n);                // adjoint/naive image = F^{-1}{y}
    out_mag.resize(img.size());
    for (std::size_t i = 0; i < img.size(); ++i) out_mag[i] = c_abs(img[i]);
}

// ---------------------------------------------------------------------------
// reconstruct_cpu: FISTA for  min (1/2)||M F x - y||^2 + lambda ||x||_1.
//
//   We use the IDENTITY sparsifying transform Psi = I here (soft-threshold the
//   image pixels themselves) because MRI angiograms and our synthetic phantom are
//   already sparse in image space -- it keeps the teaching kernel tiny while being
//   a genuine CS reconstruction. THEORY "the algorithm" shows how a wavelet/TV
//   transform slots into the same loop.
//
//   Proximal-gradient step (ISTA):   x <- prox_{t*lambda*||.||_1}( x - t * grad )
//   where grad = F^{-1}{ M (F x - y) } is the gradient of the data term and the
//   step size t = 1 (the forward op M F has spectral norm 1, so the Lipschitz
//   constant is 1 -- see THEORY). FISTA adds Nesterov momentum on top of ISTA for
//   O(1/k^2) convergence instead of O(1/k).
//
//   Every arithmetic step below calls a cs_core.h function -- the SAME code the
//   GPU kernels call -- so the CPU and GPU differ only in the FFT library.
// ---------------------------------------------------------------------------
void reconstruct_cpu(const KSpaceData& d, std::vector<float>& out_mag) {
    const int n = d.n;
    const std::size_t total = static_cast<std::size_t>(n) * n;
    const float t = 1.0f;                 // gradient step size (Lipschitz const = 1)

    // x   : current image estimate;  z : the momentum "look-ahead" point.
    // Initialize both to the zero-filled adjoint image F^{-1}{y} (a warm start).
    std::vector<Cplx> x = d.kspace;
    ifft2_cpu(x, n);
    std::vector<Cplx> z = x;
    std::vector<Cplx> x_prev = x;
    float theta = 1.0f;                   // FISTA momentum parameter t_k

    std::vector<Cplx> work(total);        // scratch for F{z}, then the gradient

    for (int it = 0; it < d.iters; ++it) {
        // ---- Gradient of the data term at the look-ahead point z -----------
        // 1) forward FFT: work = F{z}
        work = z;
        fft2_cpu(work, n);
        // 2) data-consistency residual in k-space: r = M (F z - y)
        //    (data_consistency_residual lives in cs_core.h -- shared with the GPU)
        for (std::size_t i = 0; i < total; ++i)
            work[i] = data_consistency_residual(work[i], d.kspace[i], d.mask[i]);
        // 3) inverse FFT back to image space: grad = F^{-1}{ r }
        ifft2_cpu(work, n);

        // ---- Proximal-gradient (ISTA) update of x --------------------------
        //   x = prox( z - t*grad ) = soft_threshold( z - t*grad, t*lambda )
        const float thr = t * d.lambda;
        for (std::size_t i = 0; i < total; ++i) {
            const Cplx step = c_sub(z[i], c_scale(work[i], t));  // gradient descent step
            x[i] = soft_threshold_cplx(step, thr);               // sparsity prox
        }

        // ---- FISTA momentum: extrapolate z from x_{k} and x_{k-1} ----------
        //   theta_{k+1} = (1 + sqrt(1 + 4 theta_k^2)) / 2
        //   z = x_k + ((theta_k - 1)/theta_{k+1}) (x_k - x_{k-1})
        const float theta_next = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * theta * theta));
        const float beta = (theta - 1.0f) / theta_next;
        for (std::size_t i = 0; i < total; ++i)
            z[i] = c_add(x[i], c_scale(c_sub(x[i], x_prev[i]), beta));
        theta = theta_next;
        x_prev = x;
    }

    // Final radiologist-facing image is the pixel magnitude |x|.
    out_mag.resize(total);
    for (std::size_t i = 0; i < total; ++i) out_mag[i] = c_abs(x[i]);
}
