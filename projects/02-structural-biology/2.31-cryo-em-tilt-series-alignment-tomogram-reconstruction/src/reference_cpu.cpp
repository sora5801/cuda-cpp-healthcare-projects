// ===========================================================================
// src/reference_cpu.cpp  --  Loader, alignment, ramp filter, serial WBP
// ---------------------------------------------------------------------------
// Project 2.31 : Cryo-EM Tilt-Series Alignment & Tomogram Reconstruction
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. Written to be
//   OBVIOUSLY correct -- plain readable loops, no parallelism, no cleverness --
//   so that when the GPU and CPU agree we believe the GPU. The four steps mirror
//   the real cryo-ET pipeline: align -> ramp-filter -> back-project.
//
//   Compiled by the host C++ compiler only (no CUDA here). The per-sample
//   back-projection math is shared with the GPU via wbp_core.h (see that file).
//
// READ THIS AFTER: reference_cpu.h (the science) and wbp_core.h (the shared
// math). Compare against kernels.cu (the GPU twin of backproject_cpu).
// ===========================================================================
#include "reference_cpu.h"
#include "wbp_core.h"      // sample_projection_hd, WBP_PI_F (shared CPU/GPU core)

#include <cmath>           // std::cos, std::sin, std::floor, std::fabs
#include <fstream>         // std::ifstream
#include <stdexcept>       // std::runtime_error

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// load_tilt_series: parse the text tilt-series format (see data/README.md).
//   Layout:  header "n_tilts n_det ds img world_half"
//            then n_tilts rows, each "tilt_deg  p0 p1 ... p{n_det-1}".
//   We validate aggressively so a malformed sample aborts the demo with a clear
//   message instead of silently reconstructing garbage.
// ---------------------------------------------------------------------------
TiltSeries load_tilt_series(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open tilt-series file: " + path);

    TiltSeries ts;
    if (!(in >> ts.n_tilts >> ts.n_det >> ts.ds >> ts.img >> ts.world_half))
        throw std::runtime_error(
            "bad header (expected: n_tilts n_det ds img world_half) in " + path);
    if (ts.n_tilts <= 0 || ts.n_det <= 0 || ts.img <= 0 || ts.ds <= 0.0f)
        throw std::runtime_error("non-positive geometry in " + path);

    ts.tilt.resize(static_cast<std::size_t>(ts.n_tilts));
    ts.proj.resize(static_cast<std::size_t>(ts.n_tilts) * ts.n_det);
    for (int k = 0; k < ts.n_tilts; ++k) {
        if (!(in >> ts.tilt[static_cast<std::size_t>(k)]))
            throw std::runtime_error("missing tilt angle for projection "
                                     + std::to_string(k) + " in " + path);
        float* row = &ts.proj[static_cast<std::size_t>(k) * ts.n_det];
        for (int j = 0; j < ts.n_det; ++j) {
            if (!(in >> row[j]))
                throw std::runtime_error("projection " + std::to_string(k)
                                         + " truncated in " + path);
        }
    }
    return ts;
}

// ---------------------------------------------------------------------------
// best_lag: the integer lag L in [-search, +search] that maximizes the cross-
//   correlation CC(L) = sum_j  a[j] * b[j + L] between two length-n rows.
//   A positive L means row b is shifted RIGHT relative to row a. We use double
//   accumulation, a fixed scan order, and a tie-break to the smaller |L|, so the
//   answer is unique and reproducible (deterministic stdout, PATTERNS.md sec.3).
//   This is the one primitive of cross-correlation alignment.
// ---------------------------------------------------------------------------
static int best_lag(const float* a, const float* b, int n, int search) {
    double best_cc = -1.0e300;     // correlation of the best lag so far
    int    best_L  = 0;            // that lag
    int    best_absL = search + 1; // its |lag| (for the tie-break toward zero)
    for (int L = -search; L <= search; ++L) {
        double cc = 0.0;
        // Overlap region where both a[j] and b[j+L] are valid bins.
        const int j_lo = (L < 0) ? -L : 0;
        const int j_hi = (L < 0) ? n  : (n - L);
        for (int j = j_lo; j < j_hi; ++j)
            cc += static_cast<double>(a[j]) * static_cast<double>(b[j + L]);
        const int absL = (L < 0) ? -L : L;
        if (cc > best_cc || (cc == best_cc && absL < best_absL)) {
            best_cc = cc; best_L = L; best_absL = absL;
        }
    }
    return best_L;
}

// ---------------------------------------------------------------------------
// estimate_shifts: simplified fiducial-free tilt-series ALIGNMENT (sequential
//   cross-correlation, the standard coarse pass).
//
//   THE IDEA. Each recorded projection is the true projection translated by an
//   unknown drift. We CANNOT cross-correlate a high-tilt projection directly
//   against the 0 deg view: foreshortening changes the projection's SHAPE with
//   tilt, so the correlation would lock onto feature overlap, not the drift.
//   Instead we exploit that ADJACENT tilts differ by only a few degrees, so
//   their content is nearly identical -- their cross-correlation peak IS the
//   relative drift between them. We therefore:
//     1. walk outward from the reference (the smallest |tilt| projection),
//     2. measure the relative lag between each projection and its NEIGHBOR
//        toward the reference (best_lag),
//     3. ACCUMULATE those relative lags into an absolute shift per projection.
//   This "chained" / sequential alignment is exactly the coarse cross-
//   correlation pass IMOD's `tiltxcorr` and AreTomo2 run before refinement.
//
//   The result is deterministic (integer lags, fixed walk order, double sums)
//   so stdout is byte-reproducible.
//
//   LIMITATIONS (documented honestly): real alignment also solves for in-plane
//   rotation, magnification, and the tilt-axis position, and refines with gold
//   fiducial beads (IMOD) or projection matching (AreTomo2). We teach only the
//   1-D translational, integer-precision core -- the heart of the method.
//
//   Returns the reference index (also reported in stdout). shift[ref] = 0.
// ---------------------------------------------------------------------------
int estimate_shifts(const TiltSeries& ts, int search, std::vector<int>& shift) {
    const int K = ts.n_tilts, n = ts.n_det;
    shift.assign(static_cast<std::size_t>(K), 0);

    // Reference = projection with the smallest absolute tilt angle (least
    // foreshortened, the natural origin of the alignment chain).
    int ref = 0;
    float best_abs = std::fabs(ts.tilt[0]);
    for (int k = 1; k < K; ++k) {
        const float a = std::fabs(ts.tilt[static_cast<std::size_t>(k)]);
        if (a < best_abs) { best_abs = a; ref = k; }
    }
    shift[static_cast<std::size_t>(ref)] = 0;   // reference defines shift 0

    auto rowptr = [&](int k) {
        return &ts.proj[static_cast<std::size_t>(k) * n];
    };

    // Walk DOWN from ref to 0, chaining each projection to its neighbor above it.
    // shift[k] = shift[k+1] + lag(of k relative to its already-placed neighbor).
    for (int k = ref - 1; k >= 0; --k)
        shift[static_cast<std::size_t>(k)] =
            shift[static_cast<std::size_t>(k + 1)] + best_lag(rowptr(k + 1), rowptr(k), n, search);

    // Walk UP from ref to K-1, chaining each projection to its neighbor below it.
    for (int k = ref + 1; k < K; ++k)
        shift[static_cast<std::size_t>(k)] =
            shift[static_cast<std::size_t>(k - 1)] + best_lag(rowptr(k - 1), rowptr(k), n, search);

    return ref;
}

// ---------------------------------------------------------------------------
// apply_shifts: translate each projection by -shift[k] so features line up with
//   the reference. We write aligned[k][j] = proj[k][j + shift[k]] (reading from
//   the shifted source), filling out-of-range source bins with 0. After this,
//   every projection is in a common, drift-corrected frame.
// ---------------------------------------------------------------------------
void apply_shifts(const TiltSeries& ts, const std::vector<int>& shift,
                  std::vector<float>& aligned) {
    const int K = ts.n_tilts, n = ts.n_det;
    aligned.assign(ts.proj.size(), 0.0f);
    for (int k = 0; k < K; ++k) {
        const int L = shift[static_cast<std::size_t>(k)];
        const float* src = &ts.proj[static_cast<std::size_t>(k) * n];
        float* dst       = &aligned[static_cast<std::size_t>(k) * n];
        for (int j = 0; j < n; ++j) {
            const int sj = j + L;                  // source bin we copy from
            dst[j] = (sj >= 0 && sj < n) ? src[sj] : 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// ramp_filter_cpu: Ram-Lak ramp filter by an EXPLICIT DFT (the GPU's baseline).
//
//   This is deliberately written as the slow, obvious version of EXACTLY what
//   cuFFT does in ramp_filter_gpu(): forward-transform each row, multiply every
//   spectral bin by the shared ramp weight ramp_weight_hd(), inverse-transform.
//   By computing the SAME |f| ramp (same weight function) we make the CPU and
//   GPU ramp filters the same mathematical operation -- so they agree to ~1e-4,
//   not just "roughly". (Contrast project 4.01, which filters in the spatial
//   domain; here we mirror the FFT path so the cuFFT result has a tight check.)
//
//   For a length-n real row x, the (real, even-symmetric) ramp filter is:
//       X[f]      = sum_t x[t] exp(-2*pi*i f t / n)        (forward DFT)
//       X[f]     *= ramp_weight_hd(f, nf, n, ds)           (apply ramp)
//       y[t]      = (1/n) sum_f X[f] exp(+2*pi*i f t / n)  (inverse DFT, real)
//   We exploit Hermitian symmetry (X[n-f] = conj(X[f]) for a real input) so we
//   only need bins f = 0..n/2 and can write the inverse as a cosine/sine sum.
//   Complexity O(n^2) per row -- fine for the tiny teaching sample; cuFFT does
//   the same thing in O(n log n).
//
//   NOTE: cuFFT's forward+inverse multiplies data by n, which ramp_filter_gpu()
//   divides back out; here the explicit 1/n in the inverse already normalizes,
//   so both land on the same scale.
// ---------------------------------------------------------------------------
void ramp_filter_cpu(const TiltSeries& ts, const std::vector<float>& aligned,
                     std::vector<float>& filtered) {
    const int K = ts.n_tilts, n = ts.n_det;
    const int nf = n / 2 + 1;                 // independent spectral bins (R2C)
    const float ds = ts.ds;

    filtered.assign(aligned.size(), 0.0f);
    // Temp spectrum buffers (double for accuracy in the O(n^2) sums).
    std::vector<double> re(static_cast<std::size_t>(nf));
    std::vector<double> im(static_cast<std::size_t>(nf));

    for (int k = 0; k < K; ++k) {
        const float* row = &aligned[static_cast<std::size_t>(k) * n];
        float* out       = &filtered[static_cast<std::size_t>(k) * n];

        // (1) Forward DFT for bins 0..n/2, then (2) apply the ramp weight.
        for (int f = 0; f < nf; ++f) {
            double sre = 0.0, sim = 0.0;
            const double w = -2.0 * M_PI * f / n;
            for (int t = 0; t < n; ++t) {
                const double ph = w * t;
                sre += static_cast<double>(row[t]) * std::cos(ph);
                sim += static_cast<double>(row[t]) * std::sin(ph);
            }
            const double g = static_cast<double>(ramp_weight_hd(f, nf, n, ds));
            re[static_cast<std::size_t>(f)] = sre * g;
            im[static_cast<std::size_t>(f)] = sim * g;
        }

        // (3) Inverse DFT back to real space using Hermitian symmetry: bins
        // 1..(n-1)/2 contribute twice (a bin and its conjugate mirror), DC and
        // (for even n) Nyquist contribute once. y[t] = (1/n) Re{ sum X[f] e^{+..} }.
        for (int t = 0; t < n; ++t) {
            double acc = re[0];                            // f = 0 (DC), real
            const double w = 2.0 * M_PI * t / n;
            const int last = (n % 2 == 0) ? (nf - 1) : nf; // even n: Nyquist single
            for (int f = 1; f < last; ++f) {
                const double ph = w * f;
                // X[f] e^{+i ph} + conj(X[f]) e^{-i ph} = 2 Re{X[f] e^{+i ph}}.
                acc += 2.0 * (re[static_cast<std::size_t>(f)] * std::cos(ph)
                            - im[static_cast<std::size_t>(f)] * std::sin(ph));
            }
            if (n % 2 == 0) {                              // even n: single Nyquist
                const double ph = w * (nf - 1);
                acc += re[static_cast<std::size_t>(nf - 1)] * std::cos(ph)
                     - im[static_cast<std::size_t>(nf - 1)] * std::sin(ph);
            }
            out[t] = static_cast<float>(acc / n);          // inverse-DFT 1/n norm
        }
    }
}

// ---------------------------------------------------------------------------
// compute_trig: cos/sin of every (signed) tilt angle, once, on the host.
//   Tilt angles are stored in DEGREES; we convert to radians here. Sharing this
//   precomputed table with the GPU guarantees identical trig on both sides.
// ---------------------------------------------------------------------------
void compute_trig(const TiltSeries& ts, std::vector<float>& cosv,
                  std::vector<float>& sinv) {
    const int K = ts.n_tilts;
    cosv.resize(static_cast<std::size_t>(K));
    sinv.resize(static_cast<std::size_t>(K));
    for (int k = 0; k < K; ++k) {
        const double rad = ts.tilt[static_cast<std::size_t>(k)] * M_PI / 180.0;
        cosv[static_cast<std::size_t>(k)] = static_cast<float>(std::cos(rad));
        sinv[static_cast<std::size_t>(k)] = static_cast<float>(std::sin(rad));
    }
}

// ---------------------------------------------------------------------------
// backproject_cpu: the serial weighted back-projection (the trusted baseline).
//   For every output pixel (px,py) we sum its contribution from every tilt,
//   using the SHARED sample_projection_hd() so the math is identical to the GPU.
//   Complexity: O(img^2 * n_tilts). This is the loop kernels.cu parallelizes by
//   giving each pixel its own thread.
// ---------------------------------------------------------------------------
void backproject_cpu(const TiltSeries& ts, const std::vector<float>& filtered,
                     const std::vector<float>& cosv, const std::vector<float>& sinv,
                     std::vector<float>& slice) {
    const int N = ts.img, n_det = ts.n_det, K = ts.n_tilts;
    const float ds = ts.ds, W = ts.world_half;
    const float center = 0.5f * (n_det - 1);                 // detector index of s=0
    const float scale  = WBP_PI_F / static_cast<float>(K);   // d(theta) ~ pi/n_tilts
    const float pix    = (N > 1) ? (2.0f * W / (N - 1)) : 0.0f;  // world units/pixel

    slice.assign(static_cast<std::size_t>(N) * N, 0.0f);
    for (int py = 0; py < N; ++py) {
        const float wy = -W + py * pix;                       // world y of this row
        for (int px = 0; px < N; ++px) {
            const float wx = -W + px * pix;                   // world x of this pixel
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                const float* row = &filtered[static_cast<std::size_t>(k) * n_det];
                acc += sample_projection_hd(row, n_det, wx, wy,
                                            cosv[static_cast<std::size_t>(k)],
                                            sinv[static_cast<std::size_t>(k)],
                                            ds, center);
            }
            slice[static_cast<std::size_t>(py) * N + px] = acc * scale;
        }
    }
}
