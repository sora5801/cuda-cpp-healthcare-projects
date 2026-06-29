// ===========================================================================
// src/reference_cpu.cpp  --  Periodic charge loader + CPU Ewald references
// ---------------------------------------------------------------------------
// Project 1.2 : Particle-Mesh Ewald Electrostatics
//
// ROLE IN THE PROJECT
//   The OBVIOUSLY-correct baselines the GPU is checked against. Two of them:
//     1. ewald_recip_direct_cpu -- the textbook reciprocal k-vector sum. Slow
//        (O(N * Kmax^3)) but a transparent statement of the physics: this is
//        what E_recip *is*. It validates that SPME is a good approximation.
//     2. pme_recip_cpu          -- the SPME pipeline (spread -> DFT -> convolve
//        -> sum) on the host, using the SAME pme.h math the GPU uses. This is
//        the exact twin of kernels.cu; GPU == pme_recip_cpu is our tight check.
//   Plus the real-space and self terms so we can assemble the full Ewald energy
//   and demonstrate it is INVARIANT to the splitting parameter beta.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h
//   and READ pme.h first -- the shared B-spline / fixed-point math lives there.
//
// READ THIS AFTER: reference_cpu.h, pme.h. Compare against kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"
#include "pme.h"

#include <cmath>
#include <complex>
#include <fstream>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Wrap a coordinate into the primary periodic cell [0, L). fmod can return a
// negative remainder for a negative input, so we add L once to fix the sign.
static inline double wrap_into_box(double v, double L) {
    v = std::fmod(v, L);
    if (v < 0.0) v += L;
    return v;
}

// ---------------------------------------------------------------------------
// load_system: parse "<n> <box>" then n rows of "x y z q". Wraps coordinates
// into [0, box) and checks (approximate) charge neutrality, which the periodic
// Coulomb sum requires (a non-zero net charge in a periodic box has infinite,
// ill-defined energy unless a neutralizing background is added).
// ---------------------------------------------------------------------------
System load_system(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open system file: " + path);
    System s;
    if (!(in >> s.n >> s.box) || s.n <= 0 || s.box <= 0.0)
        throw std::runtime_error("bad header (expected '<n> <box>') in " + path);
    s.x.resize(s.n); s.y.resize(s.n); s.z.resize(s.n); s.q.resize(s.n);
    double qsum = 0.0;
    for (int i = 0; i < s.n; ++i) {
        if (!(in >> s.x[i] >> s.y[i] >> s.z[i] >> s.q[i]))
            throw std::runtime_error("system data truncated in " + path);
        s.x[i] = wrap_into_box(s.x[i], s.box);   // fold into the primary cell
        s.y[i] = wrap_into_box(s.y[i], s.box);
        s.z[i] = wrap_into_box(s.z[i], s.box);
        qsum += s.q[i];
    }
    if (std::fabs(qsum) > 1e-6)
        throw std::runtime_error("system is not charge-neutral (sum q = " +
                                 std::to_string(qsum) + "); PME requires net zero charge.");
    return s;
}

// ---------------------------------------------------------------------------
// choose_params: deterministic, system-dependent PME knobs.
//   * rcut  = box/2 : the largest cutoff the minimum-image convention allows.
//   * beta  : set so erfc(beta*rcut) is small (~1e-5), i.e. the real-space sum
//             has converged by the cutoff. erfc(3.0) ~ 2.2e-5, so beta = 3/rcut.
//   * K     : grid points per axis ~ 2 per length unit, rounded UP to the next
//             multiple of 8 (FFT-friendly, and >= 2*order). Same K on all axes
//             since the box is cubic.
// Identical inputs -> identical params on CPU and GPU.
// ---------------------------------------------------------------------------
PmeParams choose_params(const System& s) {
    PmeParams p;
    p.order = PME_ORDER;
    p.rcut  = 0.5 * s.box;
    p.beta  = 3.0 / p.rcut;               // erfc(beta*rcut) = erfc(3) ~ 2e-5
    int target = static_cast<int>(std::ceil(2.0 * s.box));   // ~2 grid pts / unit
    int k = 8;
    while (k < target || k < 2 * p.order) k += 8;            // round up to mult of 8
    p.K = k;
    return p;
}

// ---------------------------------------------------------------------------
// ewald_real_cpu: short-range pairwise sum with minimum image + erfc damping.
//   For each pair (i<j) we take the nearest periodic image (minimum image), and
//   if it is within rcut, add q_i q_j erfc(beta r)/r. erfc decays like a Gaussian
//   so this converges quickly -- the whole point of the Ewald split.
//   Complexity O(N^2) here (fine for the teaching sample); production codes use
//   a neighbour list to make it O(N).
// ---------------------------------------------------------------------------
double ewald_real_cpu(const System& s, const PmeParams& p) {
    const double L = s.box, half = 0.5 * L;
    double e = 0.0;
    for (int i = 0; i < s.n; ++i) {
        for (int j = i + 1; j < s.n; ++j) {
            // Minimum-image displacement: wrap each component into [-L/2, L/2].
            double dx = s.x[i] - s.x[j];
            double dy = s.y[i] - s.y[j];
            double dz = s.z[i] - s.z[j];
            if (dx >  half) dx -= L; else if (dx < -half) dx += L;
            if (dy >  half) dy -= L; else if (dy < -half) dy += L;
            if (dz >  half) dz -= L; else if (dz < -half) dz += L;
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r < p.rcut && r > 0.0)
                e += s.q[i] * s.q[j] * std::erfc(p.beta * r) / r;
        }
    }
    return e;
}

// ---------------------------------------------------------------------------
// ewald_self: each smeared Gaussian's self-interaction, subtracted out.
//   E_self = (beta / sqrt(pi)) * sum_i q_i^2 . A constant given beta and charges.
// ---------------------------------------------------------------------------
double ewald_self(const System& s, const PmeParams& p) {
    double q2 = 0.0;
    for (int i = 0; i < s.n; ++i) q2 += s.q[i] * s.q[i];
    return p.beta / std::sqrt(M_PI) * q2;
}

// ---------------------------------------------------------------------------
// ewald_recip_direct_cpu: the GOLD-STANDARD reciprocal-space energy.
//   Sum over ALL integer wavevectors m = (mx,my,mz) != 0 up to a cutoff, with
//   k = 2*pi*m/L:
//     E_recip = (2*pi / V) * sum_{m != 0} exp(-|k|^2 / (4 beta^2)) / |k|^2 * |S(m)|^2
//   structure factor  S(m) = sum_j q_j exp(i k . r_j).
//   NOTE the prefactor is 2*pi/V with NO extra 1/2: the sum runs over the FULL
//   sphere of m (both +m and -m), which is exactly the standard textbook result.
//   We confirm this empirically -- with this prefactor the TOTAL Ewald energy is
//   invariant to beta to ~1e-7 (main.cu's physics check); a spurious 1/2 breaks
//   that invariance, which is how the constant was pinned down.
//   O(N * (2*mmax+1)^3): slow but transparently the definition of E_recip.
// ---------------------------------------------------------------------------
double ewald_recip_direct_cpu(const System& s, const PmeParams& p) {
    const double L = s.box;
    const double V = L * L * L;               // box volume
    const double two_pi = 2.0 * M_PI;
    const double beta2 = p.beta * p.beta;
    // Reciprocal cutoff: include wavevectors until the Gaussian factor is tiny.
    // exp(-(pi*|m|/(beta L))^2) < ~1e-10 when pi*|m|/(beta L) > ~5, so
    //   mmax = ceil(5 * beta * L / pi).
    int mmax = static_cast<int>(std::ceil(5.0 * p.beta * L / M_PI));
    if (mmax < 1) mmax = 1;

    double e = 0.0;
    for (int mx = -mmax; mx <= mmax; ++mx)
    for (int my = -mmax; my <= mmax; ++my)
    for (int mz = -mmax; mz <= mmax; ++mz) {
        if (mx == 0 && my == 0 && mz == 0) continue;     // the m=0 term is excluded
        const double m2 = static_cast<double>(mx * mx + my * my + mz * mz);
        // |k|^2 = (2*pi/L)^2 * m^2 ; the Ewald reciprocal weight is
        //   exp(-|k|^2/(4 beta^2)) / |k|^2.
        const double k2 = (two_pi / L) * (two_pi / L) * m2;
        const double weight = std::exp(-k2 / (4.0 * beta2)) / k2;
        // Structure factor S(m) = sum_j q_j exp(i k . r_j).
        double sr = 0.0, si = 0.0;
        for (int j = 0; j < s.n; ++j) {
            const double phase = two_pi * (mx * s.x[j] + my * s.y[j] + mz * s.z[j]) / L;
            sr += s.q[j] * std::cos(phase);
            si += s.q[j] * std::sin(phase);
        }
        e += weight * (sr * sr + si * si);
    }
    return (two_pi / V) * e;
}

double ewald_total_direct_cpu(const System& s, const PmeParams& p) {
    return ewald_real_cpu(s, p) + ewald_recip_direct_cpu(s, p) - ewald_self(s, p);
}

// ---------------------------------------------------------------------------
// spread_charges_cpu: build the SPME charge grid exactly as the GPU does.
//   For each charge, compute its scaled grid coordinate g = (coord/box)*K on each
//   axis, the lower index g0 = floor(g) and fractional offset frac = g - g0, then
//   the PME_ORDER B-spline weights (pme.h). The charge times the outer product of
//   the three axis weights is scattered onto the order^3 block of grid points,
//   wrapped periodically. We accumulate in FIXED-POINT integers (pme_to_fixed),
//   exactly mirroring the GPU's atomicAdd, then convert back to a real grid.
//   This makes the CPU and GPU grids bit-identical.
// Grid layout: row-major index = (ix*K + iy)*K + iz (ix slowest, iz fastest).
// ---------------------------------------------------------------------------
void spread_charges_cpu(const System& s, const PmeParams& p, std::vector<double>& grid) {
    const int K = p.K, order = p.order;
    const std::size_t NG = static_cast<std::size_t>(K) * K * K;
    // Integer accumulator (the exact twin of the GPU's unsigned long long grid).
    std::vector<unsigned long long> acc(NG, 0ull);

    double wx[PME_ORDER], wy[PME_ORDER], wz[PME_ORDER];
    for (int a = 0; a < s.n; ++a) {
        // Scaled fractional coordinates in [0, K).
        const double gx = (s.x[a] / s.box) * K;
        const double gy = (s.y[a] / s.box) * K;
        const double gz = (s.z[a] / s.box) * K;
        const int g0x = static_cast<int>(std::floor(gx));
        const int g0y = static_cast<int>(std::floor(gy));
        const int g0z = static_cast<int>(std::floor(gz));
        pme_bspline_weights(gx - g0x, wx);
        pme_bspline_weights(gy - g0y, wy);
        pme_bspline_weights(gz - g0z, wz);
        // Scatter q * wx[i]*wy[j]*wz[k] onto the order^3 stencil (wrapped).
        // w[i] is the weight of grid point (g0 + i) -- see pme_bspline_weights.
        for (int i = 0; i < order; ++i) {
            int ix = ((g0x + i) % K + K) % K;             // wrap periodically
            for (int j = 0; j < order; ++j) {
                int iy = ((g0y + j) % K + K) % K;
                const double wxy = s.q[a] * wx[i] * wy[j];
                for (int k = 0; k < order; ++k) {
                    int iz = ((g0z + k) % K + K) % K;
                    const std::size_t idx = (static_cast<std::size_t>(ix) * K + iy) * K + iz;
                    acc[idx] += pme_to_fixed(wxy * wz[k]); // fixed-point scatter-add
                }
            }
        }
    }
    grid.assign(NG, 0.0);
    for (std::size_t i = 0; i < NG; ++i) grid[i] = pme_fixed_to_double(acc[i]);
}

// ---------------------------------------------------------------------------
// build_influence: the SPME reciprocal "influence function" B(m)*C(m), stored
// for the HALF-COMPLEX grid the R2C FFT produces, size K*K*(K/2+1).
//   * B(m) = |b_x(mx)|^2 |b_y(my)|^2 |b_z(mz)|^2 is the B-spline correction
//            (pme_bsp_modulus2 per axis).
//   * C(m) = (2 pi / V) * exp(-k^2/(4 beta^2)) / k^2  with k = 2*pi*m/L  -- the
//            SAME coefficient as ewald_recip_direct_cpu (NO extra 1/2; see that
//            function's note on pinning the constant via beta-invariance).
//   so that  E_recip = sum_full B(m) C(m) |F[m]|^2 , matching the direct sum.
//   The half-spectrum sum in pme_recip_cpu applies the Hermitian multiplicity
//   (mult = 2 for interior bins) to recover this full-grid sum.
//   The m=0 term is set to zero (excluded from the sum).
// Layout matches cuFFT R2C output: index = (mx*K + my)*(K/2+1) + mz with the
// fast axis (mz) halved. Wavevector indices fold: m -> m-K for m > K/2.
// ---------------------------------------------------------------------------
void build_influence(const System& s, const PmeParams& p, std::vector<double>& influence) {
    const int K = p.K, order = p.order;
    const int Kh = K / 2 + 1;
    const double L = s.box, V = L * L * L;
    const double two_pi = 2.0 * M_PI;
    const double beta2 = p.beta * p.beta;

    // Precompute integer B-spline node values M_p(1..order-1) for the modulus
    // factor. Evaluating the weights at frac == 0 yields exactly these node
    // values: pme_bspline_weights(0.0, Mp) gives Mp[k] == M_p(k+1) for
    // k = 0..order-2, and Mp[order-1] == 0 (the open spline vanishes there).
    double Mp[PME_ORDER] = {0};
    pme_bspline_weights(0.0, Mp);
    (void)order;   // order == PME_ORDER; kept as a named local for readability

    influence.assign(static_cast<std::size_t>(K) * K * Kh, 0.0);
    for (int mx = 0; mx < K; ++mx) {
        const int hx = (mx <= K / 2) ? mx : mx - K;        // folded signed index
        const double bx = pme_bsp_modulus2(mx, K, Mp);
        for (int my = 0; my < K; ++my) {
            const int hy = (my <= K / 2) ? my : my - K;
            const double by = pme_bsp_modulus2(my, K, Mp);
            for (int mz = 0; mz < Kh; ++mz) {
                const int hz = mz;                          // R2C: mz in [0, K/2]
                const double bz = pme_bsp_modulus2(mz, K, Mp);
                const std::size_t idx = (static_cast<std::size_t>(mx) * K + my) * Kh + mz;
                if (hx == 0 && hy == 0 && hz == 0) { influence[idx] = 0.0; continue; }
                const double m2 = static_cast<double>(hx * hx + hy * hy + hz * hz);
                const double k2 = (two_pi / L) * (two_pi / L) * m2;
                const double C = std::exp(-k2 / (4.0 * beta2)) / k2 * (two_pi / V);
                influence[idx] = bx * by * bz * C;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// A tiny, transparent 3D DFT for the host SPME pipeline. We do NOT pull in an
// FFT library on the host (no dependency); instead we exploit SEPARABILITY -- a
// 3D DFT is three sets of 1D DFTs, one along each axis. Each 1D DFT is the naive
// O(K^2) sum (obviously correct), applied K^2 times per axis, so the whole 3D
// transform is O(3 K^4) instead of the O(K^6) of a direct 6-fold loop -- fast
// enough that the demo is snappy at the sample's K. It computes the SAME thing
// cuFFT does, so pme_recip_cpu is a faithful twin of the GPU path. (A production
// host code would call FFTW for the O(K^3 log K) version.)
//
// Forward transform convention matches cuFFT: X[m] = sum_x x[x] exp(-2pi i m x/K),
// NO normalization. Output is real-to-complex packed: we transform the full
// complex cube, then copy the non-redundant half (mz = 0..K/2) into `out`,
// matching cuFFT's R2C layout out[(mx*K + my)*Kh + mz].
// ---------------------------------------------------------------------------
// One in-place 1D DFT over `n` complex samples with stride `stride`, starting at
// base offset `base` in the flat array `a`. `scratch` is caller-provided length-n
// workspace. Forward (sign=-1) only -- all axes use the same convention.
static void dft1d(std::complex<double>* a, int n, std::size_t stride,
                  std::vector<std::complex<double>>& scratch) {
    const double two_pi = 2.0 * M_PI;
    for (int m = 0; m < n; ++m) {
        std::complex<double> acc(0.0, 0.0);
        for (int x = 0; x < n; ++x) {
            const double ph = -two_pi * static_cast<double>(m) * static_cast<double>(x) / n;
            acc += a[x * stride] * std::complex<double>(std::cos(ph), std::sin(ph));
        }
        scratch[m] = acc;
    }
    for (int m = 0; m < n; ++m) a[m * stride] = scratch[m];
}

static void dft3d_r2c(const std::vector<double>& in, int K,
                      std::vector<std::complex<double>>& out) {
    const int Kh = K / 2 + 1;
    // Work in a full complex cube (row-major: index = (ix*K + iy)*K + iz).
    std::vector<std::complex<double>> cube(static_cast<std::size_t>(K) * K * K);
    for (std::size_t i = 0; i < cube.size(); ++i) cube[i] = {in[i], 0.0};

    std::vector<std::complex<double>> scratch(K);
    // Transform along z (fastest axis, stride 1): one 1D DFT per (ix,iy) line.
    for (int ix = 0; ix < K; ++ix)
        for (int iy = 0; iy < K; ++iy) {
            const std::size_t base = (static_cast<std::size_t>(ix) * K + iy) * K;
            dft1d(&cube[base], K, 1, scratch);
        }
    // Transform along y (stride K): one 1D DFT per (ix,iz) line.
    for (int ix = 0; ix < K; ++ix)
        for (int iz = 0; iz < K; ++iz) {
            const std::size_t base = static_cast<std::size_t>(ix) * K * K + iz;
            dft1d(&cube[base], K, K, scratch);
        }
    // Transform along x (slowest axis, stride K*K): one 1D DFT per (iy,iz) line.
    for (int iy = 0; iy < K; ++iy)
        for (int iz = 0; iz < K; ++iz) {
            const std::size_t base = static_cast<std::size_t>(iy) * K + iz;
            dft1d(&cube[base], K, static_cast<std::size_t>(K) * K, scratch);
        }
    // Copy the non-redundant half (mz = 0..K/2) into the R2C-packed output.
    out.assign(static_cast<std::size_t>(K) * K * Kh, {0.0, 0.0});
    for (int mx = 0; mx < K; ++mx)
        for (int my = 0; my < K; ++my)
            for (int mz = 0; mz < Kh; ++mz)
                out[(static_cast<std::size_t>(mx) * K + my) * Kh + mz] =
                    cube[(static_cast<std::size_t>(mx) * K + my) * K + mz];
}

// ---------------------------------------------------------------------------
// pme_recip_cpu: the SPME reciprocal energy via the spread->DFT->convolve->sum
// pipeline -- the exact host twin of the GPU path.
//   E_recip = sum_full  influence[m] * |F[m]|^2
// where F is the DFT of the B-spline charge grid. Because the R2C transform only
// stores half the spectrum (Hermitian symmetry of a real grid), the bins with
// 0 < mz < K/2 represent TWO physical modes (+mz and -mz), so they are counted
// twice; mz==0 and mz==K/2 (when K even) are their own conjugate and counted once.
// We apply that multiplicity exactly so the half-spectrum sum equals the full sum.
// ---------------------------------------------------------------------------
double pme_recip_cpu(const System& s, const PmeParams& p) {
    const int K = p.K, Kh = K / 2 + 1;
    std::vector<double> grid;
    spread_charges_cpu(s, p, grid);
    std::vector<std::complex<double>> F;
    dft3d_r2c(grid, K, F);
    std::vector<double> infl;
    build_influence(s, p, infl);

    double e = 0.0;
    for (int mx = 0; mx < K; ++mx)
    for (int my = 0; my < K; ++my)
    for (int mz = 0; mz < Kh; ++mz) {
        const std::size_t idx = (static_cast<std::size_t>(mx) * K + my) * Kh + mz;
        const double mag2 = std::norm(F[idx]);              // |F|^2 = re^2 + im^2
        // Hermitian multiplicity: interior mz bins stand for two physical modes.
        const double mult = (mz == 0 || (K % 2 == 0 && mz == K / 2)) ? 1.0 : 2.0;
        e += mult * infl[idx] * mag2;
    }
    return e;
}
