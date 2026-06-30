// ===========================================================================
// src/reference_cpu.cpp  --  CPU reference: loader, RSCC, naive-DFT FSC
// ---------------------------------------------------------------------------
// Project 2.22 : Electron Density Map Analysis & Model Validation
//
// Compiled by the HOST compiler only (cl.exe / g++). It must not contain any
// CUDA syntax. The per-voxel formulas come from map_core.h (the same inline
// functions the GPU kernels call), so the reference and the GPU agree exactly
// in structure -- the whole point of the CPU baseline (CLAUDE.md §5).
//
// The FSC reference computes the 3-D Discrete Fourier Transform BY HAND (a
// separable per-axis DFT). That is deliberately slow: it is transparently the
// textbook DFT, so when cuFFT's fast transform agrees with it we trust cuFFT.
// We keep the sample grid tiny (n=16) so this reference runs in well under a
// second; THEORY.md §complexity explains why the FFT is mandatory at real map
// sizes (a 256³ map has 16.7M voxels).
// ===========================================================================
#include "reference_cpu.h"
#include "map_core.h"      // Cplx, fft_freq, shell_index, fsc_accumulate, fsc_from_sums, pearson_from_sums

#include <algorithm>       // std::swap
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// load_map: read "<n> <voxel>" then n³ floats for A then n³ floats for B.
// ---------------------------------------------------------------------------
DensityMap load_map(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open map file: " + path);
    DensityMap d;
    if (!(in >> d.n >> d.voxel_angstrom) || d.n <= 0 || d.voxel_angstrom <= 0.0)
        throw std::runtime_error("bad header (expected 'n voxel_angstrom') in " + path);

    const long long total = d.voxels();
    d.a.resize(static_cast<std::size_t>(total));
    d.b.resize(static_cast<std::size_t>(total));
    for (long long i = 0; i < total; ++i)
        if (!(in >> d.a[static_cast<std::size_t>(i)]))
            throw std::runtime_error("map A truncated in " + path);
    for (long long i = 0; i < total; ++i)
        if (!(in >> d.b[static_cast<std::size_t>(i)]))
            throw std::runtime_error("map B truncated in " + path);
    return d;
}

// ---------------------------------------------------------------------------
// rscc_cpu: Pearson correlation of the two maps over all voxels.
//   We accumulate the five sums (Σa, Σb, Σa², Σb², Σab) in one pass, then close
//   with pearson_from_sums() -- the EXACT formula the GPU reduction also uses.
//   Sums are double so the single-pass formula stays well-conditioned.
// ---------------------------------------------------------------------------
double rscc_cpu(const DensityMap& d) {
    const long long total = d.voxels();
    double Sa = 0, Sb = 0, Saa = 0, Sbb = 0, Sab = 0;
    for (long long i = 0; i < total; ++i) {
        const double av = d.a[static_cast<std::size_t>(i)];
        const double bv = d.b[static_cast<std::size_t>(i)];
        Sa  += av;
        Sb  += bv;
        Saa += av * av;
        Sbb += bv * bv;
        Sab += av * bv;
    }
    return pearson_from_sums(static_cast<double>(total), Sa, Sb, Saa, Sbb, Sab);
}

// ---------------------------------------------------------------------------
// dft3d: the 3-D Discrete Fourier Transform of a real n³ map, BY HAND.
//   F(kx,ky,kz) = Σ_{x,y,z} f(x,y,z) · exp(-2πi (kx·x + ky·y + kz·z)/n)
//   Computed as a SEPARABLE transform: DFT along x for every (y,z) line, then
//   along y, then along z. That is O(n⁴) total (n³ voxels × n work × 3 axes) --
//   fine for the tiny n=16 sample, and a transparent stand-in for what cuFFT
//   does in O(n³ log n). Output is the full complex cube in the maps' C-order.
// ---------------------------------------------------------------------------
static void dft3d(const std::vector<float>& f, int n, std::vector<Cplx>& out) {
    const std::size_t total = static_cast<std::size_t>(n) * n * n;
    // Two complex scratch cubes we ping-pong between as we transform each axis.
    std::vector<Cplx> cur(total), nxt(total);
    for (std::size_t i = 0; i < total; ++i) { cur[i].re = f[i]; cur[i].im = 0.0; }

    // Precompute the twiddle factors W[k] = exp(-2πi k / n) for k = 0..n-1, so
    // the inner loop is a table lookup instead of a transcendental call.
    std::vector<double> wcos(n), wsin(n);
    for (int k = 0; k < n; ++k) {
        const double ang = -2.0 * M_PI * k / n;
        wcos[k] = std::cos(ang);
        wsin[k] = std::sin(ang);
    }

    // dft_axis: 1-D DFT applied along the axis whose neighbouring samples are
    //   `stride` voxels apart (x: stride 1, y: stride n, z: stride n²). Each line
    //   is the n samples at base + j*stride for j=0..n-1; a line's `base` is any
    //   flat index whose component ALONG this axis is 0.
    auto dft_axis = [&](const std::vector<Cplx>& src, std::vector<Cplx>& dst,
                        std::size_t stride) {
        const int nn = n;
        for (std::size_t base = 0; base < total; ++base) {
            // Keep only the start voxel of each line (axis-index == 0).
            if ((base / stride) % static_cast<std::size_t>(nn) != 0) continue;
            for (int kk = 0; kk < nn; ++kk) {
                double re = 0.0, im = 0.0;
                for (int j = 0; j < nn; ++j) {
                    const Cplx s = src[base + static_cast<std::size_t>(j) * stride];
                    const int w = (kk * j) % nn;             // twiddle table index
                    // complex multiply  s · W[w]  =  (s.re + i s.im)(wcos + i wsin)
                    re += s.re * wcos[w] - s.im * wsin[w];
                    im += s.re * wsin[w] + s.im * wcos[w];
                }
                Cplx& o = dst[base + static_cast<std::size_t>(kk) * stride];
                o.re = re;
                o.im = im;
            }
        }
    };

    dft_axis(cur, nxt, 1);                          std::swap(cur, nxt);  // along x
    dft_axis(cur, nxt, static_cast<std::size_t>(n)); std::swap(cur, nxt); // along y
    dft_axis(cur, nxt, static_cast<std::size_t>(n) * n); std::swap(cur, nxt); // along z
    out.swap(cur);
}

// ---------------------------------------------------------------------------
// fsc_cpu: FSC curve via the naive 3-D DFT of each map, then shell binning.
//   1. F1 = DFT(a), F2 = DFT(b).
//   2. For every reciprocal-space voxel, find its shell s = round(|k|) and add
//      its three FSC accumulands (cross, |F1|², |F2|²) to that shell -- using
//      the SHARED fsc_accumulate() so the GPU gets identical sums.
//   3. FSC[s] = cross / sqrt(p1·p2) per shell (shared fsc_from_sums()).
// ---------------------------------------------------------------------------
void fsc_cpu(const DensityMap& d, std::vector<double>& fsc,
             std::vector<long long>& shell_count) {
    const int n = d.n;
    std::vector<Cplx> F1, F2;
    dft3d(d.a, n, F1);
    dft3d(d.b, n, F2);

    const int n_shells = max_shell(n);              // shared bound (map_core.h)
    std::vector<double> cross(n_shells, 0.0), p1(n_shells, 0.0), p2(n_shells, 0.0);
    shell_count.assign(n_shells, 0);

    // Bin every voxel into its spherical shell, accumulating the three sums.
    for (int z = 0; z < n; ++z) {
        const int kz = fft_freq(z, n);
        for (int y = 0; y < n; ++y) {
            const int ky = fft_freq(y, n);
            for (int x = 0; x < n; ++x) {
                const int kx = fft_freq(x, n);
                const int s = shell_index(kx, ky, kz);
                const std::size_t idx =
                    (static_cast<std::size_t>(z) * n + y) * n + x;
                fsc_accumulate(F1[idx], F2[idx], &cross[s], &p1[s], &p2[s]);
                ++shell_count[s];
            }
        }
    }

    fsc.assign(n_shells, 0.0);
    for (int s = 0; s < n_shells; ++s)
        fsc[s] = fsc_from_sums(cross[s], p1[s], p2[s]);
}

// ---------------------------------------------------------------------------
// resolution_at_threshold: walk shells from low to high frequency; return the
//   last shell index that stays at/above `threshold` before the curve first
//   drops below it. Skips empty shells (count 0). Shell 0 (DC) is always 1.0,
//   so the loop starts at shell 1.
// ---------------------------------------------------------------------------
int resolution_at_threshold(const std::vector<double>& fsc,
                            const std::vector<long long>& shell_count,
                            double threshold) {
    int last_good = 0;
    for (std::size_t s = 1; s < fsc.size(); ++s) {
        if (shell_count[s] == 0) continue;       // no voxels in this shell
        if (fsc[s] >= threshold) last_good = static_cast<int>(s);
        else break;                              // first crossing -> stop
    }
    return last_good;
}

// ---------------------------------------------------------------------------
// shell_to_res: frequency shell -> resolution in Å (period = box / frequency).
// ---------------------------------------------------------------------------
double shell_to_res(int shell, int n, double voxel_angstrom) {
    if (shell <= 0) return std::numeric_limits<double>::infinity();
    return (static_cast<double>(n) * voxel_angstrom) / shell;
}
