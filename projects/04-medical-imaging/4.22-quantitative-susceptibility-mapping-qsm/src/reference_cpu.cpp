// ===========================================================================
// src/reference_cpu.cpp  --  CPU QSM reference (direct 3-D DFT, no cuFFT)
// ---------------------------------------------------------------------------
// Project 4.22 : Quantitative Susceptibility Mapping (QSM)
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the cuFFT-based GPU result is checked against. It
//   is written to be OBVIOUSLY correct: a plain O(N^2) discrete Fourier transform
//   (a direct sum over all voxels -- no FFT butterflies, no library) bracketing
//   the SAME per-bin dipole/inversion math the GPU uses (from qsm_core.h). So the
//   two paths differ ONLY in HOW they compute the transform (direct DFT here vs
//   cuFFT there); when they agree we believe the GPU.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// WHY A DIRECT DFT (and why it is O(N^2))
//   The DFT of an N-voxel volume is X[k] = sum_r x[r] * exp(-2*pi*i * (k.r)/dims).
//   Done as a literal double loop over (k, r) it costs O(N^2) -- unusable at
//   256^3 but perfectly fine on the tiny teaching volume, and transparently
//   correct. That O(N^2) cost is precisely the motivation for the O(N log N) FFT
//   that cuFFT runs on the GPU.
//
// READ THIS AFTER: reference_cpu.h, qsm_core.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::cos, std::sin, std::sqrt
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// A separable-per-axis pi constant (double precision). Using a named constant
// keeps the twiddle-factor expressions below readable.
static const double TWO_PI = 6.283185307179586476925286766559;

// ---------------------------------------------------------------------------
// dft3d_forward: naive forward 3-D DFT of a REAL volume.
//   in  : real volume x[r], dimensions (nx,ny,nz)
//   out : complex spectrum X[k], same dimensions, laid out identically to `in`
//         (bin (kx,ky,kz) at index (kz*ny+ky)*nx+kx). Sized by the caller.
//
//   X[kx,ky,kz] = sum over (x,y,z) of
//                 in[x,y,z] * exp(-2*pi*i * (kx*x/nx + ky*y/ny + kz*z/nz))
//
// We separate the exponent into three per-axis phase terms and use
// exp(-i*theta) = cos(theta) - i*sin(theta). This is the textbook definition;
// cuFFT computes the identical sum (up to the FFT algorithm's operation order).
// Complexity: O(N^2) -- fine for the tiny demo volume, hopeless at scan scale.
// ---------------------------------------------------------------------------
static void dft3d_forward(const Volume& in, std::vector<Complex>& out) {
    const int nx = in.nx, ny = in.ny, nz = in.nz;
    const int N = nx * ny * nz;
    out.assign(static_cast<std::size_t>(N), cplx(0.0, 0.0));

    for (int kz = 0; kz < nz; ++kz)
    for (int ky = 0; ky < ny; ++ky)
    for (int kx = 0; kx < nx; ++kx) {
        double acc_re = 0.0, acc_im = 0.0;      // accumulate this output bin
        for (int z = 0; z < nz; ++z) {
            // Per-axis phase increments -- pulled out of the inner loops so the
            // innermost loop is cheap. theta = 2*pi*(kx*x/nx + ky*y/ny + kz*z/nz).
            const double pz = TWO_PI * static_cast<double>(kz) * z / nz;
            for (int y = 0; y < ny; ++y) {
                const double py = TWO_PI * static_cast<double>(ky) * y / ny;
                for (int x = 0; x < nx; ++x) {
                    const double px = TWO_PI * static_cast<double>(kx) * x / nx;
                    const double theta = px + py + pz;
                    const double v = in.vox[in.idx(x, y, z)];
                    // exp(-i*theta) = cos(theta) - i*sin(theta); scale by v.
                    acc_re += v * std::cos(theta);
                    acc_im -= v * std::sin(theta);
                }
            }
        }
        out[static_cast<std::size_t>(in.idx(kx, ky, kz))] = cplx(acc_re, acc_im);
    }
}

// ---------------------------------------------------------------------------
// dft3d_inverse_real: naive inverse 3-D DFT, returning the REAL part.
//   in  : complex spectrum X[k]
//   out : real volume x[r], dimensions copied from `dims`
//
//   x[x,y,z] = (1/N) * sum over (kx,ky,kz) of
//              X[kx,ky,kz] * exp(+2*pi*i * (kx*x/nx + ky*y/ny + kz*z/nz))
//
// Note the +i sign and the 1/N normalization (the forward DFT above is
// unnormalized, so the inverse carries the 1/N -- exactly cuFFT's convention,
// where we divide by N ourselves). For a spectrum that is Hermitian-symmetric
// (as ours are, coming from real inputs through a real-valued kernel weight) the
// imaginary part of the inverse is ~0; we keep only the real part and the tiny
// imaginary residue is a good round-off check.
// ---------------------------------------------------------------------------
static void dft3d_inverse_real(const std::vector<Complex>& in, const Volume& dims,
                               Volume& out) {
    const int nx = dims.nx, ny = dims.ny, nz = dims.nz;
    const int N = nx * ny * nz;
    const double inv_n = 1.0 / static_cast<double>(N);
    out.nx = nx; out.ny = ny; out.nz = nz;
    out.vox.assign(static_cast<std::size_t>(N), 0.0);

    for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x) {
        double acc_re = 0.0;                    // we only need the real part
        for (int kz = 0; kz < nz; ++kz) {
            const double pz = TWO_PI * static_cast<double>(kz) * z / nz;
            for (int ky = 0; ky < ny; ++ky) {
                const double py = TWO_PI * static_cast<double>(ky) * y / ny;
                for (int kx = 0; kx < nx; ++kx) {
                    const double px = TWO_PI * static_cast<double>(kx) * x / nx;
                    const double theta = px + py + pz;
                    const Complex X = in[static_cast<std::size_t>(dims.idx(kx, ky, kz))];
                    // Re{ X * exp(+i theta) } = X.re*cos - X.im*(-sin)... expand:
                    //   (X.re + i X.im)(cos + i sin) -> real = X.re*cos - X.im*sin
                    acc_re += X.re * std::cos(theta) - X.im * std::sin(theta);
                }
            }
        }
        out.vox[static_cast<std::size_t>(dims.idx(x, y, z))] = acc_re * inv_n;
    }
}

// ---------------------------------------------------------------------------
// apply_kspace_weight: multiply every spectrum bin by a REAL per-bin weight
// w(k) computed from the dipole kernel. This is the heart of every method here
// (forward or inverse); only the choice of weight differs. The weight comes from
// the SHARED qsm_core.h functions, so the GPU applies the identical numbers.
//   spec : in/out complex spectrum
//   dims : grid dimensions (for signed_freq and indexing)
//   mode : 0 = forward dipole D(k); 1 = TKD 1/D_thr; 2 = Tikhonov D/(D^2+a)
//   param: thr (mode 1) or alpha (mode 2); unused for mode 0
// ---------------------------------------------------------------------------
static void apply_kspace_weight(std::vector<Complex>& spec, const Volume& dims,
                                int mode, double param) {
    const int nx = dims.nx, ny = dims.ny, nz = dims.nz;
    for (int kz = 0; kz < nz; ++kz) {
        const double fz = static_cast<double>(signed_freq(kz, nz)) / nz;  // scaled freq
        for (int ky = 0; ky < ny; ++ky) {
            const double fy = static_cast<double>(signed_freq(ky, ny)) / ny;
            for (int kx = 0; kx < nx; ++kx) {
                const double fx = static_cast<double>(signed_freq(kx, nx)) / nx;
                const double D = dipole_kernel(fx, fy, fz);   // shared math
                double w;
                if (mode == 0)      w = D;                          // forward
                else if (mode == 1) w = tkd_reciprocal(D, param);   // TKD inverse
                else                w = tikhonov_exact_weight(D, param); // Wiener
                Complex& c = spec[static_cast<std::size_t>(dims.idx(kx, ky, kz))];
                c = cscale(c, w);   // real scale of a complex bin
            }
        }
    }
}

// ---------------------------------------------------------------------------
// make_field_from_chi: forward dipole model. DFT(chi) -> multiply by D(k) -> IDFT.
// (Contract in reference_cpu.h.)
// ---------------------------------------------------------------------------
Volume make_field_from_chi(const Volume& chi) {
    std::vector<Complex> spec;
    dft3d_forward(chi, spec);                 // DFT(chi)
    apply_kspace_weight(spec, chi, /*mode=*/0, 0.0);  // multiply by D(k)
    Volume field;
    dft3d_inverse_real(spec, chi, field);     // IDFT -> field-shift map
    return field;
}

// ---------------------------------------------------------------------------
// reconstruct_tkd_cpu: TKD inverse. DFT(field) -> * 1/D_thr -> IDFT.
// ---------------------------------------------------------------------------
Volume reconstruct_tkd_cpu(const Volume& field, double thr) {
    std::vector<Complex> spec;
    dft3d_forward(field, spec);
    apply_kspace_weight(spec, field, /*mode=*/1, thr);   // TKD reciprocal
    Volume chi;
    dft3d_inverse_real(spec, field, chi);
    return chi;
}

// ---------------------------------------------------------------------------
// reconstruct_tikhonov_cpu: closed-form Wiener filter. DFT -> * D/(D^2+a) -> IDFT.
// ---------------------------------------------------------------------------
Volume reconstruct_tikhonov_cpu(const Volume& field, double alpha) {
    std::vector<Complex> spec;
    dft3d_forward(field, spec);
    apply_kspace_weight(spec, field, /*mode=*/2, alpha);  // exact Tikhonov weight
    Volume chi;
    dft3d_inverse_real(spec, field, chi);
    return chi;
}

// ---------------------------------------------------------------------------
// reconstruct_tikhonov_iter_cpu: iterative gradient descent (the GPU's twin).
//   1. DFT(field) -> Ffield (fixed data spectrum)
//   2. Start Fchi = 0
//   3. For each iteration, update EVERY bin with the shared tikhonov_grad_step().
//   4. IDFT(Fchi) -> chi
// Because the Tikhonov objective decouples per bin (D is diagonal in k-space),
// each bin runs an independent 1-D gradient descent; with a valid step size it
// converges to the closed-form weight above. This is the loop the GPU parallel-
// izes across bins (one thread per bin) -- see kernels.cu.
// ---------------------------------------------------------------------------
Volume reconstruct_tikhonov_iter_cpu(const Volume& field, double alpha,
                                     double step, int iters) {
    const int nx = field.nx, ny = field.ny, nz = field.nz;
    const int N = nx * ny * nz;

    std::vector<Complex> Ffield;
    dft3d_forward(field, Ffield);                       // fixed data spectrum

    // Current estimate of chi's spectrum, initialized to zero everywhere.
    std::vector<Complex> Fchi(static_cast<std::size_t>(N), cplx(0.0, 0.0));

    for (int it = 0; it < iters; ++it) {
        for (int kz = 0; kz < nz; ++kz) {
            const double fz = static_cast<double>(signed_freq(kz, nz)) / nz;
            for (int ky = 0; ky < ny; ++ky) {
                const double fy = static_cast<double>(signed_freq(ky, ny)) / ny;
                for (int kx = 0; kx < nx; ++kx) {
                    const double fx = static_cast<double>(signed_freq(kx, nx)) / nx;
                    const double D = dipole_kernel(fx, fy, fz);
                    const int i = field.idx(kx, ky, kz);
                    // One shared gradient step for this bin (CPU == GPU math).
                    Fchi[static_cast<std::size_t>(i)] =
                        tikhonov_grad_step(Fchi[static_cast<std::size_t>(i)],
                                           Ffield[static_cast<std::size_t>(i)],
                                           D, alpha, step);
                }
            }
        }
    }

    Volume chi;
    dft3d_inverse_real(Fchi, field, chi);
    return chi;
}

// ---------------------------------------------------------------------------
// load_volume: parse the tiny text volume format (see data/README.md).
//   header: "<nx> <ny> <nz>"  then nx*ny*nz doubles (x fastest). Row-major.
// ---------------------------------------------------------------------------
Volume load_volume(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open volume file: " + path);
    Volume v;
    if (!(in >> v.nx >> v.ny >> v.nz) || v.nx <= 0 || v.ny <= 0 || v.nz <= 0)
        throw std::runtime_error("bad header (expected 'nx ny nz') in " + path);
    v.vox.resize(static_cast<std::size_t>(v.nx) * v.ny * v.nz);
    for (std::size_t i = 0; i < v.vox.size(); ++i)
        if (!(in >> v.vox[i]))
            throw std::runtime_error("volume data truncated in " + path);
    return v;
}

// ---------------------------------------------------------------------------
// rms: root-mean-square magnitude of a volume (one scalar).
// ---------------------------------------------------------------------------
double rms(const Volume& v) {
    if (v.vox.empty()) return 0.0;
    double acc = 0.0;
    for (double x : v.vox) acc += x * x;
    return std::sqrt(acc / static_cast<double>(v.vox.size()));
}

// ---------------------------------------------------------------------------
// rms_diff: RMS of the voxelwise difference of two equal-sized volumes. Returns
// a large sentinel on a size mismatch so a shape bug cannot masquerade as
// agreement.
// ---------------------------------------------------------------------------
double rms_diff(const Volume& a, const Volume& b) {
    if (a.vox.size() != b.vox.size() || a.vox.empty()) return 1.0e300;
    double acc = 0.0;
    for (std::size_t i = 0; i < a.vox.size(); ++i) {
        const double d = a.vox[i] - b.vox[i];
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(a.vox.size()));
}
