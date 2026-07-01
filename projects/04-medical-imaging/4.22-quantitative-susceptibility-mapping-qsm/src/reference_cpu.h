// ===========================================================================
// src/reference_cpu.h  --  Volume model, k-space grid, and the CPU QSM reference
// ---------------------------------------------------------------------------
// Project 4.22 : Quantitative Susceptibility Mapping (QSM)
//
// WHAT THIS PROJECT COMPUTES
//   Quantitative Susceptibility Mapping (QSM) turns the PHASE of a gradient-echo
//   MRI scan into a map of tissue magnetic SUSCEPTIBILITY chi(r) -- a physical
//   property that distinguishes iron-rich deep-brain nuclei, calcifications,
//   veins (deoxygenated blood), and myelin. The phase gives a local field-shift
//   map delta-B(r); chi and delta-B are related by convolution with the magnetic
//   DIPOLE kernel. In k-space that convolution is a simple multiply:
//
//       Fhat_field[k] = D(k) * Fhat_chi[k],   D(k) = 1/3 - kz^2/|k|^2   (B0 || z)
//
//   RECONSTRUCTION = the INVERSE: recover chi from the field map by undoing the
//   multiply by D(k). This is ill-posed because D(k) = 0 on a double cone (the
//   "magic angle"), so a naive division by D(k) explodes into streaking noise.
//   We implement the two canonical fixes:
//     * TKD  (Threshold-based K-space Division): clamp |D| away from 0, divide.
//     * Tikhonov-regularized least squares: min ||D.Fchi - Ffield||^2 + a||Fchi||^2,
//       solved both in closed form (a Wiener filter) and by ITERATIVE gradient
//       descent (the structure real MEDI-style solvers use).
//
// WHY A GPU / cuFFT
//   Every method is bracketed by 3-D Fourier transforms of the whole volume, and
//   the iterative method runs O(100) of them (forward + inverse per iteration) on
//   volumes up to 256^3. That is exactly what cuFFT accelerates. This CPU
//   reference instead does a plain, obviously-correct O(N^2) discrete Fourier
//   transform (a direct sum over all voxels) on a TINY volume, so we can verify
//   the cuFFT-based GPU result bin-for-bin. The per-bin dipole/inversion math is
//   SHARED with the GPU via qsm_core.h, so the two paths differ ONLY in the FFT.
//
//   This is a pure C++ header (no CUDA): kernels.cu / main.cu reuse the Volume
//   struct and the same synthetic-field builder, so both paths share inputs.
//
// READ THIS BEFORE: qsm_core.h (the shared per-bin math), kernels.cuh.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "qsm_core.h"   // Complex, dipole_kernel(), tkd_reciprocal(), Tikhonov

// A real-valued 3-D scalar volume (a susceptibility map or a field map) stored
// as a flat row-major array with x fastest, then y, then z:
//   vox[(z*ny + y)*nx + x]  = value at grid point (x,y,z).
// We use double throughout: QSM inversion divides by tiny dipole values, so
// precision matters, and the teaching volumes are small enough that double is
// free. nx,ny,nz are the grid dimensions (voxel counts) along each axis.
struct Volume {
    int nx = 0, ny = 0, nz = 0;      // dimensions (voxels) along x, y, z
    std::vector<double> vox;         // size nx*ny*nz, row-major (x fastest)
    int size() const { return nx * ny * nz; }
    // Flatten a 3-D index to the 1-D storage index (documented above).
    int idx(int x, int y, int z) const { return (z * ny + y) * nx + x; }
};

// ---- k-space coordinate helper --------------------------------------------
//
// signed_freq: map an FFT bin index i in [0, n) to its SIGNED integer frequency.
//   The DFT packs frequencies as 0,1,...,n/2-1, then the negative half. So bins
//   above n/2 represent negative frequencies: bin i -> i - n. Using the signed
//   frequency (not the raw index) is what makes the dipole kernel symmetric and
//   physically correct. Returns a value in roughly [-n/2, n/2).
inline int signed_freq(int i, int n) { return (i <= n / 2) ? i : i - n; }

// ---- Forward model: build the measured field map from a known chi -----------
//
// make_field_from_chi: apply the dipole forward model in k-space to a
// susceptibility volume, producing the field-shift volume that a scanner would
// measure. Steps: DFT(chi) -> multiply each bin by dipole_kernel(k) -> IDFT.
// Used by main.cu to synthesize the demo INPUT from a known ground-truth chi, so
// we can later check the reconstruction recovers it. Deterministic.
//   chi : the (synthetic) ground-truth susceptibility volume
// Returns the field-shift volume (same dimensions), real-valued.
Volume make_field_from_chi(const Volume& chi);

// ---- The CPU reconstruction references (the trusted baselines) --------------

// reconstruct_tkd_cpu: recover chi from a field map by Threshold-based K-space
// Division. DFT(field) -> multiply each bin by tkd_reciprocal(D(k), thr) -> IDFT.
//   field : the measured field-shift volume (input)
//   thr   : TKD threshold (e.g. 0.15)
// Returns the reconstructed susceptibility volume. This is the direct, one-shot
// method (no iteration) and its GPU twin is verified against it.
Volume reconstruct_tkd_cpu(const Volume& field, double thr);

// reconstruct_tikhonov_cpu: recover chi by the CLOSED-FORM Tikhonov (Wiener)
// filter, DFT(field) -> multiply by tikhonov_exact_weight(D(k), alpha) -> IDFT.
//   field : the measured field-shift volume
//   alpha : Tikhonov regularization weight (e.g. 0.05)
// This is the exact minimizer that the ITERATIVE GPU path should converge to; we
// use it to check that convergence.
Volume reconstruct_tikhonov_cpu(const Volume& field, double alpha);

// reconstruct_tikhonov_iter_cpu: recover chi by ITERATIVE gradient descent on the
// Tikhonov objective (the structure real iterative QSM uses). DFT(field) once,
// then `iters` gradient steps per bin via tikhonov_grad_step(), then IDFT.
//   field : the measured field-shift volume
//   alpha : Tikhonov weight
//   step  : gradient-descent step size
//   iters : number of gradient iterations
// The GPU runs the SAME iteration; this reference verifies it exactly.
Volume reconstruct_tikhonov_iter_cpu(const Volume& field, double alpha,
                                     double step, int iters);

// ---- I/O ------------------------------------------------------------------

// Load a field-map volume from the tiny text sample format (see data/README):
//   header line: "<nx> <ny> <nz>"
//   then nx*ny*nz whitespace-separated doubles (x fastest, then y, then z).
// Throws std::runtime_error on a malformed/missing file so demos fail loudly.
Volume load_volume(const std::string& path);

// ---- Scalar summaries used for the deterministic report --------------------

// rms: root-mean-square of a volume's values (a single magnitude number).
double rms(const Volume& v);

// rms_diff: RMS of the voxelwise difference between two equal-sized volumes -- a
// single "how far apart are these two reconstructions" number. Used both to
// verify GPU vs CPU and to score a reconstruction against the ground-truth chi.
double rms_diff(const Volume& a, const Volume& b);
