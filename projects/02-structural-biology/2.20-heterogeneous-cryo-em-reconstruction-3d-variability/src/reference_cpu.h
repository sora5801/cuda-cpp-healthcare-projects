// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for 3D Variability (3DVA)
// ---------------------------------------------------------------------------
// Project 2.20 : Heterogeneous Cryo-EM Reconstruction (3D Variability)
//   (a REDUCED-SCOPE teaching version -- see ../THEORY.md "Where this sits in
//    the real world" for why, and what the full cryoDRGN VAE adds.)
//
// WHAT THIS PROJECT COMPUTES (the one-sentence version)
//   A cryo-EM dataset of one "flexible" protein is really a CLOUD of slightly
//   different 3D density maps -- one conformation per particle. cryoSPARC's 3D
//   Variability Analysis (3DVA) finds the few directions of LARGEST variation
//   in that cloud by Principal Component Analysis (PCA) on the volumes. PC1 is
//   "the main way this molecule moves"; each particle's coordinate along PC1 is
//   its position in that motion. This project computes exactly that, on the GPU,
//   via the covariance eigenvalue problem.
//
// WHY A SEPARATE PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the plain host compiler (cl.exe / g++) and
//   must NOT see any CUDA/__global__ syntax, so its prototypes live HERE, not in
//   kernels.cuh. main.cu (nvcc) and reference_cpu.cpp (cl.exe) both include this
//   header so they agree on the data model and the function signatures.
//
// THE SHARED __host__ __device__ CORE (PATTERNS.md §2)
//   The *per-element math* that BOTH the CPU reference and the GPU kernels must
//   compute identically (centering a volume, one Gram-matrix dot product, one
//   projection coefficient) is written ONCE here as HD-decorated inline
//   functions, so CPU and GPU run byte-for-byte-identical arithmetic. Keep this
//   header free of __global__ and CUDA-only types so the host compiler is happy.
//
// READ THIS BEFORE: reference_cpu.cpp (uses these), kernels.cuh/.cu (the GPU
//   path), main.cu (the 5-step driver). The science is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <cstddef>   // std::size_t
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// HD: the "__host__ __device__" portability macro (PATTERNS.md §2).
//   When this header is pulled in by nvcc (which defines __CUDACC__) we decorate
//   the shared math so it can run on BOTH the CPU and inside a kernel. When the
//   plain host compiler includes it, those CUDA keywords do not exist, so the
//   macro expands to nothing. One source of truth, two compilers, same math.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// VolumeSet -- the in-memory dataset (the "particle volumes").
//   In real 3DVA each particle image is first back-projected into a small 3D
//   density map (a "volume"). We skip the imaging step (that is project 2.xx
//   reconstruction's job) and start from N such volumes, each a G x G x G cube
//   FLATTENED to a length-D row, D = G^3. So `vox` is an N x D matrix, row-major:
//   vox[p*D + v] is voxel v of particle p.
//
//   We also carry `truth[p]`, the GROUND-TRUTH conformational coordinate used to
//   synthesize particle p (e.g. how far a density blob has slid along z). It is
//   NOT used by the algorithm -- it is the hidden answer we check 3DVA recovers
//   (PATTERNS.md §6: embed a known answer in synthetic data).
// ---------------------------------------------------------------------------
struct VolumeSet {
    int N = 0;                      // number of particle volumes (rows)
    int G = 0;                      // grid edge length; cube is G x G x G
    int D = 0;                      // voxels per volume = G*G*G (columns)
    std::vector<double> vox;        // [N*D] density values, row-major (double for exactness)
    std::vector<double> truth;      // [N]   ground-truth conformational coordinate
};

// load_volumes: parse the tiny text sample (see data/README.md for the format).
//   Throws std::runtime_error on any malformed input so demos fail loudly.
VolumeSet load_volumes(const std::string& path);

// ===========================================================================
// THE SHARED __host__ __device__ MATH CORE
// Each function is the single definition used by both the CPU reference and the
// GPU kernel, guaranteeing identical results.
// ===========================================================================

// centered_value: voxel v of particle p AFTER subtracting the dataset mean.
//   3DVA / PCA operates on MEAN-SUBTRACTED data: we care about how each volume
//   DEVIATES from the average shape, not the average shape itself. This helper
//   is the atom of that operation -- both the Gram matrix and the projections
//   are built from centered values, so we define it once.
//     vox  : [N*D] the raw volume matrix (row-major)
//     mean : [D]   per-voxel mean over all N particles
//     p    : particle (row) index, 0..N-1
//     v    : voxel (column) index, 0..D-1
//     D    : voxels per volume (row stride)
HD inline double centered_value(const double* vox, const double* mean,
                                int p, int v, int D) {
    // Row p starts at offset p*D; voxel v is the v-th entry of that row.
    return vox[(std::size_t)p * D + v] - mean[v];
}

// gram_entry: one entry G[i][j] of the N x N covariance (Gram) matrix.
//   THE KEY TRICK (eigenfaces / snapshot method, THEORY §The algorithm):
//   PCA wants the eigenvectors of the D x D covariance C = (1/N) X^T X, but D is
//   huge (D = G^3) while N is small. The nonzero spectrum of C equals that of
//   the N x N matrix Gtil = (1/N) X X^T (X is the N x D CENTERED data). We
//   diagonalize the SMALL Gtil instead. gram_entry computes one dot product
//   between centered particle i and centered particle j, divided by N:
//     Gtil[i][j] = (1/N) * sum_v ( Xc[i][v] * Xc[j][v] )
//   It is symmetric (Gtil[i][j] == Gtil[j][i]); a thread computes one (i,j).
//     vox  : [N*D] raw volumes;  mean : [D] per-voxel mean
//     i,j  : the two particle indices;  N : particle count;  D : voxels/volume
HD inline double gram_entry(const double* vox, const double* mean,
                            int i, int j, int N, int D) {
    double acc = 0.0;                       // running dot product of two centered rows
    for (int v = 0; v < D; ++v) {
        double xi = centered_value(vox, mean, i, v, D);  // Xc[i][v]
        double xj = centered_value(vox, mean, j, v, D);  // Xc[j][v]
        acc += xi * xj;                     // accumulate the inner product
    }
    return acc / (double)N;                 // covariance normalization (1/N)
}

// project_particle: the latent coordinate of particle p along a chosen mode.
//   Once we have a volume-space principal component u (a length-D unit vector,
//   "the way the molecule moves"), each particle's position in that motion is
//   the projection of its CENTERED volume onto u:
//       z[p] = sum_v ( Xc[p][v] * u[v] )
//   This single number per particle IS the 3DVA "latent coordinate" -- exactly
//   what cryoSPARC plots to show a continuous conformational trajectory.
//     vox  : [N*D] raw volumes;  mean : [D] mean;  u : [D] the mode (unit vector)
//     p    : particle index;  D : voxels/volume
HD inline double project_particle(const double* vox, const double* mean,
                                  const double* u, int p, int D) {
    double z = 0.0;                         // running projection coefficient
    for (int v = 0; v < D; ++v) {
        z += centered_value(vox, mean, p, v, D) * u[v];  // Xc[p][v] * u[v]
    }
    return z;
}

// ===========================================================================
// CPU REFERENCE (declared here, defined in reference_cpu.cpp)
// The host path mirrors the GPU path so the demo can VERIFY they agree.
// ===========================================================================

// compute_mean: per-voxel mean volume, mean[v] = (1/N) sum_p vox[p][v].
//   Output `mean` is resized to D. This is the "average shape" we subtract.
void compute_mean(const VolumeSet& vs, std::vector<double>& mean);

// build_gram_cpu: fill the N x N covariance/Gram matrix (row-major) on the CPU
//   using gram_entry above. This is the matrix we eigendecompose.
void build_gram_cpu(const VolumeSet& vs, const std::vector<double>& mean,
                    std::vector<double>& gram);

// jacobi_eigen_symmetric: classic Jacobi eigensolver for a small symmetric
//   matrix -- our READABLE, dependency-free reference for cuSOLVER's Dsyevd.
//     A      : [n*n] symmetric input (row-major); not modified
//     n      : matrix dimension (here n = N particles)
//     eval   : [n]   eigenvalues, returned ASCENDING (to match cuSOLVER)
//     evec   : [n*n] eigenvectors, COLUMN k = eigenvector k (column-major)
//   See THEORY "How we verify correctness" for why Jacobi is the right teaching
//   reference (transparent, deterministic, no LAPACK dependency).
void jacobi_eigen_symmetric(const std::vector<double>& A, int n,
                            std::vector<double>& eval, std::vector<double>& evec);

// lift_to_volume_pc: turn an eigenvector of the SMALL Gram matrix into the
//   actual volume-space principal component u (length D), then normalize it.
//   Math (THEORY §The algorithm): if w is a unit eigenvector of Gtil = (1/N)XX^T
//   with eigenvalue lambda>0, then u = X^T w / ||X^T w|| is the matching unit
//   eigenvector of the big covariance C = (1/N)X^T X. We build X^T w voxel by
//   voxel from centered data, then scale to unit length.
//     vs    : the dataset;  mean : [D] mean
//     gevec : [N*N] Gram eigenvectors (column-major);  k : which column (mode)
//     u     : [D] output volume-space PC (resized, unit-normalized)
void lift_to_volume_pc(const VolumeSet& vs, const std::vector<double>& mean,
                       const std::vector<double>& gevec, int k,
                       std::vector<double>& u);

// project_all_cpu: latent coordinate of every particle along mode u (length N),
//   using project_particle above. Returns z[p] for p = 0..N-1.
void project_all_cpu(const VolumeSet& vs, const std::vector<double>& mean,
                     const std::vector<double>& u, std::vector<double>& z);
