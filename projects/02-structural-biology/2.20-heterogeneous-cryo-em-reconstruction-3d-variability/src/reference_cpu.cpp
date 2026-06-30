// ===========================================================================
// src/reference_cpu.cpp  --  Plain-C++ reference for 3D Variability (3DVA)
// ---------------------------------------------------------------------------
// Project 2.20 : Heterogeneous Cryo-EM Reconstruction (3D Variability)
//
// ROLE
//   The transparent, GPU-free baseline. It computes the SAME 3DVA result as the
//   GPU path -- mean volume, N x N covariance (Gram) matrix, eigendecomposition,
//   lift to a volume-space principal component, and per-particle projections --
//   so main.cu can VERIFY the GPU against it within a documented tolerance.
//
//   The per-element math (centering, Gram dot product, projection) is NOT
//   re-derived here: it is included from reference_cpu.h as shared
//   __host__ __device__ helpers, so CPU and GPU compute identical values.
//   What lives here is the host-side ORCHESTRATION plus the Jacobi eigensolver
//   (our readable stand-in for cuSOLVER's Dsyevd).
//
// READ THIS AFTER: reference_cpu.h (the data model + shared math). The GPU
//   mirror is kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::fabs, std::sqrt
#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream
#include <stdexcept>   // std::runtime_error
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// load_volumes: parse the tiny committed sample (see data/README.md).
//   FORMAT (whitespace-separated, '#' starts a comment line):
//     line 1 : three ints  N G D        (D must equal G*G*G; checked)
//     next N : D doubles each            -> one flattened G^3 volume per row
//     last 1 : N doubles                 -> ground-truth conformational coords
//   We parse defensively: any shortfall throws so the demo fails loudly rather
//   than silently computing on garbage.
// ---------------------------------------------------------------------------
VolumeSet load_volumes(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open volume file: " + path);

    // Read the whole file into a token stream, skipping '#' comment lines so the
    // sample can document itself inline.
    std::vector<double> tok;            // every numeric token, in order
    std::string line;
    while (std::getline(in, line)) {
        // Strip anything from a '#' onward (inline or full-line comment).
        const std::size_t hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        std::istringstream ls(line);
        double x;
        while (ls >> x) tok.push_back(x);
    }
    if (tok.size() < 3) throw std::runtime_error("volume file truncated (need header)");

    VolumeSet vs;
    std::size_t k = 0;                  // cursor into the token stream
    vs.N = (int)tok[k++];               // particle count
    vs.G = (int)tok[k++];               // grid edge
    vs.D = (int)tok[k++];               // voxels per volume
    if (vs.N <= 1 || vs.G <= 0 || vs.D <= 0)
        throw std::runtime_error("volume header has non-positive / degenerate sizes");
    if (vs.D != vs.G * vs.G * vs.G)
        throw std::runtime_error("header inconsistent: D must equal G*G*G");

    // We need N*D voxel values then N truth values after the header.
    const std::size_t need = (std::size_t)vs.N * vs.D + (std::size_t)vs.N;
    if (tok.size() - k < need)
        throw std::runtime_error("volume file has fewer values than the header promises");

    vs.vox.resize((std::size_t)vs.N * vs.D);
    for (std::size_t i = 0; i < vs.vox.size(); ++i) vs.vox[i] = tok[k++];
    vs.truth.resize(vs.N);
    for (int p = 0; p < vs.N; ++p) vs.truth[p] = tok[k++];
    return vs;
}

// ---------------------------------------------------------------------------
// compute_mean: per-voxel average over all N particles, mean[v]=(1/N)sum_p vox.
//   This is the "average shape". Subtracting it (centering) is what turns raw
//   volumes into the deviations PCA explains.
// ---------------------------------------------------------------------------
void compute_mean(const VolumeSet& vs, std::vector<double>& mean) {
    mean.assign(vs.D, 0.0);                          // accumulator, one per voxel
    for (int p = 0; p < vs.N; ++p)
        for (int v = 0; v < vs.D; ++v)
            mean[v] += vs.vox[(std::size_t)p * vs.D + v];  // sum over particles
    const double inv = 1.0 / (double)vs.N;
    for (int v = 0; v < vs.D; ++v) mean[v] *= inv;   // divide by N -> mean
}

// ---------------------------------------------------------------------------
// build_gram_cpu: the N x N covariance/Gram matrix (row-major), using the
//   shared gram_entry(). Symmetric, so we compute the upper triangle and mirror
//   it -- exactly half the dot products of the naive loop.
// ---------------------------------------------------------------------------
void build_gram_cpu(const VolumeSet& vs, const std::vector<double>& mean,
                    std::vector<double>& gram) {
    const int N = vs.N;
    gram.assign((std::size_t)N * N, 0.0);
    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {                // upper triangle incl. diagonal
            const double g = gram_entry(vs.vox.data(), mean.data(), i, j, N, vs.D);
            gram[(std::size_t)i * N + j] = g;        // G[i][j]
            gram[(std::size_t)j * N + i] = g;        // mirror -> G[j][i]
        }
    }
}

// ---------------------------------------------------------------------------
// jacobi_eigen_symmetric: the classic cyclic Jacobi rotation method.
//   IDEA: repeatedly zero off-diagonal entries with 2x2 rotations; the
//   accumulated rotations converge the matrix to diagonal (the eigenvalues) and
//   the product of rotations to the eigenvectors. O(n^3) per sweep, a handful of
//   sweeps -- perfect for our small N x N Gram matrix and fully transparent.
//   We finish by SORTING ascending so the order matches cuSOLVER's Dsyevd, which
//   makes the eigenvalue-by-eigenvalue comparison in main.cu meaningful.
// ---------------------------------------------------------------------------
void jacobi_eigen_symmetric(const std::vector<double>& A, int n,
                            std::vector<double>& eval, std::vector<double>& evec) {
    // Work on a mutable copy M (row-major). V accumulates the eigenvectors and
    // starts as the identity.
    std::vector<double> M = A;
    std::vector<double> V((std::size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i) V[(std::size_t)i * n + i] = 1.0;

    const int    MAX_SWEEPS = 100;       // ample for our small matrices
    const double EPS = 1e-300;           // below this an off-diagonal is "zero"

    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {
        // off = sqrt(2 * sum of squared upper off-diagonals): convergence gauge.
        double off = 0.0;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q)
                off += M[(std::size_t)p * n + q] * M[(std::size_t)p * n + q];
        off = std::sqrt(2.0 * off);
        if (off < 1e-15) break;          // converged: matrix is (numerically) diagonal

        // One sweep: rotate away every off-diagonal (p,q), p<q.
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = M[(std::size_t)p * n + q];
                if (std::fabs(apq) < EPS) continue;        // already zero, skip
                const double app = M[(std::size_t)p * n + p];
                const double aqq = M[(std::size_t)q * n + q];
                // Rotation angle that zeros M[p][q] (standard Jacobi formula).
                const double tau = (aqq - app) / (2.0 * apq);
                const double t   = (tau >= 0.0 ? 1.0 : -1.0)
                                 / (std::fabs(tau) + std::sqrt(1.0 + tau * tau));
                const double c   = 1.0 / std::sqrt(1.0 + t * t);   // cos(theta)
                const double s   = t * c;                          // sin(theta)

                // Apply the rotation to columns p and q of M ( M <- M J ).
                for (int k = 0; k < n; ++k) {
                    const double mkp = M[(std::size_t)k * n + p];
                    const double mkq = M[(std::size_t)k * n + q];
                    M[(std::size_t)k * n + p] = c * mkp - s * mkq;
                    M[(std::size_t)k * n + q] = s * mkp + c * mkq;
                }
                // Apply to rows p and q of M ( M <- J^T M ).
                for (int k = 0; k < n; ++k) {
                    const double mpk = M[(std::size_t)p * n + k];
                    const double mqk = M[(std::size_t)q * n + k];
                    M[(std::size_t)p * n + k] = c * mpk - s * mqk;
                    M[(std::size_t)q * n + k] = s * mpk + c * mqk;
                }
                // Accumulate the rotation into the eigenvector matrix V.
                for (int k = 0; k < n; ++k) {
                    const double vkp = V[(std::size_t)k * n + p];
                    const double vkq = V[(std::size_t)k * n + q];
                    V[(std::size_t)k * n + p] = c * vkp - s * vkq;
                    V[(std::size_t)k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    // Diagonal of M = eigenvalues; column k of V = eigenvector k. Sort ascending.
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    // Insertion sort by eigenvalue (n is tiny; clarity over speed).
    for (int a = 1; a < n; ++a) {
        const int key = order[a];
        const double kv = M[(std::size_t)key * n + key];
        int b = a - 1;
        while (b >= 0 && M[(std::size_t)order[b] * n + order[b]] > kv) {
            order[b + 1] = order[b];
            --b;
        }
        order[b + 1] = key;
    }

    eval.assign(n, 0.0);
    evec.assign((std::size_t)n * n, 0.0);
    for (int newc = 0; newc < n; ++newc) {
        const int oldc = order[newc];
        eval[newc] = M[(std::size_t)oldc * n + oldc];
        // Copy eigenvector (column oldc of V) into column newc of evec
        // (column-major output to match cuSOLVER's layout).
        for (int r = 0; r < n; ++r)
            evec[(std::size_t)newc * n + r] = V[(std::size_t)r * n + oldc];
    }
}

// ---------------------------------------------------------------------------
// lift_to_volume_pc: build u = X^T w / ||X^T w|| (THEORY §The algorithm).
//   gevec column k is the Gram eigenvector w (length N). We form X^T w voxel by
//   voxel -- for each voxel v, sum over particles p of (centered vox[p][v])*w[p]
//   -- then normalize. The result is the volume-space principal component: a
//   density map you could open in ChimeraX showing "how the molecule moves".
// ---------------------------------------------------------------------------
void lift_to_volume_pc(const VolumeSet& vs, const std::vector<double>& mean,
                       const std::vector<double>& gevec, int k,
                       std::vector<double>& u) {
    const int N = vs.N, D = vs.D;
    u.assign(D, 0.0);
    // X^T w : accumulate each particle's centered volume weighted by w[p].
    for (int p = 0; p < N; ++p) {
        const double w = gevec[(std::size_t)k * N + p];     // column-major: w[p]
        for (int v = 0; v < D; ++v)
            u[v] += centered_value(vs.vox.data(), mean.data(), p, v, D) * w;
    }
    // Normalize to unit length so projections are in consistent units.
    double nrm = 0.0;
    for (int v = 0; v < D; ++v) nrm += u[v] * u[v];
    nrm = std::sqrt(nrm);
    if (nrm > 0.0) {
        const double inv = 1.0 / nrm;
        for (int v = 0; v < D; ++v) u[v] *= inv;
    }
    // SIGN CONVENTION: an eigenvector is defined only up to sign (u and -u are
    // both valid PCs). We fix the sign so the largest-magnitude voxel is
    // positive -- this makes the printed result deterministic regardless of
    // which sign cuSOLVER vs Jacobi happened to return.
    int amax = 0;
    for (int v = 1; v < D; ++v)
        if (std::fabs(u[v]) > std::fabs(u[amax])) amax = v;
    if (u[amax] < 0.0)
        for (int v = 0; v < D; ++v) u[v] = -u[v];
}

// ---------------------------------------------------------------------------
// project_all_cpu: z[p] = projection of centered particle p onto mode u.
//   Uses the shared project_particle() so it matches the GPU bit-for-bit.
// ---------------------------------------------------------------------------
void project_all_cpu(const VolumeSet& vs, const std::vector<double>& mean,
                     const std::vector<double>& u, std::vector<double>& z) {
    z.assign(vs.N, 0.0);
    for (int p = 0; p < vs.N; ++p)
        z[p] = project_particle(vs.vox.data(), mean.data(), u.data(), p, vs.D);
}
