// ===========================================================================
// src/cofold.h  --  Shared (host + device) co-folding diffusion math
// ---------------------------------------------------------------------------
// Project 2.14 : Protein-Ligand Co-Folding (reduced-scope teaching version)
//
// WHAT THIS PROJECT COMPUTES
//   A *diffusion* model that, in a single reverse process, denoises the 3-D
//   coordinates of a JOINT protein+ligand token sequence -- i.e. it "co-folds"
//   the complex (predicts the protein pocket geometry AND the ligand binding
//   pose together) instead of docking a ligand into a frozen protein. This is
//   the toy, fully-transparent analogue of Boltz-1 / AlphaFold3 (catalog 2.14):
//   the architecture (a denoising loop whose every step is a self-attention
//   pass over the joint token sequence) is faithful; the network weights are
//   replaced by a fixed, analytic "score" so the math is legible and the CPU
//   and GPU agree to ~machine precision. The full learned model is a 🔴 Active
//   R&D system we describe in THEORY "Where this sits in the real world".
//
// THE TOKENS
//   We represent the complex as N TOKENS, each with:
//     * a 3-D position  x in R^3                (what we denoise)
//     * a frozen target  x*  (its native position; the planted answer)
//     * a TYPE flag: protein-backbone token vs. ligand-atom token.
//   Protein and ligand tokens share one sequence, so a single attention pass
//   reasons about protein-protein, ligand-ligand AND protein<->ligand (the
//   "cross-attention" that couples pocket and pose) at once.
//
// THE MODEL (one reverse diffusion step), per query token i:
//   1. ATTENTION (the GPU bottleneck, run every step). We use GEOMETRIC
//        (distance-kernel) attention: a query attends most to keys whose CURRENT
//        position is closest to it, with a small same-type bonus:
//          logit(i,j) = -||x_i - x_j||^2 / (2*temp^2)  +  type_bias*[type_i==type_j]
//        softmax over j gives weights a_ij. This is the radial-basis (RBF) form
//        of attention; it is equivalent up to a constant to scaled dot-product
//        attention on the position features (||q-k||^2 = |q|^2 + |k|^2 - 2 q.k),
//        and it makes the "attend to your own neighbourhood" behaviour explicit
//        -- exactly what couples a token to the right part of the structure.
//        We aggregate a *geometric target* t_i = sum_j a_ij * x*_j -- where the
//        attention "wants" token i to sit, given where its neighbours belong.
//   2. SCORE / denoiser:  the predicted clean position is x0_hat = t_i, so the
//        score (gradient of log-density) points from the noisy x_i toward t_i.
//   3. DDPM UPDATE (DDIM-style deterministic sampler): move x_i a controlled
//        fraction of the way from x_i to x0_hat for this noise level. Iterated
//        over T steps, x converges to the native complex x*  -- a reverse
//        diffusion that "folds" the noise cloud into the bound pose.
//
//   Because step (1)'s weights are a fixed function of positions+types (no
//   learned parameters), the whole reverse process is deterministic and the
//   CPU reference and GPU kernel run byte-compatible math (rd.h idiom):
//   COFOLD_HD = __host__ __device__ under nvcc, nothing under the host compiler.
//
// WHY A GPU
//   Each denoising step is an attention pass: every token attends to every
//   other token -> O(N^2 * d) work *per step*, repeated for T steps. That
//   all-pairs, per-step structure is exactly what GPUs eat for breakfast (one
//   block per query token, threads cooperate over the keys; cf. PATTERNS.md
//   "score one query vs N items" + "ensemble / per-step loop"). In real
//   co-folding this attention (FlashAttention2) dominates the runtime.
//
// READ THIS AFTER: nothing -- start here, then reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cmath>     // std::sqrt, std::exp (host); device uses the CUDA intrinsics
#include <cstddef>   // std::size_t

// HD-macro idiom (PATTERNS.md §2): under nvcc, decorate the per-token math so it
// compiles for BOTH host and device; under the plain host compiler the
// decorators simply vanish. Keep this header free of __global__ and CUDA-only
// types so reference_cpu.cpp (host compiler) can include it unchanged.
#ifdef __CUDACC__
#define COFOLD_HD __host__ __device__
#else
#define COFOLD_HD
#endif

// ---------------------------------------------------------------------------
// Compile-time problem dimension. Tiny on purpose (this is a teaching demo):
//   * D_POS = 3 : we denoise real 3-D coordinates (x, y, z).
// A constant (not a runtime param) so the shared math has fixed-size stack
// arrays -- which both the host loop and a single GPU thread hold in registers
// without dynamic allocation.
// ---------------------------------------------------------------------------
static constexpr int D_POS = 3;   // x, y, z

// A token type. We keep it an int (not an enum class) so it is trivially
// copyable across the host/device boundary inside POD structs.
//   TYPE_PROTEIN : a protein backbone (C-alpha) token.
//   TYPE_LIGAND  : a ligand heavy-atom token.
// The type enters attention as a small bias that makes like-types attend a
// touch more strongly to each other -- a stand-in for the rich learned
// chemistry embeddings in a real co-folding model.
static constexpr int TYPE_PROTEIN = 0;
static constexpr int TYPE_LIGAND  = 1;

// ---------------------------------------------------------------------------
// CofoldParams: everything the reverse diffusion needs, loaded from the sample
// file. POD (plain old data) so it copies by value into a kernel argument.
// ---------------------------------------------------------------------------
struct CofoldParams {
    int   n_tokens;     // total tokens = n_protein + n_ligand
    int   n_protein;    // how many of the first tokens are protein
    int   n_ligand;     // the remaining tokens are ligand
    int   steps;        // T: number of reverse-diffusion (denoising) steps
    double temp;        // attention "temperature": logits are divided by temp*sqrt(d).
                        //   smaller -> sharper (more peaked) attention.
    double step_frac;   // DDIM step size: fraction of (x0_hat - x) moved per step,
                        //   in (0,1]. The reverse schedule's effective rate.
    double type_bias;   // additive logit bonus when query & key share a type.
    int    seed;        // RNG seed for the initial noised positions (reproducible).
    double noise_scale; // std-dev of the Gaussian noise added to x* to make x_T.
};

// ---------------------------------------------------------------------------
// EXP for both worlds. On the device we want the fast hardware exp; on the host
// std::exp. We wrap it so the shared softmax below reads identically on both.
//   Using the SAME function on both sides keeps results bit-compatible up to
//   the platform's libm/intrinsic differences (covered by our 1e-3 tolerance).
// ---------------------------------------------------------------------------
COFOLD_HD inline double cofold_exp(double v) {
#ifdef __CUDA_ARCH__
    return exp(v);          // device: CUDA's double-precision exp intrinsic
#else
    return std::exp(v);     // host: standard library exp
#endif
}

COFOLD_HD inline double cofold_sqrt(double v) {
#ifdef __CUDA_ARCH__
    return sqrt(v);
#else
    return std::sqrt(v);
#endif
}

// ---------------------------------------------------------------------------
// attention_logit: the unnormalized score that query token i assigns to key
//   token j BEFORE softmax, using GEOMETRIC (RBF) attention:
//       logit = -||x_i - x_j||^2 / (2 * temp^2)  +  type_bias * [type_i==type_j]
//
//   The first term is a Gaussian distance kernel: it is largest (== 0) when the
//   two tokens coincide and falls off with separation, so a query attends most
//   strongly to keys near its CURRENT position. `temp` is the kernel bandwidth
//   (the "temperature"): small temp -> sharp, peaked attention (each token
//   attends almost only to its own neighbourhood); large temp -> diffuse
//   averaging. The second term is the chemistry-aware coupling: same-type tokens
//   (protein-protein, ligand-ligand) get an additive bonus, a stand-in for the
//   learned type embeddings of a real co-folding transformer.
//
//   (Why this is "really" attention: expand the square,
//    -||q-k||^2 = 2 q.k - |q|^2 - |k|^2. The 2 q.k is exactly scaled
//    dot-product attention; the -|q|^2 term is constant across keys j so it
//    cancels in the softmax, and -|k|^2 is the standard "key norm" correction.
//    RBF attention is the dot-product attention you get from position features.)
//
//   qx..qz / kx..kz: the two tokens' current positions. same_type: 1 if they
//   share a type, else 0. P: parameters (temp, type_bias).
// ---------------------------------------------------------------------------
COFOLD_HD inline double attention_logit(double qx, double qy, double qz,
                                        double kx, double ky, double kz,
                                        int same_type, const CofoldParams& P) {
    const double dx = qx - kx, dy = qy - ky, dz = qz - kz;
    const double dist2 = dx * dx + dy * dy + dz * dz;     // squared separation
    const double rbf = -dist2 / (2.0 * P.temp * P.temp);  // Gaussian distance kernel
    return rbf + P.type_bias * (double)same_type;         // + same-type bonus
}

// ---------------------------------------------------------------------------
// ddim_blend: one deterministic DDIM-style coordinate update for a single axis.
//   Given the current (noisy) coordinate `x_cur` and the network's predicted
//   clean target `x0_hat`, move a fixed fraction `step_frac` of the way:
//       x_next = x_cur + step_frac * (x0_hat - x_cur)
//   Iterating this T times is a geometric contraction toward x0_hat -- the
//   reverse diffusion trajectory. We use the deterministic (no added noise)
//   sampler so the demo's output is reproducible (PATTERNS.md §3); the
//   stochastic DDPM sampler would add a Gaussian term here, which we describe
//   in THEORY but deliberately omit for determinism.
// ---------------------------------------------------------------------------
COFOLD_HD inline double ddim_blend(double x_cur, double x0_hat, double step_frac) {
    return x_cur + step_frac * (x0_hat - x_cur);
}

// ---------------------------------------------------------------------------
// denoise_token: THE per-token reverse-diffusion step (the one true formula,
//   shared by CPU and GPU). For query token i it:
//     (a) builds q_i's feature,
//     (b) sweeps all keys j: computes logits, tracks the max (for a numerically
//         stable softmax), accumulates exp-weights and the weighted sum of the
//         keys' NATIVE targets x*_j,
//     (c) forms the predicted clean position x0_hat = sum_j a_ij x*_j,
//     (d) blends the current x_i toward x0_hat (DDIM) and writes x_next.
//
//   Inputs (all device-or-host pointers, length n_tokens * D_POS unless noted):
//     i        : the query token index this call updates
//     pos      : current positions  x   (read; the frozen state for this step)
//     target   : native positions   x*  (read-only; the planted answer)
//     types    : per-token type  [n_tokens]
//     P        : parameters
//   Output:
//     pos_next : where token i's updated 3-D position is written (in place-safe
//                because we read `pos` and write a SEPARATE `pos_next` buffer --
//                double buffering, exactly like the stencil ping-pong).
//
//   Numerical note: the softmax uses the standard max-subtraction trick so
//   exp() never overflows. The reduction over j is done in a FIXED index order
//   (0..n-1) identically on host and device, so the floating-point sum is the
//   same order on both -> agreement to ~1e-12 per step (drift to ~1e-3 over T
//   steps from FMA differences; see THEORY "Numerical considerations").
// ---------------------------------------------------------------------------
COFOLD_HD inline void denoise_token(int i, const double* pos, const double* target,
                                    const int* types, const CofoldParams& P,
                                    double* pos_next) {
    // (a) query token i's CURRENT position + type.
    const double qx = pos[i * D_POS + 0];
    const double qy = pos[i * D_POS + 1];
    const double qz = pos[i * D_POS + 2];
    const int qtype = types[i];

    // First pass over keys: find the maximum logit for stable softmax.
    double max_logit = -1.0e300;   // effectively -infinity for doubles
    for (int j = 0; j < P.n_tokens; ++j) {
        const int same = (types[j] == qtype) ? 1 : 0;
        const double l = attention_logit(qx, qy, qz,
                                         pos[j * D_POS + 0], pos[j * D_POS + 1],
                                         pos[j * D_POS + 2], same, P);
        if (l > max_logit) max_logit = l;
    }

    // Second pass: accumulate softmax denominator and the weighted target sum.
    double denom = 0.0;                 // sum_j exp(logit_j - max)
    double acc[D_POS] = {0.0, 0.0, 0.0}; // sum_j w_j * x*_j  (the geometric target)
    for (int j = 0; j < P.n_tokens; ++j) {
        const int same = (types[j] == qtype) ? 1 : 0;
        const double l = attention_logit(qx, qy, qz,
                                         pos[j * D_POS + 0], pos[j * D_POS + 1],
                                         pos[j * D_POS + 2], same, P);
        const double w = cofold_exp(l - max_logit);   // unnormalized weight
        denom += w;
        // Aggregate the NATIVE target of key j -- "attention points token i to
        // where its trusted neighbours belong" -> the predicted clean pose.
        acc[0] += w * target[j * D_POS + 0];
        acc[1] += w * target[j * D_POS + 1];
        acc[2] += w * target[j * D_POS + 2];
    }

    // (c) predicted clean position x0_hat = normalized weighted target.
    const double inv = (denom > 0.0) ? (1.0 / denom) : 0.0;
    const double x0x = acc[0] * inv;
    const double x0y = acc[1] * inv;
    const double x0z = acc[2] * inv;

    // (d) DDIM blend the current position toward x0_hat; write to the next buffer.
    pos_next[i * D_POS + 0] = ddim_blend(qx, x0x, P.step_frac);
    pos_next[i * D_POS + 1] = ddim_blend(qy, x0y, P.step_frac);
    pos_next[i * D_POS + 2] = ddim_blend(qz, x0z, P.step_frac);
}

// ---------------------------------------------------------------------------
// rmsd_to_target: root-mean-square deviation between current positions and the
//   native targets, over all tokens. This is our SCIENCE-level success metric
//   (PATTERNS.md §4): a real pose-prediction tool reports ligand-RMSD to the
//   crystal pose, and "good" is < 2 Angstrom. Here, a correct reverse diffusion
//   drives RMSD -> ~0, recovering the planted complex. Pure host/device math so
//   both sides can call it; we report it from the host in main.cu.
// ---------------------------------------------------------------------------
COFOLD_HD inline double rmsd_to_target(const double* pos, const double* target,
                                       int n_tokens) {
    double s = 0.0;
    for (int i = 0; i < n_tokens; ++i)
        for (int c = 0; c < D_POS; ++c) {
            const double d = pos[i * D_POS + c] - target[i * D_POS + c];
            s += d * d;
        }
    return cofold_sqrt(s / (double)(n_tokens > 0 ? n_tokens : 1));
}
