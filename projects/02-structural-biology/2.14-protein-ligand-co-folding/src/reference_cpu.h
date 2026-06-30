// ===========================================================================
// src/reference_cpu.h  --  Co-folding parameters, complex loader, CPU reference
// ---------------------------------------------------------------------------
// Project 2.14 : Protein-Ligand Co-Folding (reduced-scope teaching version)
//
// Pure C++ (no CUDA). The per-token reverse-diffusion math lives in cofold.h;
// kernels.cu reuses CofoldParams and that same math, so the GPU reproduces the
// CPU result. The CPU path is the trusted baseline the GPU is verified against.
//
// READ THIS AFTER: cofold.h (the shared math).  Pairs with reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "cofold.h"   // CofoldParams, denoise_token, rmsd_to_target, D_POS

// ---------------------------------------------------------------------------
// Complex: the loaded protein-ligand system. We keep the native (target)
// positions and the per-token types here; the noisy starting positions are
// generated from these by init_positions().
//   target : flat [n_tokens * D_POS] native coordinates (the planted answer).
//   types  : [n_tokens] TYPE_PROTEIN / TYPE_LIGAND per token.
// ---------------------------------------------------------------------------
struct Complex {
    CofoldParams        P;        // dimensions + diffusion schedule
    std::vector<double> target;   // native positions x*  (flat, row-major)
    std::vector<int>    types;    // per-token type
};

// Load a Complex from the one-token-per-line sample format (see data/README.md):
//   header line:  n_protein n_ligand steps temp step_frac type_bias seed noise_scale
//   then one line per token:  type x* y* z*
// Throws std::runtime_error on a malformed file so demos fail loudly.
Complex load_complex(const std::string& path);

// init_positions: build the NOISED starting coordinates x_T by adding
// deterministic pseudo-random Gaussian noise (std = noise_scale) to every native
// coordinate, using a fixed seed so the demo is reproducible. This is the
// "forward diffusion" endpoint: a cloud of noise that the reverse process must
// fold back into the bound complex. Fills `pos` to length n_tokens * D_POS.
void init_positions(const Complex& C, std::vector<double>& pos);

// simulate_cpu: the CPU reference reverse diffusion. Advances `pos` through
// `steps` denoising steps in place (double-buffered, calling denoise_token per
// token per step). After it returns, `pos` holds the final predicted complex.
void simulate_cpu(const Complex& C, std::vector<double>& pos);
