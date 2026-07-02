// ===========================================================================
// src/kernels.cuh  --  GPU DTI-GNN interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 7.2 : Drug-Target Interaction Prediction (GNN)
//
// THE BIG IDEA (two classic GPU patterns, PATTERNS.md sec 1)
//   The forward pass is three GPU stages, each embarrassingly parallel:
//
//   (A) MESSAGE PASSING  -- a GATHER over graph edges. One thread per NODE. Each
//       thread reads its CSR neighbour row (adj_off[i]..adj_off[i+1]), SUMS the
//       neighbours' feature vectors, and pushes the sum through the shared
//       linear layer + ReLU (gnn.h). Nodes are independent, so no atomics: each
//       output node is written by exactly one thread. We double-buffer node
//       features (cur -> nxt) and swap between the GNN_T rounds -- the same
//       ping-pong idea as a stencil solver (PATTERNS.md sec 1, "stencil").
//
//   (B) POOLING          -- one thread per DRUG sums its nodes' final features
//       into a length-F embedding (graph readout). Independent per drug.
//
//   (C) PAIR SCORING     -- one thread per (drug, protein) PAIR computes the DTI
//       probability = sigmoid(dot(drug_emb, prot_emb)/F). This is the
//       "independent jobs" pattern (PATTERNS.md sec 1) -- D*P independent dot
//       products, exactly the shape of large virtual-screening scoring.
//
//   The weights (W, bias, Wp, bp) are tiny and read by every thread, so we keep
//   them in CONSTANT memory (broadcast cache) -- the same trick as the Tanimoto
//   flagship's query fingerprint (1.12).
//
//   kernels.cu defines the kernels + the host wrapper dti_gpu(); main.cu calls
//   dti_gpu() and verifies its output against dti_cpu() (reference_cpu.cpp).
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, gnn.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // Dataset, GnnModel (pure C++, safe to include in .cu)

// ---------------------------------------------------------------------------
// message_pass_kernel: ONE round of message passing. One thread per node.
//   feat_in  : [total_nodes*F] current node features (round input)
//   feat_out : [total_nodes*F] next node features (round output)
//   adj_off  : [total_nodes+1] CSR row pointers
//   adj      : [total_edges]   neighbour global-node indices
//   total_nodes : guard for the ragged last block
//   round    : which round's weights to use (indexes constant-memory W/bias)
// Thread (blockIdx.x, threadIdx.x) owns node i = blockIdx.x*blockDim.x+threadIdx.x.
// ---------------------------------------------------------------------------
__global__ void message_pass_kernel(const float* __restrict__ feat_in,
                                     float* __restrict__ feat_out,
                                     const int* __restrict__ adj_off,
                                     const int* __restrict__ adj,
                                     int total_nodes, int round);

// ---------------------------------------------------------------------------
// pool_kernel: graph-level sum pooling. One thread per drug.
//   feat     : [total_nodes*F] final node features (after all rounds)
//   node_off : [D+1] CSR-style drug->node ranges
//   emb      : [D*F] output drug embeddings
// ---------------------------------------------------------------------------
__global__ void pool_kernel(const float* __restrict__ feat,
                            const int* __restrict__ node_off,
                            float* __restrict__ emb, int D);

// ---------------------------------------------------------------------------
// protein_encode_kernel: project each protein descriptor into embedding space.
//   prot : [P*F] protein descriptor vectors
//   pemb : [P*F] output protein embeddings (linear+ReLU via constant Wp/bp)
// One thread per protein.
// ---------------------------------------------------------------------------
__global__ void protein_encode_kernel(const float* __restrict__ prot,
                                       float* __restrict__ pemb, int P);

// ---------------------------------------------------------------------------
// score_kernel: DTI probability for each drug x protein pair. One thread/pair.
//   emb   : [D*F] drug embeddings
//   pemb  : [P*F] protein embeddings
//   score : [D*P] output probabilities, drug-major (score[drug*P + p])
// Thread flat index j in [0, D*P) -> drug = j/P, protein = j%P.
// ---------------------------------------------------------------------------
__global__ void score_kernel(const float* __restrict__ emb,
                             const float* __restrict__ pemb,
                             float* __restrict__ score, int D, int P);

// ---------------------------------------------------------------------------
// dti_gpu: host wrapper -- runs the whole forward pass on the GPU and returns
// the drug embeddings (emb, [D*F]) and DTI score matrix (score, [D*P]). The
// end-to-end GPU time (all kernels, excluding H2D/D2H copies) is returned via
// *kernel_ms. main.cu compares emb/score against the CPU reference.
// ---------------------------------------------------------------------------
void dti_gpu(const Dataset& d, const GnnModel& m,
             std::vector<float>& emb, std::vector<float>& score, float* kernel_ms);
