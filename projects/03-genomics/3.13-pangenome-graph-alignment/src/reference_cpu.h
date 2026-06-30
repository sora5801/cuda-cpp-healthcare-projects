// ===========================================================================
// src/reference_cpu.h  --  Data model, scoring, graph loader & CPU reference
// ---------------------------------------------------------------------------
// Project 3.13 : Pangenome Graph Alignment
//
// WHAT THIS PROJECT COMPUTES
//   A LINEAR reference genome is one string; a PANGENOME is a *graph* of many
//   genomes. Each node carries a short DNA segment; directed edges say which
//   segments may follow which. A "bubble" like
//
//          ___ [G] ___
//         /            \
//     [ACG]            [TT]        the "[C] vs [G]" branch is a SNP: two alleles
//         \___ [C] ___/            -- each path through the graph is a haplotype.
//
//   encodes that the population carries either a C or a G at that locus. We align
//   one query read to the WHOLE graph at once with a generalised local Smith-
//   Waterman: we want the best-scoring local alignment of the read against ANY
//   path through the graph. The score matrix becomes one score block PER node:
//
//     H[v][i][j] = best local-alignment score that consumes query[0..i) and
//                  ends at column j inside node v's segment.
//
//   It is the ordinary SW recurrence INSIDE a node; the only new idea is at a
//   node's FIRST column (j = 1), whose "diagonal" and "left" neighbours live in
//   the LAST column of v's PREDECESSOR nodes -- so we take a max over all
//   predecessors. Nodes are processed in TOPOLOGICAL order so every predecessor
//   block is finished before we fill v. (This is exactly how vg / gssw align
//   reads to variation graphs; see THEORY.md "Where this sits in the real world".)
//
// WHY A GPU
//   Inside one node the recurrence has the same top/left/top-left dependency as
//   linear SW, so the cells on one ANTI-DIAGONAL (i+j const) are mutually
//   independent and fill in parallel -- the "wavefront" (cf. flagship 3.01). The
//   graph just threads that wavefront through nodes in topological order. (The
//   2024 SC pangenome paper in the catalog GPU-accelerates a *different* stage --
//   force-directed graph LAYOUT; here we accelerate the alignment DP, the part a
//   learner meets first. THEORY.md "real world" connects the two.)
//
//   This pure-C++ header is shared by reference_cpu.cpp, main.cu, and kernels.cu.
//   The per-cell recurrence lives in ONE __host__ __device__ function
//   (cell_score, below; PATTERNS.md §2) so the CPU reference and the GPU kernel
//   run BYTE-FOR-BYTE identical integer math -- making verification EXACT, not
//   approximate.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh.  READ BEFORE: kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// HD: the host/device decoration switch (PATTERNS.md §2). When this header is
// compiled by nvcc (__CUDACC__ defined) the per-cell math is marked
// __host__ __device__ so it can be called from BOTH the CPU reference and the
// GPU kernel. When compiled by the plain host C++ compiler (reference_cpu.cpp),
// the decorators do not exist, so we expand HD to nothing.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// Linear-gap scoring. INTEGERS keep CPU and GPU results bit-identical (integer
// adds and max() commute and never round) -- so we can verify to tolerance ZERO.
constexpr int MATCH    =  2;   // reward for aligning two identical nucleotides
constexpr int MISMATCH = -1;   // penalty for aligning two different nucleotides
constexpr int GAP      = -2;   // penalty for one inserted/deleted nucleotide
constexpr char ALPHABET[] = "ACGT";  // code 0..3 <-> nucleotide letter

// ---------------------------------------------------------------------------
// cell_score: THE one true Smith-Waterman recurrence for a single cell.
//   Given the three already-computed neighbour scores and whether the two
//   residues at this cell are equal, return H[i][j]. The max-with-0 ("floor")
//   is what makes the alignment LOCAL: a negative running score restarts a
//   fresh alignment instead of dragging the total down. Both the CPU loop and
//   the GPU kernel call THIS function, guaranteeing identical results.
//
//   diag : H of the top-left neighbour (extend the alignment by a match/mismatch)
//   up   : H of the top neighbour      (a gap in the graph path -> consume query)
//   left : H of the left neighbour     (a gap in the query      -> consume graph)
//   is_match : true if query residue == graph residue at this cell
// ---------------------------------------------------------------------------
HD inline int cell_score(int diag, int up, int left, bool is_match) {
    const int s     = is_match ? MATCH : MISMATCH;   // substitution score
    const int d_ext = diag + s;                      // extend alignment diagonally
    const int u_ext = up   + GAP;                     // gap on the graph path
    const int l_ext = left + GAP;                     // gap in the query
    int v = 0;                                        // the 0 floor (local restart)
    if (d_ext > v) v = d_ext;
    if (u_ext > v) v = u_ext;
    if (l_ext > v) v = l_ext;
    return v;
}

// ---------------------------------------------------------------------------
// Graph: a pangenome graph in compressed (GFA-like) form.
//   Sequences of all nodes are concatenated into one buffer `seq`; node v owns
//   seq[seq_off[v] .. seq_off[v]+seq_len[v]). Edges are stored as a CSR-style
//   adjacency on PREDECESSORS (we always look "backwards" to predecessors when
//   filling a node's first column). Nodes are pre-sorted topologically by the
//   loader, so the index v IS the topological rank: every predecessor of v has a
//   SMALLER index than v. That single invariant is what lets us fill blocks in
//   plain ascending-v order and know all dependencies are already done.
// ---------------------------------------------------------------------------
struct Graph {
    int num_nodes = 0;                 // |V|
    int total_bases = 0;               // sum of all node lengths = seq.size()
    std::vector<uint8_t> seq;          // [total_bases] concatenated node segments (codes 0..3)
    std::vector<int> seq_off;          // [num_nodes] start of node v inside seq
    std::vector<int> seq_len;          // [num_nodes] length of node v's segment

    // Predecessor adjacency in CSR form: the predecessors of node v are
    //   pred_idx[ pred_off[v] .. pred_off[v+1] ).
    std::vector<int> pred_off;         // [num_nodes+1] CSR row pointers
    std::vector<int> pred_idx;         // [num_edges] predecessor node ids (all < v)

    // Human-readable node names (for printing the recovered path), e.g. "n3".
    std::vector<std::string> name;     // [num_nodes]
};

// A loaded alignment problem: one query read + the graph it is aligned to.
struct Problem {
    std::vector<uint8_t> query;        // [qlen] encoded read (codes 0..3)
    int qlen = 0;                      // query length
    Graph graph;                       // the pangenome graph
};

// ---------------------------------------------------------------------------
// GraphDP: the filled score matrices for every node, laid out flat so the CPU
// and GPU share the exact same memory model.
//   Node v occupies a (qlen+1) x (Lv+1) block (rows = query incl. row 0, columns
//   = node segment incl. column 0). Blocks are concatenated; block_off[v] is the
//   flat start of node v's block, and the row stride of block v is (Lv+1). So
//   cell (i, j) of node v is at  block_off[v] + i*(seq_len[v]+1) + j.
//   We keep the WHOLE matrix (not just two diagonals) so the host can trace the
//   optimal path back after the fill -- the GPU teaching point is the parallel
//   FILL, not the serial traceback.
// ---------------------------------------------------------------------------
struct GraphDP {
    std::vector<int> H;                // flat score cells for all node blocks
    std::vector<int> block_off;        // [num_nodes+1] flat offset of each block
    int qlen = 0;                      // rows per block = qlen+1
};

// Compute the per-node block layout for a graph + query (shared by CPU and GPU
// so both index H identically). Fills dp.block_off and dp.qlen and sizes dp.H.
void layout_blocks(const Problem& p, GraphDP& dp);

// One recovered local alignment of the query against a path through the graph.
struct PathAlignment {
    int score = 0;                     // best local score over ALL nodes/cells
    int end_node = -1;                 // node holding the max cell
    int end_i = 0, end_j = 0;          // (query row, node column) of the max cell
    int length = 0;                    // number of alignment columns
    int identities = 0;                // columns where the two residues match
    std::string node_path;             // node names visited, e.g. "n0>n2>n4"
    std::string q_line, m_line, t_line;// aligned query / markers / graph path
};

// Load a problem from the tiny text graph format (see data/README.md). Throws
// std::runtime_error on a malformed file. The loader verifies the node order is
// a valid topological order (every edge points forward: source index < dest).
Problem load_problem(const std::string& path);

// CPU reference: fill every node's score block in topological order. This is the
// trusted, obviously-correct baseline the GPU wavefront is checked against (every
// cell must match exactly). See reference_cpu.cpp.
void graph_sw_cpu(const Problem& p, GraphDP& dp);

// Recover the best local alignment + node path from filled blocks (host-side;
// done once on whichever matrix we display). Deterministic tie-breaking.
PathAlignment traceback(const Problem& p, const GraphDP& dp);
