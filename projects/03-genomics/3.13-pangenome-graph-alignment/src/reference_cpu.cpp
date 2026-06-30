// ===========================================================================
// src/reference_cpu.cpp  --  Trusted serial graph-SW + loader + traceback
// ---------------------------------------------------------------------------
// Project 3.13 : Pangenome Graph Alignment
//
// (1) load_problem()  : parse the tiny text graph + read (see data/README.md),
//                       encode nucleotides, build CSR predecessor lists, and
//                       VERIFY the file lists nodes in a valid topological order.
// (2) layout_blocks() : compute the flat (qlen+1) x (Lv+1) block layout shared
//                       by the CPU reference and the GPU kernel.
// (3) graph_sw_cpu()  : the obviously-correct serial DP the GPU is checked on.
// (4) traceback()     : recover the best local alignment + node path from a
//                       filled set of blocks (host only; not the GPU teaching
//                       point, so done once on the host).
//
// Compiled by the host C++ compiler only (no CUDA syntax). See reference_cpu.h.
// The per-cell recurrence (cell_score) is shared with the GPU kernel via the
// HD macro in reference_cpu.h, so CPU and GPU blocks are bit-identical.
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>   // std::reverse, std::count, std::max
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

// ---------------------------------------------------------------------------
// encode: map a nucleotide character to its 0..3 code (or throw on a bad letter).
//   Keeping sequences as small integer codes (not chars) makes the device data
//   compact and the match test a single integer compare.
// ---------------------------------------------------------------------------
static uint8_t encode(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:
            throw std::runtime_error(std::string("non-ACGT character in sequence: '") + c + "'");
    }
}

// Encode a whole DNA string, tolerating stray whitespace/carriage returns.
static std::vector<uint8_t> encode_seq(const std::string& s) {
    std::vector<uint8_t> v;
    v.reserve(s.size());
    for (char c : s) {
        if (c == '\r' || c == ' ' || c == '\t') continue;
        v.push_back(encode(c));
    }
    return v;
}

// ---------------------------------------------------------------------------
// load_problem: parse the tiny text graph format. Lines (order-independent
// except that a node must be declared before it is referenced by an edge):
//     # comment                      -- ignored
//     Q <DNA>                        -- the query read (exactly one)
//     N <name> <DNA>                 -- a node, in TOPOLOGICAL order of appearance
//     E <src_name> <dst_name>        -- a directed edge src -> dst
//
//   We assign each node an index in DECLARATION order, then require every edge
//   to point FORWARD (src index < dst index). That makes declaration order a
//   valid topological order, which the DP relies on (predecessors finish first).
//   See data/README.md for the full grammar and an annotated example.
// ---------------------------------------------------------------------------
Problem load_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open graph file: " + path);

    Problem p;
    Graph& g = p.graph;
    std::unordered_map<std::string, int> id_of;   // node name -> index
    std::vector<std::string> seqs;                // per-node raw DNA (concatenated later)
    bool have_query = false;
    // Edges are collected as (src,dst) index pairs first, then turned into CSR.
    std::vector<std::pair<int, int>> edges;

    std::string line;
    while (std::getline(in, line)) {
        // Strip a trailing CR (Windows line endings) and skip blank/comment lines.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ls(line);
        std::string tag;
        if (!(ls >> tag)) continue;                 // blank line
        if (tag.empty() || tag[0] == '#') continue; // comment

        if (tag == "Q") {                            // the query read
            std::string dna;
            if (!(ls >> dna)) throw std::runtime_error("Q line missing sequence in " + path);
            p.query = encode_seq(dna);
            p.qlen  = static_cast<int>(p.query.size());
            have_query = true;
        } else if (tag == "N") {                     // a node
            std::string nm, dna;
            if (!(ls >> nm >> dna)) throw std::runtime_error("N line needs <name> <DNA> in " + path);
            if (id_of.count(nm)) throw std::runtime_error("duplicate node name '" + nm + "' in " + path);
            const int idx = static_cast<int>(g.name.size());
            id_of[nm] = idx;
            g.name.push_back(nm);
            seqs.push_back(dna);
        } else if (tag == "E") {                     // a directed edge
            std::string a, b;
            if (!(ls >> a >> b)) throw std::runtime_error("E line needs <src> <dst> in " + path);
            auto ia = id_of.find(a), ib = id_of.find(b);
            if (ia == id_of.end() || ib == id_of.end())
                throw std::runtime_error("edge references an undeclared node in " + path);
            // Topological invariant: an edge must point from a smaller index to a
            // larger one. If not, the file is not topologically sorted and the DP
            // (which fills nodes in ascending index order) would read an unfilled
            // predecessor. We reject it loudly rather than silently miscompute.
            if (ia->second >= ib->second)
                throw std::runtime_error("edge '" + a + "->" + b +
                    "' is not forward: list nodes in topological order in " + path);
            edges.emplace_back(ia->second, ib->second);
        } else {
            throw std::runtime_error("unknown line tag '" + tag + "' in " + path);
        }
    }
    if (!have_query) throw std::runtime_error("no Q (query) line in " + path);
    if (g.name.empty()) throw std::runtime_error("no N (node) lines in " + path);
    if (p.qlen == 0) throw std::runtime_error("empty query in " + path);

    g.num_nodes = static_cast<int>(g.name.size());

    // Concatenate node sequences into one buffer and record per-node offsets.
    g.seq_off.assign(g.num_nodes, 0);
    g.seq_len.assign(g.num_nodes, 0);
    for (int v = 0; v < g.num_nodes; ++v) {
        std::vector<uint8_t> coded = encode_seq(seqs[v]);
        if (coded.empty()) throw std::runtime_error("node '" + g.name[v] + "' has empty sequence in " + path);
        g.seq_off[v] = static_cast<int>(g.seq.size());
        g.seq_len[v] = static_cast<int>(coded.size());
        g.seq.insert(g.seq.end(), coded.begin(), coded.end());
    }
    g.total_bases = static_cast<int>(g.seq.size());

    // Build the CSR PREDECESSOR adjacency. We need, for each node v, the list of
    // nodes u with an edge u->v. Standard counting-sort CSR construction:
    //   pass 1: count predecessors per v.   pass 2: prefix-sum into pred_off.
    //   pass 3: scatter each edge's source into its destination's slot.
    g.pred_off.assign(g.num_nodes + 1, 0);
    for (const auto& e : edges) g.pred_off[e.second + 1]++;          // count into [v+1]
    for (int v = 0; v < g.num_nodes; ++v) g.pred_off[v + 1] += g.pred_off[v]; // prefix sum
    g.pred_idx.assign(edges.size(), 0);
    std::vector<int> cursor(g.pred_off.begin(), g.pred_off.end() - 1); // write cursor per v
    for (const auto& e : edges) g.pred_idx[cursor[e.second]++] = e.first;

    return p;
}

// ---------------------------------------------------------------------------
// layout_blocks: assign each node v a contiguous (qlen+1) x (Lv+1) flat block.
//   Both the CPU reference and the GPU kernel call this so they index H the same
//   way. block_off[v] is where node v's block starts; block_off[num_nodes] is
//   the total number of cells (= dp.H.size()).
// ---------------------------------------------------------------------------
void layout_blocks(const Problem& p, GraphDP& dp) {
    const Graph& g = p.graph;
    dp.qlen = p.qlen;
    dp.block_off.assign(g.num_nodes + 1, 0);
    for (int v = 0; v < g.num_nodes; ++v) {
        const int rows = p.qlen + 1;            // query rows incl. the 0th row
        const int cols = g.seq_len[v] + 1;      // node columns incl. the 0th column
        dp.block_off[v + 1] = dp.block_off[v] + rows * cols;
    }
    dp.H.assign(static_cast<std::size_t>(dp.block_off[g.num_nodes]), 0);
}

// ---------------------------------------------------------------------------
// first_column_neighbours: the heart of GRAPH alignment.
//   For node v's first content column (j = 1), the "left" and "diagonal"
//   neighbours are not inside v -- they live in the LAST column (j = Lu) of each
//   PREDECESSOR u. We take the MAX over predecessors:
//       diag(v, i) = max over predecessors u of  H[u][i-1][Lu]
//       left(v, i) = max over predecessors u of  H[u][i  ][Lu]
//   A node with NO predecessor (a graph source) uses 0 for both -- so a local
//   alignment may start fresh at the beginning of any source node. This single
//   rule is the only difference from linear Smith-Waterman.
//
//   Returns the pair (diag_in, left_in) for cell (i, j=1) of node v. Used by BOTH
//   the CPU reference (here) and conceptually mirrored in the GPU kernel, which
//   precomputes these boundary columns on the host side (see kernels.cu).
// ---------------------------------------------------------------------------
static void first_column_neighbours(const Problem& p, const GraphDP& dp,
                                    int v, int i, int& diag_in, int& left_in) {
    const Graph& g = p.graph;
    diag_in = 0;   // graph-source default: a fresh local start (the 0 floor)
    left_in = 0;
    for (int e = g.pred_off[v]; e < g.pred_off[v + 1]; ++e) {
        const int u  = g.pred_idx[e];
        const int Lu = g.seq_len[u];
        const int Wu = Lu + 1;                       // predecessor block row stride
        const int base = dp.block_off[u];
        // last column of predecessor u, at rows i-1 (diagonal) and i (left)
        const int up_diag = dp.H[base + (i - 1) * Wu + Lu];
        const int up_left = dp.H[base +  i      * Wu + Lu];
        if (up_diag > diag_in) diag_in = up_diag;
        if (up_left > left_in) left_in = up_left;
    }
}

// ---------------------------------------------------------------------------
// graph_sw_cpu: fill every node's block, nodes in ASCENDING index order (which
// the loader guaranteed is a topological order). Within a node we fill plainly
// row by row, column by column -- the obvious serial baseline. The GPU twin
// fills the SAME cells but sweeps each node's block as an anti-diagonal
// wavefront (kernels.cu); both call cell_score(), so the blocks come out equal.
// ---------------------------------------------------------------------------
void graph_sw_cpu(const Problem& p, GraphDP& dp) {
    const Graph& g = p.graph;
    layout_blocks(p, dp);                 // (re)allocates and zeroes dp.H

    for (int v = 0; v < g.num_nodes; ++v) {
        const int L = g.seq_len[v];        // node segment length
        const int W = L + 1;               // block row stride (cols incl. col 0)
        const int base = dp.block_off[v];  // flat start of this node's block
        const int soff = g.seq_off[v];     // start of this node's bases in g.seq

        // Row 0 and column 0 are already 0 (the SW init); start at i=1, j=1.
        for (int i = 1; i <= p.qlen; ++i) {
            const uint8_t q = p.query[i - 1];   // query residue at this row
            for (int j = 1; j <= L; ++j) {
                int diag, up, left;
                if (j == 1) {
                    // First content column: pull diag/left from predecessor blocks.
                    first_column_neighbours(p, dp, v, i, diag, left);
                    up = dp.H[base + (i - 1) * W + j];   // "up" is still inside v
                } else {
                    diag = dp.H[base + (i - 1) * W + (j - 1)];
                    up   = dp.H[base + (i - 1) * W +  j     ];
                    left = dp.H[base +  i      * W + (j - 1)];
                }
                const bool is_match = (q == g.seq[soff + (j - 1)]);
                dp.H[base + i * W + j] = cell_score(diag, up, left, is_match);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// best_predecessor_for_first_col: during traceback we need to know WHICH
// predecessor u (and at which of its cells) supplied the winning diag/left value
// at a first column. We re-derive it deterministically: scan predecessors in CSR
// order and return the first u achieving the recorded incoming score. Returns -1
// if none matches (which means the alignment started fresh at this source node).
// ---------------------------------------------------------------------------
static int best_predecessor(const Problem& p, const GraphDP& dp,
                            int v, int i, int want, bool diagonal) {
    const Graph& g = p.graph;
    for (int e = g.pred_off[v]; e < g.pred_off[v + 1]; ++e) {
        const int u  = g.pred_idx[e];
        const int Lu = g.seq_len[u];
        const int Wu = Lu + 1;
        const int base = dp.block_off[u];
        const int row = diagonal ? (i - 1) : i;     // diag uses i-1, left uses i
        if (dp.H[base + row * Wu + Lu] == want) return u;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// traceback: find the global max cell over ALL node blocks (the local-alignment
// endpoint), then walk backwards -- inside a node by the usual diag/up/left
// preference, and ACROSS a node boundary (when j reaches 1) by hopping to the
// predecessor that supplied the winning incoming score. Builds the aligned
// query/marker/graph strings and the visited node path. Deterministic: first max
// cell in (node, i, j) scan order; preference diagonal > up > left; first
// matching predecessor in CSR order.
// ---------------------------------------------------------------------------
PathAlignment traceback(const Problem& p, const GraphDP& dp) {
    const Graph& g = p.graph;
    PathAlignment a;

    // 1) Locate the maximum-scoring cell across every node block.
    for (int v = 0; v < g.num_nodes; ++v) {
        const int L = g.seq_len[v], W = L + 1, base = dp.block_off[v];
        for (int i = 1; i <= p.qlen; ++i)
            for (int j = 1; j <= L; ++j) {
                const int h = dp.H[base + i * W + j];
                if (h > a.score) { a.score = h; a.end_node = v; a.end_i = i; a.end_j = j; }
            }
    }
    if (a.score == 0) {                       // no positive-scoring local alignment
        a.node_path = "(none)";
        return a;
    }

    // 2) Walk back from the max cell until we reach a 0 (the local start).
    std::string q, mk, t;
    std::vector<int> path_nodes;              // node indices, end -> start
    int v = a.end_node, i = a.end_i, j = a.end_j;
    path_nodes.push_back(v);

    while (i > 0 && j > 0) {
        const int L = g.seq_len[v], W = L + 1, base = dp.block_off[v];
        const int soff = g.seq_off[v];
        const int h = dp.H[base + i * W + j];
        if (h == 0) break;                    // reached the local-alignment start

        const uint8_t qres = p.query[i - 1];
        const uint8_t tres = g.seq[soff + (j - 1)];
        const bool is_match = (qres == tres);
        const int s = is_match ? MATCH : MISMATCH;

        // Recover the three incoming scores (inside the node, or across the
        // boundary when j == 1) exactly as the fill computed them.
        int diag, up, left;
        if (j == 1) {
            first_column_neighbours(p, dp, v, i, diag, left);
            up = dp.H[base + (i - 1) * W + j];
        } else {
            diag = dp.H[base + (i - 1) * W + (j - 1)];
            up   = dp.H[base + (i - 1) * W +  j     ];
            left = dp.H[base +  i      * W + (j - 1)];
        }

        // Preference diagonal > up > left makes the path deterministic.
        if (h == diag + s) {                  // DIAGONAL: align qres with tres
            q  += ALPHABET[qres];
            t  += ALPHABET[tres];
            mk += is_match ? '|' : '.';
            if (j == 1) {                     // crossing into a predecessor node
                const int u = best_predecessor(p, dp, v, i, diag, /*diagonal=*/true);
                if (u < 0) { --i; --j; }      // started at a source: step to (i-1, 0)
                else { v = u; i -= 1; j = g.seq_len[u]; path_nodes.push_back(v); }
            } else { --i; --j; }
        } else if (h == up + GAP) {           // UP: gap on the graph path
            q  += ALPHABET[qres];
            t  += '-';
            mk += ' ';
            --i;                              // "up" never crosses a node boundary
        } else {                              // LEFT: gap in the query
            q  += '-';
            t  += ALPHABET[tres];
            mk += ' ';
            if (j == 1) {                     // crossing into a predecessor node
                const int u = best_predecessor(p, dp, v, i, left, /*diagonal=*/false);
                if (u < 0) { --j; }           // started at a source: step to col 0
                else { v = u; j = g.seq_len[u]; path_nodes.push_back(v); }
            } else { --j; }
        }
    }

    // 3) We built everything end-to-start; reverse to read 5'->3' / start->end.
    std::reverse(q.begin(),  q.end());
    std::reverse(mk.begin(), mk.end());
    std::reverse(t.begin(),  t.end());
    std::reverse(path_nodes.begin(), path_nodes.end());

    a.q_line = q; a.m_line = mk; a.t_line = t;
    a.length = static_cast<int>(q.size());
    a.identities = static_cast<int>(std::count(mk.begin(), mk.end(), '|'));

    std::string path;
    for (std::size_t k = 0; k < path_nodes.size(); ++k) {
        if (k) path += ">";
        path += g.name[path_nodes[k]];
    }
    a.node_path = path;
    return a;
}
