# Data — 7.2 Drug-Target Interaction Prediction (GNN)

## Committed sample (`sample/dti_sample.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** batched molecular graphs + protein descriptors (`scripts/make_synthetic.py`) |
| License | Public domain (CC0) — synthetic |
| Size | < 2 KB |
| Contents | 6 drug graphs (3–5 atoms each, 24 atoms total) × 4 protein targets, feature width `F=8` |

### File format

```
D P                      # number of drug graphs, number of protein targets
true_drug true_prot      # implanted ground-truth interaction (0-based indices)
                         # --- for each drug d in 0..D-1: ---
n_d k_d                  #   n_d atoms, then k_d undirected bonds
<n_d rows of F floats>   #   initial atom feature vectors (length F each)
<k_d rows of "u v">      #   bonds as LOCAL atom indices (0..n_d-1)
                         # --- then for each protein p in 0..P-1: ---
<F floats>               #   protein descriptor vector (length F)
```

- **Atoms** are graph nodes; each carries a length-`F` feature vector (a stand-in
  for one-hot atom type + degree + charge, etc.). Bonds are **undirected** and are
  expanded into both directions at load time; a **self-loop** is added to every
  node so it keeps its own feature during aggregation (standard GNN practice).
- **`F` and the number of message-passing rounds `T` must match `src/gnn.h`**
  (`GNN_F`, `GNN_T`). `make_synthetic.py` mirrors them.
- **`true_drug true_prot`** is the pair the fixed-weight model ranks highest —
  see "Provenance & honesty" — which the demo then recovers.

## Full dataset

Real DTI benchmarks measure binding affinity (Kd/Ki) or binary interaction for
drug–target pairs; you featurize each drug's molecular graph and encode each
protein, then write the format above:

- **BindingDB** — <https://www.bindingdb.org/> (~2.9M measured affinities).
- **ChEMBL** — <https://www.ebi.ac.uk/chembl/> (>20M bioactivity records).
- **Davis** — kinase inhibitor affinities (442 kinases × 68 drugs).
- **KIBA** — integrated kinase-inhibitor bioactivity benchmark.

Toolkits that build the graphs and TRAIN the full model:
**DeepPurpose** (<https://github.com/kexinhuang12345/DeepPurpose>), **TorchDrug**
(<https://github.com/DeepGraphLearning/torchdrug>), **DGL-LifeSci**
(<https://github.com/awslabs/dgl-lifesci>). Bigger synthetic batch (no download):
`python scripts/make_synthetic.py --drugs 64 --proteins 16`.

## Provenance & honesty

The sample is **synthetic** — small hand-built chain graphs and descriptor
vectors, with **no clinical meaning**. The network weights are **untrained**
(deterministically seeded; see `src/gnn.h`), so the scores are *illustrative of
the machinery only*, not binding predictions. To keep the "ground truth"
label honest, `make_synthetic.py` runs the exact same fixed-weight forward pass
the C++ code uses and writes whichever drug×protein pair the model ranks highest
as `true_drug true_prot`. The demo then "recovers" that pair — validating the
GNN + pairwise-scoring pipeline end-to-end, not any real drug–target affinity.
