# Data — 3.18 Protein Language Model Inference

## Committed sample (`sample/protein_sample.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** peptide (`scripts/make_synthetic.py`) |
| License | Public domain (CC0) — it is synthetic |
| Size | < 1 KB |
| Contents | 1 protein sequence (24 residues) + the model shape |

### File format

```
<d_model> <n_heads>     # embedding width and number of attention heads
<sequence>              # the protein, one char per residue (20 canonical AAs)
```

The default sample is:

```
32 4
MKTAYIAKQRQISFVKSHFSRQLE
```

i.e. a **24-residue** peptide processed by a single self-attention block with
`d_model = 32`, `n_heads = 4`, so `d_head = 8`. The sequence uses only the 20
canonical amino acids (alphabet `ACDEFGHIKLMNPQRSTVWY`), matching `AA_ALPHABET`
in `src/attention_math.h`.

> **Important:** the model *weights* are **not** in this file. Every embedding and
> projection weight is generated deterministically from an integer hash inside
> the C++ code (`embed_value` / `weight_value` in `src/attention_math.h`), so the
> CPU and GPU build byte-identical tensors with no data download. The only input
> is the sequence above.

This tiny file lets `demo/run_demo` run **offline, with zero downloads** — a hard
requirement for every project (CLAUDE.md §8). A longer synthetic peptide:
`python scripts/make_synthetic.py --len 64`.

## Full dataset (real protein language models)

To run a *real* PLM you need (a) a trained model and (b) input sequences. This
project teaches the attention *math*, not the trained weights; for the real thing:

- **Trained models — fair-esm** (<https://github.com/facebookresearch/esm>):
  ESM-2 (8M–15B params) and ESMFold. Weights download via the `torch.hub` /
  `transformers` APIs (hundreds of MB to tens of GB). **Not redistributed here.**
- **Sequences — UniRef50/90** (<https://www.uniprot.org/help/uniref>): the PLM
  training corpus (FASTA, many GB).
- **ESM Metagenomic Atlas** (<https://esmatlas.com/>): 700M predicted structures.
- **PDB** (<https://www.rcsb.org/>) and **CATH/SCOP** (<https://www.cathdb.info/>):
  structural validation sets for ESMFold.

`scripts/download_data.ps1` / `.sh` print these pointers; they download nothing
(the demo is fully self-contained on synthetic weights).

## Provenance & honesty

The sample sequence is a **synthetic, biologically-meaningless** peptide, and the
model weights are **synthetic deterministic hashes**, not trained parameters.
They exist to make the attention forward pass concrete, reproducible, and
verifiable (CPU vs GPU). No output here has any biological or clinical meaning.
