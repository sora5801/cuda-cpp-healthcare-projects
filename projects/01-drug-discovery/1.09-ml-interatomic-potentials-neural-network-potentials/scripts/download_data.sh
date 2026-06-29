#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.9 : ML Interatomic Potentials (Neural Network Potentials)
#
# CONTRACT (CLAUDE.md sec 8): idempotent, documented, prints source URL +
# expected size, and NEVER bypasses credentials/registration. The committed
# synthetic sample is enough to run the demo offline; the real NNP training sets
# below are large research datasets you fetch only if you want to go further.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.9 -- ML Interatomic Potentials (NNP)"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed tiny sample (data/sample/water_cluster.xyzc) is SYNTHETIC and"
echo "is all the demo needs. For a larger synthetic cluster, run:"
echo "    python scripts/make_synthetic.py --mols 64"
echo
echo "REAL NNP training datasets (quantum-chemistry energies/forces):"
echo "  ANI-1ccx  CCSD(T) energies, ~500k conformers   https://github.com/isayev/ANI1ccx_dataset"
echo "  SPICE     DFT energies+forces, drugs+peptides   https://github.com/openmm/spice-dataset"
echo "  rMD17     revised MD17 force benchmark           https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038"
echo "  OE62      62k organic molecules, DFT energetics  (verify current URL)"
echo
echo "These are HDF5/archive downloads (hundreds of MB to GB). This teaching"
echo "project does NOT train on them -- it uses a fixed, manufactured network to"
echo "demonstrate the descriptor + per-atom MLP pipeline. To USE the real data you"
echo "would: (1) read geometries/energies, (2) train weights (TorchANI/NequIP/MACE),"
echo "(3) export the weights into AtomicNet. See README 'Prior art & further reading'."
echo
echo "Idempotent fetch pattern when wiring a real set:"
echo "  1) skip download if the file already exists with the right checksum"
echo "  2) print source URL + expected size + SHA256"
echo "  3) for credentialed sets, print registration instructions ONLY (never bypass)"
