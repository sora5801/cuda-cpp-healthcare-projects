#!/usr/bin/env python3
# ===========================================================================
# scripts/make_synthetic.py  --  Generate the synthetic sample network
# ---------------------------------------------------------------------------
# Project 6.6 : Neuronal Network Simulation (Biophysical)
#
# WHY THIS EXISTS
#   Real neuronal morphologies (NeuroMorpho.Org) and published models (ModelDB)
#   are large and license-encumbered, so this repo ships a TINY, clearly
#   SYNTHETIC network so the demo runs offline with zero downloads. This script
#   writes that sample deterministically; it takes no randomness, so the file --
#   and therefore demo/expected_output.txt -- is byte-stable across machines.
#
# WHAT IT WRITES  (single whitespace-separated line, parsed by load_network):
#   ncell ncomp dt steps v_rest n_stim i_stim gAxial wSyn tauSyn
#     ncell   number of neurons in the ring
#     ncomp   compartments per neuron (soma + dendrites), must be <= NN_MAX_COMP (8)
#     dt      integration timestep in ms
#     steps   number of timesteps (run length = steps*dt ms)
#     v_rest  resting membrane voltage in mV
#     n_stim  number of leading cells given a startup depolarisation
#     i_stim  startup depolarisation added to those somata in mV
#     gAxial  inter-compartment (axial) coupling conductance, mS/cm^2
#     wSyn    excitatory synaptic conductance added per presynaptic spike, mS/cm^2
#     tauSyn  synaptic conductance decay time constant, ms
#
#   The defaults are tuned so the kicked cells fire, their spikes propagate one
#   hop per synaptic delay around the ring, and the run captures a clean
#   travelling wave of activity -- an interpretable, verifiable result.
#
# USAGE
#   python scripts/make_synthetic.py                 # writes data/sample/network.txt
#   python scripts/make_synthetic.py --ncell 64      # a bigger ring
# ===========================================================================
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder
OUT = ROOT / "data" / "sample" / "network.txt"


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic neuronal-network sample.")
    ap.add_argument("--ncell",  type=int,   default=16,   help="neurons in the ring")
    ap.add_argument("--ncomp",  type=int,   default=4,    help="compartments per neuron (<=8)")
    ap.add_argument("--dt",     type=float, default=0.025, help="timestep (ms)")
    ap.add_argument("--steps",  type=int,   default=4000, help="number of timesteps")
    ap.add_argument("--v-rest", type=float, default=-65.0, help="resting voltage (mV)")
    ap.add_argument("--n-stim", type=int,   default=1,    help="leading cells kicked")
    ap.add_argument("--i-stim", type=float, default=45.0, help="startup depolarisation (mV)")
    ap.add_argument("--gaxial", type=float, default=0.30, help="axial coupling (mS/cm^2)")
    ap.add_argument("--wsyn",   type=float, default=0.90, help="synaptic weight (mS/cm^2)")
    ap.add_argument("--tausyn", type=float, default=2.0,  help="synaptic decay tau (ms)")
    ap.add_argument("--out",    default=str(OUT), help="output path")
    args = ap.parse_args()

    if args.ncomp > 8:
        raise SystemExit("ncomp must be <= 8 (NN_MAX_COMP in src/neuron.h)")

    fields = [args.ncell, args.ncomp, args.dt, args.steps, args.v_rest,
              args.n_stim, args.i_stim, args.gaxial, args.wsyn, args.tausyn]
    line = " ".join(f"{v:g}" for v in fields)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(line + "\n", encoding="utf-8")
    print(f"[make_synthetic] wrote {args.out}  "
          f"(ncell={args.ncell}, ncomp={args.ncomp}, steps={args.steps}; SYNTHETIC)")


if __name__ == "__main__":
    main()
