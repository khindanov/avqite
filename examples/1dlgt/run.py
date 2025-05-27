#!/usr/bin/env python
import argparse
import os
import sys
import warnings

import numpy
from mpi4py import MPI

from avqite.ansatz import ansatzSinglePool
from avqite.model import model

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--rcut",
    type=float,
    default=1.0e-2,
    help="McLachlan distance cut-off. dflt: 0.01",
)
parser.add_argument(
    "-f",
    "--fcut",
    type=float,
    default=1.0e-2,
    help="invidual unitary cut-off ratio. dflt: 0.01",
)
parser.add_argument(
    "-m",
    "--maxadd",
    type=int,
    default=5,
    help="Max. allowed unitaries to be added at one iteration. dflt: 5.",
)
parser.add_argument(
    "-n",
    "--maxntheta",
    type=int,
    default=-1,
    help="Max. total allowed unitaries to be added. dflt: -1 (unlimited).",
)
parser.add_argument(
    "-g",
    "--gsdegeneracy",
    type=int,
    default=1,
    help="Ground state degeneracy. dflt: 1.",
)
parser.add_argument(
    "--eigvals",
    type=int,
    default=7,
    help="Eigenvalues to be calculated by ED. dflt: 7.",
)
parser.add_argument(
    "--invmode",
    type=int,
    default=1,
    help="matrix inversion mode [0: inv, 1: cg ]. dflt: 1.",
)
parser.add_argument(
    "--noarpack", action="store_true", help="Use eigh instead of arpack. dflt: False."
)
parser.add_argument(
    "--notetras", action="store_true", help="Do not use tetras. dflt: False."
)
parser.add_argument(
    "--localh", action="store_true", help="Use local_expect for h. dflt: False."
)
parser.add_argument(
    "-b",
    "--bound",
    type=float,
    default=10,
    help="Bounds for dtheta/dt: [-b, b]. dflt: 10",
)
parser.add_argument(
    "--delta",
    type=float,
    default=1e-4,
    help="Tikhonov parameter. dflt: 1e-4. Nagative value switch on lsq.",
)
parser.add_argument(
    "-v",
    "--vtol",
    type=float,
    default=1e-4,
    help="Tolerance for grandient. dflt: 1e-4.",
)
parser.add_argument(
    "-t", "--dt", type=float, default=0.02, help="Time step size. dflt: 0.02"
)
parser.add_argument(
    "--tmax", type=float, default=1e7, help="Maximal time for simulations. dflt: 1e7"
)
parser.add_argument(
    "--tf", type=float, default=numpy.inf, help="Final time. dflt: numpy.inf."
)
parser.add_argument(
    "-md",
    "--model_dir",
    type=str,
    default="1d-tfim/",
    help="Model directory for incar file and save.",
)
parser.add_argument("--filename", type=str, default="N8g0.1", help="Filename.")
parser.add_argument(
    "-o",
    "--optimize",
    type=str,
    default="greedy",
    help="Optimizer to use when computing tensor contractions.",
)
parser.add_argument(
    "-s",
    "--simplify_sequence",
    type=str,
    default="ADCRS",
    help="Simplification sequence to use by quimb.",
)
parser.add_argument("-be", "--backend", type=str, default="None", help="Backend.")

model_dir = os.path.dirname(os.path.abspath(__file__))

args = parser.parse_args()

mdl = model(localh=args.localh, filename=args.filename, model_dir=model_dir)
ans = ansatzSinglePool(
    mdl,
    rcut=args.rcut,
    fcut=args.fcut,
    max_add=args.maxadd,
    maxntheta=args.maxntheta,
    bounds=[-args.bound, args.bound],
    dt=args.dt,
    invmode=args.invmode,
    delta=args.delta,
    vtol=args.vtol,
    tetras=not args.notetras,
    tmax=args.tmax,
    tf=args.tf,
    filename=args.filename,
    model_dir=model_dir,
    optimize=args.optimize,
    simplify_sequence=args.simplify_sequence,
    backend=args.backend,
)

comm = MPI.COMM_WORLD
m_rank = comm.Get_rank()

ans.run()

if m_rank == 0:
    print("costs:", ans._e)
    ans.psave_ansatz_simp()
    ans.psave_ansatz_inp()
