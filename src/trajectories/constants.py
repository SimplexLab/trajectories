from math import cos, sin

import numpy as np
import torch
from torchjd.aggregation import (
    IMTLG,
    MGDA,
    AlignedMTL,
    CAGrad,
    DualProj,
    GradDrop,
    Mean,
    NashMTL,
    PCGrad,
    Random,
    UPGrad,
)

from trajectories.objectives import (
    ConvexQuadraticForm,
    ElementWiseQuadratic,
    Multinorm,
    QuadraticForm,
)

AGGREGATORS = {
    "upgrad": UPGrad(reg_eps=1e-7, norm_eps=1e-9),
    "mgda": MGDA(),
    "cagrad": CAGrad(c=0.5),
    "nashmtl": NashMTL(n_tasks=2, optim_niter=1),
    "nashmtl20": NashMTL(n_tasks=20, optim_niter=1),
    "graddrop": GradDrop(),
    "imtl_g": IMTLG(),
    "aligned_mtl": AlignedMTL(),
    "dualproj": DualProj(),
    "pcgrad": PCGrad(),
    "random": Random(),
    "mean": Mean(),
}
LR_MULTIPLIERS = {
    "upgrad": 1.0,
    "mgda": 1.0,
    "cagrad": 1.0,
    "nashmtl": 2.0,
    "nashmtl20": 2.0,
    "graddrop": 0.5,
    "imtl_g": 2.0,
    "aligned_mtl": 1.0,
    "dualproj": 1.0,
    "pcgrad": 0.5,
    "random": 1.0,
    "mean": 1.0,
}
AGGREGATOR_ORDER = {
    "upgrad": 9,
    "mgda": 1,
    "cagrad": 5,
    "nashmtl": 7,
    "nashmtl20": 7,
    "graddrop": 3,
    "imtl_g": 4,
    "aligned_mtl": 8,
    "dualproj": 2,
    "random": 6,
    "mean": 0,
    # No location for PCGrad as it's equivalent to UPGrad with 2 objectives
}
LATEX_NAMES = {
    "upgrad": r"$\mathcal A_{\mathrm{UPGrad}}$ (ours)",
    "mgda": r"$\mathcal A_{\mathrm{MGDA}}$",
    "cagrad": r"$\mathcal A_{\mathrm{CAGrad}}$",
    "nashmtl": r"$\mathcal A_{\mathrm{Nash-MTL}}$",
    "nashmtl20": r"$\mathcal A_{\mathrm{Nash-MTL}}$",
    "graddrop": r"$\mathcal A_{\mathrm{GradDrop}}$",
    "imtl_g": r"$\mathcal A_{\mathrm{IMTL-G}}$",
    "aligned_mtl": r"$\mathcal A_{\mathrm{Aligned-MTL}}$",
    "dualproj": r"$\mathcal A_{\mathrm{DualProj}}$",
    "pcgrad": r"$\mathcal A_{\mathrm{PCGrad}}$",
    "random": r"$\mathcal A_{\mathrm{RGW}}$",
    "mean": r"$\mathcal A_{\mathrm{Mean}}$",
}

THETA = np.pi / 16

OBJECTIVES = {
    "EWQ": ElementWiseQuadratic(2),
    "CQF": ConvexQuadraticForm(
        Bs=[
            torch.tensor([[cos(THETA), -sin(THETA)], [sin(THETA), cos(THETA)]])
            @ torch.diag(torch.tensor([1.0, 0.1])),
            torch.tensor([[cos(THETA), sin(THETA)], [-sin(THETA), cos(THETA)]])
            @ torch.diag(torch.tensor([torch.sqrt(torch.tensor(3.0)), 0.1])),
        ],
        us=[torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])],
    ),
    "QF": QuadraticForm(
        As=[torch.tensor([[2.0, -1.0], [-1.0, 2.0]]), torch.tensor([[6.0, -3.0], [-3.0, 6.0]])],
        us=[torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])],
    ),
    "MN2": Multinorm(torch.tensor([1.0, 10.0])),
    "MN20": Multinorm(torch.arange(1, 21)),
}
BASE_LEARNING_RATES = {
    "EWQ": 0.075,
    "CQF": 0.05,
    "QF": 0.01,
    "MN2": 0.02,
    "MN20": 0.005,
}
INITIAL_POINTS = {
    "EWQ": [
        [3.0, -2.0],
        [0.0, -3.0],
        [-4.0, 4.0],
        [-3.0, 4.0],
        [-3.5, -0.75],
    ],
    "CQF": [
        [0.5, 0.5],
        [-1.0, 7.0],
        [0.0, 0.0],
        [1.0, 6.0],
    ],
    "QF": [
        [0.5, 0.5],
        [0.0, 0.0],
        [0.5, -0.5],
    ],
    "MN2": [
        [0.0, 0.0],
        [-5.0, 5.0],
        [10.0, 5.0],
        [10.0, 0.0],
        [20.0, 0.0],
    ],
    "MN20": [
        [0.0] * 20,
    ],
}
N_ITERS = {
    "EWQ": 50,
    "CQF": 500,
    "QF": 50,
    "MN2": 50,
    "MN20": 500,
}
