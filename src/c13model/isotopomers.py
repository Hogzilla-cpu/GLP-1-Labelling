"""Isotopomer bookkeeping.

An isotopomer distribution of an ``n``-carbon molecule is a vector of length
``2**n``.  Bit ``k-1`` of the index is set when carbon ``k`` is 13C, so for
glutamate index ``0b01000`` (8) is [4-13C]glutamate and ``0b11111`` (31) is
[U-13C]glutamate.  Each vector sums to 1.
"""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np


def index_of(labeled_carbons: Sequence[int]) -> int:
    """Isotopomer index for a set of labeled carbons (1-based carbon numbers)."""
    return sum(1 << (c - 1) for c in labeled_carbons)


def unlabeled(n_carbons: int, natural_abundance: float = 0.0) -> np.ndarray:
    """Distribution of an unlabeled pool (optionally at natural 13C abundance)."""
    p = natural_abundance
    idx = np.arange(2**n_carbons)
    n_lab = np.array([bin(i).count("1") for i in idx])
    return (p**n_lab) * ((1 - p) ** (n_carbons - n_lab))


def point(n_carbons: int, labeled_carbons: Sequence[int]) -> np.ndarray:
    """Distribution containing a single isotopomer."""
    x = np.zeros(2**n_carbons)
    x[index_of(labeled_carbons)] = 1.0
    return x


class Transform:
    """Carbon-atom transition for a reaction.

    ``mapping`` has one entry per product carbon (product C1 first); each entry
    is ``(substrate_number, substrate_carbon)``, both 0- and 1-based
    respectively.  Product distributions are computed exactly by enumerating
    all substrate isotopomer combinations once and caching the product index.
    """

    def __init__(self, substrate_sizes: Sequence[int], mapping: Sequence[tuple[int, int]]):
        self.substrate_sizes = tuple(substrate_sizes)
        self.n_out = len(mapping)
        shape = tuple(2**n for n in substrate_sizes)
        prod_idx = np.zeros(shape, dtype=np.intp)
        for combo in itertools.product(*(range(s) for s in shape)):
            out = 0
            for pos, (s, c) in enumerate(mapping):
                if combo[s] >> (c - 1) & 1:
                    out |= 1 << pos
            prod_idx[combo] = out
        self._prod_idx = prod_idx.ravel()

    def __call__(self, *dists: np.ndarray) -> np.ndarray:
        weights = dists[0]
        for d in dists[1:]:
            weights = np.multiply.outer(weights, d)
        return np.bincount(self._prod_idx, weights=np.ravel(weights), minlength=2**self.n_out)


def fractional_enrichment(x: np.ndarray, carbon: int) -> np.ndarray:
    """13C fraction at ``carbon``. ``x`` may be (2**n,) or (..., 2**n)."""
    mask = (np.arange(x.shape[-1]) >> (carbon - 1)) & 1
    return x @ mask


def multiplets(x: np.ndarray, carbon: int, n_carbons: int) -> dict[str, np.ndarray]:
    """Fractions of the 13C signal at ``carbon`` split by neighbour labeling.

    Returns keys ``S`` (no labeled neighbour), ``D_left`` (only carbon-1
    labeled), ``D_right`` (only carbon+1 labeled) and ``Q`` (both neighbours
    labeled; a doublet of doublets, seen as a triplet when both couplings are
    equal, e.g. glutamate C3).  Values are fractions of the total 13C signal at
    that carbon (NaN where the carbon is unlabeled).
    """
    idx = np.arange(x.shape[-1])
    here = (idx >> (carbon - 1)) & 1
    left = (idx >> (carbon - 2)) & 1 if carbon > 1 else np.zeros_like(idx)
    right = (idx >> carbon) & 1 if carbon < n_carbons else np.zeros_like(idx)
    total = x @ here
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "S": (x @ (here * (1 - left) * (1 - right))) / total,
            "D_left": (x @ (here * left * (1 - right))) / total,
            "D_right": (x @ (here * (1 - left) * right)) / total,
            "Q": (x @ (here * left * right)) / total,
        }
