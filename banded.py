"""Banded symmetric positive-definite solves for the frame stiffness (reverse Cuthill-McKee order).

A 2D frame stiffness matrix of n free DOFs has a half-bandwidth of a few dozen once the DOFs are
renumbered, so a banded Cholesky (LAPACK pbtrf / pbtrs via scipy) costs ~ n u^2 instead of n^3 / 3
for the dense factorisation. The pattern (which entries can be non-zero) is fixed by the element
connectivity, so the renumbering and the scatter maps are computed once per model and reused for every
P-Delta iteration (K - Kg(N)) and every load combination.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee


class BandedSPD:
    """Pattern of an n x n symmetric matrix given by (rows, cols) index pairs (both triangles or either)."""

    def __init__(self, n: int, rows: np.ndarray, cols: np.ndarray):
        self.n = n
        pat = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
        pat = pat + pat.T
        self.perm = np.asarray(reverse_cuthill_mckee(pat, symmetric_mode=True), dtype=np.intp)   # new -> old
        self.pos = np.empty(n, dtype=np.intp)
        self.pos[self.perm] = np.arange(n)                                                        # old -> new
        p, q = self.pos[pat.tocoo().row], self.pos[pat.tocoo().col]
        self.u = int(np.max(np.abs(p - q))) if len(p) else 0

    def scatter(self, rows: np.ndarray, cols: np.ndarray):
        """Flat indices into the (u+1, n) upper band storage for entries (rows, cols) of the original
        matrix; entries below the diagonal (in the new order) map to -1 (symmetric, not stored)."""
        p, q = self.pos[np.asarray(rows)], self.pos[np.asarray(cols)]
        flat = (self.u + p - q) * self.n + q
        return np.where(p <= q, flat, -1)

    def band(self, values: np.ndarray, flat: np.ndarray) -> np.ndarray:
        """Upper band storage from values at the flat indices of scatter() (-1 entries skipped)."""
        ab = np.zeros((self.u + 1) * self.n)
        keep = flat >= 0
        np.add.at(ab, flat[keep], values[keep])
        return ab.reshape(self.u + 1, self.n)

    def factor(self, ab: np.ndarray):
        """Cholesky of the band; raises numpy.linalg.LinAlgError if not positive definite."""
        return cholesky_banded(ab, lower=False, check_finite=False)

    def solve(self, c: np.ndarray, b: np.ndarray) -> np.ndarray:
        """x = A^-1 b in the original numbering (b: (n,) or (n, k))."""
        y = cho_solve_banded((c, False), b[self.perm], check_finite=False)
        return y[self.pos]

    def matvec_band(self, ab: np.ndarray, x: np.ndarray) -> np.ndarray:
        """A x for A in upper band storage (original numbering)."""
        xn, n, u = x[self.perm], self.n, self.u
        y = ab[u] * xn
        for k in range(1, u + 1):
            d = ab[u - k, k:]                     # A[i, i + k], i = 0 .. n - k - 1
            y[:-k] += d * xn[k:]
            y[k:] += d * xn[:-k]
        return y[self.pos]
