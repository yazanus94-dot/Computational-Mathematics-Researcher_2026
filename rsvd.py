"""Randomized low-rank approximation (Halko-Martinsson-Tropp) with power iterations."""
import numpy as np


def make_matrix(m, n, decay=0.5, rng=None):
    """Matrix with known singular values sigma_j = (j+1)^(-decay*4)... exponential-free polynomial decay."""
    rng = np.random.default_rng(rng)
    r = min(m, n)
    U, _ = np.linalg.qr(rng.standard_normal((m, r)))
    V, _ = np.linalg.qr(rng.standard_normal((n, r)))
    s = (np.arange(1, r + 1)) ** (-decay * 4.0)
    return (U * s) @ V.T, s


def range_finder(A, k, p=10, q=0, rng=None):
    """Orthonormal Q (m x (k+p)) approximating range(A); q power iterations with re-orthonormalisation."""
    rng = np.random.default_rng(rng)
    Omega = rng.standard_normal((A.shape[1], k + p))
    Q, _ = np.linalg.qr(A @ Omega)
    for _ in range(q):
        W, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ W)
    return Q


def rsvd(A, k, p=10, q=0, rng=None):
    """Return U, s, Vt of rank k."""
    Q = range_finder(A, k, p, q, rng)
    B = Q.T @ A
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    return (Q @ Ub)[:, :k], s[:k], Vt[:k]


def frob_bound(s, k, p):
    """Halko et al. Thm 10.5 expectation bound for E||A - QQ^T A||_F (q=0)."""
    return np.sqrt(1 + k / (p - 1)) * np.sqrt(np.sum(s[k:] ** 2))


def spectral_error(A, Q):
    return np.linalg.norm(A - Q @ (Q.T @ A), 2)
