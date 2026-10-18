import numpy as np
from rsvd import *

def test_exact_low_rank_recovered():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((200, 8)) @ rng.standard_normal((8, 150))
    U, s, Vt = rsvd(A, 8, p=5, rng=1)
    assert np.linalg.norm(A - (U * s) @ Vt) / np.linalg.norm(A) < 1e-10

def test_expected_frobenius_bound_holds():
    A, s = make_matrix(300, 200, 0.3, rng=0)
    k, p = 10, 8
    errs = [np.linalg.norm(A - (lambda Q: Q @ (Q.T @ A))(range_finder(A, k, p, 0, rng=t)), 'fro')
            for t in range(40)]
    assert np.mean(errs) <= frob_bound(s, k, p)

def test_power_iterations_reduce_spectral_error():
    A, s = make_matrix(400, 300, 0.2, rng=2)
    k = 10
    e = [np.mean([spectral_error(A, range_finder(A, k, 5, q, rng=t)) for t in range(10)]) for q in (0, 1, 2)]
    assert e[0] > e[1] > e[2]
    assert e[2] < 1.5 * s[k]          # near-optimal (Eckart-Young: s[k] for rank k+... )

def test_singular_values_accuracy():
    A, s = make_matrix(300, 300, 0.5, rng=3)
    _, sr, _ = rsvd(A, 5, p=10, q=2, rng=4)
    assert np.allclose(sr, s[:5], rtol=1e-6)
