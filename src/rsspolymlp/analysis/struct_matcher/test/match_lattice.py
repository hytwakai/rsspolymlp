import math

import numpy as np
import spglib
from sympy import ZZ, Matrix
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.normalforms import smith_normal_decomp


def lattice_transformation(A, B):
    """NB = MA"""
    C = A @ np.linalg.inv(B)
    P, q = approximate_rational_matrix(C)

    # Find M satisfying M @ P == 0 (mod q)
    M = find_integer_matrix(P, q)
    MP = M @ P
    assert np.all(MP % q == 0)

    N = M @ C
    A0 = M @ A
    _, M_niggli = reduced_niggle(A0)
    M = M_niggli @ M
    N = M_niggli @ N
    return M, N


def find_integer_matrix(P, q):
    """Find an integer matrix M minimizing |det(M)|,
    subject to M @ P being divisible by q elementwise.
    """
    P = np.asarray(P, dtype=np.int64)

    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square matrix.")
    if q <= 0:
        raise ValueError("q must be a positive integer.")

    n = P.shape[0]

    # Smith normal form with transformation matrices
    dm = DomainMatrix.from_Matrix(Matrix(P.tolist())).convert_to(ZZ)
    D, U, V = smith_normal_decomp(dm)
    """a = (
        np.array(U.to_Matrix().tolist(), dtype=int)
        @ P
        @ np.array(V.to_Matrix().tolist(), dtype=int)
    )"""

    D = D.to_Matrix()
    U = U.to_Matrix()
    V = V.to_Matrix()

    # Compute K = diag(q / gcd(q, d_i))
    k = [q // math.gcd(q, abs(int(D[i, i]))) for i in range(n)]
    K = Matrix.diag(*k)
    M = K @ U

    # Verify the result
    MP = M * Matrix(P.tolist())
    assert all(int(x) % q == 0 for x in MP)
    assert M.det() != 0

    return np.array(M.tolist(), dtype=object)


def approximate_rational_matrix(C, max_denominator=100, tol=1e-8):
    C = np.asarray(C, dtype=float)

    for q in range(1, max_denominator + 1):
        P = np.rint(C * q).astype(np.int64)

        if np.max(np.abs(C - P / q)) <= tol:
            return P, q


def reduced_niggle(A):
    """A_niggli = MA"""
    A_niggli = spglib.niggli_reduce(
        A,
        eps=1e-5,
    )
    A = np.asarray(A, dtype=np.float64)
    A_niggli = np.asarray(A_niggli, dtype=np.float64)
    M_float = A_niggli @ np.linalg.inv(A)
    M = np.rint(M_float).astype(np.int64)
    assert np.allclose(M @ A, A_niggli, atol=1e-7)
    assert abs(round(np.linalg.det(M))) == 1
    return A_niggli, M
