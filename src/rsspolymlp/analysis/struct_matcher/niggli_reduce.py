import numpy as np


def get_niggli_reduced_cell(axis, tol: float = 1e-5):
    """
    Get the Niggli-reduced cell using a numerically stable algorithm.

    References
    ----------
    1. I. Krivy and B. Gruber, Acta Cryst. A 32, 297-298 (1976).
    2. R. W. Grosse-Kunstleve, N. K. Sauter, and P. D. Adams, Acta Cryst. A 60, 1-6 (2004).

    Parameters
    ----------
    axis : (3, 3) array_like
        Lattice vectors defining the unit cell in Cartesian coordinates.
        Each row represents one lattice vector (a, b, or c).
        Equivalent to np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]).
    tol : float, default=1e-5
        Numerical tolerance for the reduction procedure.

    Returns
    -------
    axis_niggli : (3, 3) ndarray
        Niggli-reduced lattice vectors, with each row representing one lattice vector.
    """

    def get_G_components(G):
        return (
            G[0, 0],
            G[1, 1],
            G[2, 2],
            2 * G[1, 2],
            2 * G[0, 2],
            2 * G[0, 1],
        )

    def convert_G_and_P(C, G, P):
        G = C.T @ G @ C
        P = P @ C
        return G, P

    G: np.ndarray = np.dot(axis, axis.T)
    P = np.eye(3, dtype=int)
    e = tol

    iteration = 100
    for n_iter in range(iteration):
        if n_iter == iteration - 1:
            raise RuntimeError("Niggli reduction did not converge.")

        aa, bb, cc, bc, ac, ab = get_G_components(G)

        # A1: Main condition: aa <= bb; Special condition: |bc| < |ac|
        if bb + e < aa or (abs(aa - bb) < e and abs(bc) > abs(ac) + e):
            C = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
            G, P = convert_G_and_P(C, G, P)
            aa, bb, cc, bc, ac, ab = get_G_components(G)

        # A2: Main condition: bb <= cc; Special condition: |ac| < |ab|
        if (cc + e < bb) or (abs(bb - cc) < e and abs(ac) > abs(ab) + e):
            C = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
            G, P = convert_G_and_P(C, G, P)
            continue

        ell = 0 if abs(bc) < e else bc / abs(bc)
        m = 0 if abs(ac) < e else ac / abs(ac)
        n = 0 if abs(ab) < e else ab / abs(ab)

        # A3: Main condition: bc > 0, ac > 0, ab > 0 (Type I)
        if ell * m * n == 1:
            i = -1 if ell == -1 else 1
            j = -1 if m == -1 else 1
            k = -1 if n == -1 else 1
            C = np.diag((i, j, k))
            G, P = convert_G_and_P(C, G, P)

        # A4: Main condition: bc <= 0, ac <= 0, ab <= 0 (Type II)
        if ell * m * n in (0, -1):
            i = -1 if ell == 1 else 1
            j = -1 if m == 1 else 1
            k = -1 if n == 1 else 1
            if i * j * k == -1:
                # Keep the proper rotation
                if n == 0:
                    k = -1
                elif m == 0:
                    j = -1
                elif ell == 0:
                    i = -1
            C = np.diag((i, j, k))
            G, P = convert_G_and_P(C, G, P)

        aa, bb, cc, bc, ac, ab = get_G_components(G)

        # A5: Main condition: |bc| <= bb;
        # Special conditions: if bc = bb: ab/2 <= ac,
        # elif bc = -bb: ab = 0 (otherwise, convert to Type I)
        if (
            abs(bc) > bb + e
            or (abs(bc - bb) < e and ab - e > 2 * ac)
            or (abs(bc + bb) < e and -e > ab)
        ):
            # Basis transformation: c' = c - sign(bc) * b
            #   cc' = cc + bb - |bc|
            #   bc' = bc/2 - sign(bc) * bb
            #   ac' = ac/2 - sign(bc) * ab/2
            # If bc = bb and ab > 2 * ac:
            #   The transformed cell is Type I and satisfies |ab'| / 2 <= |ac'|,
            #   because 2 * |ac'| = 2 * |ac - ab| > ab.
            # If bc = -bb and ab < 0:
            #   ac' = ac + ab < 0 and bc' = bb > 0.
            #   The transformed cell is Type I.
            # A6 and A7 follow by analogous arguments.
            C = np.array([[1, 0, 0], [0, 1, -bc / abs(bc)], [0, 0, 1]])
            G, P = convert_G_and_P(C, G, P)
            continue

        # A6: Main condition: |ac| <= aa;
        # Special conditions: if ac = aa: ab/2 <= bc,
        # elif ac = -aa: ab = 0 (otherwise, convert to Type I)
        if (
            abs(ac) > aa + e
            or (abs(aa - ac) < e and ab - e > 2 * bc)
            or (abs(aa + ac) < e and -e > ab)
        ):
            # Basis transformation: c' = c - sign(ac) * a
            #   cc' = cc + aa - |ac|
            #   ac' = ac/2 - sign(ac) * aa
            #   bc' = bc/2 - sign(ac) * ab/2
            C = np.array([[1, 0, -ac / abs(ac)], [0, 1, 0], [0, 0, 1]])
            G, P = convert_G_and_P(C, G, P)
            continue

        # A7: Main condition: |ab| <= aa;
        # Special conditions: if ab = aa: ac/2 <= bc,
        # elif ab = -aa: ac = 0 (otherwise, convert to Type I)
        if (
            abs(ab) > aa + e
            or (abs(aa - ab) < e and ac - e > 2 * bc)
            or (abs(aa + ab) < e and -e > ac)
        ):
            # Basis transformation: b' = b - sign(ab) * a
            #   bb' = bb + aa - |ab|
            #   ab' = ab/2 - sign(ab) * aa
            #   bc' = bc/2 - sign(ab) * ac/2
            C = np.array([[1, -ab / abs(ab), 0], [0, 1, 0], [0, 0, 1]])
            G, P = convert_G_and_P(C, G, P)
            continue

        # A8: Main condition: |bc| + |ac| + |ab| <= aa + bb (Type II);
        # Special condition: if |bc| + |ac| + |ab| = aa + bb: aa <= |ac| + |ab/2|
        # (otherwise, convert to Type I)
        if -e > bc + ac + ab + aa + bb or (
            abs(bc + ac + ab + aa + bb) < e < ab + (aa + ac) * 2
        ):
            # Basis transformation: c' = a + b + c
            #   cc' = cc + aa + bb + ab + ac + bc
            #   ac' = 2 * aa + ab + ac (special condition: ac' > 0)
            #   bc' = 2 * bb + ab + bc (special condition: bc' < 0)
            C = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])
            G, P = convert_G_and_P(C, G, P)
            continue

        break

    axis_niggli = P.T @ axis
    return axis_niggli, P.T
