import numpy as np

from rsspolymlp.analysis.struct_matcher.chiral_spg import get_chiral_spg
from rsspolymlp.common.property import PropUtil


def invert_and_swap_positions(lattice, positions, spg_number, symprec):
    """Return all position arrays reachable by inverting/swapping
    crystallographically equivalent lattice axes."""
    # Axis lengths (a, b, c) and angles (α, β, γ)
    prop = PropUtil(lattice, positions)
    abc_angle = np.asarray(prop.abc, dtype=float)  # (6,)
    abc, angles = abc_angle[:3], abc_angle[3:]
    tol = symprec * 10.0

    angle_similar = np.isclose(angles[:, None], angles[None, :], atol=tol)
    length_similar = np.isclose(abc[:, None], abc[None, :], atol=tol)

    near_90_flag = np.isclose(angles, 90.0, atol=tol)
    same_axis_flag = np.any(
        length_similar & angle_similar[:3, :3] & ~np.eye(3, dtype=bool), axis=1
    )

    # Inverting atomic positions
    if allow_all_invert(spg_number):
        invert_values = [np.array([1, 1, 1]), np.array([-1, -1, -1])]
    else:
        invert_values = [np.array([1, 1, 1])]
    invert_values = invert_positions(invert_values, near_90_flag)

    # Swapping equivalent axes.
    swap_values = [np.array([0, 1, 2])]
    swap_values = swap_positions(swap_values, same_axis_flag)

    return invert_values, swap_values


def allow_all_invert(spg_number):
    chiral_spg = get_chiral_spg()
    return spg_number not in chiral_spg


def invert_positions(invert_values, near_90_flag):
    _invert_values = invert_values.copy()
    for val in _invert_values:
        _val = val.copy()
        if np.all(near_90_flag):
            for pattern in [1, 2, 4]:
                mask = np.array([(pattern >> i) & 1 for i in range(3)], dtype=bool)
                _val[mask] = -_val[mask]
                invert_values.append(_val)
        elif np.sum(near_90_flag) == 2:
            idx = np.argmax(~near_90_flag)
            _val[idx] = -_val[idx]
            invert_values.append(_val)
        # else: ⇒ only the original array
    return invert_values


def swap_positions(swap_values, same_axis_flag):
    _swap_values = swap_values.copy()
    for val in _swap_values:
        _val = val.copy()
        active_cols = np.nonzero(same_axis_flag)[0]
        if len(active_cols) == 3:
            # If all 3 are equivalent: generate all 6 non‑trivial permutations
            perms = np.array(
                [[0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]], dtype=int
            )
            swap_values.extend([row for row in perms])
        elif len(active_cols) == 2:
            _val[active_cols[1]], _val[active_cols[0]] = (
                _val[active_cols[0]],
                _val[active_cols[1]],
            )
            swap_values.append(_val)
        elif len(active_cols) <= 1:
            # No axis is considered equivalent ⇒ do nothing (only original is used)
            pass
    return swap_values
