import numpy as np

from rsspolymlp.analysis.struct_matcher.chiral_spg import get_chiral_spg
from rsspolymlp.common.property import PropUtil


def allow_all_invert(spg_number):
    chiral_spg = get_chiral_spg()
    return spg_number not in chiral_spg


def invert_positions(pos_candidates, same_angle_90_flag):
    original_candidates = pos_candidates.copy()
    for pos in original_candidates:
        if np.all(same_angle_90_flag):
            for pattern in [1, 2, 4]:
                _pos = pos.copy()
                mask = np.array([(pattern >> i) & 1 for i in range(3)], dtype=bool)
                _pos[:, mask] = (-_pos[:, mask]) % 1.0
                pos_candidates.append(_pos)
        elif np.any(same_angle_90_flag):
            _pos = pos.copy()
            idx = np.argmax(~same_angle_90_flag)
            _pos[:, idx] = (-_pos[:, idx]) % 1.0
            pos_candidates.append(_pos)
        # else: all False ⇒ only the original array
    return pos_candidates


def swap_positions(pos_candidates, same_axis_flag):
    original_candidates = pos_candidates.copy()
    for pos in original_candidates:
        _pos = pos.copy()
        active_cols = np.nonzero(same_axis_flag)[0]
        if len(active_cols) == 3:
            # If all 3 are equivalent: generate all 6 non‑trivial permutations
            perms = np.array(
                [[0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]], dtype=int
            )
            for perm in perms:
                pos_candidates.append(_pos[:, perm])
        elif len(active_cols) == 2:
            _pos[:, (active_cols[1], active_cols[0])] = _pos[
                :, (active_cols[0], active_cols[1])
            ]
            pos_candidates.append(_pos)
        elif len(active_cols) <= 1:
            # No axis is considered equivalent ⇒ do nothing (only original is used)
            pass
    return pos_candidates


def invert_and_permute_positions(lattice, positions, spg_number, symprec):
    """Return all position arrays reachable by inverting/swapping
    crystallographically equivalent lattice axes."""
    # Axis lengths (a, b, c) and angles (α, β, γ)
    prop = PropUtil(lattice, positions)
    abc_angle = np.asarray(prop.axis_to_abc, dtype=float)  # (6,)
    abc, angles = abc_angle[:3], abc_angle[3:]
    tol = symprec * 10.0

    angle_similar = np.isclose(angles[:, None], angles[None, :], atol=tol)
    near_90 = np.isclose(angles, 90.0, atol=tol)
    near_90_pair = near_90[:, None] & near_90[None, :]
    length_similar = np.isclose(abc[:, None], abc[None, :], atol=tol)

    same_angle_90_flag = np.any(
        angle_similar & near_90_pair & ~np.eye(3, dtype=bool), axis=1
    )
    same_axis_flag = np.any(
        length_similar & angle_similar[:3, :3] & ~np.eye(3, dtype=bool), axis=1
    )

    # Inverting atomic positions
    if allow_all_invert(spg_number):
        pos_candidates = [positions.copy(), -positions.copy() % 1.0]
    else:
        pos_candidates = [positions.copy()]
    pos_candidates = invert_positions(pos_candidates, same_angle_90_flag)

    # Swapping equivalent axes.
    pos_candidates = swap_positions(pos_candidates, same_axis_flag)

    return pos_candidates
