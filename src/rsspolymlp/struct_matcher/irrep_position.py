import itertools

import numpy as np

from myutils import property_util


class IrrepPos:
    """Identify irreducible atomic positions in a periodic cell.

    Parameters
    ----------
    symprec : float, optional
        Numerical tolerance when comparing fractional coordinates (default: 1e-3).
    """

    def __init__(self, symprec: float = 1e-3):
        """Init method."""
        self.symprec = float(symprec)

    def irrep_positions(self, axis, positions):
        """Return a irreducible representation of atomic positions
        and two recommended `symprec` values.

        Parameters
        ----------
        axis      - lattice vectors as a (3, 3) array, and
        positions - fractional coordinates as an (N, 3) array.

        Returns
        -------
        irrep_position : ndarray
            One-dimensional vector [X_a, X_b, X_c] that uniquely
            identifies the structure up to the tolerance `symprec`.
        recommend_symprecs : list[float]
            Two recommended `symprec` values that is useful as
            starting points for subsequent analyses.
        """

        lattice = np.asarray(axis, dtype=float)
        pos = np.asarray(positions, dtype=float)

        # Trivial case: single‑atom cell → nothing to do
        if pos.shape[0] == 1:
            return [], []

        pos_candidates = self.invert_and_swap_positions(lattice, pos)
        irrep_position = None
        get_recom_symprec = False

        for pos in pos_candidates:
            pos_cls_id, snapped_pos = self.assign_clusters(pos)

            if not get_recom_symprec:
                diffs = self.inter_cluster_diffs(snapped_pos, pos_cls_id)
                recommend_symprecs = self._recommended_symprec(diffs)
                get_recom_symprec = True

            pos_candidates, id_candidates = self.centroid_positions(
                snapped_pos, pos_cls_id
            )

            for i, h, g in itertools.product(
                range(len(pos_candidates[0])),
                range(len(pos_candidates[1])),
                range(len(pos_candidates[2])),
            ):
                pos = np.stack(
                    [pos_candidates[0][i], pos_candidates[1][h], pos_candidates[2][g]],
                    axis=1,
                )
                ids = np.stack(
                    [id_candidates[0][i], id_candidates[1][h], id_candidates[2][g]],
                    axis=1,
                )

                sorted_pos = self._sort_positions(pos, ids)
                flat_pos = sorted_pos.T.reshape(-1)

                if irrep_position is None:
                    irrep_position = flat_pos
                else:
                    irrep_position = self._choose_lex_larger_one(
                        irrep_position, flat_pos
                    )

        return irrep_position, recommend_symprecs

    def invert_and_swap_positions(self, lattice, positions):
        """Return all position arrays reachable by inverting/swapping
        crystallographically equivalent lattice axes."""
        # Axis lengths (a, b, c) and angles (α, β, γ)
        prop = property_util.PropUtil(lattice, positions)
        abc_angle = np.asarray(prop.axis_to_abc, dtype=float)  # (6,)
        abc, angle = abc_angle[:3], abc_angle[3:]

        tol = self.symprec * 10.0

        same_angle = np.isclose(angle[:, None], angle[None, :], atol=tol)
        close_to_90 = np.isclose(angle, 90.0, atol=tol)
        close_to_90_pair = close_to_90[:, None] & close_to_90[None, :]
        same_angle_90 = same_angle & close_to_90_pair
        np.fill_diagonal(same_angle_90, False)
        same_angle_90_flag = np.any(same_angle_90, axis=1)

        same_len = np.isclose(abc[:, None], abc[None, :], atol=tol)
        same_axis = same_len & same_angle
        np.fill_diagonal(same_axis, False)
        same_axis_flag = np.any(same_axis, axis=1)

        inversion_pos = -positions.copy() % 1.0
        pos_candidates = [positions.copy(), inversion_pos]

        # Inverting atomic positions
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
                idx = np.argmax(same_angle_90_flag)
                _pos[:, idx] = (-_pos[:, idx]) % 1.0
                pos_candidates.append(_pos)
            # else: all False ⇒ only the original array

        # Swapping equivalent axes.
        # If all 3 are equivalent: generate all 6 non‑trivial permutations.
        # Otherwise swap just the pair(s) that are equivalent.
        original_candidates = pos_candidates.copy()
        for pos in original_candidates:
            _pos = pos.copy()
            if np.all(same_axis_flag):
                perms = np.array(
                    [[0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]], dtype=int
                )
                for perm in perms:
                    pos_candidates.append(_pos[:, perm])
            elif np.any(same_axis_flag):
                active_cols = np.nonzero(same_axis_flag)[0]
                _pos[:, (active_cols[1], active_cols[0])] = _pos[
                    :, (active_cols[0], active_cols[1])
                ]
                pos_candidates.append(_pos)
            # else: all False ⇒ only the original array

        return pos_candidates

    def assign_clusters(self, positions: np.ndarray):
        positions_copy = positions.copy()
        pos_cls_id = np.zeros_like(positions_copy, dtype=np.int32)

        for axis in range(3):
            coord = positions_copy[:, axis]
            sort_idx = np.argsort(coord, kind="mergesort")
            coord_sorted = coord[sort_idx]

            # Cyclic forward difference (wrap unit cell)
            gap = np.roll(coord_sorted, -1) - coord_sorted
            gap[-1] += 1.0

            # New cluster starts where gap > symprec
            new_cluster = gap > self.symprec
            pos_cls_id_1d = np.empty_like(coord_sorted, dtype=int)
            pos_cls_id_1d[0] = 0
            pos_cls_id_1d[1:] = np.cumsum(new_cluster[:-1])
            if not new_cluster[-1]:  # Merge first & last if wrapped
                coord_sorted[pos_cls_id_1d == pos_cls_id_1d[-1]] -= 1
                pos_cls_id_1d[pos_cls_id_1d == pos_cls_id_1d[-1]] = 0
            pos_cls_id[sort_idx, axis] = pos_cls_id_1d
            positions_copy[sort_idx, axis] = coord_sorted

        return pos_cls_id, positions_copy

    def centroid_positions(self, positions, pos_cls_id):
        n_atoms = positions.shape[0]
        positions_cent = positions - np.mean(positions, axis=0)

        rep_pos_candidates = []
        rep_id_candidates = []

        for axis in range(3):
            pos = positions_cent[:, axis].copy()
            cluster_id = pos_cls_id[:, axis]
            id_max = np.max(cluster_id)

            pos_candidates = [pos]
            id_candidates = [cluster_id]
            max_all = [np.max(pos)]

            for h in range(id_max):
                size = np.sum(cluster_id == h)
                pos = pos - size / n_atoms
                pos[cluster_id == h] += 1
                pos_candidates.append(pos)
                id_candidates.append((cluster_id - h - 1) % (id_max + 1))
                max_all.append(np.max(pos))

            max_val = np.max(max_all)
            max_idx = np.where(np.isclose(max_all, max_val, atol=self.symprec))[0]

            rep_positions = []
            rep_ids = []
            target_idx = None
            for idx in max_idx:
                if target_idx is None:
                    target_idx = idx
                    rep_positions.append(pos_candidates[idx])
                    rep_ids.append(id_candidates[idx])
                    continue

                chosen_idx = self._choose_lex_larger_index(
                    pos_candidates, target_idx, idx
                )
                if chosen_idx is None:
                    rep_positions.append(pos_candidates[idx])
                    rep_ids.append(id_candidates[idx])
                elif chosen_idx == target_idx:
                    pass
                else:
                    rep_positions = [pos_candidates[chosen_idx]]
                    rep_ids = [id_candidates[chosen_idx]]
                    target_idx = idx

            rep_pos_candidates.append(rep_positions)
            rep_id_candidates.append(rep_ids)

        return rep_pos_candidates, rep_id_candidates

    def _choose_lex_larger_index(self, positions_all: np.ndarray, target_idx, idx):
        """Return the lexicographically larger of two 1-D vectors *A* and *B*.
        If the two vectors differ by less than *symprec* in all components,
        they are considered equal and *None* is returned.
        """
        pos_target = np.sort(positions_all[target_idx], kind="mergesort")
        pos_ref = np.sort(positions_all[idx], kind="mergesort")
        diff = pos_target - pos_ref
        non_zero = np.where(np.abs(diff) > self.symprec)[0]
        if non_zero.size:
            return target_idx if diff[non_zero[-1]] > 0 else idx
        return None  # Equivalent within tolerance

    def _choose_lex_larger_one(self, A: np.ndarray, B: np.ndarray):
        diff = A - B
        non_zero = np.where(np.abs(diff) > self.symprec)[0]
        if non_zero.size:
            return A if diff[non_zero[0]] > 0 else B
        return (A + B) / 2  # Equivalent within tolerance

    def _sort_positions(self, positions: np.ndarray, pos_cls_id: np.ndarray):
        # Stable lexicographic sort by (ids_x, ids_y, ids_z)
        sort_idx_final = np.lexsort(
            (pos_cls_id[:, 2], pos_cls_id[:, 1], pos_cls_id[:, 0])
        )
        sorted_positions = positions[sort_idx_final]
        return sorted_positions

    def inter_cluster_diffs(self, positions: np.ndarray, pos_cls_id: np.ndarray):
        positions_copy = positions.copy()

        diffs_all = np.zeros_like(positions_copy, dtype=float)
        for axis in range(3):
            pos_cls_id_1d = pos_cls_id[:, axis]
            pos_1d = positions_copy[:, axis]
            sort_idx = np.argsort(pos_1d, kind="mergesort")
            coord_sorted = pos_1d[sort_idx]
            pos_cls_id_sorted = pos_cls_id_1d[sort_idx]
            cluster_sizes = np.bincount(pos_cls_id_sorted)

            n_cluster = pos_cls_id_sorted[-1] + 1
            if n_cluster == 1:
                continue  # Nothing to do on this axis

            # Weighted average → cluster centre
            cluster_centres = np.bincount(
                pos_cls_id_sorted, weights=coord_sorted, minlength=n_cluster
            )
            cluster_centres = np.divide(
                cluster_centres, cluster_sizes, where=cluster_sizes > 0
            )

            # Distance between consecutive centres (cyclic)
            centre_gap = np.roll(cluster_centres, -1) - cluster_centres
            centre_gap[-1] += 1.0
            diffs_all[sort_idx, axis] = centre_gap[pos_cls_id_sorted]

        return diffs_all

    def _recommended_symprec(self, diffs: np.ndarray):
        """Determine and return two suitable symprec values derived from the distances
        between identified clusters."""
        orders = np.full_like(diffs, fill_value=np.nan, dtype=float)
        nonzero_mask = diffs > 0
        orders[nonzero_mask] = np.log10(diffs[nonzero_mask])

        orders_valid = orders[:, ~np.all(np.isnan(orders), axis=0)]
        max_orders = np.nanmax(orders_valid, axis=0)
        min_order = np.nanmin(max_orders)

        return [10 ** (min_order - 1), 10 ** (min_order - 2)]
