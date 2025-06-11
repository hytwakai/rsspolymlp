from collections import Counter

import numpy as np

from rsspolymlp.analysis.struct_matcher.invert_and_swap import (
    invert_and_swap_positions,
)


class IrrepPosition:
    """Identify irreducible atomic positions in a periodic cell.

    Parameters
    ----------
    symprec : List[float], optional
        Numerical tolerance when comparing fractional coordinates (default: 1e-5).
    """

    def __init__(self, symprec: list[float] = [1e-5, 1e-5, 1e-5]):
        """Init method."""
        self.symprec = np.array(symprec)
        self.invert_values = None
        self.swap_values = None

    def irrep_positions(self, axis, positions, elements, spg_number):
        """Derive a irreducible representation of atomic positions.

        Parameters
        ----------
        axis : (3, 3) array_like
            Lattice vectors defining the unit cell. Each row represents
            a lattice vector (a, b, or c) in Cartesian coordinates. Equivalent to
            np.array([a, b, c]), where each of a, b, and c is a 3-element vector.
        positions : (N, 3) array_like
            Fractional atomic coordinates within the unit cell.
            Each row represents the (x, y, z) coordinate of an atom.
        elements : (N,) array_like
            Chemical element symbols corresponding to each atomic position.

        Returns
        -------
        irrep_position : ndarray
            One-dimensional vector [X_a, X_b, X_c] that uniquely identifies
            the structure up to the tolerance `symprec`.
        sorted_elements : ndarray
            Chemical element symbols sorted in alphabetical order,
            corresponding to the order of fractional atomic coordinates in irrep_position.
        """

        _lattice = np.asarray(axis, dtype=float)
        _positions = np.asarray(positions, dtype=float)
        _elements = np.asarray(elements, dtype=str)

        # Trivial case: single‑atom cell → nothing to do
        if _positions.shape[0] == 1:
            return np.array([0, 0, 0]), _elements

        self.invert_values, self.swap_values = invert_and_swap_positions(
            _lattice, _positions, spg_number, self.symprec
        )

        unique_elements = np.sort(np.unique(_elements))
        types = np.array([np.where(unique_elements == el)[0][0] for el in _elements])

        counts = Counter(elements)
        min_count = min(counts.values())
        least_elements = [el for el, cnt in counts.items() if cnt == min_count]
        target_element = sorted(least_elements)[0]
        target_type = np.where(unique_elements == target_element)[0][0]
        types = (types - target_type) % (np.max(types) + 1)

        red_pos_cands = self.reduced_translation(_positions, types)

        irrep_position = None
        irrep_cls_id = None
        id_max = np.max(red_pos_cands[0]["cluster_id"], axis=0) + 1
        for pos_cand in red_pos_cands:
            for target_idx in pos_cand["cands_idx"]:
                _pos = pos_cand["positions"].copy()
                _cls_id = pos_cand["cluster_id"].copy()
                trans_pos = _pos - _pos[target_idx]
                trans_cls_id = np.mod(_cls_id - _cls_id[target_idx], id_max).astype(int)

                reduced_perm_positions, sorted_cls_id = self.reduced_permutation(
                    trans_pos, types, trans_cls_id
                )

                judge = self._compare_lex_order(irrep_position, reduced_perm_positions)
                if judge == 0:
                    irrep_position = (irrep_position + reduced_perm_positions) / 2
                elif judge == 1:
                    irrep_position = reduced_perm_positions
                    irrep_cls_id = sorted_cls_id

        for swap_val in self.swap_values:
            if np.array_equal(swap_val, [0, 1, 2]):
                continue

            _pos = irrep_position.copy()
            _cls_id = irrep_cls_id.copy()
            _pos[:, [0, 1, 2]] = _pos[:, swap_val]
            _cls_id[:, [0, 1, 2]] = _cls_id[:, swap_val]
            reduced_perm_positions, sorted_cls_id = self.reduced_permutation(
                _pos, types, _cls_id
            )

            judge = self._compare_lex_order(irrep_position, reduced_perm_positions)
            if judge == 0:
                irrep_position = (irrep_position + reduced_perm_positions) / 2
            elif judge == 1:
                irrep_position = reduced_perm_positions
                irrep_cls_id = sorted_cls_id

        irrep_position = irrep_position.T.reshape(-1)
        sort_idx = np.argsort(types)
        sorted_elements = _elements[sort_idx]

        return irrep_position, sorted_elements

    def reduced_translation(
        self,
        positions: np.ndarray,
        types: np.ndarray,
    ):
        _positions = positions.copy()
        cluster_id, snapped_pos = self.assign_clusters(_positions, types)

        red_pos_cands = []

        mask = types == 0
        for invert_val in self.invert_values:
            pos = np.zeros_like(_positions)
            _cluster_id = np.zeros_like(_positions, dtype=np.int32)
            for axis, val in enumerate(invert_val):
                if val == 1:
                    pos[:, axis] = snapped_pos[:, axis]
                    _cluster_id[:, axis] = cluster_id[:, axis]
                else:
                    pos[:, axis] = snapped_pos[:, axis + 3]
                    _cluster_id[:, axis] = cluster_id[:, axis + 3]

            pos_sub = pos[mask]
            cluster_id_sub = _cluster_id[mask]

            if pos_sub.shape[0] == 1:
                red_pos_cands.append(
                    {
                        "positions": pos,
                        "cluster_id": _cluster_id,
                        "cands_idx": np.where(mask)[0][[0]],
                    }
                )
                continue

            distance_cluster = np.zeros_like(pos_sub, dtype=float)
            for ax in range(3):
                cls_id = cluster_id_sub[:, ax]

                id_bins = np.bincount(cls_id)
                with np.errstate(divide="ignore", invalid="ignore"):
                    centres = np.bincount(cls_id, weights=pos_sub[:, ax]) / id_bins

                sort_idx = np.argsort(centres)
                centres = centres[sort_idx]

                valid_mask = np.isfinite(centres)
                centres_valid = centres[valid_mask]
                centre_gap = np.roll(centres, -1) - centres
                centre_gap_valid = np.roll(centres_valid, -1) - centres_valid
                centre_gap_valid[-1] += 1.0

                centre_gap = np.full_like(centres, np.nan)
                centre_gap[valid_mask] = centre_gap_valid
                centre_gap[sort_idx] = centre_gap

                distance_cluster[:, ax] = centre_gap[cls_id]

            sum_vals = np.sum(distance_cluster, axis=1)
            max_val = np.max(sum_vals)
            cands_idx_sub = np.where(
                np.isclose(sum_vals, max_val, atol=np.max(self.symprec))
            )[0]
            cands_idx = np.where(mask)[0][cands_idx_sub]
            red_pos_cands.append(
                {"positions": pos, "cluster_id": _cluster_id, "cands_idx": cands_idx}
            )

        return red_pos_cands

    def reduced_permutation(
        self, positions: np.ndarray, types: np.ndarray, cluster_id: np.ndarray
    ):
        pos = positions.copy()
        cls_id = cluster_id.copy()
        near_zero_mask = np.isclose(np.abs(pos), 0, atol=1e-8)
        pos[near_zero_mask] = 0
        nonzero_mask = ~(cls_id == 0)
        pos[nonzero_mask] %= 1.0

        # Stable lexicographic sort by (ids_x, ids_y, ids_z)
        sort_idx = np.lexsort((cls_id[:, 2], cls_id[:, 1], cls_id[:, 0], types))
        reduced_perm_positions = pos[sort_idx]
        sorted_cls_id = cls_id[sort_idx]

        return reduced_perm_positions, sorted_cls_id

    def assign_clusters(self, positions: np.ndarray, types: np.ndarray):
        """
        Assigns cluster IDs along each axis; atoms at identical positions share the same ID.
        """
        _pos = positions.copy()
        _types = types.copy()

        invert_list = [False]
        if self.invert_values is not None and any(
            np.any(v == -1) for v in self.invert_values
        ):
            invert_list = [False, True]

        cluster_id, snapped_positions = self._assign_clusters_by_type(
            _pos, _types, invert_list
        )
        cluster_id2 = self._relabel_clusters_by_centres(
            snapped_positions, _types, cluster_id
        )
        return cluster_id2, snapped_positions

    def _assign_clusters_by_type(self, positions, types, invert_list=[False]):
        """Assigns cluster IDs by element and axis."""
        if len(invert_list) == 1:
            cluster_id = np.full_like(positions, -1, dtype=np.int32)
            snapped_positions = np.zeros_like(positions)
        else:
            n_rows, n_cols = positions.shape
            cluster_id = np.full((n_rows, n_cols * 2), -1, dtype=np.int32)
            snapped_positions = np.zeros((n_rows, n_cols * 2), dtype=positions.dtype)

        for invert in invert_list:
            if not invert:
                _positions = positions.copy()
                target_idx = slice(0, 3)
            else:
                _positions = -positions.copy() % 1.0
                target_idx = slice(3, 6)

            start_id = np.zeros((3))

            # Group atoms by element type
            for type_n in range(np.max(types) + 1):
                mask = types == type_n
                pos_sub = _positions[mask]
                idx_sub = np.where(mask)[0]

                sort_idx = np.argsort(pos_sub, axis=0, kind="mergesort")
                coord_sorted = np.take_along_axis(pos_sub, sort_idx, axis=0)

                # Compute forward differences with periodic wrapping
                gap = np.roll(coord_sorted, -1, axis=0) - coord_sorted
                gap[-1, :] += 1.0

                # New cluster starts where gap > symprec
                is_new_cluster = gap > self.symprec
                cluster_id_sorted = np.empty_like(coord_sorted, dtype=np.int32)
                cluster_id_sorted[0, :] = start_id
                cluster_id_sorted[1:, :] = (
                    np.cumsum(is_new_cluster[:-1, :], axis=0) + start_id
                )

                # Merge last cluster if gap is small (periodic condition)
                merge_mask = ~is_new_cluster[-1, :]
                for ax in np.where(merge_mask)[0]:
                    max_id = cluster_id_sorted[-1, ax]
                    merged = cluster_id_sorted[:, ax] == max_id
                    coord_sorted[merged, ax] -= 1.0
                    cluster_id_sorted[merged, ax] = start_id[ax]

                # Restore original order
                cluster_id_sub = np.empty_like(coord_sorted, dtype=np.int32)
                coord_unsort_sub = np.empty_like(coord_sorted)
                for ax in range(3):
                    cluster_id_sub[sort_idx[:, ax], ax] = cluster_id_sorted[:, ax]
                    coord_unsort_sub[sort_idx[:, ax], ax] = coord_sorted[:, ax]

                cluster_id[idx_sub, target_idx] = cluster_id_sub
                snapped_positions[idx_sub, target_idx] = coord_unsort_sub
                start_id = np.max(cluster_id_sub, axis=0) + 1

        return cluster_id, snapped_positions

    def _relabel_clusters_by_centres(self, positions, types, cluster_id):
        """
        Relabels cluster IDs so that cluster centers are ordered in ascending position.
        Different element types within the same center are assigned separate IDs.
        """
        cluster_id2 = np.full_like(positions, -1, dtype=np.int32)

        for ax in range(positions.shape[1]):
            cls_id = cluster_id[:, ax]
            coord = positions[:, ax]

            # The index of `centres` corresponds directly to the cluster ID
            centres = np.bincount(cls_id, weights=coord) / np.bincount(cls_id)

            # Assign a element type to each cluster
            _, unique_idx = np.unique(cls_id, return_index=True)
            cluster_types = types[unique_idx]

            # Create cluster IDs based on centre positions only (ignoring types)
            sort_idx = np.argsort(centres)
            centres_sorted = centres[sort_idx]
            gap = np.roll(centres_sorted, -1) - centres_sorted
            gap[-1] += 1.0
            is_new_cluster = gap > self.symprec[ax % 3]
            centre_cls_id = np.zeros_like(centres_sorted, dtype=np.int32)
            centre_cls_id[1:] = np.cumsum(is_new_cluster[:-1])
            if not is_new_cluster[-1]:
                centre_cls_id[centre_cls_id == centre_cls_id[-1]] = 0

            # Map cluster center IDs back to their atomic order
            centre_cls_id_origin = np.empty_like(centre_cls_id)
            centre_cls_id_origin[sort_idx] = centre_cls_id

            # Reassign new cluster IDs to each atom based on reordered clusters:
            # primary key = center ID, secondary key = element type
            reorder_cluster_ids = np.lexsort((cluster_types, centre_cls_id_origin))
            for new_id, old_id in enumerate(reorder_cluster_ids):
                cluster_id2[cls_id == old_id, ax] = new_id

        return cluster_id2

    def _compare_lex_order(self, A: np.ndarray, B: np.ndarray):
        """
        Compare two 1D vectors A and B lexicographically with tolerance `symprec`.

        Returns:
            -1 if A < B,
            1 if A > B,
            0 if A ≈ B within tolerance
        """
        if A is None:
            return 1

        A_flat = A.T.reshape(-1)
        B_flat = B.T.reshape(-1)
        diff = A_flat - B_flat
        diff_abs = np.abs(diff)
        length = diff.shape[0] // 3

        symprec_array = np.concatenate(
            [
                np.full(length, self.symprec[0]),
                np.full(length, self.symprec[1]),
                np.full(length, self.symprec[2]),
            ]
        )
        non_zero = np.where(diff_abs > symprec_array)[0]
        if not non_zero.size:
            return 0
        return -1 if diff[non_zero[0]] < 0 else 1
