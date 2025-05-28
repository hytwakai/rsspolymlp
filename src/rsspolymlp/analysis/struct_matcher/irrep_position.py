import itertools

import numpy as np

from rsspolymlp.analysis.struct_matcher.invert_and_permute import (
    invert_and_permute_positions,
)


class IrrepPosition:
    """Identify irreducible atomic positions in a periodic cell.

    Parameters
    ----------
    symprec : float, optional
        Numerical tolerance when comparing fractional coordinates (default: 1e-3).
    """

    def __init__(self, symprec: float = 1e-3, original_element_order: bool = False):
        """Init method."""
        self.symprec = float(symprec)
        self.original_element_order = original_element_order
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
            Chemical element symbols sorted in the same order as used
            in the irreducible representation.
        """

        _lattice = np.asarray(axis, dtype=float)
        _positions = np.asarray(positions, dtype=float)
        _elements = np.asarray(elements, dtype=str)

        # Trivial case: single‑atom cell → nothing to do
        if _positions.shape[0] == 1:
            return np.array([0, 0, 0]), _elements

        if self.original_element_order:
            # Preserve the order of elements as they appear in the input
            _, idx = np.unique(_elements, return_index=True)
            unique_ordered = _elements[np.sort(idx)]
            types = np.array([np.where(unique_ordered == el)[0][0] for el in _elements])
        else:
            # Assign types based on lexicographic (alphabetical) order of elements
            unique_sorted = np.sort(np.unique(_elements))
            types = np.array([np.where(unique_sorted == el)[0][0] for el in _elements])

        self.invert_values, self.swap_values = invert_and_permute_positions(
            _lattice, _positions, spg_number, self.symprec
        )

        pos_cands_all, id_cands_all = self.centroid_positions(_positions, types)

        irrep_position = None
        for n, pos_cands in enumerate(pos_cands_all):
            for i, h, g in itertools.product(
                range(len(pos_cands[0])),
                range(len(pos_cands[1])),
                range(len(pos_cands[2])),
            ):
                pos2 = np.stack(
                    [
                        pos_cands[0][i],
                        pos_cands[1][h],
                        pos_cands[2][g],
                    ],
                    axis=1,
                )
                ids = np.stack(
                    [
                        id_cands_all[n][0][i],
                        id_cands_all[n][1][h],
                        id_cands_all[n][2][g],
                    ],
                    axis=1,
                )

                sorted_pos = self._sort_positions(pos2, types, ids)
                irrep_position = self._choose_lex_smaller_one(
                    irrep_position, sorted_pos
                )

        for swap_val in self.swap_values:
            _pos = irrep_position.copy()
            if np.array_equal(swap_val, [0, 1, 2]):
                continue
            _pos[:, [0, 1, 2]] = _pos[:, swap_val]
            irrep_position = self._choose_lex_smaller_one(irrep_position, _pos)

        irrep_position = irrep_position.T.reshape(-1)
        sort_idx = np.argsort(types)
        sorted_elements = _elements[sort_idx]

        return irrep_position, sorted_elements

    def centroid_positions(self, positions, types):
        _positions = positions.copy()
        pos_cls_id, snapped_pos = self.assign_clusters(_positions, types)

        pos_cands_all = []
        id_cands_all = []
        for invert_val in self.invert_values:
            pos = np.zeros_like(_positions)
            _pos_cls_id = np.zeros_like(_positions, dtype=np.int32)
            for axis, val in enumerate(invert_val):
                if val == 1:
                    pos[:, axis] = snapped_pos[:, axis]
                    _pos_cls_id[:, axis] = pos_cls_id[:, axis]
                else:
                    pos[:, axis] = snapped_pos[:, axis + 3]
                    _pos_cls_id[:, axis] = pos_cls_id[:, axis + 3]

            positions_cent = pos - np.mean(pos, axis=0)

            pos_cands = []
            id_cands = []
            for axis in range(3):
                pos = positions_cent[:, axis].copy()
                cluster_id = _pos_cls_id[:, axis]
                id_max = np.max(cluster_id)

                pos_cands_axis = [pos]
                id_cands_axis = [cluster_id]
                max_all = [np.max(pos)]
                for target_id in range(id_max):
                    size = np.sum(cluster_id == target_id)
                    pos = pos - size / positions_cent.shape[0]
                    pos[cluster_id == target_id] += 1
                    pos_cands_axis.append(pos)
                    id_cands_axis.append((cluster_id - target_id - 1) % (id_max + 1))
                    max_all.append(np.max(pos))

                max_val = np.max(max_all)
                cands_idx = np.where(np.isclose(max_all, max_val, atol=self.symprec))[0]
                pos_2d = np.array(pos_cands_axis)[cands_idx]
                id_2d = np.array(id_cands_axis)[cands_idx]
                pos_cands.append(pos_2d)
                id_cands.append(id_2d)

            pos_cands_all.append(pos_cands)
            id_cands_all.append(id_cands)
        # pos_cands_all: List[List[np.ndarray]]
        #     - Outer list: M transformation patterns (original + inverted + swapped)
        #     - Inner list: 3 elements (per axis)
        #     - Each np.ndarray: shape = (cands_idx_i, N_atom), where i = 0 (x), 1 (y), 2 (z)

        all_max_vals = []
        for i, _pos_cands in enumerate(pos_cands_all):
            _all_max_val = 0
            for axis in range(3):
                max_vals = np.max(_pos_cands[axis], axis=1)
                max_max_val = np.max(max_vals)
                _all_max_val += max_max_val
            all_max_vals.append(_all_max_val)

        all_max_vals = np.array(all_max_vals)
        max_all_max_vals = np.max(all_max_vals)
        reduced_idx = np.where(
            np.isclose(all_max_vals, max_all_max_vals, atol=self.symprec)
        )[0]

        reduced_pos_cands = [pos_cands_all[i] for i in reduced_idx]
        reduced_id_cands = [id_cands_all[i] for i in reduced_idx]

        return reduced_pos_cands, reduced_id_cands

    def assign_clusters(self, positions: np.ndarray, types: np.ndarray):
        """
        Assigns cluster IDs along each axis based on element type; same-type atoms at identical
        positions share the same ID. The IDs are then relabeled by cluster center and element type
        to prepare for subsequent centroid calculations.
        """
        _pos = positions.copy()
        _types = types.copy()

        invert_list = [False]
        if self.invert_values is not None and any(
            np.any(v == -1) for v in self.invert_values
        ):
            invert_list = [False, True]

        pos_cls_id, snapped_positions = self._assign_clusters_by_type(
            _pos, _types, invert_list
        )
        pos_cls_id2 = self._relabel_clusters_by_centres(
            snapped_positions, _types, pos_cls_id
        )

        return pos_cls_id2, snapped_positions

    def _assign_clusters_by_type(self, positions, types, invert_list=[False]):
        """Assigns cluster IDs by element and axis."""
        if len(invert_list) == 1:
            pos_cls_id = np.full_like(positions, -1, dtype=np.int32)
            snapped_positions = np.zeros_like(positions)
        else:
            n_rows, n_cols = positions.shape
            pos_cls_id = np.full((n_rows, n_cols * 2), -1, dtype=np.int32)
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
                pos_cls_id_sorted = np.empty_like(coord_sorted, dtype=np.int32)
                pos_cls_id_sorted[0, :] = start_id
                pos_cls_id_sorted[1:, :] = (
                    np.cumsum(is_new_cluster[:-1, :], axis=0) + start_id
                )

                # Merge last cluster if gap is small (periodic condition)
                merge_mask = ~is_new_cluster[-1, :]
                for ax in np.where(merge_mask)[0]:
                    max_id = pos_cls_id_sorted[-1, ax]
                    merged = pos_cls_id_sorted[:, ax] == max_id
                    coord_sorted[merged, ax] -= 1.0
                    pos_cls_id_sorted[merged, ax] = start_id[ax]

                # Restore original order
                pos_cls_id_sub = np.empty_like(coord_sorted, dtype=np.int32)
                coord_unsort_sub = np.empty_like(coord_sorted)
                for ax in range(3):
                    pos_cls_id_sub[sort_idx[:, ax], ax] = pos_cls_id_sorted[:, ax]
                    coord_unsort_sub[sort_idx[:, ax], ax] = coord_sorted[:, ax]

                pos_cls_id[idx_sub, target_idx] = pos_cls_id_sub
                snapped_positions[idx_sub, target_idx] = coord_unsort_sub
                start_id = np.max(pos_cls_id_sub, axis=0) + 1

        return pos_cls_id, snapped_positions

    def _relabel_clusters_by_centres(self, positions, types, pos_cls_id):
        """
        Relabels cluster IDs so that cluster centers are ordered in ascending position.
        Different element types within the same center are assigned separate IDs.
        """
        pos_cls_id2 = np.full_like(positions, -1, dtype=np.int32)

        for ax in range(positions.shape[1]):
            cls_id = pos_cls_id[:, ax]
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
            is_new_cluster = gap > self.symprec
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
                pos_cls_id2[cls_id == old_id, ax] = new_id

        return pos_cls_id2

    def _sort_positions(
        self, positions: np.ndarray, types: np.ndarray, pos_cls_id: np.ndarray
    ):
        # Stable lexicographic sort by (ids_x, ids_y, ids_z)
        sort_idx = np.lexsort(
            (pos_cls_id[:, 2], pos_cls_id[:, 1], pos_cls_id[:, 0], types)
        )
        sorted_positions = positions[sort_idx]
        return sorted_positions

    def _choose_lex_smaller_one(self, A: np.ndarray, B: np.ndarray):
        if A is None:
            return B
        A_flat = A.T.reshape(-1)
        B_flat = B.T.reshape(-1)
        result = self._compare_lex_order(A_flat, B_flat)
        if result == 0:
            return (A + B) / 2
        return A if result == -1 else B

    def _compare_lex_order(self, A: np.ndarray, B: np.ndarray):
        """
        Compare two 1D vectors A and B lexicographically with tolerance `symprec`.

        Returns:
            -1 if A < B,
            1 if A > B,
            0 if A ≈ B within tolerance
        """
        diff = A - B
        non_zero = np.where(np.abs(diff) > self.symprec)[0]
        if not non_zero.size:
            return 0
        return -1 if diff[non_zero[0]] < 0 else 1
