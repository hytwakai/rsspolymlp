from collections import Counter

import numpy as np

from rsspolymlp.analysis.struct_matcher.lattice_redrep import get_reduced_lattice


class PositionRepReducer:
    """Identify the reduced crystal structure representation in a periodic cell."""

    def __init__(
        self,
        symprec: list[float] = [1e-4, 1e-4, 1e-4],
        standardize_axis: bool = False,
        original_axis: bool = False,
        cartesian_coords: bool = True,
    ):
        """Init method."""
        self.symprec = np.array(symprec)
        self.standardize_axis = standardize_axis
        self.original_axis = original_axis
        self.cartesian_coords = cartesian_coords

    def get_reduced_structure_representation(
        self, axis, positions, elements, spg_number
    ):
        """Derive the reduced representation of a crystal structure.

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
        reduced_positions : ndarray
            One-dimensional vector [X_a, X_b, X_c] that uniquely identifies
            the structure up to the tolerance `symprec`.
        sorted_elements : ndarray
            Chemical element symbols ordered starting from the least frequent element
            (alphabetically first in case of ties), corresponding to the order of
            atomic coordinates.
        """

        self.axis = np.asarray(axis, dtype=float)
        self.positions = np.asarray(positions, dtype=float)
        self.elements = np.asarray(elements, dtype=str)

        if self.standardize_axis:
            volume = abs(np.linalg.det(self.axis))
            _axis = self.axis / ((volume / len(elements)) ** (1 / 3))
        else:
            _axis = self.axis

        metric_tensor_half, axis_cands, signed_permutation_cands = (
            get_reduced_lattice(
                _axis,
                spg_number,
                self.symprec,
                self.original_axis,
            )
        )

        # Trivial case: single‑atom cell → nothing to do
        if self.positions.shape[0] == 1:
            return metric_tensor_half, np.array([0, 0, 0]), self.elements

        if self.cartesian_coords:
            a, b, c = np.array(axis_cands[0])
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            norm_c = np.linalg.norm(c)
            self.axis_abc = np.array([norm_a, norm_b, norm_c])

        reduced_positions, sorted_elements = self.get_reduced_positions(
            self.positions,
            self.elements,
            axis_cands,
            signed_permutation_cands,
        )

        return metric_tensor_half, reduced_positions, sorted_elements

    def get_reduced_positions(
        self,
        positions,
        elements,
        axis_cands,
        signed_permutation_cands,
    ):
        """Derive a reduced representation of atomic positions."""
        unique_elements = np.sort(np.unique(elements))
        types = np.array([np.where(unique_elements == el)[0][0] for el in elements])

        counts = Counter(elements)
        min_count = min(counts.values())
        least_elements = [el for el, cnt in counts.items() if cnt == min_count]
        target_element = sorted(least_elements)[0]
        target_type = np.where(unique_elements == target_element)[0][0]
        types = (types - target_type) % (np.max(types) + 1)

        sort_idx = np.argsort(types)
        sorted_elements = elements[sort_idx]
        sorted_types = types[sort_idx]
        positions = positions[sort_idx, :]

        position_cands = self.position_candidates(
            positions, sorted_types, axis_cands, signed_permutation_cands
        )

        reduced_perm_cands = []
        target_vals = []
        for pos_cand in position_cands:
            for target_idx in pos_cand["cands_idx"]:
                _pos = pos_cand["positions"].copy()
                _cls_id = pos_cand["cluster_id"].copy()
                reduced_perm_positions = self.reduced_permutation(
                    target_idx, _pos, sorted_types, _cls_id
                )
                if self.cartesian_coords:
                    reduced_perm_positions = (
                        reduced_perm_positions @ pos_cand["axis"]
                    )
                reduced_perm_cands.append(reduced_perm_positions.T.reshape(-1))

                mask = types == 0
                target_vals.append(
                    reduced_perm_positions[0 : len(mask) - 1, :].T.reshape(-1)
                )

        target_vals = np.array(target_vals)
        reduced_perm_cands = np.array(reduced_perm_cands)
        reduced_positions = self.reduced_translation(
            target_vals,
            reduced_perm_cands,
        )

        return reduced_positions, sorted_elements

    def position_candidates(
        self,
        positions: np.ndarray,
        types: np.ndarray,
        axis_cands: np.ndarray,
        signed_permutation_cands: np.ndarray,
    ):
        _positions = positions.copy()
        cluster_id, positions_by_axis = self.assign_clusters(
            _positions, signed_permutation_cands
        )

        mask = types == 0
        position_cands = []
        for idx, signed in enumerate(signed_permutation_cands):
            _pos = np.zeros_like(_positions)
            _cls_id = np.zeros_like(_positions, dtype=np.int32)
            for axis, val in enumerate(signed):
                target_axis = np.where(val != 0)[0][0]
                sign = val[target_axis]
                if sign == 1:
                    _pos[:, axis] = positions_by_axis[:, target_axis]
                    _cls_id[:, axis] = cluster_id[:, target_axis]
                else:
                    _pos[:, axis] = positions_by_axis[:, target_axis + 3]
                    _cls_id[:, axis] = cluster_id[:, target_axis + 3]
            position_cands.append(
                {
                    "positions": _pos,
                    "cluster_id": _cls_id,
                    "cands_idx": np.where(mask)[0],
                    "axis": axis_cands[idx],
                }
            )
        return position_cands

    def reduced_permutation(
        self,
        target_idx: int,
        positions: np.ndarray,
        types: np.ndarray,
        cluster_id: np.ndarray,
    ):
        pos = positions.copy()
        cls_id = cluster_id.copy()
        id_max = np.max(cls_id, axis=0) + 1

        pos = pos - pos[target_idx]
        pos %= 1.0
        cls_id = np.mod(cls_id - cls_id[target_idx], id_max).astype(int)

        pos = np.delete(pos, target_idx, axis=0)
        cls_id = np.delete(cls_id, target_idx, axis=0).astype(int)
        types = np.delete(types, target_idx, axis=0)

        for ax in range(3):
            near_zero_mask = cls_id[:, ax] == 0
            vals = pos[near_zero_mask, ax]
            dist_to_0 = vals
            dist_to_1 = 1.0 - vals
            pos[near_zero_mask, ax] = np.where(dist_to_0 < dist_to_1, vals, vals - 1.0)

        # Stable lexicographic sort by (ids_z, ids_y, ids_x)
        sort_idx = np.lexsort((cls_id[:, 0], cls_id[:, 1], cls_id[:, 2], types))
        reduced_perm_positions = pos[sort_idx]

        return reduced_perm_positions

    def reduced_translation(
        self,
        target_vals: np.ndarray,
        reduced_perm_cands: np.ndarray,
    ):
        _reduced_perm_cands = reduced_perm_cands
        _target_vals = target_vals

        atom_num = int(_target_vals.shape[1] / 3)
        atom_list = list(range(int(_target_vals.shape[1] / 3)))
        for axis in range(3):
            for atom_idx in atom_list:
                target_idx = atom_idx + axis * atom_num
                sort_idx = np.argsort(-_target_vals[:, target_idx])
                _target_vals = _target_vals[sort_idx, :]
                _reduced_perm_cands = _reduced_perm_cands[sort_idx, :]

                sorted_one_coord = _target_vals[:, target_idx]
                max_coord = sorted_one_coord[0]

                is_near_max = np.where(
                    np.abs(sorted_one_coord - max_coord) <= self.symprec[axis]
                )[0]
                _target_vals = _target_vals[: is_near_max[-1] + 1, :]
                _reduced_perm_cands = _reduced_perm_cands[: is_near_max[-1] + 1, :]
                if _reduced_perm_cands.shape[0] == 1:
                    break

        reduced_perm_cands = _reduced_perm_cands[0, :]
        return reduced_perm_cands

    def assign_clusters(
        self,
        positions: np.ndarray,
        signed_permutation_cands: np.ndarray,
    ):
        """
        Assigns cluster IDs along each axis; atoms at identical positions share the same ID.
        """
        invert_list = [False]
        cluster_id = np.full_like(positions, -1, dtype=np.int32)
        positions_by_axis = np.zeros_like(positions)
        if any(np.any(v == -1) for v in signed_permutation_cands):
            invert_list = [False, True]
            n_rows, n_cols = positions.shape
            cluster_id = np.full((n_rows, n_cols * 2), -1, dtype=np.int32)
            positions_by_axis = np.zeros((n_rows, n_cols * 2), dtype=positions.dtype)

        for invert in invert_list:
            if not invert:
                _positions = positions.copy()
                target_idx = slice(0, 3)
            else:
                _positions = -positions.copy() % 1.0
                target_idx = slice(3, 6)
            positions_by_axis[:, target_idx] = _positions

            sort_idx = np.argsort(_positions, axis=0, kind="mergesort")
            pos_sorted = np.take_along_axis(_positions, sort_idx, axis=0)

            # Compute forward differences with periodic wrapping
            gap = np.roll(pos_sorted, -1, axis=0) - pos_sorted
            gap[-1, :] += 1.0
            if self.cartesian_coords:
                gap = gap * self.axis_abc

            # New cluster starts where gap > symprec
            is_new_cluster = gap > self.symprec
            cluster_id_sorted = np.zeros_like(_positions, dtype=np.int32)
            cluster_id_sorted[1:, :] = np.cumsum(is_new_cluster[:-1, :], axis=0)

            # Merge last cluster if gap is small (periodic condition)
            merge_mask = ~is_new_cluster[-1, :]
            for ax in np.where(merge_mask)[0]:
                max_id = cluster_id_sorted[-1, ax]
                merged = cluster_id_sorted[:, ax] == max_id
                pos_sorted[merged, ax] -= 1.0
                cluster_id_sorted[merged, ax] = 0

            # Restore original order
            cluster_id_sub = np.empty_like(pos_sorted, dtype=np.int32)
            pos_unsort_sub = np.empty_like(pos_sorted)
            for ax in range(3):
                cluster_id_sub[sort_idx[:, ax], ax] = cluster_id_sorted[:, ax]
                pos_unsort_sub[sort_idx[:, ax], ax] = pos_sorted[:, ax]

            cluster_id[:, target_idx] = cluster_id_sub
            positions_by_axis[:, target_idx] = pos_unsort_sub

        return cluster_id, positions_by_axis
