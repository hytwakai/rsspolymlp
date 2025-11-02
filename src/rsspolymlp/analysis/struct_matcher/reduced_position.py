from collections import Counter

import numpy as np

from rsspolymlp.analysis.struct_matcher.chiral_spg import is_chiral
from rsspolymlp.analysis.struct_matcher.invert_and_swap import (
    metric_tensor_transform,
    signed_permutation_matrices,
)
from rsspolymlp.common.property import PropUtil, get_metric_tensor


class StructRepReducer:
    """Identify the reduced crystal structure representation in a periodic cell.

    Parameters
    ----------
    symprec : List[float], optional
        Numerical tolerance when comparing fractional coordinates (default: 1e-5).
    """

    def __init__(
        self,
        symprec: list[float] = [1e-4, 1e-4, 1e-4],
        standardize_axis: bool = False,
        cartesian_coords: bool = True,
    ):
        """Init method."""
        self.symprec = np.array(symprec)
        self.standardize_axis = standardize_axis
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
            Chemical element symbols sorted in alphabetical order,
            corresponding to the order of fractional atomic coordinates in reduced_positions.
        """

        self.axis = np.asarray(axis, dtype=float)
        self.positions = np.asarray(positions, dtype=float)
        self.elements = np.asarray(elements, dtype=str)

        if self.standardize_axis:
            volume = abs(np.linalg.det(self.axis))
            _axis = self.axis / (volume ** (1 / 3))
        else:
            _axis = self.axis

        prop = PropUtil(_axis, self.positions)
        self.abc_angle = np.asarray(prop.abc, dtype=float)
        metric_tensor = get_metric_tensor(self.abc_angle)

        reduced_axis, signed_permutation_cands = self.get_reduced_axis(
            metric_tensor,
            spg_number,
        )

        # Trivial case: single‑atom cell → nothing to do
        if self.positions.shape[0] == 1:
            return reduced_axis, np.array([0, 0, 0]), self.elements

        reduced_positions, sorted_elements = self.get_reduced_positions(
            self.positions,
            self.elements,
            signed_permutation_cands,
        )

        return metric_tensor, reduced_positions, sorted_elements

    def get_reduced_axis(self, metric_tensor, spg_number):
        proper_matrices, improper_matrices = signed_permutation_matrices()
        proper_G_transform, improper_G_transform = metric_tensor_transform()
        if not is_chiral(spg_number):
            all_signed_permutation_matrices = np.concatenate(
                [proper_matrices, improper_matrices], axis=0
            )
            all_G_transform = np.concatenate(
                [proper_G_transform, improper_G_transform], axis=0
            )
        else:
            all_signed_permutation_matrices = proper_matrices
            all_G_transform = proper_G_transform

        all_metric_tensor = []
        for P in all_G_transform:
            trans_metric_tensor = P @ metric_tensor
            all_metric_tensor.append(trans_metric_tensor)
        all_metric_tensor = np.array(all_metric_tensor)

        reduced_axis_cands = all_metric_tensor
        signed_permutation_cands = all_signed_permutation_matrices
        for idx in range(6):
            max_metric = np.min(reduced_axis_cands[:, idx])
            is_near_max = np.where(
                np.abs(reduced_axis_cands[:, idx] - max_metric) <= self.symprec[idx % 3]
            )[0]

            reduced_axis_cands = reduced_axis_cands[is_near_max, :]
            signed_permutation_cands = signed_permutation_cands[is_near_max, :]
            if reduced_axis_cands.shape[0] == 1:
                break

        reduced_axis = reduced_axis_cands[0]
        return reduced_axis, signed_permutation_cands

    def get_reduced_positions(self, positions, elements, signed_permutation_cands):
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
            positions, sorted_types, signed_permutation_cands
        )

        reduced_perm_cands = []
        for pos_cand in position_cands:
            for target_idx in pos_cand["cands_idx"]:
                _pos = pos_cand["positions"].copy()
                _cls_id = pos_cand["cluster_id"].copy()
                reduced_perm_positions = self.reduced_permutation(
                    target_idx, _pos, sorted_types, _cls_id
                )
                if self.cartesian_coords:
                    reduced_perm_positions = (
                        reduced_perm_positions * self.abc_angle[0:3]
                    )
                reduced_perm_cands.append(reduced_perm_positions[1:, :].T.reshape(-1))

        reduced_perm_cands = np.array(reduced_perm_cands)
        reduced_positions = self.reduced_translation(reduced_perm_cands)

        return reduced_positions, sorted_elements

    def position_candidates(
        self,
        positions: np.ndarray,
        types: np.ndarray,
        signed_permutation_cands: np.ndarray,
    ):
        _positions = positions.copy()
        cluster_id, snapped_pos = self.assign_clusters(
            _positions, types, signed_permutation_cands
        )

        position_cands = []

        mask = types == 0
        for cand in signed_permutation_cands:
            _pos = np.zeros_like(_positions)
            _cls_id = np.zeros_like(_positions, dtype=np.int32)
            for axis, val in enumerate(cand):
                target_axis = np.where(val != 0)[0][0]
                sign = val[target_axis]
                if sign == 1:
                    _pos[:, axis] = snapped_pos[:, target_axis]
                    _cls_id[:, axis] = cluster_id[:, target_axis]
                else:
                    _pos[:, axis] = snapped_pos[:, target_axis + 3]
                    _cls_id[:, axis] = cluster_id[:, target_axis + 3]
            position_cands.append(
                {
                    "positions": _pos,
                    "cluster_id": _cls_id,
                    "cands_idx": np.where(mask)[0],
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
        cls_id = np.mod(cls_id - cls_id[target_idx], id_max).astype(int)
        for ax in range(3):
            pos[:, ax] %= 1.0

            near_zero_mask = cls_id[:, ax] == 0
            vals = pos[near_zero_mask, ax]
            dist_to_0 = vals
            dist_to_1 = 1.0 - vals
            pos[near_zero_mask, ax] = np.where(dist_to_0 < dist_to_1, vals, vals - 1.0)

        # Stable lexicographic sort by (ids_x, ids_y, ids_z)
        sort_idx = np.lexsort((cls_id[:, 2], cls_id[:, 1], cls_id[:, 0], types))
        reduced_perm_positions = pos[sort_idx]

        return reduced_perm_positions

    def reduced_translation(self, reduced_perm_cands: np.ndarray):
        atom_num3 = reduced_perm_cands.shape[1]
        for idx in range(atom_num3):
            axis_idx = idx // atom_num3
            sort_idx = np.argsort(-reduced_perm_cands[:, idx])
            _reduced_perm_cands = reduced_perm_cands[sort_idx, :]
            sorted_one_coord = reduced_perm_cands[sort_idx, idx]
            max_coord = sorted_one_coord[0]

            is_near_max = np.where(
                np.abs(sorted_one_coord - max_coord) <= self.symprec[axis_idx]
            )[0]
            reduced_perm_cands = _reduced_perm_cands[: is_near_max[-1] + 1, :]
            if reduced_perm_cands.shape[0] == 1:
                break

        reduced_perm_cands = reduced_perm_cands[0, :]
        return reduced_perm_cands

    def assign_clusters(
        self,
        positions: np.ndarray,
        types: np.ndarray,
        signed_permutation_cands: np.ndarray,
    ):
        """
        Assigns cluster IDs along each axis; atoms at identical positions share the same ID.
        """
        _pos = positions.copy()
        _types = types.copy()

        invert_list = [False]
        if any(np.any(v == -1) for v in signed_permutation_cands):
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
                if self.cartesian_coords:
                    gap = gap * self.abc_angle[0:3]

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
