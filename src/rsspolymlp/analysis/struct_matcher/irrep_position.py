import itertools

import numpy as np

from rsspolymlp.analysis.struct_matcher.invert_and_permute import (
    invert_and_permute_positions,
)


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

    def irrep_positions(self, axis, positions, elements, spg_number):
        """Return a irreducible representation of atomic positions
        and two recommended `symprec` values.

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

        _, idx = np.unique(_elements, return_index=True)
        unique_ordered = _elements[np.sort(idx)]
        types = np.array([np.where(unique_ordered == el)[0][0] for el in _elements])

        pos_cands1 = invert_and_permute_positions(
            _lattice, _positions, spg_number, self.symprec
        )

        irrep_position = None
        for pos1 in pos_cands1:
            pos_cls_id, snapped_pos = self.assign_clusters(pos1, types)

            pos_cands2, id_cands = self.centroid_positions(
                snapped_pos, types, pos_cls_id
            )

            for i, h, g in itertools.product(
                range(len(pos_cands2[0])),
                range(len(pos_cands2[1])),
                range(len(pos_cands2[2])),
            ):
                pos2 = np.stack(
                    [
                        pos_cands2[0][i],
                        pos_cands2[1][h],
                        pos_cands2[2][g],
                    ],
                    axis=1,
                )
                ids = np.stack(
                    [id_cands[0][i], id_cands[1][h], id_cands[2][g]],
                    axis=1,
                )

                sorted_pos = self._sort_positions(pos2, types, ids)
                flat_pos = sorted_pos.T.reshape(-1)

                irrep_position = self._choose_lex_smaller_one(irrep_position, flat_pos)

        sort_idx = np.argsort(types)
        sorted_elements = _elements[sort_idx]

        return irrep_position, sorted_elements

    def assign_clusters(self, positions: np.ndarray, types: np.ndarray):
        _pos = positions.copy()
        _types = types.copy()

        pos_cls_id, coord_unsort = self._assign_clusters_by_type(_pos, _types)
        pos_cls_id2 = self._relabel_clusters_by_centres(
            coord_unsort, _types, pos_cls_id
        )

        return pos_cls_id2, coord_unsort

    def _assign_clusters_by_type(self, positions, types):
        pos_cls_id = np.full_like(positions, -1, dtype=np.int32)
        coord_unsort = np.zeros_like(positions)
        start_id = np.zeros((3))

        for type_n in range(np.max(types) + 1):
            mask = types == type_n
            pos_sub = positions[mask]
            idx_sub = np.where(mask)[0]

            sort_idx = np.argsort(pos_sub, axis=0, kind="mergesort")
            coord_sorted = np.take_along_axis(pos_sub, sort_idx, axis=0)

            # Cyclic forward difference (wrap unit cell)
            gap = np.roll(coord_sorted, -1, axis=0) - coord_sorted
            gap[-1, :] += 1.0

            # New cluster starts where gap > symprec
            is_new_cluster = gap > self.symprec
            pos_cls_id_sorted = np.empty_like(coord_sorted, dtype=np.int32)
            pos_cls_id_sorted[0, :] = start_id
            pos_cls_id_sorted[1:, :] = (
                np.cumsum(is_new_cluster[:-1, :], axis=0) + start_id
            )

            merge_mask = ~is_new_cluster[-1, :]
            for ax in np.where(merge_mask)[0]:
                max_id = pos_cls_id_sorted[-1, ax]
                merged = pos_cls_id_sorted[:, ax] == max_id
                coord_sorted[merged, ax] -= 1.0
                pos_cls_id_sorted[merged, ax] = start_id[ax]

            pos_cls_id_sub = np.empty_like(coord_sorted, dtype=np.int32)
            coord_unsort_sub = np.empty_like(coord_sorted)
            for ax in range(3):
                pos_cls_id_sub[sort_idx[:, ax], ax] = pos_cls_id_sorted[:, ax]
                coord_unsort_sub[sort_idx[:, ax], ax] = coord_sorted[:, ax]

            pos_cls_id[idx_sub, :] = pos_cls_id_sub
            coord_unsort[idx_sub, :] = coord_unsort_sub
            start_id = np.max(pos_cls_id_sub, axis=0) + 1

        return pos_cls_id, coord_unsort

    def _relabel_clusters_by_centres(self, positions, types, pos_cls_id):
        pos_cls_id2 = np.full_like(positions, -1, dtype=np.int32)

        for ax in range(3):
            cls_id = pos_cls_id[:, ax]
            coord = positions[:, ax]

            # The index of `centres` corresponds directly to the cluster ID
            centres = np.bincount(cls_id, weights=coord) / np.bincount(cls_id)

            # Select a representative type for each cluster (first occurrence)
            _, unique_idx = np.unique(cls_id, return_index=True)
            cluster_types = types[unique_idx]

            sort_idx = np.argsort(centres)
            centres_sorted = centres[sort_idx]
            gap = np.roll(centres_sorted, -1) - centres_sorted
            gap[-1] += 1.0
            is_new_cluster = gap > self.symprec
            centre_cls_id = np.zeros_like(centres_sorted, dtype=np.int32)
            centre_cls_id[1:] = np.cumsum(is_new_cluster[:-1])
            if not is_new_cluster[-1]:
                centre_cls_id[centre_cls_id == centre_cls_id[-1]] = 0

            # Map cluster center IDs back to their original order.
            centre_cls_id_origin = np.empty_like(centre_cls_id)
            centre_cls_id_origin[sort_idx] = centre_cls_id

            # Reassign new cluster IDs to each atom based on reordered clusters:
            # primary key = center ID, secondary key = element type
            reorder_cluster_ids = np.lexsort((cluster_types, centre_cls_id_origin))
            for new_id, old_id in enumerate(reorder_cluster_ids):
                pos_cls_id2[cls_id == old_id, ax] = new_id
        return pos_cls_id2

    def centroid_positions(self, positions, types, pos_cls_id):
        rep_pos_cands = []
        rep_id_cands = []
        positions_cent = positions - np.mean(positions, axis=0)

        for axis in range(3):
            pos = positions_cent[:, axis].copy()
            cluster_id = pos_cls_id[:, axis]
            id_max = np.max(cluster_id)

            pos_cands = [pos]
            id_cands = [cluster_id]
            max_all = [np.max(pos)]
            for target_id in range(id_max):
                size = np.sum(cluster_id == target_id)
                pos = pos - size / positions_cent.shape[0]
                pos[cluster_id == target_id] += 1
                pos_cands.append(pos)
                id_cands.append((cluster_id - target_id - 1) % (id_max + 1))
                max_all.append(np.max(pos))

            max_val = np.max(max_all)
            cands_idx = np.where(np.isclose(max_all, max_val, atol=self.symprec))[0]

            rep_positions = []
            rep_ids = []
            target_idx = None
            for idx in cands_idx:
                if target_idx is None:
                    target_idx = idx
                    rep_positions.append(pos_cands[idx])
                    rep_ids.append(id_cands[idx])
                    continue

                chosen_idx = self._choose_lex_smaller_index(
                    pos_cands, types, target_idx, idx
                )

                if chosen_idx is None:
                    rep_positions.append(pos_cands[idx])
                    rep_ids.append(id_cands[idx])
                elif chosen_idx == target_idx:
                    continue
                else:
                    rep_positions = [pos_cands[chosen_idx]]
                    rep_ids = [id_cands[chosen_idx]]
                    target_idx = idx

            rep_pos_cands.append(rep_positions)
            rep_id_cands.append(rep_ids)

        return rep_pos_cands, rep_id_cands

    def _sort_positions(
        self, positions: np.ndarray, types: np.ndarray, pos_cls_id: np.ndarray
    ):
        # Stable lexicographic sort by (ids_x, ids_y, ids_z)
        sort_idx = np.lexsort(
            (pos_cls_id[:, 2], pos_cls_id[:, 1], pos_cls_id[:, 0], types)
        )
        sorted_positions = positions[sort_idx]
        return sorted_positions

    def _choose_lex_smaller_index(self, positions_all: np.ndarray, types, idx1, idx2):
        sort_idx1 = np.lexsort((positions_all[idx1], types))
        A = positions_all[idx1][sort_idx1]
        sort_idx2 = np.lexsort((positions_all[idx2], types))
        B = positions_all[idx2][sort_idx2]
        result = self._compare_lex_order(A, B)
        if result == 0:
            return None
        return idx1 if result == -1 else idx2

    def _choose_lex_smaller_one(self, A: np.ndarray, B: np.ndarray):
        if A is None:
            return B
        result = self._compare_lex_order(A, B)
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
