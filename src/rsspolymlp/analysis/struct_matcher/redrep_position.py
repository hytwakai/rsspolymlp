import time
from collections import Counter

import numpy as np

from rsspolymlp.analysis.struct_matcher.redrep_lattice import get_reduced_lattice


class PositionRepReducer:
    """Identify the reduced crystal structure representation."""

    def __init__(
        self,
        axis: np.ndarray,
        positions: np.ndarray,
        elements: np.ndarray,
        spg_number: int,
        symprec: list[float] = [1e-4, 1e-4, 1e-4],
        normalize_axis: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize PositionRepReducer class.

        Parameters
        ----------
        axis : (3, 3) array_like
            Lattice vectors defining the unit cell in Cartesian coordinates.
            Each row represents one lattice vector (a, b, or c).
            Equivalent to np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]).
        positions : (N, 3) array_like
            Fractional atomic coordinates within the unit cell.
            Each row represents the coordinates of one atom.
        elements : (N,) array_like
            Chemical element symbols corresponding to the atomic positions.
        spg_number : int
            Space group number of the structure.
        symprec : list of float, default=[1e-4, 1e-4, 1e-4]
            Tolerances used for structural matching along the three lattice
            directions.
        normalize_axis : bool, default=False
            If True, scale the lattice vectors so that the unit-cell volume
            per atom is equal to one.
        """

        self.axis = np.asarray(axis, dtype=float)
        self.positions = np.asarray(positions, dtype=float)
        self.elements = np.asarray(elements, dtype=str)
        self.spg_number = spg_number
        self.symprec = np.array(symprec)
        self.normalize_axis = normalize_axis
        if self.normalize_axis:
            volume = abs(np.linalg.det(self.axis))
            self.axis = self.axis / ((volume / len(self.elements)) ** (1 / 3))
        self.verbose = verbose
        self.axis_abc = None

    def get_reduced_structure_representation(self) -> dict:
        """Derive the reduced representation of a crystal structure.

        Returns
        -------
        result : dict
            Dictionary containing the following entries:
            - metric_tensor : ndarray, shape (6,)
                Independent components of the matrix square root of the
                metric tensor, ordered as [aa, bb, cc, ab, ac, bc].
            - axis : ndarray
                Lattice vectors.
            - axis_abc : ndarray, shape (3,)
                Lengths of the three lattice vectors.
            - reduced_positions : ndarray
                One-dimensional vector [X_a, X_b, X_c] that uniquely identifies
                the structure up to the tolerance `symprec`.
            - all_positions : list[dict]
                List of dictionaries, each containing:
                - perm_mat : ndarray, shape (3, 3)
                    Permutation matrix.
                - pos_reps : list of ndarray
                    List of position representations.
            - elements : ndarray
                Chemical element symbols ordered starting from the least frequent element
                (alphabetically first in case of ties), corresponding to the order of
                atomic coordinates.
        """

        t_lattice = time.time()
        metric_tensor_sqrt, axis_cands, signed_permutation_cands = get_reduced_lattice(
            self.axis,
            self.spg_number,
            self.symprec,
        )
        if self.verbose:
            print("[struct_matcher.redrep_position]")
            print(f"  Lattice: {round((time.time() - t_lattice) * 1000, 2)} ms")

        self.axis = axis_cands[0]
        a, b, c = np.array(self.axis)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        norm_c = np.linalg.norm(c)
        self.axis_abc = np.array([norm_a, norm_b, norm_c])

        # Trivial case: single‑atom cell → nothing to do
        if self.positions.shape[0] == 1:
            reduced_positions = []
            all_positions = []
            sorted_elements = self.elements
        else:
            reduced_positions, all_positions, sorted_elements = (
                self.get_reduced_positions(
                    signed_permutation_cands,
                )
            )

        return {
            "metric_tensor": metric_tensor_sqrt,
            "axis": self.axis,
            "axis_abc": self.axis_abc,
            "reduced_positions": reduced_positions,
            "all_positions": all_positions,
            "elements": sorted_elements,
        }

    def get_reduced_positions(
        self,
        signed_permutation_cands,
        axis_abc=None,
    ):
        """Derive a reduced representation of atomic positions."""
        if self.axis_abc is None and axis_abc is not None:
            self.axis_abc = axis_abc
        else:
            a, b, c = np.array(self.axis)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            norm_c = np.linalg.norm(c)
            self.axis_abc = np.array([norm_a, norm_b, norm_c])

        elements = self.elements.copy()
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

        t_pos_cands = time.time()
        positions = self.positions.copy()
        sorted_positions = positions[sort_idx, :]
        position_cands = self._position_candidates(
            sorted_positions, sorted_types, signed_permutation_cands
        )
        if self.verbose:
            print(f"  Candidates: {round((time.time() - t_pos_cands) * 1000, 2)} ms")
            print(f"    N_atoms: {len(positions)}")
            print(f"    N_trans: {len(position_cands)}")

        t_permutation = time.time()
        all_positions = []
        reduced_perm_cands = []
        target_vals = []
        for pos_cand in position_cands:
            redperm_positions_list = self._reduced_permutations(
                pos_cand["cands_idx"],
                pos_cand["positions"],
                sorted_types,
                pos_cand["cluster_id"],
            )
            mask = np.where(sorted_types == 0)[0]
            for redperm_positions in redperm_positions_list:
                reduced_perm_cands.append(redperm_positions.T.reshape(-1))
                target_vals.append(
                    redperm_positions[0 : len(mask), :].T.reshape(-1)
                )
            all_positions.append(
                {
                    "perm_mat": pos_cand["perm_mat"],
                    "pos_reps": redperm_positions_list,
                }
            )
        if self.verbose:
            print(f"  Permutation: {round((time.time() - t_permutation) * 1000, 2)} ms")

        t_translation = time.time()
        reduced_perm_cands = np.array(reduced_perm_cands)
        target_vals = np.array(target_vals)
        reduced_positions = self._reduced_translation(
            reduced_perm_cands,
            target_vals,
        )
        if self.verbose:
            print(f"  Translation: {round((time.time() - t_translation) * 1000, 2)} ms")

        return reduced_positions, all_positions, sorted_elements

    def _position_candidates(
        self,
        positions: np.ndarray,
        types: np.ndarray,
        signed_permutation_cands: np.ndarray,
    ):
        _positions = positions.copy()
        cluster_id, positions_by_axis = self._assign_clusters(
            _positions, signed_permutation_cands
        )

        mask = types == 0
        position_cands = []
        for perm_mat in signed_permutation_cands:
            _pos = np.zeros_like(_positions)
            _cls_id = np.zeros_like(_positions, dtype=np.int32)
            for axis, val in enumerate(perm_mat):
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
                    "perm_mat": perm_mat,
                }
            )
        return position_cands

    def _reduced_permutations(
        self,
        target_indices: np.ndarray,
        positions: np.ndarray,
        types: np.ndarray,
        cluster_id: np.ndarray,
    ) -> list[np.ndarray]:
        target_indices = np.asarray(target_indices, dtype=int)
        positions = np.asarray(positions)
        types = np.asarray(types)
        cluster_id = np.asarray(cluster_id)

        n_cands = len(target_indices)
        n_atoms = len(positions)

        # Calculate relative positions and cluster IDs for all candidates.
        # (N_cands, N_atoms, 3)
        pos = positions[None, :, :] - positions[target_indices, None, :]
        pos %= 1.0
        id_max = np.max(cluster_id, axis=0) + 1
        cls_id = np.mod(
            cluster_id[None, :, :] - cluster_id[target_indices, None, :],
            id_max,
        )

        # Remove the target atom from each candidate.
        keep_mask = np.broadcast_to(
            np.arange(n_atoms)[None, :] != target_indices[:, None],
            (n_cands, n_atoms),
        )  # (N_cands, N_atoms)
        pos = pos[keep_mask].reshape(n_cands, n_atoms - 1, 3)
        cls_id = cls_id[keep_mask].reshape(n_cands, n_atoms - 1, 3)

        # Correct positions near the periodic boundary.
        pos[(cls_id == 0) & (pos > 0.5)] -= 1.0
        pos = pos * self.axis_abc

        # Sort each candidate lexicographically.
        redperm_positions = []
        for (
            pos_i,
            cls_id_i,
        ) in zip(pos, cls_id):
            sort_idx = np.lexsort(
                (cls_id_i[:, 0], cls_id_i[:, 1], cls_id_i[:, 2], types[1:])
            )
            redperm_positions.append(pos_i[sort_idx])
        return redperm_positions

    def _reduced_translation(
        self,
        reduced_perm_cands: np.ndarray,
        target_vals: np.ndarray,
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

    def _assign_clusters(
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

            # New cluster starts where gap > symprec
            gap_normalized = gap * self.axis_abc
            is_new_cluster = gap_normalized > self.symprec
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
