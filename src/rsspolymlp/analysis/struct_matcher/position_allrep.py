from collections import Counter

import numpy as np

from rsspolymlp.analysis.struct_matcher.lattice_redrep import get_reduced_lattice


class PositionRepGenerator:
    """Identify the reduced crystal structure representation in a periodic cell."""

    def __init__(
        self,
        symprec: list[float] = [1e-3, 1e-3, 1e-3],
        standardize_axis: bool = False,
        original_axis: bool = False,
    ):
        """Init method."""
        self.symprec = np.array(symprec)
        self.standardize_axis = standardize_axis
        self.original_axis = original_axis

    def get_all_structure_representations(self, axis, positions, elements, spg_number):
        """Derive all representations of a crystal structure.

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
        all_positions : list[ndarray]
            List of (N, 3) arrays.
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

        metric_tensor_half, axis_cands, signed_permutation_cands = get_reduced_lattice(
            _axis,
            spg_number,
            self.symprec,
            self.original_axis,
        )

        # Trivial case: single‑atom cell → nothing to do
        if self.positions.shape[0] == 1:
            return metric_tensor_half, np.array([0, 0, 0]), self.elements

        all_positions, sorted_elements = self.get_all_positions(
            self.positions,
            self.elements,
            axis_cands,
            signed_permutation_cands,
        )

        return metric_tensor_half, all_positions, sorted_elements

    def get_all_positions(
        self,
        positions,
        elements,
        axis_cands,
        signed_permutation_cands,
    ):
        unique_elements = np.sort(np.unique(elements))
        types = np.array([np.where(unique_elements == el)[0][0] for el in elements])

        counts = Counter(elements)
        min_count = min(counts.values())
        least_elements = [el for el, cnt in counts.items() if cnt == min_count]
        target_element = sorted(least_elements)[0]
        target_type = np.where(unique_elements == target_element)[0][0]
        types = (types - target_type) % (np.max(types) + 1)

        sort_idx = np.argsort(types)
        elements = elements[sort_idx]
        types = types[sort_idx]
        positions = positions[sort_idx, :]

        position_cands = self.position_candidates(
            positions, types, axis_cands, signed_permutation_cands
        )

        all_positions = []
        for pos_cand in position_cands:
            for target_idx in pos_cand["cands_idx"]:
                _pos = pos_cand["positions"].copy()
                _pos = _pos - _pos[target_idx]
                position_onerep = _pos % 1.0
                all_positions.append(position_onerep)

        return all_positions, elements

    def position_candidates(
        self,
        positions: np.ndarray,
        types: np.ndarray,
        axis_cands: np.ndarray,
        signed_permutation_cands: np.ndarray,
    ):
        invert_list = [False]
        positions_by_axis = np.zeros_like(positions)
        if any(np.any(v == -1) for v in signed_permutation_cands):
            invert_list = [False, True]
            n_rows, n_cols = positions.shape
            positions_by_axis = np.zeros((n_rows, n_cols * 2), dtype=positions.dtype)
        for invert in invert_list:
            if not invert:
                positions_by_axis[:, slice(0, 3)] = positions.copy()
            else:
                positions_by_axis[:, slice(3, 6)] = -positions.copy() % 1.0

        mask = types == 0
        cands_idx = np.where(mask)[0]
        position_cands = []
        for idx, cand in enumerate(signed_permutation_cands):
            _pos = np.zeros_like(positions)
            for axis, val in enumerate(cand):
                target_axis = np.where(val != 0)[0][0]
                sign = val[target_axis]
                if sign == 1:
                    _pos[:, axis] = positions_by_axis[:, target_axis]
                else:
                    _pos[:, axis] = positions_by_axis[:, target_axis + 3]
            position_cands.append(
                {
                    "positions": _pos,
                    "cands_idx": cands_idx,
                    "axis": axis_cands[idx],
                }
            )
        return position_cands
