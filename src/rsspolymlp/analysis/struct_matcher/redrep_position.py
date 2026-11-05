import itertools
import time
from collections import Counter
from typing import Optional

import numpy as np

from rsspolymlp.analysis.struct_matcher.chiral_spg import is_chiral


class PositionRepReducer:
    """Identify the reduced crystal structure representation."""

    def __init__(
        self,
        axis: np.ndarray,
        positions: np.ndarray,
        elements: np.ndarray,
        spg_number: int,
        tol_sm: float = 1e-2,
        tol: float = 1e-4,
        normalize_axis: bool = False,
        transformation_matrices: Optional[np.ndarray] = None,
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
        tol : float
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
        self.tol_sm = tol_sm
        self.tol = tol
        self.normalize_axis = normalize_axis
        if self.normalize_axis:
            volume = abs(np.linalg.det(self.axis))
            self.axis = self.axis / ((volume / len(self.elements)) ** (1 / 3))
        self.transformation_matrices = transformation_matrices
        self.verbose = verbose
        self.axis_abc = None
        if self.verbose:
            print("code: struct_matcher.redrep_position")
            print("  tol:", tol)

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
                the structure up to the tolerance `tol`.
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

        G = self.axis @ self.axis.T
        metric_tensor = np.array([G[0, 0], G[1, 1], G[2, 2], G[1, 2], G[0, 2], G[0, 1]])
        if self.transformation_matrices is None:
            t_lattice = time.time()
            transformation_matrices = enumerate_metric_symmetries(
                self.axis, self.axis, self.spg_number, tol=self.tol_sm
            )
            if self.verbose:
                print(f"  Lattice: {round((time.time() - t_lattice) * 1000, 2)} ms")

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
                    transformation_matrices,
                )
            )

        return {
            "metric_tensor": metric_tensor,
            "axis": self.axis,
            "axis_abc": self.axis_abc,
            "reduced_positions": reduced_positions,
            "all_positions": all_positions,
            "elements": sorted_elements,
        }

    def get_reduced_positions(
        self,
        transformation_matrices,
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
        positions = positions[sort_idx, :]
        position_cands = []
        for trans_mat in transformation_matrices:
            _positions = positions.copy()
            _positions = _positions @ np.linalg.inv(trans_mat)
            _positions %= 1.0
            cluster_id, position_cand = assign_clusters(
                _positions, self.axis_abc, self.tol
            )
            position_cands.append(
                {
                    "positions": position_cand,
                    "cluster_id": cluster_id,
                    "perm_mat": trans_mat,
                }
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
                pos_cand["positions"],
                sorted_types,
                pos_cand["cluster_id"],
            )
            mask = np.where(sorted_types == 0)[0]
            for redperm_positions in redperm_positions_list:
                reduced_perm_cands.append(redperm_positions[:, ::-1].T.reshape(-1))
                target_vals.append(redperm_positions[0 : len(mask), ::-1].T.reshape(-1))
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
        reduced_positions = reduced_positions.reshape(3, -1)[::-1, :].reshape(-1)
        if self.verbose:
            print(f"  Translation: {round((time.time() - t_translation) * 1000, 2)} ms")

        return reduced_positions, all_positions, sorted_elements

    def _reduced_permutations(
        self,
        positions: np.ndarray,
        types: np.ndarray,
        cluster_id: np.ndarray,
    ) -> list[np.ndarray]:
        target_indices = np.where(types == 0)[0]
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
                    np.abs(sorted_one_coord - max_coord) <= self.tol
                )[0]
                _target_vals = _target_vals[: is_near_max[-1] + 1, :]
                _reduced_perm_cands = _reduced_perm_cands[: is_near_max[-1] + 1, :]
                if _reduced_perm_cands.shape[0] == 1:
                    break

        reduced_perm_cands = _reduced_perm_cands[0, :]
        return reduced_perm_cands


def assign_clusters(
    positions: np.ndarray,
    axis_abc: np.ndarray,
    tol: float,
    verbose: bool = False,
):
    """
    Assigns cluster IDs along each axis; atoms at identical positions share the same ID.
    """
    _positions = positions.copy()
    sort_idx = np.argsort(_positions, axis=0, kind="mergesort")
    pos_sorted = np.take_along_axis(_positions, sort_idx, axis=0)

    # Compute forward differences with periodic wrapping
    gap = np.roll(pos_sorted, -1, axis=0) - pos_sorted
    gap[-1, :] += 1.0

    # New cluster starts where gap > tol
    gap_normalized = gap * axis_abc
    proper_tol = []
    for idx_axis in range(3):
        proper_tol.append(find_proper_tolerance(gap_normalized.T[idx_axis], tol))
    is_new_cluster = gap_normalized > np.array(proper_tol)
    cluster_id_sorted = np.zeros_like(_positions, dtype=np.int32)
    cluster_id_sorted[1:, :] = np.cumsum(is_new_cluster[:-1, :], axis=0)
    if verbose:
        print("Position differences (Ang.):")
        print(np.round(gap_normalized.T, 3).tolist()[0])
        print(np.round(gap_normalized.T, 3).tolist()[1])
        print(np.round(gap_normalized.T, 3).tolist()[2])
        print("Proper tolerances:")
        print(proper_tol)

    # Merge last cluster if gap is small (periodic condition)
    merge_mask = ~is_new_cluster[-1, :]
    for ax in np.where(merge_mask)[0]:
        max_id = cluster_id_sorted[-1, ax]
        merged = cluster_id_sorted[:, ax] == max_id
        pos_sorted[merged, ax] -= 1.0
        cluster_id_sorted[merged, ax] = 0

    # Restore original order
    cluster_id = np.empty_like(pos_sorted, dtype=np.int32)
    snapped_positions = np.empty_like(pos_sorted)
    for ax in range(3):
        cluster_id[sort_idx[:, ax], ax] = cluster_id_sorted[:, ax]
        snapped_positions[sort_idx[:, ax], ax] = pos_sorted[:, ax]

    return cluster_id, snapped_positions


def find_proper_tolerance(
    differences,
    tol: float,
    window: float = 0.1,
    range: list = [0.5, 1.5],
    num_points: int = 19,
):
    differences = np.sort(np.asarray(differences).ravel())
    tolerance_range = [range[0] * tol, range[1] * tol]
    tolerances = np.linspace(*tolerance_range, num_points)

    tol_window = tol * window
    densities = np.array(
        [
            np.count_nonzero(
                (differences >= t - tol_window) & (differences <= t + tol_window)
            )
            for t in tolerances
        ]
    )
    min_indices = np.where(densities == densities.min())[0]
    idx = min_indices[np.argmin(np.abs(tolerances[min_indices] - tol))]
    return float(tolerances[idx])


def enumerate_metric_symmetries(
    axis_1: np.ndarray,
    axis_2: np.ndarray,
    spg_number: int,
    tol: float = 1e-3,
) -> list[np.ndarray]:
    """Enumerate unimodular integer matrices satisfying M @ G_1 @ M.T = G_2."""
    G_1 = axis_1 @ axis_1.T
    G_2 = axis_2 @ axis_2.T
    if not (np.allclose(G_1, G_1.T, atol=tol) and np.allclose(G_2, G_2.T, atol=tol)):
        raise ValueError("G must be symmetric.")

    norm_a = G_1[0, 0] ** 0.5
    norm_b = G_1[1, 1] ** 0.5
    norm_c = G_1[2, 2] ** 0.5
    e_aa = tol * 2 * norm_a
    e_bb = tol * 2 * norm_b
    e_cc = tol * 2 * norm_c
    e_bc = tol * (norm_b + norm_c)
    e_ac = tol * (norm_a + norm_c)
    e_ab = tol * (norm_a + norm_b)

    # Enumerate candidate column vectors.
    candidates = [
        enumerate_integer_vectors(G_1, G_2[i, i], [e_aa, e_bb, e_cc][i])
        for i in range(3)
    ]
    matrices = []

    # Enumerate combinations of the three columns.
    for m1, m2, m3 in itertools.product(*candidates):
        if abs(m1 @ G_1 @ m2 - G_2[0, 1]) > e_ab:
            continue
        if abs(m1 @ G_1 @ m3 - G_2[0, 2]) > e_ac:
            continue
        if abs(m2 @ G_1 @ m3 - G_2[1, 2]) > e_bc:
            continue
        M = np.column_stack((m1, m2, m3))

        det = round(np.linalg.det(M))
        if abs(det) != 1:
            continue
        if is_chiral(spg_number) and det == -1:
            continue
        matrices.append(M)

    return matrices


def enumerate_integer_vectors(
    G: np.ndarray,
    q: float,
    tol: float = 1e-5,
) -> np.ndarray:
    """Enumerate integer vectors satisfying v.T @ G @ v ~= q."""
    G = np.asarray(G, dtype=float)

    # Upper bound from the minimum eigenvalue.
    lambda_min = np.linalg.eigvalsh(G)[0]
    if lambda_min <= 0:
        raise ValueError("G must be positive definite.")

    vectors = []
    bound = int(np.ceil(np.sqrt((q + tol) / lambda_min)))
    for v in itertools.product(range(-bound, bound + 1), repeat=3):
        v = np.asarray(v, dtype=int)
        if abs(v @ G @ v - q) <= tol:
            vectors.append(v)

    return np.asarray(vectors, dtype=int).reshape(-1, 3)
