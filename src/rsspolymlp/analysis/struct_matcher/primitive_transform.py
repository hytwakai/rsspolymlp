"""Recover a primitive-cell transformation from translational symmetry.

Convention:
    axis stores lattice vectors as rows.
    L_primitive = inv(M) @ L_original
    frac_primitive = frac_original @ M

The implementation is intended for structures whose translational symmetry is
exact up to a user-specified Cartesian tolerance. It finds candidate
translations by mapping one reference atom to atoms of the same elements,
checks each candidate against the complete structure, and reconstructs M using
an exact integer row-lattice HNF calculation.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form


@dataclass
class PrimitiveCell:
    """Result of primitive-cell transformation recovery."""

    M: np.ndarray  # integer 3x3 matrix
    index: int  # |det(M)|, number of original cells in primitive cell
    translations: np.ndarray  # valid translations in original fractional coordinates
    axis: np.ndarray  # inv(M) @ original lattice matrix
    positions: np.ndarray  # unique sites in primitive-cell fractional coordinates
    elements: np.ndarray  # elements corresponding to primitive_positions
    mapped_positions: (
        np.ndarray
    )  # all original sites transformed by M, wrapped to [0, 1)


def get_primitive_cell(
    axis: np.ndarray,
    positions: np.ndarray,
    elements: np.ndarray | list[str],
    tol: float = 1e-2,
    translation_tol: float | None = None,
    integer_tol: float = 1e-2,
    verbose: bool = True,
) -> PrimitiveCell:
    """Find a primitive-cell transformation from lattice and atom arrays.

    Parameters
    ----------
    axis
        Shape-(3, 3) matrix whose rows are the lattice vectors.
    positions
        Shape-(N, 3) array of fractional coordinates.
    elements
        Length-N sequence identifying each atom elements (e.g. ``["Si", "Si"]``).
    tol
        Cartesian distance tolerance in angstrom for atom matching.
    translation_tol
        Cartesian tolerance for deduplicating candidate translations.
    integer_tol
        Tolerance for checking that ``index * translation`` is integer-valued.

    Returns
    -------
    PrimitiveCell
        Integer transformation M, index |det(M)|, valid translations, and the
        transformed lattice and fractional coordinates.

    Notes
    -----
    - This implementation assumes the input structure is translationally
      periodic without changing elements.
    - For disordered/partial-occupancy sites, convert to an explicit ordered
      representation before using this function.
    - For strongly skewed, unreduced lattices, the +/-1 image replication used
      by the KD-tree may be insufficient; reduce the lattice first or replace
      the image search with an exact nearest-image routine.
    """
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if translation_tol is None:
        translation_tol = tol
    if translation_tol <= 0:
        raise ValueError("translation_tol must be positive.")

    axis = np.asarray(axis, dtype=float)
    positions = np.mod(np.asarray(positions, dtype=float), 1.0)
    elements = np.asarray(elements, dtype=str)
    if axis.shape != (3, 3):
        raise ValueError("axis must have shape (3, 3).")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")
    if len(elements) != len(positions):
        raise ValueError("elements and positions must have the same length.")
    n_sites = len(positions)
    if n_sites == 0:
        raise ValueError("Cannot analyze an empty structure.")

    # Select a reference atom from the least-populated elements to minimize the
    # number of translation candidates.
    uniq_elements, counts = np.unique(elements, return_counts=True)
    reference_element = uniq_elements[np.argmin(counts)]
    reference_id = int(np.flatnonzero(elements == reference_element)[0])
    reference_frac = positions[reference_id]
    reference_targets = np.flatnonzero(elements == reference_element)

    trees = _build_periodic_elements_trees(positions, elements, axis)
    candidate_translations = [
        np.mod((positions[j] - reference_frac), 1.0) for j in reference_targets
    ]

    valid_translations: list[np.ndarray] = []
    for translation in candidate_translations:
        shifted_positions = np.mod(positions + translation, 1.0)
        shifted_cart = shifted_positions @ axis
        if not _translation_is_valid(shifted_cart, elements, trees, tol):
            continue
        duplicate = any(
            _periodic_translation_distance(translation, old, axis) <= translation_tol
            for old in valid_translations
        )
        if not duplicate:
            valid_translations.append(translation)

    translations = np.asarray(valid_translations, dtype=float)
    if len(translations) == 0:
        raise RuntimeError("No valid translational symmetry was found.")

    index = len(translations)
    if any(int(count) % index != 0 for count in counts):
        raise ValueError(
            f"Detected {index} translations, but this does not divide all elements counts "
            f"{counts.tolist()}. Consider tightening tol."
        )

    M = _get_transformation_matrix(translations, index, integer_tol=integer_tol)
    primitive_lattice = np.linalg.inv(M.astype(float)) @ axis
    mapped_positions = np.mod(positions @ M, 1.0)
    primitive_positions, primitive_elements = _extract_primitive_positions(
        mapped_positions, elements, primitive_lattice, tol
    )

    expected_n_sites = n_sites // index
    if len(primitive_positions) != expected_n_sites:
        raise RuntimeError(
            "The recovered primitive cell has an unexpected number of unique sites: "
            f"found {len(primitive_positions)}, expected {expected_n_sites}. "
            "Check the tolerance and translation detection."
        )

    return PrimitiveCell(
        M=M,
        index=index,
        translations=translations,
        axis=primitive_lattice,
        positions=primitive_positions,
        elements=primitive_elements,
        mapped_positions=mapped_positions,
    )


def _build_periodic_elements_trees(
    positions: np.ndarray,
    elements: np.ndarray,
    axis: np.ndarray,
) -> dict[str, tuple[cKDTree, np.ndarray]]:
    """Build periodic Cartesian KD-trees."""
    image_shifts = np.asarray(
        list(product((-1, 0, 1), repeat=3)), dtype=float
    )  # shape-(27, 3)
    trees: dict[str, tuple[cKDTree, np.ndarray]] = {}
    for element in np.unique(elements):
        target_ids = np.flatnonzero(elements == element)
        base = positions[target_ids]
        images = (base[:, None, :] + image_shifts[None, :, :]).reshape(-1, 3)
        image_to_atom_id = np.repeat(np.arange(len(target_ids)), len(image_shifts))
        cart_images = images @ axis
        trees[str(element)] = (cKDTree(cart_images), image_to_atom_id)
    return trees


def _translation_is_valid(
    shifted_cart: np.ndarray,
    elements: np.ndarray,
    trees: dict[str, tuple[cKDTree, np.ndarray]],
    tol: float,
) -> bool:
    """Check whether a translation maps all sites onto same-elements sites."""
    for element in np.unique(elements):
        target_ids = np.flatnonzero(elements == element)
        tree, image_to_atom_id = trees[str(element)]
        target_count = len(target_ids)
        candidate_rows: list[int] = []
        candidate_cols: list[int] = []

        neighbor_lists = tree.query_ball_point(shifted_cart[target_ids], r=tol)
        for shifted_atom_id, match_image_idxs in enumerate(neighbor_lists):
            if not match_image_idxs:
                return False
            match_atom_ids = np.unique(
                image_to_atom_id[np.asarray(match_image_idxs, dtype=int)]
            )
            candidate_rows.extend([shifted_atom_id] * len(match_atom_ids))
            candidate_cols.extend(match_atom_ids.tolist())

        graph = csr_matrix(
            (
                np.ones(len(candidate_rows), dtype=np.int8),
                (candidate_rows, candidate_cols),
            ),
            shape=(target_count, target_count),
        )
        matching = maximum_bipartite_matching(graph, perm_type="column")
        if matching.shape[0] != target_count or np.any(matching < 0):
            return False
    return True


def _periodic_translation_distance(
    t1: np.ndarray, t2: np.ndarray, axis: np.ndarray
) -> float:
    delta = np.asarray(t1, dtype=float) - np.asarray(t2, dtype=float)
    shifts = np.asarray(list(product((-1, 0, 1), repeat=3)), dtype=float)
    cart = (delta[None, :] + shifts) @ axis
    return float(np.min(np.linalg.norm(cart, axis=1)))


def _get_transformation_matrix(
    translations: np.ndarray, index: int, integer_tol: float = 1e-2
) -> np.ndarray:
    """Get transformation matrix M (L_original = M @ L_primitive)."""

    def _lattice_basis(generators: np.ndarray) -> np.ndarray:
        """Return an integer basis for the lattice generated by matrix rows."""
        h_col = hermite_normal_form(
            Matrix(np.asarray(generators, dtype=object).T.tolist())
        )
        h_row = h_col.T
        if h_row.shape != (3, 3):
            raise ValueError("Failed to obtain a rank-3 integer lattice basis.")
        return np.asarray(h_row.tolist(), dtype=object)

    # By Lagrange's theorem, multiplying each translation vector by the group order (index)
    # yields an integer vector.
    integer_translations = np.rint(index * translations).astype(np.int64)

    # Check whether index * translations is close to integer vectors.
    residual = np.abs(index * translations - integer_translations)
    if np.any(residual > integer_tol * index):
        raise ValueError(
            "Translations are not consistent with denominator index. "
            f"Maximum residual in index*t: {residual.max():.3e}."
        )

    # Construct integer generators for the lattice containing all translations.
    generators = np.vstack([index * np.eye(3, dtype=np.int64), integer_translations])

    # H / index gives a basis of the translation lattice in the original
    # fractional coordinates (L_primitive = (H / index) @ L_original).
    H = _lattice_basis(generators)
    H_rational = Matrix(H.tolist())
    M_rational = index * H_rational.inv()

    if any(value.q != 1 for value in M_rational):
        raise ValueError("Recovered transformation is not an integer matrix.")

    M = np.asarray(M_rational.tolist(), dtype=np.int64)

    # The absolute determinant of M gives the ratio of the original-cell
    # volume to the primitive-cell volume, which equals the group order (index).
    det_M = int(round(np.linalg.det(M)))
    if abs(det_M) != index:
        raise ValueError(
            f"Transformation matrix determinant {det_M} is inconsistent with index {index}."
        )

    # Verify that every observed translation becomes an integer lattice
    # translation when transformed by M.
    transformed = translations @ M
    residual_M = np.abs(transformed - np.rint(transformed))
    if np.any(residual_M > max(integer_tol, 10 * np.finfo(float).eps)):
        raise ValueError(
            "Transformation matrix does not map all observed translations to integer vectors."
        )
    return M


def _extract_primitive_positions(
    mapped_positions: np.ndarray,
    elements: np.ndarray,
    primitive_lattice: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge equivalent sites in the primitive lattice using mean coordinates."""
    matched_positions: list[list[np.ndarray]] = []
    primitive_elements: list[str] = []
    for coord, element in zip(mapped_positions, elements):
        coord = np.asarray(coord, dtype=float)
        is_duplicate = False
        for positions, old_element in zip(matched_positions, primitive_elements):
            if element != old_element:
                continue
            if (
                _periodic_cartesian_distance_frac(
                    coord, positions[0], primitive_lattice
                )
                <= tol
            ):
                positions.append(coord)
                is_duplicate = True
                break
        if not is_duplicate:
            matched_positions.append([coord])
            primitive_elements.append(str(element))

    primitive_positions = []
    shifts = np.asarray(list(product((-1, 0, 1), repeat=3)), dtype=int)
    for cluster in matched_positions:
        # Find periodic images nearest to the reference coordinate.
        cluster_array = np.asarray(cluster, dtype=float)
        delta = cluster_array - cluster[0]
        fracs_candidates = delta[:, None, :] + shifts[None, :, :]
        distances = np.linalg.norm(fracs_candidates, axis=2)
        best_shifts = shifts[np.argmin(distances, axis=1)]

        # Average coordinates after applying periodic corrections.
        unwrapped = cluster_array + best_shifts
        mean_coord = np.mean(unwrapped, axis=0) % 1.0
        primitive_positions.append(mean_coord)

    return (
        np.asarray(primitive_positions, dtype=float).reshape(-1, 3),
        np.asarray(primitive_elements, dtype=str),
    )


def _periodic_cartesian_distance_frac(
    frac1: np.ndarray, frac2: np.ndarray, axis: np.ndarray
) -> float:
    """Minimum Cartesian distance between two fractional coordinates."""
    delta = np.asarray(frac1, dtype=float) - np.asarray(frac2, dtype=float)
    shifts = np.asarray(list(product((-1, 0, 1), repeat=3)), dtype=float)
    cart = (delta[None, :] + shifts) @ axis
    return float(np.min(np.linalg.norm(cart, axis=1)))
