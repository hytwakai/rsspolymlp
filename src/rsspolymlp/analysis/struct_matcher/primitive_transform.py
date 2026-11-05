"""Get a primitive cell from translational symmetry."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form


@dataclass
class PrimitiveCell:
    """Result of primitive-cell transformation recovery."""

    axis: np.ndarray  # inv(M) @ original lattice matrix
    positions: np.ndarray  # unique sites in primitive-cell fractional coordinates
    elements: np.ndarray  # elements corresponding to primitive_positions
    M: np.ndarray  # integer 3x3 matrix
    index: int  # |det(M)|, number of original cells in primitive cell
    translations: np.ndarray  # valid translations in original fractional coordinates
    mapped_positions: (
        np.ndarray
    )  # all original sites transformed by M, wrapped to [0, 1)


def get_primitive_cell(
    axis: np.ndarray,
    positions: np.ndarray,
    elements: np.ndarray | list[str],
    tol: float = 1e-2,
    integer_tol: float = 1e-2,
    verbose: bool = False,
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
    integer_tol
        Tolerance for checking that ``index * translation`` is integer-valued.

    Returns
    -------
    PrimitiveCell
        Integer transformation M, index |det(M)|, valid translations, and the
        transformed lattice and fractional coordinates.

    Notes
    -----
    - For strongly skewed, unreduced lattices, the +/-1 image replication used
      by the KD-tree may be insufficient; reduce the lattice first or replace
      the image search with an exact nearest-image routine.
    """
    if tol <= 0:
        raise ValueError("tol must be positive.")

    axis = np.asarray(axis, dtype=float)
    positions = np.mod(np.asarray(positions, dtype=float), 1.0)
    elements = np.asarray(elements, dtype=str)
    if axis.shape != (3, 3):
        raise ValueError("axis must have shape (3, 3).")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")
    if len(elements) != len(positions):
        raise ValueError("elements and positions must have the same length.")

    # Select a reference atom from the least-populated elements to minimize the
    # number of translation candidates.
    uniq_elements, counts = np.unique(elements, return_counts=True)
    reference_element = uniq_elements[np.argmin(counts)]
    reference_id = int(np.flatnonzero(elements == reference_element)[0])
    reference_frac = positions[reference_id]
    reference_targets = np.flatnonzero(elements == reference_element)

    images = _build_periodic_elements_images(positions, elements, axis, tol * 2)
    candidate_translations = [
        np.mod((positions[j] - reference_frac), 1.0) for j in reference_targets
    ]

    valid_translations: list[np.ndarray] = []
    for translation in candidate_translations:
        shifted_positions = np.mod(positions + translation, 1.0)
        shifted_cart = shifted_positions @ axis
        is_valid, mean_translation = _translation_is_valid(
            shifted_cart, elements, images, tol
        )
        if not is_valid:
            continue
        translation += mean_translation @ np.linalg.inv(axis)
        duplicate = any(
            _periodic_translation_distance(translation, old, axis) <= tol
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
        mapped_positions,
        elements,
        primitive_lattice,
        index,
    )

    expected_n_sites = len(positions) // index
    if len(primitive_positions) != expected_n_sites:
        raise RuntimeError(
            "The recovered primitive cell has an unexpected number of unique sites: "
            f"found {len(primitive_positions)}, expected {expected_n_sites}. "
            "Check the tolerance and translation detection."
        )

    return PrimitiveCell(
        axis=primitive_lattice,
        positions=primitive_positions,
        elements=primitive_elements,
        M=M,
        index=index,
        translations=translations,
        mapped_positions=mapped_positions,
    )


def _build_periodic_elements_images(
    positions: np.ndarray,
    elements: np.ndarray,
    axis: np.ndarray,
    tol: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build periodic Cartesian images."""
    inv_axis = np.linalg.inv(axis)
    frac_tol = tol * np.linalg.norm(inv_axis, axis=0)

    images_dict = {}
    for element in np.unique(elements):
        target_ids = np.flatnonzero(elements == element)
        base = positions[target_ids]

        images = []
        image_to_atom_id = []
        for atom_id, position in enumerate(base):
            shift_candidates = [
                ([1] if x < frac_tol[i] else [])
                + [0]
                + ([-1] if x > 1.0 - frac_tol[i] else [])
                for i, x in enumerate(position)
            ]
            atom_shifts = np.array(list(product(*shift_candidates)), dtype=float)
            images.append(position + atom_shifts)
            image_to_atom_id.extend([atom_id] * len(atom_shifts))
        images_dict[str(element)] = (
            np.concatenate(images) @ axis,
            np.asarray(image_to_atom_id),
        )
    return images_dict


def _translation_is_valid(
    shifted_cart: np.ndarray,
    elements: np.ndarray,
    images_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    tol: float,
) -> tuple[bool, np.ndarray]:
    """Check whether a translation maps all sites onto same-elements sites."""
    translations = []

    for element in np.unique(elements):
        target_ids = np.flatnonzero(elements == element)
        target_cart = shifted_cart[target_ids]
        images, image_to_atom_id = images_dict[str(element)]
        target_count = len(target_ids)

        candidate_rows = []
        candidate_cols = []
        for i, cart in enumerate(target_cart):
            distances_sq = np.sum((cart - images) ** 2, axis=1)
            match_atom_ids = np.unique(image_to_atom_id[distances_sq <= tol**2])
            if len(match_atom_ids) == 0:
                return False, np.array([])
            candidate_rows.extend([i] * len(match_atom_ids))
            candidate_cols.extend(match_atom_ids.tolist())

        graph = csr_matrix(
            (
                np.ones(len(candidate_rows), dtype=np.int8),
                (candidate_rows, candidate_cols),
            ),
            shape=(target_count, target_count),
        )
        matching = maximum_bipartite_matching(graph, perm_type="column")
        if np.any(matching < 0):
            return False, np.array([])

        translations.extend(target_cart[matching] - target_cart)
    return True, np.mean(translations, axis=0)


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
    index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge equivalent sites in the primitive lattice using mean coordinates."""
    primitive_positions = []
    primitive_elements = []

    for element in np.unique(elements):
        target_ids = np.flatnonzero(elements == element)
        target_positions = mapped_positions[target_ids] % 1.0
        pos_diff = target_positions[:, None, :] - target_positions[None, :, :]
        pos_diff -= np.round(pos_diff)

        cart_diff = pos_diff @ primitive_lattice
        distances = np.linalg.norm(cart_diff, axis=2)

        neighbor_ids = np.argsort(distances, axis=1)
        unassigned = np.ones(len(target_positions), dtype=bool)
        mean_positions = []
        while np.any(unassigned):
            i = np.flatnonzero(unassigned)[0]

            # Select index nearest unassigned atoms
            ids = neighbor_ids[i]
            ids = ids[unassigned[ids]][:index]
            if len(ids) < index:
                break
            mean_position = (target_positions[i] - pos_diff[i, ids]).mean(axis=0) % 1.0
            mean_positions.append(mean_position)
            unassigned[ids] = False
        mean_positions = np.array(mean_positions)

        primitive_positions.extend(mean_positions)
        primitive_elements.extend([str(element)] * len(mean_positions))

    return (
        np.asarray(primitive_positions, dtype=float),
        np.asarray(primitive_elements, dtype=str),
    )
