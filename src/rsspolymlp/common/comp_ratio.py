from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CompositionResult:
    elements: np.ndarray
    types: np.ndarray
    unique_elements: np.ndarray
    atom_counts: np.ndarray
    comp_ratio: tuple[int, ...]
    comp_str: str


def compute_composition(
    elements: list[str], element_order: Optional[list[str]] = None
) -> CompositionResult:
    """
    Compute reduced composition ratio and element counts.

    Parameters
    ----------
    elements : list of str
        List of element symbols (e.g., ['Bi', 'Bi', 'Ba']).
    element_order : list of str, optional
        List of elements to use in specified order.
        If None, elements are ordered by first appearance.

    Returns
    -------
    CompositionResult
        Data class containing sorted elements, atom counts, and reduced ratio.
    """
    _elements = np.array(elements)
    if element_order is not None:
        sorted_elements = np.array(element_order)
    else:
        unique_elements, first_indices = np.unique(_elements, return_index=True)
        sorted_elements = unique_elements[np.argsort(first_indices)]

    label_to_type = {el: i for i, el in enumerate(sorted_elements)}
    types = np.array([label_to_type[el] for el in elements])
    atom_counts = np.array(
        [np.count_nonzero(_elements == el) for el in sorted_elements]
    )

    if not np.any(atom_counts):
        raise ValueError("No valid elements found in the structure.")

    gcd = np.gcd.reduce(atom_counts)
    reduced_comp_ratio = tuple(int(x) for x in (atom_counts // gcd))
    comp_str = "".join(
        f"{el}{n}" for el, n in zip(sorted_elements, reduced_comp_ratio) if n != 0
    )

    return CompositionResult(
        elements=elements,
        types=types,
        unique_elements=sorted_elements,
        atom_counts=atom_counts,
        comp_ratio=reduced_comp_ratio,
        comp_str=comp_str,
    )
