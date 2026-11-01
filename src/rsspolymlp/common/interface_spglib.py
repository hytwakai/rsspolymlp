"""Interfaces for spglib."""

import numpy as np
import spglib


class SymCell:
    """Class for using spglib functions."""

    def __init__(
        self,
        axis: np.ndarray,
        positions: np.ndarray,
        elements: np.ndarray,
        symprec: float = 1e-4,
    ):
        """
        Init method.

        Parameters
        ----------
        axis : (3, 3) array_like
            Each row represents one lattice vector (a, b, or c).
            Equivalent to np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]).
        positions : (N, 3) array_like
            Fractional atomic coordinates within the unit cell.
            Each row represents the coordinates of one atom.
        elements : (N,) array_like
            Chemical element symbols corresponding to the atomic positions.
        """

        axis = np.array(axis)
        positions = np.array(positions)
        elements = np.array(elements)
        element_types = {}
        types = []
        for e in elements:
            if e not in element_types:
                element_types[e] = len(element_types)
            types.append(element_types[e])

        self.n_types = len(set(elements))
        self.cell = (
            axis,
            positions,
            types,
        )

        self.element_map = dict()
        for e, t in zip(elements, types):
            self.element_map[t] = e

        self.symprec = symprec

    def primitive_cell(self) -> tuple:
        """Primitive cell."""
        try:
            lattice1, position1, types1 = spglib.find_primitive(
                self.cell, symprec=self.symprec
            )
        except Exception:
            raise TypeError

        type_list = list(range(0, self.n_types))

        position1_tmp, types1_tmp = [], []
        for t_ref in type_list:
            for i, t in enumerate(types1):
                if t == t_ref:
                    position1_tmp.append(position1[i])
                    types1_tmp.append(t)
        position1, types1 = np.array(position1_tmp), types1_tmp
        elements1 = [self.element_map[t] for t in types1]
        return lattice1, position1, elements1

    def refine_cell(self, standardize_cell=False) -> tuple:
        """Refine cell."""
        if standardize_cell is False:
            try:
                lattice1, position1, types1 = spglib.refine_cell(
                    self.cell, symprec=self.symprec
                )
            except Exception:
                raise TypeError
        else:
            try:
                lattice1, position1, types1 = spglib.standardize_cell(
                    self.cell, symprec=self.symprec
                )
            except Exception:
                raise TypeError

        type_list = list(range(0, self.n_types))

        position1_tmp, types1_tmp = [], []
        for t_ref in type_list:
            for i, t in enumerate(types1):
                if t == t_ref:
                    position1_tmp.append(position1[i])
                    types1_tmp.append(t)
        position1, types1 = np.array(position1_tmp), types1_tmp
        elements1 = [self.element_map[t] for t in types1]
        return lattice1, position1, elements1

    def get_spacegroup(self):
        """Return space group."""
        try:
            spg = spglib.get_spacegroup(self.cell, symprec=self.symprec)
        except spglib._spglib.SpglibCppError:
            spg = None
        return spg

    def get_spacegroup_multiple_prec(self, symprecs=[1e-2, 1e-3, 1e-4, 1e-5]):
        """Return list of space groups using multiple precisions."""
        return [spglib.get_spacegroup(self.cell, symprec=p) for p in symprecs]
