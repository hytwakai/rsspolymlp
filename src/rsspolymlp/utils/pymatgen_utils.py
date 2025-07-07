import numpy as np

from pymatgen.analysis.prototypes import AflowPrototypeMatcher
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from rsspolymlp.common.property import PropUtil

np.set_printoptions(precision=15)


class PymatUtil:

    def __init__(self):
        pass

    def parameter_to_pymat_st(self, lattice, frac_coords, species):
        """
        Parameters
        ----------
        lattice : parameters of lattice (3*3)
        coords : parameters of atomic postions
        species : string atom_type
        Returns: pymatgen.core.structure
        """
        return Structure(lattice=lattice, species=species, coords=frac_coords)

    def space_group(self, pymat_st, symprec_in=10**-5):
        """pymat_st : pymatgen.core.structure"""
        finder = SpacegroupAnalyzer(pymat_st, symprec=symprec_in, angle_tolerance=-1.0)
        spac = finder.get_space_group_symbol()
        return spac

    def refine_cell(self, pymat_st, symprec_in=10**-5):
        finder = SpacegroupAnalyzer(pymat_st, symprec=symprec_in, angle_tolerance=-1.0)
        pymat_st = finder.get_refined_structure()
        return pymat_st

    def primitive_cell(self, pymat_st, symprec_in=10**-5):
        finder = SpacegroupAnalyzer(pymat_st, symprec=symprec_in, angle_tolerance=-1.0)
        pymat_st = finder.get_primitive_standard_structure()
        return pymat_st

    def input_poscar(self, file):
        """
        Parameters
        ----------
        file : string (poscar)
        Returns pymatgen.core.structure
        """
        pymat_st = Structure.from_file(file)
        return pymat_st

    def input_cif(self, file, site_tolerance=1e-4):
        """
        Parameters
        ----------
        file : string (cif file)
        Returns pymatgen.core.structure
        """
        parser = CifParser(file, site_tolerance=site_tolerance)
        pymat_st = parser.parse_structures(primitive=True)[0]
        return pymat_st

    def output_poscar(self, pymat_st, file_name="./POSCAR"):
        writer = Poscar(pymat_st).get_string(significant_figures=16)
        with open(file_name, "w") as f:
            print(writer, file=f)

    def match_str(
        self,
        pymat_st1,
        pymat_st2,
        ltol=0.1,
        stol=0.1,
        angle_tol=1,
        primitive_cell=True,
        skip_structure_reduction=False,
    ):
        """
        Parameters
        ----------
        pymat_st1 : pymatgen.core.structure
        pymat_st2 : pymatgen.core.structure
        Returns True or False
        """
        inst = StructureMatcher(
            ltol=ltol, stol=stol, angle_tol=angle_tol, primitive_cell=primitive_cell
        )
        judge = inst.fit(
            struct1=pymat_st1,
            struct2=pymat_st2,
            symmetric=False,
            skip_structure_reduction=skip_structure_reduction,
        )
        return judge

    def prototype(self, pymat_st):
        """Returns Prototype of target structure"""
        proto = AflowPrototypeMatcher()
        structure_type = proto.get_prototypes(pymat_st)
        return structure_type

    def get_parameters(self, pymat_st):
        lat = pymat_st.lattice.matrix
        coo = pymat_st.frac_coords
        prop = PropUtil(lat, coo)
        _res = {}
        _res["axis"] = lat
        _res["abc"] = prop.abc
        _res["positions"] = coo.tolist()
        _res["elements"] = [str(el).split()[-1] for el in pymat_st.species]
        unique_elements = list(dict.fromkeys(_res["elements"]))
        label_map = {el: i for i, el in enumerate(unique_elements)}
        _res["types"] = [label_map[el] for el in _res["elements"]]
        _res["distance"] = prop.least_distance
        _res["volume"] = prop.volume
        return _res

    def supercell(self, pymat_st, matrix):
        """Returns pymatgen.core.structure (supercell)"""
        pymat_st.make_supercell(matrix)
        return pymat_st

    def get_niggli(self, pymat_st):
        params = self.get_parameters(pymat_st)
        lat = params["abc"]
        lattice = Lattice.from_parameters(
            a=float(lat[0]),
            b=float(lat[1]),
            c=float(lat[2]),
            alpha=lat[3],
            beta=lat[4],
            gamma=lat[5],
        )
        lattice_niggli = lattice.get_niggli_reduced_lattice()
        # abc = lattice_niggli.abc
        # angles = lattice_niggli.angles
        # print(abc, angles)
        return lattice_niggli

    def least_distance(self, pymat_st):
        dist_mat = pymat_st.distance_matrix
        dist_mat_refine = np.where(dist_mat > 1e-10, dist_mat, np.inf)
        distance_min = np.min(dist_mat_refine)
        return distance_min
