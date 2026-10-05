"""Interfaces for vasp."""

import re
import xml.etree.ElementTree as ET
from typing import Union

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.units import EVtoKbar
from rsspolymlp.common.atomic_energy import atomic_energy


def parse_properties_from_vasprun(
    vasprun: str,
    verbose: bool = False,
    units: str = "eV",
) -> tuple:
    """Parse vasprun.xml files and return structures and properties."""
    try:
        v = Vasprun(vasprun)
    except Exception:
        if verbose:
            print("- Not readable:", vasprun, flush=True)
        raise ValueError
    if not _is_convergent(v):
        if verbose:
            print("- Not convergent:", vasprun, flush=True)
        raise ValueError

    # Get cohosive energy
    e = v.energy
    for element in v.structure.elements:
        e -= atomic_energy(element)

    if units == "eV":
        sigma = v.stress * v.structure.volume / EVtoKbar
    elif units == "GPa":
        sigma = v.stress / 10
    elif units == "kbar":
        sigma = v.stress
    else:
        raise ValueError(f"Unsupported units: {units}")

    return v.structure, (e, v.forces, sigma)


class Vasprun:
    """Class for parsing vasprun.xml from single-point calculation."""

    def __init__(self, name: str, root=None):
        """Init method."""
        if root is None:
            self._root = ET.parse(name).getroot()
        else:
            self._root = root
        self._calc = self._root.find("calculation")

        self._energy = None
        self._forces = None
        self._stress = None
        self._structure = None
        self._name = name

    def get_energy_smearing_delta(self) -> float:
        """Parse vasprun and return smearing delta F."""
        e = self._calc.find("energy")
        return float(e[2].text)

    @property
    def energy(self) -> float:
        """Parse vasprun and return energy.

        Return
        ------
        energy: float.
        """
        if self._energy is not None:
            return self._energy
        e = self._calc.find("energy")
        self._energy = float(e[1].text)
        return self._energy

    @property
    def forces(self) -> np.ndarray:
        """Parse vasprun and return forces.

        Return
        ------
        forces: shape=(3, n_atom)
        """
        if self._forces is not None:
            return self._forces
        f = self._root.find(".//*[@name='forces']")
        self._forces = self._varray_to_nparray(f).T
        return self._forces

    @property
    def stress(self) -> np.ndarray:
        """Parse vasprun and return stress tensor in kbar.

        Return
        ------
        stress: shape=(3, 3) in kbar.
        """
        if self._stress is not None:
            return self._stress
        f = self._root.find(".//*[@name='stress']")
        self._stress = self._varray_to_nparray(f)
        return self._stress

    @property
    def properties(self) -> tuple:
        """Return properties."""
        return (self.energy, self.forces, self.stress)

    @property
    def structure(self) -> PolymlpStructure:
        """Parse vasprun and return structure."""
        if self._structure is not None:
            return self._structure

        st = self._root.find(".//*[@name='finalpos']")
        st1 = st.find(".//*[@name='basis']")
        st2 = st.find(".//*[@name='positions']")
        st3 = st.find(".//*[@name='volume']")
        st4 = self._root.findall(".//*[@name='atomtypes']/set/rc")
        st5 = self._root.findall(".//*[@name='atoms']/set/rc")

        axis = self._varray_to_nparray(st1).T
        positions = self._varray_to_nparray(st2).T
        volume = float(st3.text)

        tmp1 = self._read_rc_set(st4)
        n_atoms = [int(x) for x in list(np.array(tmp1)[:, 0])]

        tmp2 = self._read_rc_set(st5)
        elements = list(np.array(tmp2)[:, 0])
        elements = ["Zr" if e == "r" else e for e in elements]
        types = [int(x) - 1 for x in list(np.array(tmp2)[:, 1])]

        # if valence:
        #     valence_dict = dict()
        #     for d in tmp1:
        #         valence_dict[d[1]] = float(d[3])
        #     valence = [valence_dict[e] for e in self.elements]
        # else:
        #     valence = None

        self._structure = PolymlpStructure(
            axis,
            positions,
            n_atoms,
            elements,
            types,
            volume,
            name=self._name,
        )
        return self._structure

    def get_scstep(self) -> np.ndarray:
        """Return SC step."""
        scsteps = self._root.find("calculation").findall("scstep")
        e_history = []
        for sc in scsteps:
            e0 = sc.find("energy").find(".//*[@name='e_0_energy']")
            e_history.append(float(e0.text))
        return np.array(e_history)

    def _varray_to_nparray(self, varray):
        """Convert varray to numpy array."""
        nparray = [[float(x) for x in v1.text.split()] for v1 in varray]
        return np.array(nparray)

    def _read_rc_set(self, obj):
        """Read rc_set."""
        return [[c.text.replace(" ", "") for c in rc.findall("c")] for rc in obj]


class Poscar:
    """Class for parsing POSCAR."""

    def __init__(self, filename: str, selective_dynamics: bool = False):
        """Init method."""
        self._parse(filename, selective_dynamics=selective_dynamics)

    def _parse(self, filename: str, selective_dynamics: bool = False):
        """Parse POSCAR file."""
        f = open(filename, "r")
        lines = f.readlines()
        f.close()

        comment = lines[0].replace("\n", "")
        axis_const = float(lines[1].split()[0])
        axis1 = [float(x) for x in lines[2].split()[0:3]]
        axis2 = [float(x) for x in lines[3].split()[0:3]]
        axis3 = [float(x) for x in lines[4].split()[0:3]]
        axis = np.c_[axis1, axis2, axis3] * axis_const

        if len(re.findall(r"[a-z,A-Z]+", lines[5])) > 0:
            uniq_elements = lines[5].split()
            n_atoms = [int(x) for x in lines[6].split()]
            n_line = 7
        else:
            uniq_elements = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            n_atoms = [int(x) for x in lines[5].split()]
            n_line = 6

        elements, types = [], []
        for i, n in enumerate(n_atoms):
            for j in range(n):
                types.append(i)
                elements.append(uniq_elements[i])

        if selective_dynamics:
            # sd = lines[begin_nline]
            n_line += 1

        # coord_type = lines[n_line].split()[0]
        n_line += 1

        positions = []
        for i in range(sum(n_atoms)):
            pos = [float(x) for x in lines[n_line].split()[0:3]]
            positions.append(pos)
            n_line += 1
        positions = np.array(positions).T
        volume = np.linalg.det(axis)

        self._structure = PolymlpStructure(
            axis,
            positions,
            n_atoms,
            elements,
            types,
            volume,
            comment=comment,
            name=filename,
        )

    @property
    def structure(self) -> PolymlpStructure:
        """Return structure."""
        return self._structure


def read_doscar(name):
    """Parse DOSCAR file."""
    f = open(name)
    lines = f.readlines()
    f.close()

    e_fermi = float(lines[5].split()[3])
    dos = []
    for l1 in lines[6:]:
        str1 = l1.split()[:2]
        vals = [float(str1[0]) - e_fermi, float(str1[1])]
        dos.append(vals)
    return np.array(dos)


def _exist_no_errors(logfile: str):
    """Check errors in VASP calculation."""
    try:
        f = open(logfile)
        lines = f.readlines()
        f.close()
    except Exception:
        return False

    for line in lines:
        if (
            "Your highest band is occupied" in line
            or "Error EDDDAV: Call to ZHEGV failed" in line
            or "WARNING: Sub-Space-Matrix is not hermitian in DAV" in line
            or "WARNING: CNORMN: search vector ill defined" in line
        ):
            return False
    return True


def _is_convergent(vasprun_xml: Union[str, Vasprun], tol: float = 1e-3):
    """Check if VASP calculation is convergent."""
    if isinstance(vasprun_xml, Vasprun):
        vasp = vasprun_xml
    else:
        try:
            vasp = Vasprun(vasprun_xml)
        except Exception:
            return False

    try:
        e_history = vasp.get_scstep()
        if abs(e_history[-1] - e_history[-2]) < tol:
            return True
        return False
    except Exception:
        return True


class VaspErrorCheck:
    """Class for checking errors in VASP calculations."""

    def __init__(self):
        """Init method."""
        pass

    def test_no_errors(self, logfiles: Union[str, list]):
        """Test if calculations finish successfully."""
        if isinstance(logfiles, str):
            return _exist_no_errors(logfiles)
        return np.array([_exist_no_errors(log) for log in logfiles])

    def test_convergence(self, vaspruns: Union[str, list], tol: float = 1e-3):
        """Test if calculations converge."""
        if isinstance(vaspruns, str):
            return _is_convergent(vaspruns)
        return np.array([_is_convergent(vasp) for vasp in vaspruns])
