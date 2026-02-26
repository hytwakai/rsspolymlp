import numpy as np
import scipy

from phonopy import Phonopy
from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.sscha.harmonic_reciprocal import HarmonicReciprocal
from pypolymlp.calculator.sscha.sscha_restart import Restart
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.units import EVtoGPa, EVtoKJmol
from pypolymlp.utils.phonopy_utils import structure_to_phonopy_cell
from pypolymlp.utils.spglib_utils import construct_basis_cell
from pypolymlp.utils.symfc_utils import construct_basis_fractional_coordinates
from rsspolymlp.utils.opt_sscha.harmonic_real import HarmonicReal


class SSCHAProperty:

    def __init__(
        self,
        cell: PolymlpStructure,
        pot: str,
        n_samples: int = 1000,
        yamlfile: str = "./sscha_results.yaml",
        fc2file: str = "./fc2.hdf5",
        pressure: float = 0,
    ):
        self._structure = cell
        self._n_samples = n_samples
        self._pressure = pressure

        self._res = Restart(yamlfile, fc2hdf5=fc2file, pot=pot)
        self._fc2 = self._res.force_constants
        self._ev_to_kjmol = EVtoKJmol / self._res.n_unitcells
        self._prop = Properties(pot=self._res.polymlp)

        self._sscha_enegies = None
        self._sscha_forces = None
        self._sscha_stress_tensors = None

    def run(self):
        self._ph_real = HarmonicReal(
            self._structure,
            self._prop,
            n_unitcells=self._res.n_unitcells,
            fc2=self._fc2,
        )
        self._ph_real.run(temp=self._res.temperature, n_samples=self._n_samples)

        self._phonopy = Phonopy(
            structure_to_phonopy_cell(self._structure),
            self._res.supercell_matrix,
            nac_params=None,
        )
        self._ph_recip = HarmonicReciprocal(
            self._phonopy,
            self._prop,
            fc2=self._fc2,
        )
        self._ph_recip.compute_thermal_properties(
            temp=self._res.temperature, qmesh=(10, 10, 10)
        )

        self.calc_sscha_energies()
        self.calc_sscha_forces()
        self.calc_sscha_stress_tensors()

    def calc_sscha_energies(self):
        free_energies = self._ph_recip.free_energy + self._ph_real.anharmonic_potentials
        self._sscha_enegies = (
            free_energies + self._ph_real.static_potential
        ) / self._ev_to_kjmol
        self._sscha_enegies += self._pressure * self._structure.volume / EVtoGPa

    def calc_sscha_forces(self):
        forces_from_fc2 = [
            self._forces_from_fc2(d.T.reshape(-1)) for d in self._ph_real.displacements
        ]
        forces_from_fc2 = np.array(forces_from_fc2)
        _sample_forces = self._ph_real.forces.transpose(0, 2, 1)
        sample_forces = _sample_forces.reshape(
            _sample_forces.shape[0], _sample_forces.shape[1] * _sample_forces.shape[2]
        )
        res_forces = sample_forces - forces_from_fc2
        self._sscha_forces = res_forces.reshape(res_forces.shape[0], -1, 3).transpose(
            0, 2, 1
        )

    def calc_sscha_stress_tensors(self):
        stress_indices = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]

        sample_forces = self._ph_real.forces
        sample_stresses = self._ph_real.stresses
        disps = self._ph_real.displacements
        # positions_cartesian = self._structure.axis @ self._structure.positions
        # disps += positions_cartesian

        self._sscha_stress_tensors = np.zeros((sample_stresses.shape[0], 6))
        for k, (i, j) in enumerate(stress_indices):
            correction = 0.5 * (
                np.sum(sample_forces[:, i, :] * disps[:, j, :], axis=1)
                + np.sum(sample_forces[:, j, :] * disps[:, i, :], axis=1)
            )
            self._sscha_stress_tensors[:, k] = sample_stresses[:, k] - correction

    def position_opt(self, basis_f):
        # sample_forces.shape = (N_samp, 3, N_atom)
        # fc2.shape = (N_atom, N_atom, 3, 3)
        _sample_forces = self.self._ph_real.forces.transpose(0, 2, 1)

        # fc2.shape = (N_atom*3, N_atom*3)
        # basis_f.shape = (N_atom*3, sym)
        N3 = self._fc2.shape[0] * self._fc2.shape[2]
        fc2 = np.transpose(self._fc2, (0, 2, 1, 3))
        fc2 = np.reshape(fc2, (N3, N3))
        fc2_sym = fc2 @ basis_f

        forces_from_fc2 = [
            self._forces_from_fc2(d.T.reshape(-1)) for d in self._ph_real.displacements
        ]
        forces_from_fc2 = np.array(forces_from_fc2)
        sample_forces = _sample_forces.reshape(
            _sample_forces.shape[0], _sample_forces.shape[1] * _sample_forces.shape[2]
        )
        xTx = np.zeros((fc2_sym.shape[1], fc2_sym.shape[1]))
        xTy = np.zeros(fc2_sym.shape[1])
        l2_norm = 0
        for i in range(forces_from_fc2.shape[0]):
            res_force = sample_forces[i] - forces_from_fc2[i]
            l2_norm += np.sum(res_force**2)
            xTx += fc2_sym.T @ fc2_sym
            xTy += fc2_sym.T @ res_force
        print("L2 norm =", l2_norm**0.5, flush=True)

        move_eq_position, _, _, _ = scipy.linalg.lstsq(xTx, xTy, check_finite=True)
        move_eq_position = (move_eq_position.reshape(1, -1) @ basis_f.T).reshape(-1)

        forces_from_fc2 = [
            self._forces_from_fc2(d.T.reshape(-1) - move_eq_position)
            for d in self._ph_real.displacements
        ]
        forces_from_fc2 = np.array(forces_from_fc2)
        l2_norm = 0
        for i in range(forces_from_fc2.shape[0]):
            res_force = sample_forces[i] - forces_from_fc2[i]
            l2_norm += np.sum(res_force**2)
        print("L2 norm (after optimization) =", l2_norm**0.5, flush=True)

        max_displacement = np.max(np.abs(move_eq_position))
        print("max_displecement =", max_displacement, "(Ang.)", flush=True)
        move_eq_position = move_eq_position.reshape(-1, 3).transpose(1, 0)
        move_eq_position = self._structure.axis_inv @ move_eq_position
        move_eq_position[np.abs(move_eq_position) < 1e-8] = 0

        return move_eq_position, max_displacement

    def _forces_from_fc2(self, disp):
        # disp.shape = (3, N_atom)
        N3 = self._fc2.shape[0] * self._fc2.shape[2]
        fc2 = np.transpose(self._fc2, (0, 2, 1, 3))
        fc2 = np.reshape(fc2, (N3, N3))
        return -fc2 @ disp

    def estimate_derivatives_standard_deviation(self):
        self.run()

        e = self._sscha_enegies
        f = self._sscha_forces  # (n_samples, 3, n_atom)
        s = self._sscha_stress_tensors  # (n_samples, 6)

        sd_e = np.sqrt(np.var(e))
        sd_f = np.sqrt(np.var(f, axis=0))
        sd_s = np.sqrt(np.var(s, axis=0))

        basis_f = construct_basis_fractional_coordinates(self._structure)
        if basis_f is not None:
            prod = (
                -sd_f.transpose(0, 2, 1) @ self._structure.axis
            )  # (n_samples, n_atom, 3)
            derivatives_f = basis_f.T @ prod.reshape(prod.shape[0], -1).T
        else:
            derivatives_f = None

        sigma = np.zeros((s.shape[0], 3, 3), dtype=s.dtype)
        sigma[:, 0, 0] = s[:, 0]  # xx
        sigma[:, 1, 1] = s[:, 1]  # yy
        sigma[:, 2, 2] = s[:, 2]  # zz
        sigma[:, 0, 1] = sigma[:, 1, 0] = s[:, 3]  # xy
        sigma[:, 1, 2] = sigma[:, 2, 1] = s[:, 4]  # yz
        sigma[:, 0, 2] = sigma[:, 2, 0] = s[:, 5]  # zx

        basis_axis, _ = construct_basis_cell(self._structure)
        self._structure.axis_inv = np.linalg.inv(self._structure.axis)
        prod = -sigma @ self._structure.axis_inv.T
        derivatives_s = basis_axis.T @ prod.reshape(prod.shape[0], -1).T

        if derivatives_f is None:
            sd_derivatives_f = None
        else:
            sd_derivatives_f = np.sqrt(np.var(derivatives_f, axis=1))
        sd_derivatives_s = np.sqrt(np.var(derivatives_s, axis=1))

        return sd_e, sd_f, sd_s, sd_derivatives_f, sd_derivatives_s

    @property
    def sscha_energy(self):
        if self._sscha_enegies is None:
            return None
        sscha_energy = np.average(self._sscha_enegies)
        return sscha_energy

    @property
    def sscha_force(self):
        if self._sscha_forces is None:
            return None
        sscha_force = np.mean(self._sscha_forces, axis=0)
        return sscha_force

    @property
    def sscha_stress_tensor(self):
        if self._sscha_stress_tensors is None:
            return None
        sscha_stress_tensor = np.mean(self._sscha_stress_tensors, axis=0)
        return sscha_stress_tensor
