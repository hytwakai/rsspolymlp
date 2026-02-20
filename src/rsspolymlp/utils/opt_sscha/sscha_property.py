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
    ):
        self._structure = cell
        self._res = Restart(yamlfile, fc2hdf5=fc2file, pot=pot)
        self._fc2 = self._res.force_constants
        self._ev_to_kjmol = EVtoKJmol / self._res.n_unitcells

        prop = Properties(pot=self._res.polymlp)
        self._ph_real = HarmonicReal(
            self._structure,
            prop,
            n_unitcells=self._res.n_unitcells,
            fc2=self._fc2,
        )
        self._ph_real.run(temp=self._res.temperature, n_samples=n_samples)

        self._phonopy = Phonopy(
            structure_to_phonopy_cell(self._structure),
            self._res.supercell_matrix,
            nac_params=None,
        )
        self._ph_recip = HarmonicReciprocal(
            self._phonopy,
            prop,
            fc2=self._fc2,
        )
        self._ph_recip.compute_thermal_properties(
            temp=self._res.temperature, qmesh=(10, 10, 10)
        )

    def sscha_energy(self, pressure):
        free_energy = (
            self._ph_recip.free_energy + self._ph_real.average_anharmonic_potential
        )
        sscha_energy = (
            free_energy + self._ph_real.static_potential
        ) / self._ev_to_kjmol
        sscha_energy += pressure * self._structure.volume / EVtoGPa
        return sscha_energy

    def sscha_force(self):
        forces_from_fc2 = [
            self._forces_from_fc2(d.T.reshape(-1)) for d in self._ph_real.displacements
        ]
        forces_from_fc2 = np.array(forces_from_fc2)
        _sample_forces = self._ph_real.forces.transpose(0, 2, 1)
        sample_forces = _sample_forces.reshape(
            _sample_forces.shape[0], _sample_forces.shape[1] * _sample_forces.shape[2]
        )
        res_forces = sample_forces - forces_from_fc2
        sscha_force = np.mean(res_forces, axis=0)
        sscha_force = sscha_force.reshape(-1, 3).transpose(1, 0)
        return sscha_force

    def sscha_stress(self):
        stress_indices = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]
        sscha_stress = []
        _sample_forces = self._ph_real.forces
        _stresses = self._ph_real._stresses
        disps = self._ph_real.displacements
        positions_cartesian = self._structure.axis @ self._structure.positions
        disps += positions_cartesian
        for i, j in stress_indices:
            sscha_stress.append(
                np.mean(
                    _stresses[:, stress_indices.index((i, j))]
                    - 0.5
                    * (
                        np.sum(_sample_forces[:, i, :] * disps[:, j, :], axis=1)
                        + np.sum(_sample_forces[:, j, :] * disps[:, i, :], axis=1)
                    )
                )
            )
        return np.array(sscha_stress)

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
        e = self._ph_real.full_potentials / self._ev_to_kjmol
        f = self._ph_real.forces  # (n_samples, 3, n_atom)
        s = self._ph_real.stresses  # (n_samples, 6)

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
