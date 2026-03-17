from typing import Optional

import numpy as np
import scipy

from phonopy import Phonopy
from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.sscha.harmonic_reciprocal import HarmonicReciprocal
from pypolymlp.calculator.sscha.sscha_data import SSCHAData
from pypolymlp.calculator.sscha.sscha_io import save_sscha_yaml
from pypolymlp.calculator.sscha.sscha_params import SSCHAParams
from pypolymlp.calculator.sscha.sscha_restart import Restart
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.units import EVtoKJmol
from pypolymlp.utils.phonopy_utils import (
    phonopy_cell_to_structure,
    structure_to_phonopy_cell,
)
from pypolymlp.utils.spglib_utils import construct_basis_cell
from pypolymlp.utils.symfc_utils import construct_basis_fractional_coordinates
from rsspolymlp.utils.opt_sscha.harmonic_real import HarmonicReal


class SSCHAProperty:

    def __init__(
        self,
        yamlfile: str,  # ./sscha_results.yaml
        fc2file: str,  # ./fc2.hdf5
        cell: Optional[PolymlpStructure] = None,
        pot: Optional[list] = None,
        n_samples: Optional[int] = 1000,
        pressure: Optional[float] = 0,
    ):
        self._fc2file = fc2file
        self._n_samples = n_samples
        self._pressure = pressure

        self._res = Restart(yamlfile, fc2hdf5=self._fc2file, pot=pot)
        if cell is None:
            self._structure = self._res.unitcell
        else:
            self._structure = cell
        self._fc2 = self._res.force_constants
        self._parameters = self._res.parameters
        self._supercell_matrix = self._res.supercell_matrix
        self._n_unitcells = int(round(np.linalg.det(self._supercell_matrix)))
        self._temperature = self._res.temperature
        self._ev_to_kjmol = EVtoKJmol / self._res.n_unitcells

        self._prop = Properties(pot=self._res.polymlp)
        self._sscha_enegies = None
        self._sscha_forces = None
        self._sscha_stress_tensors = None

    def run(self, save_data_file: Optional[str] = None):
        self._phonopy = Phonopy(
            structure_to_phonopy_cell(self._structure),
            self._supercell_matrix,
        )
        supercell_polymlp = phonopy_cell_to_structure(self._phonopy.supercell)
        supercell_polymlp.masses = self._phonopy.supercell.masses
        supercell_polymlp.supercell_matrix = self._supercell_matrix
        supercell_polymlp.n_unitcells = self._n_unitcells

        self._ph_real = HarmonicReal(
            supercell_polymlp,
            self._prop,
            n_unitcells=self._n_unitcells,
            fc2=self._fc2,
        )
        self._ph_real.run(temp=self._temperature, n_samples=self._n_samples)

        self._ph_recip = HarmonicReciprocal(
            self._phonopy,
            self._prop,
            fc2=self._fc2,
        )
        self._ph_recip.compute_thermal_properties(
            temp=self._temperature, qmesh=(10, 10, 10)
        )

        self.calc_sscha_energies()
        self.calc_sscha_forces()
        self.calc_sscha_stress_tensors()

        if save_data_file is not None:
            sscha_data = SSCHAData(
                temperature=self._temperature,
                static_potential=self._ph_real.static_potential,  # kJ/mol
                harmonic_potential=self._ph_real.average_harmonic_potential,  # kJ/mol
                harmonic_free_energy=self._ph_recip.free_energy,  # kJ/mol
                average_potential=self._ph_real.average_full_potential,  # kJ/mol
                anharmonic_free_energy=self._ph_real.average_anharmonic_potential,
                entropy=self._ph_recip.entropy,  # J/K/mol
                harmonic_heat_capacity=self._ph_recip.heat_capacity,  # J/K/mol
                static_forces=self._ph_real.static_forces,  # eV/ang
                average_forces=self._ph_real.average_forces,  # eV/ang
            )
            sscha_params = SSCHAParams(
                unitcell=self._structure,
                supercell_matrix=self._supercell_matrix,
                supercell=supercell_polymlp,
                pot=self._res.polymlp,
                temp=self._parameters["temperature"],
                n_samples_init=self._n_samples,
                n_samples_final=None,
                tol=None,
                max_iter=None,
                mixing=None,
                mesh=self._parameters["mesh_phonon"],
                init_fc_algorithm="file",
                init_fc_file=self._fc2file,
            )
            save_sscha_yaml(sscha_params, [sscha_data], filename=save_data_file)

    def calc_sscha_energies(self):
        free_energies = self._ph_recip.free_energy + self._ph_real.anharmonic_potentials
        self._sscha_enegies = (
            free_energies + self._ph_real.static_potential
        ) / self._ev_to_kjmol

    def calc_sscha_forces(self):
        # displacements.shape = (N_samples, 3, N_atom)
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
        f_fc2 = [
            self._forces_from_fc2(d.T.reshape(-1)) for d in self._ph_real.displacements
        ]
        f_fc2 = np.array(f_fc2)
        f_fc2 = f_fc2.reshape(f_fc2.shape[0], f_fc2.shape[1] // 3, 3).transpose(
            0, 2, 1
        )  # (N_samples, 3, N_atom)
        sample_stresses = self._ph_real.stresses
        disps = self._ph_real.displacements

        self._sscha_stress_tensors = np.zeros((sample_stresses.shape[0], 6))
        stress_indices = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]
        for k, (i, j) in enumerate(stress_indices):
            correction = 0.5 * (
                np.sum(f_fc2[:, i, :] * disps[:, j, :], axis=1)
                + np.sum(f_fc2[:, j, :] * disps[:, i, :], axis=1)
            )
            self._sscha_stress_tensors[:, k] = sample_stresses[:, k] - correction

    def position_opt(self):
        # basis_f.shape: (N_atom*3, N_sym)
        # fc2.shape: (N_atom, N_atom, 3, 3) -> (N_atom*3, N_atom*3)
        basis_f = construct_basis_fractional_coordinates(self._structure)
        N3 = self._fc2.shape[0] * self._fc2.shape[2]
        fc2 = np.transpose(self._fc2, (0, 2, 1, 3))
        fc2 = np.reshape(fc2, (N3, N3))
        fc2_sym = fc2 @ basis_f

        # sample_forces.shape: (N_samples, 3, N_atom)
        sample_forces = self._ph_real.forces.transpose(0, 2, 1)
        sample_forces = sample_forces.reshape(
            sample_forces.shape[0], sample_forces.shape[1] * sample_forces.shape[2]
        )
        forces_from_fc2 = [
            self._forces_from_fc2(d.T.reshape(-1)) for d in self._ph_real.displacements
        ]
        forces_from_fc2 = np.array(forces_from_fc2)

        res_force = np.mean(sample_forces - forces_from_fc2, axis=0)
        l2_norm = np.sum(res_force**2)
        print("L2 norm =", l2_norm**0.5, flush=True)

        xTx = fc2_sym.T @ fc2_sym
        xTy = fc2_sym.T @ res_force

        pos_coeff, _, _, _ = scipy.linalg.lstsq(xTx, xTy, check_finite=True)
        mv_poseq_ang = (pos_coeff @ basis_f.T).reshape(-1)

        forces_from_fc2 = [
            self._forces_from_fc2(d.T.reshape(-1) - mv_poseq_ang)
            for d in self._ph_real.displacements
        ]
        forces_from_fc2 = np.array(forces_from_fc2)
        res_force = np.mean(sample_forces - forces_from_fc2, axis=0)
        l2_norm = np.sum(res_force**2)
        print("L2 norm (after optimization) =", l2_norm**0.5, flush=True)

        max_disp_ang = np.max(np.abs(mv_poseq_ang))
        print("max_displecement =", max_disp_ang, "(Ang.)", flush=True)

        mv_poseq_ang = mv_poseq_ang.reshape(-1, 3).transpose(1, 0)
        mv_poseq_frac = self._structure.axis_inv @ mv_poseq_ang
        mv_poseq_frac[np.abs(mv_poseq_ang) < 1e-8] = 0
        print("mv_poseq_frac:")
        print(mv_poseq_frac)

        return mv_poseq_frac, max_disp_ang

    def _forces_from_fc2(self, disp):
        N3 = self._fc2.shape[0] * self._fc2.shape[2]
        fc2 = np.transpose(self._fc2, (0, 2, 1, 3))
        fc2 = np.reshape(fc2, (N3, N3))
        return -fc2 @ disp  # (N_atom*3,)

    def estimate_derivatives_standard_deviation(self):
        self.run()

        e = self._sscha_enegies
        f = self._sscha_forces  # (N_samples, 3, N_atom)
        s = self._sscha_stress_tensors  # (N_samples, 6)

        sd_e = np.sqrt(np.var(e))
        sd_f = np.sqrt(np.var(f, axis=0))
        sd_s = np.sqrt(np.var(s, axis=0))

        basis_f = construct_basis_fractional_coordinates(self._structure)
        if basis_f is not None:
            prod = -f.transpose(0, 2, 1)  # (N_samples, N_atom, 3)
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
