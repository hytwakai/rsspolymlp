import collections
import math

import numpy as np

from lammps import lammps


class LammpsUtil:

    def __init__(
        self,
        x: list,
        element: str,
        pot: list[str] = [],
        other_pot=False,
        verbose=False,
        logfile_name=None,
    ):

        if verbose is True:
            args = []
        else:
            args = ["-sc", "none"]

        # lammps str
        lx = x[0]
        xy = x[1] * math.cos(x[5] * math.pi / 180)
        if x[1] ** 2 - xy**2 > 0:
            ly = math.sqrt(x[1] ** 2 - xy**2)
        else:
            ly = np.nan
        xz = x[2] * math.cos(x[4] * math.pi / 180)
        yz = (x[1] * x[2] * math.cos(x[3] * math.pi / 180) - xy * xz) / ly
        if x[2] ** 2 - xz**2 - yz**2 > 0:
            lz = math.sqrt(x[2] ** 2 - xz**2 - yz**2)
        else:
            lz = np.nan
        if np.isnan(lz) is True or np.isnan(ly) is True:
            raise ValueError("Abnormal input structure.")

        lmp = lammps(cmdargs=args)

        if logfile_name is not None:
            lmp.command(f"log {str(logfile_name)}")
        else:
            lmp.command("log log.equil")
        lmp.command("thermo 100")
        lmp.command("atom_style atomic")
        lmp.command("units metal")
        lmp.command("boundary p p p")
        lmp.command("box tilt large")
        lmp.command("neigh_modify every 1 delay 0 check yes one 100000 page 1000000")

        lmp.command(f"region region1 prism 0 {lx} 0 {ly} 0 {lz} {xy} {xz} {yz}")
        lmp.command("create_box 1 region1")

        if not int(len(x) / 3) - 2 == 0:
            for i in range(int(len(x) / 3) - 2):
                lmp.command(
                    "create_atoms 1 single "
                    + str(x[3 * i + 6] * lx + x[3 * i + 7] * xy + x[3 * i + 8] * xz)
                    + " "
                    + str(x[3 * i + 7] * ly + x[3 * i + 8] * yz)
                    + " "
                    + str(x[3 * i + 8] * lz)
                    + " remap yes"
                )

        lmp.command("mass * " + str(self.lammps_mass(element)))
        self.lmp = lmp

        if not other_pot:
            if len(pot) > 1:
                lmp.command("pair_style hybrid/overlay polymlp polymlp")
                lmp.command(
                    "pair_coeff * * polymlp 1 " + str(pot[0]) + " " + str(element)
                )
                lmp.command(
                    "pair_coeff * * polymlp 2 " + str(pot[1]) + " " + str(element)
                )
            else:
                lmp.command("pair_style polymlp")
                lmp.command("pair_coeff * * " + str(pot[0]) + " " + str(element))

    def minimize(
        self,
        box=False,
        count_maxiter=10000,
        count_maxeval=100000,
        press=0,  # bar (10^5 Pa)
        ftol=1e-6,
    ):
        lmp = self.lmp
        if box == "tri":
            lmp.command("fix f1 all box/relax tri " + str(press))
        elif box == "iso":
            lmp.command("fix f1 all box/relax iso " + str(press))
        elif box == "aniso":
            lmp.command("fix f1 all box/relax aniso " + str(press))
        lmp.command(f"minimize 0.0 {ftol} {int(count_maxiter)} {int(count_maxeval)}")

    def get_results(self):
        lmp = self.lmp

        lmp.command("run 0")
        lmp.command("thermo 1")
        lmp.command(
            "thermo_style custom step temp pe press " + "pxx pyy pzz pxy pxz pyz"
        )
        lmp.command("thermo_modify norm no")
        lmp.command("variable energy equal pe")
        lmp.command("variable pxx0 equal pxx")
        lmp.command("variable pyy0 equal pyy")
        lmp.command("variable pzz0 equal pzz")
        lmp.command("variable pyz0 equal pyz")
        lmp.command("variable pxz0 equal pxz")
        lmp.command("variable pxy0 equal pxy")
        lmp.command("run 0")

        n = lmp.get_natoms()

        energy = lmp.extract_variable("energy", "all", 0)
        energy = energy / n
        forces = lmp.extract_atom("f", 3)
        forces = np.array(
            [[forces[i][0], forces[i][1], forces[i][2]] for i in range(n)]
        )

        pxx = self.lmp.extract_variable("pxx0", None, 0)
        pyy = self.lmp.extract_variable("pyy0", None, 0)
        pzz = self.lmp.extract_variable("pzz0", None, 0)
        pyz = self.lmp.extract_variable("pyz0", None, 0)
        pxz = self.lmp.extract_variable("pxz0", None, 0)
        pxy = self.lmp.extract_variable("pxy0", None, 0)
        stress = np.array([pxx, pyy, pzz, pyz, pxz, pxy])

        struct = self.get_structure()
        struct[6:] = struct[6:] % 1.0

        lmp.close()

        return energy, forces, stress, struct

    def get_structure(self):
        lmp = self.lmp
        box = lmp.extract_box()
        xlo, ylo, zlo = box[0]
        xhi, yhi, zhi = box[1]
        lx = xhi - xlo
        ly = yhi - ylo
        lz = zhi - zlo
        xy, yz, xz = box[2:5]

        n = lmp.get_natoms()
        # n_atomtypes = lmp.extract_global("ntypes", 0)

        types = lmp.extract_atom("type", 0)
        types1 = [types[i] - 1 for i in range(n)]
        n_atoms = collections.Counter(types1)
        n_atoms = list(n_atoms.values())

        cds = lmp.extract_atom("x", 3)
        positions_c = [[cds[i][0], cds[i][1], cds[i][2]] for i in range(n)]
        positions_c = np.array(
            [np.array(pos) - np.array(box[0]) for pos in positions_c]
        ).T
        positions_c[2] = positions_c[2] / lz
        positions_c[1] = (positions_c[1] - positions_c[2] * yz) / ly
        positions_c[0] = (
            positions_c[0] - xy * positions_c[1] - xz * positions_c[2]
        ) / lx
        positions_c = positions_c.T

        a_scalar = lx
        b_scalar = math.sqrt(xy**2 + ly**2)
        c_scalar = math.sqrt(xz**2 + yz**2 + lz**2)
        ab_angle = math.degrees(math.acos(xy / b_scalar))
        bc_angle = math.degrees(math.acos((xy * xz + ly * yz) / (b_scalar * c_scalar)))
        ca_angle = math.degrees(math.acos(xz / c_scalar))

        axis_abc = np.array(
            [a_scalar, b_scalar, c_scalar, bc_angle, ca_angle, ab_angle]
        )
        struct = np.append(axis_abc, positions_c)
        return struct

    def lammps_mass(self, element):
        mass_dict = {
            "Ag": 107.87,
            "Al": 26.98,
            "As": 74.92,
            "Au": 196.97,
            "Ba": 137.33,
            "Be": 9.01,
            "Bi": 208.98,
            "Ca": 40.08,
            "Cd": 112.4,
            "Cr": 52.0,
            "Cs": 132.9,
            "Cu": 63.55,
            "Ga": 69.72,
            "Ge": 72.64,
            "Hf": 178.49,
            "Hg": 200.6,
            "In": 114.82,
            "Ir": 192.2,
            "K": 39.1,
            "La": 138.9,
            "Li": 6.94,
            "Mg": 24.31,
            "Mo": 95.95,
            "Na": 22.99,
            "Nb": 92.91,
            "Os": 190.2,
            "P": 30.97,
            "Pb": 207.2,
            "Pd": 106.4,
            "Pt": 195.1,
            "Rb": 85.47,
            "Re": 186.2,
            "Rh": 102.91,
            "Ru": 101.1,
            "Sb": 121.76,
            "Sc": 45.0,
            "Si": 28.0855,
            "Sn": 118.7,
            "Sr": 87.6,
            "Ta": 180.95,
            "Te": 127.6,
            "Ti": 47.87,
            "Tl": 204.4,
            "V": 50.94,
            "W": 183.84,
            "Y": 88.9,
            "Zn": 65.4,
            "Zr": 91.22,
        }

        return mass_dict[element]
