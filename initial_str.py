import argparse
import glob
import math
import os

import numpy as np
from go_tools import variable
from scipy.linalg import cholesky


class generate_initial_structure:

    def __init__(
        self,
        elements: list = ['Z'],
        n_atoms: int = 4,
        max_str: int = 100,
        atomic_length: float = 3.0,
        least_distance: float = 0.5,
        pre_str_count: int = 0,
    ):
        self.elements = elements
        self.n_atoms = n_atoms
        self.max_str = max_str
        self.atomic_length = atomic_length
        self.least_distance = least_distance * self.atomic_length
        self.str_count = pre_str_count

    def random_structure(self):
        atom_num = sum(self.n_atoms)
        vol_up = atom_num * (4 * math.pi * ((self.atomic_length / 2) ** 3) / 3) * 100 / 10
        vol_under = atom_num * 4 * math.pi * ((self.atomic_length / 2) ** 3) / 3
        vol_under_root = (vol_under ** (1 / 3)) ** 2
        vol_up_root = (vol_up ** (1 / 3)) ** 2
        vol_diff = vol_up_root - vol_under_root

        penalty = 0
        iteration = 1
        num_samples = 1000
        while True:
            print(f'----- Iteration {iteration} -----')
            rand_g123 = np.sort(np.random.rand(num_samples, 3), axis=1)
            # rand_g123 = penalty / 100 + (rand_g123 * (1 - penalty / 100))
            rand_g456 = np.random.rand(num_samples, 3)
            random_num_set = np.concatenate([rand_g123, rand_g456], axis=1)

            # Niggli reduced cell
            G1 = vol_under_root + random_num_set[:, 0] * vol_diff
            G2 = vol_under_root + random_num_set[:, 1] * vol_diff
            G3 = vol_under_root + random_num_set[:, 2] * vol_diff
            G4 = -G1 / 2 + random_num_set[:, 3] * G1
            G5 = random_num_set[:, 4] * G1 / 2
            G6 = random_num_set[:, 5] * G2 / 2
            G_sets = np.stack([G1, G4, G5, G4, G2, G6, G5, G6, G3], axis=1)
            valid_g_sets = G_sets[(G1 + G2 + 2 * G4) >= (2 * G5 + 2 * G6)]
            sym_g_sets = valid_g_sets.reshape(valid_g_sets.shape[0], 3, 3)
            print(f'< generate {sym_g_sets.shape[0]} random structures >')

            # Obtaining axis values (axis.T) and random atomic positions
            L_matrices = np.array([cholesky(mat, lower=False) for mat in sym_g_sets])
            fixed_position = np.zeros([valid_g_sets.shape[0], 3, 1])
            random_atomic_postion = np.random.rand(valid_g_sets.shape[0], 3, (atom_num - 1))
            random_atomic_postion = np.concatenate([fixed_position, random_atomic_postion], axis=2)

            # Screening the structures based on interatomic distances
            dist_sets = np.array(
                [
                    self.nearest_neighbor_atomic_distance(lat, coo)
                    for lat, coo in zip(L_matrices, random_atomic_postion)
                ]
            )
            valid_l_matrices = L_matrices[dist_sets > self.least_distance]
            valid_positons = random_atomic_postion[dist_sets > self.least_distance]
            print(f'< screened {valid_l_matrices.shape[0]} random structures >')

            for axis, positions in zip(valid_l_matrices, valid_positons):
                self.str_count += 1
                self.write_poscar(axis, positions, f"initial_str/POSCAR_{self.str_count}")
                if self.str_count == self.max_str:
                    return
            iteration += 1
            penalty += 1

    def nearest_neighbor_atomic_distance(self, lat, coo):
        cartesian_coo = lat @ coo
        c1 = cartesian_coo

        image_x, image_y, image_z = np.meshgrid(
            np.arange(-1, 1.1), np.arange(-1, 1.1), np.arange(-1, 1.1), indexing="ij"
        )
        image_matrix = (
            np.stack([image_x, image_y, image_z], axis=-1).reshape(-1, 3).T
        )  # (3, num_images)

        parallel_move = lat @ image_matrix
        parallel_move = np.tile(
            parallel_move[:, None, :], (1, c1.shape[-1], 1)
        )  # (3, N, num_images)
        c2_all = cartesian_coo[:, :, None] + parallel_move
        z = (c1[:, None, :, None] - c2_all[:, :, None, :]) ** 2  # (3, N, N, num_images)

        _dist_mat = np.sqrt(np.sum(z, axis=0))  # (N, N, num_images)
        dist_mat = np.min(_dist_mat, axis=-1)  # (N, N)

        dist_mat_refine = np.where(dist_mat > 1e-10, dist_mat, np.inf)
        distance_min = np.min(dist_mat_refine)
        if np.isinf(distance_min):
            _dist_mat = np.where(_dist_mat > 1e-10, _dist_mat, np.inf)
            distance_min = np.min(_dist_mat)

        return distance_min

    def write_poscar(self, axis, positions, filename="POSCAR"):
        """Write structure in POSCAR file."""
        with open(filename, "w") as f:
            print("Initial structure for RSS", file=f)
            print("1.0", file=f)
            for n in axis.T:
                print(
                    "  ",
                    "{0:15.15f}".format(n[0]),
                    "{0:15.15f}".format(n[1]),
                    "{0:15.15f}".format(n[2]),
                    file=f,
                )
            for idx, n in enumerate(self.n_atoms):
                if n > 0:
                    print(self.elements[idx], " ", end="", file=f)
                else:
                    print("Z  ", end="", file=f)
            print("", file=f)
            for n in self.n_atoms:
                print(n, " ", end="", file=f)
            print("", file=f)
            print("Direct", file=f)
            for n in positions.T:
                print(
                    "  ",
                    "{0:15.15f}".format(float(n[0])),
                    "{0:15.15f}".format(float(n[1])),
                    "{0:15.15f}".format(float(n[2])),
                    file=f,
                )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=str, nargs="+", default=None)
    parser.add_argument("--n_atoms", type=int, nargs="+", default=None)
    parser.add_argument("--max_str", type=int, default=1000)
    parser.add_argument("--least_distance", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs("initial_str", exist_ok=True)
    max_str = args.max_str
    pre_str_count = len(glob.glob("initial_str/*"))

    if max_str > pre_str_count:
        elements = args.elements
        atomic_length = None
        for element in elements:
            _atomic_length = variable.atom_variable(element)
            if atomic_length is None:
                atomic_length = _atomic_length
            elif atomic_length < _atomic_length:
                atomic_length = _atomic_length

        gen_str = generate_initial_structure(
            elements,
            args.n_atoms,
            max_str,
            atomic_length=atomic_length,
            least_distance=args.least_distance,
            pre_str_count=pre_str_count,
        )
        gen_str.random_structure()
