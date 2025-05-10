import glob
import os

import numpy as np
from scipy.linalg import cholesky

from rsspolymlp.common.parse_arg import ParseArgument


def run():
    args = ParseArgument.get_initial_structure_args()
    os.makedirs("initial_struct", exist_ok=True)

    # Generate new structures if necessary
    pre_str_count = len(glob.glob("initial_struct/*"))
    if args.num_init_str > pre_str_count:
        gen_str = GenerateRandomStructure(
            args.elements,
            args.atom_counts,
            args.num_init_str,
            least_distance=args.least_distance,
            pre_str_count=pre_str_count,
        )
        gen_str.random_structure(min_volume=args.min_volume, max_volume=args.max_volume)


def nearest_neighbor_atomic_distance(lattice, frac_coo):
    """
    Calculate the nearest neighbor atomic distance within a periodic lattice.

    Parameters
    ----------
    lattice : ndarray (3,3)
        Lattice matrix.
    frac_coo : ndarray (3, N)
        Atomic coordinates in fractional coordinates.

    Returns
    -------
    distance_min : float
        Minimum atomic distance considering periodic boundary conditions.
    """

    cartesian_coo = lattice @ frac_coo
    c1 = cartesian_coo

    # Generate periodic image translations along x, y, and z
    image_x, image_y, image_z = np.meshgrid(
        np.arange(-1, 1.1), np.arange(-1, 1.1), np.arange(-1, 1.1), indexing="ij"
    )
    image_matrix = (
        np.stack([image_x, image_y, image_z], axis=-1).reshape(-1, 3).T
    )  # (3, num_images)

    # Compute the translations due to periodic images
    parallel_move = lattice @ image_matrix
    parallel_move = np.tile(
        parallel_move[:, None, :], (1, c1.shape[-1], 1)
    )  # (3, N, num_images)
    c2_all = cartesian_coo[:, :, None] + parallel_move

    # Compute squared distances between all pairs of atoms in all periodic images
    z = (c1[:, None, :, None] - c2_all[:, :, None, :]) ** 2  # (3, N, N, num_images)
    _dist_mat = np.sqrt(np.sum(z, axis=0))  # (N, N, num_images)

    # Find the minimum distance for each pair
    dist_mat = np.min(_dist_mat, axis=-1)  # (N, N)
    dist_mat_refine = np.where(dist_mat > 1e-10, dist_mat, np.inf)
    distance_min = np.min(dist_mat_refine)

    # Handle self-interaction case
    if np.isinf(distance_min):
        _dist_mat = np.where(_dist_mat > 1e-10, _dist_mat, np.inf)
        distance_min = np.min(_dist_mat)

    return distance_min


class GenerateRandomStructure:
    """Class for creating initial random structures for RSS."""

    def __init__(
        self,
        elements,
        atom_counts,
        max_str: int = 5000,
        least_distance: float = 0.0,
        pre_str_count: int = 0,
    ):
        """
        Initialize the structure generation parameters.

        Parameters
        ----------
        elements : list
            List of element symbols.
        atom_counts : list
            List of the number of atoms for each element.
        max_str : int, optional
            Maximum number of structures to generate (default: 5000).
        least_distance : float, optional
            Minimum allowed atomic distance in unit of angstrom (default: 0.0).
        pre_str_count : int, optional
            Initial structure count.
        """

        self.elements = elements
        self.atom_counts = atom_counts
        self.max_str = max_str
        self.least_distance = least_distance
        self.str_count = pre_str_count

    def random_structure(self, min_volume=0, max_volume=100):
        """
        Generate random structures while ensuring minimum interatomic distance constraints.
        """
        atom_num = sum(self.atom_counts)

        # Define initial structure constraints
        vol_constraint_max = max_volume * atom_num  # A^3
        vol_constraint_min = min_volume * atom_num  # A^3
        axis_constraint = ((atom_num ** (1 / 3)) * 8) ** 2

        iteration = 1
        num_samples = 2000
        while True:
            print(f"----- Iteration {iteration} -----")

            # Define volume constraints based on atomic packing fraction
            rand_g123 = np.sort(np.random.rand(num_samples, 3), axis=1)
            rand_g456 = np.random.rand(num_samples, 3)
            random_num_set = np.concatenate([rand_g123, rand_g456], axis=1)

            # Construct valid Niggli-reduced cells
            G1 = random_num_set[:, 0] * axis_constraint
            G2 = random_num_set[:, 1] * axis_constraint
            G3 = random_num_set[:, 2] * axis_constraint
            G4 = -G1 / 2 + random_num_set[:, 3] * G1
            G5 = random_num_set[:, 4] * G1 / 2
            G6 = random_num_set[:, 5] * G2 / 2
            G_sets = np.stack([G1, G4, G5, G4, G2, G6, G5, G6, G3], axis=1)
            valid_g_sets = G_sets[(G1 + G2 + 2 * G4) >= (2 * G5 + 2 * G6)]
            sym_g_sets = valid_g_sets.reshape(valid_g_sets.shape[0], 3, 3)
            print(f"< generate {sym_g_sets.shape[0]} random structures >")

            # Convert lattice tensors to Cartesian lattice matrices
            L_matrices = np.array([cholesky(mat, lower=False) for mat in sym_g_sets])
            volumes = np.abs(np.linalg.det(L_matrices))
            valid_l_matrices = L_matrices[
                (volumes >= vol_constraint_min) & (volumes <= vol_constraint_max)
            ]
            fixed_position = np.zeros([valid_l_matrices.shape[0], 3, 1])
            random_atomic_position = np.random.rand(
                valid_l_matrices.shape[0], 3, (atom_num - 1)
            )
            valid_positions = np.concatenate(
                [fixed_position, random_atomic_position], axis=2
            )
            print(
                f"< screened {valid_l_matrices.shape[0]} random structures (volume) >"
            )

            if self.least_distance > 0:
                # Filter structures based on interatomic distance constraints
                distance_sets = np.array(
                    [
                        nearest_neighbor_atomic_distance(lat, coo)
                        for lat, coo in zip(valid_l_matrices, random_atomic_position)
                    ]
                )
                valid_l_matrices = valid_l_matrices[distance_sets > self.least_distance]
                valid_positions = valid_positions[distance_sets > self.least_distance]
                print(
                    f"< screened {valid_l_matrices.shape[0]} random structures (least distance) >"
                )

            # Save valid structures
            for axis, positions in zip(valid_l_matrices, valid_positions):
                self.str_count += 1
                self.write_poscar(
                    axis, positions, f"initial_struct/POSCAR_{self.str_count}"
                )
                if self.str_count == self.max_str:
                    return
            iteration += 1

    def write_poscar(self, axis, positions, filename="POSCAR"):
        """Write the generated structure to a VASP POSCAR file."""
        with open(filename, "w") as f:
            print("Generated by rss_polymlp", file=f)
            print("1.0", file=f)
            for n in axis.T:
                print(
                    "  ",
                    "{0:15.15f}".format(n[0]),
                    "{0:15.15f}".format(n[1]),
                    "{0:15.15f}".format(n[2]),
                    file=f,
                )
            for idx, n in enumerate(self.atom_counts):
                if n > 0:
                    print(self.elements[idx], " ", end="", file=f)
                else:
                    print("Z  ", end="", file=f)
            print("", file=f)
            for n in self.atom_counts:
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
    run()
