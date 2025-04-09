import numpy as np
from scipy.linalg import cholesky


def nearest_neighbor_atomic_distance(lat, coo):
    """
    Calculate the nearest neighbor atomic distance within a periodic lattice.

    Parameters
    ----------
    lat : ndarray (3,3)
        Lattice matrix.
    coo : ndarray (3, N)
        Atomic coordinates in fractional coordinates.

    Returns
    -------
    distance_min : float
        Minimum atomic distance considering periodic boundary conditions.
    """

    cartesian_coo = lat @ coo
    c1 = cartesian_coo

    # Generate periodic image translations along x, y, and z
    image_x, image_y, image_z = np.meshgrid(
        np.arange(-1, 1.1), np.arange(-1, 1.1), np.arange(-1, 1.1), indexing="ij"
    )
    image_matrix = (
        np.stack([image_x, image_y, image_z], axis=-1).reshape(-1, 3).T
    )  # (3, num_images)

    # Compute the translations due to periodic images
    parallel_move = lat @ image_matrix
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


class GenerateInitialStructure:
    """Class for creating initial random structures for RSS."""

    def __init__(
        self,
        elements,
        n_atoms,
        max_str: int = 100,
        least_distance: float = 0.5,
        pre_str_count: int = 0,
        penalty: bool = False,
    ):
        """
        Initialize the structure generation parameters.

        Parameters
        ----------
        elements : list
            List of element symbols.
        n_atoms : list
            List of the number of atoms for each element.
        max_str : int, optional
            Maximum number of structures to generate (default: 100).
        least_distance : float, optional
            Minimum allowed atomic distance in unit of angstrom (default: 0.5).
        pre_str_count : int, optional
            Initial structure count (default: 0).
        penalty : bool, optional
            Enables a penalty term to bias the generation of lattice parameters (default: False).
        """

        self.elements = elements
        self.n_atoms = n_atoms
        self.max_str = max_str
        self.least_distance = least_distance
        self.str_count = pre_str_count
        self.penalty = penalty

    def random_structure(self):
        """
        Generate random structures while ensuring minimum interatomic distance constraints.
        """
        atom_num = sum(self.n_atoms)

        # Define volume constraints
        vol_up = 100 * atom_num  # A^3
        vol_up_root = (vol_up ** (1 / 3)) ** 2

        penalty = 0
        iteration = 1
        num_samples = 1000
        while True:
            print(f"----- Iteration {iteration} -----")

            # Define volume constraints based on atomic packing fraction
            rand_g123 = np.sort(np.random.rand(num_samples, 3), axis=1)
            if not self.penalty:
                rand_g123 = penalty / 100 + (rand_g123 * (1 - penalty / 100))
            rand_g456 = np.random.rand(num_samples, 3)
            random_num_set = np.concatenate([rand_g123, rand_g456], axis=1)

            # Construct valid Niggli-reduced cells
            G1 = random_num_set[:, 0] * vol_up_root
            G2 = random_num_set[:, 1] * vol_up_root
            G3 = random_num_set[:, 2] * vol_up_root
            G4 = -G1 / 2 + random_num_set[:, 3] * G1
            G5 = random_num_set[:, 4] * G1 / 2
            G6 = random_num_set[:, 5] * G2 / 2
            G_sets = np.stack([G1, G4, G5, G4, G2, G6, G5, G6, G3], axis=1)
            valid_g_sets = G_sets[(G1 + G2 + 2 * G4) >= (2 * G5 + 2 * G6)]
            sym_g_sets = valid_g_sets.reshape(valid_g_sets.shape[0], 3, 3)
            print(f"< generate {sym_g_sets.shape[0]} random structures >")

            # Convert lattice tensors to Cartesian lattice matrices
            L_matrices = np.array([cholesky(mat, lower=False) for mat in sym_g_sets])
            fixed_position = np.zeros([valid_g_sets.shape[0], 3, 1])
            random_atomic_position = np.random.rand(
                valid_g_sets.shape[0], 3, (atom_num - 1)
            )
            random_atomic_position = np.concatenate(
                [fixed_position, random_atomic_position], axis=2
            )

            # Filter structures based on interatomic distance constraints
            dist_sets = np.array(
                [
                    nearest_neighbor_atomic_distance(lat, coo)
                    for lat, coo in zip(L_matrices, random_atomic_position)
                ]
            )
            valid_l_matrices = L_matrices[dist_sets > self.least_distance]
            valid_positions = random_atomic_position[dist_sets > self.least_distance]
            print(f"< screened {valid_l_matrices.shape[0]} random structures >")

            # Save valid structures
            for axis, positions in zip(valid_l_matrices, valid_positions):
                self.str_count += 1
                self.write_poscar(
                    axis, positions, f"initial_str/POSCAR_{self.str_count}"
                )
                if self.str_count == self.max_str:
                    return
            iteration += 1
            penalty += 1

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
