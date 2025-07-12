import numpy as np
from scipy.linalg import cholesky

from rsspolymlp.common.property import PropUtil
from rsspolymlp.utils.vasp_util.write_poscar import write_poscar


class GenerateRandomStructure:
    """Class for creating initial random structures for RSS."""

    def __init__(
        self,
        element_list,
        atom_counts,
        max_str: int = 5000,
        least_distance: float = 0.0,
        pre_str_count: int = 0,
    ):
        """
        Initialize the structure generation parameters.

        Parameters
        ----------
        element_list : list
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

        self.element_list = element_list
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
                        PropUtil(lat.T, coo.T).least_distance
                        for lat, coo in zip(valid_l_matrices, valid_positions)
                    ]
                )
                valid_l_matrices = valid_l_matrices[distance_sets > self.least_distance]
                valid_positions = valid_positions[distance_sets > self.least_distance]
                print(
                    f"< screened {valid_l_matrices.shape[0]} random structures (least distance) >"
                )

            # Save valid structures
            elements = []
            for i, el in enumerate(self.element_list):
                for h in range(self.atom_counts[i]):
                    elements.append(el)
            for axis, positions in zip(valid_l_matrices, valid_positions):
                self.str_count += 1
                write_poscar(
                    axis,
                    positions,
                    elements,
                    f"initial_struct/POSCAR_{self.str_count}",
                )
                if self.str_count == self.max_str:
                    return
            iteration += 1
