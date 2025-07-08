import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.spglib_utils import SymCell
from rsspolymlp.common.composition import compute_composition


class PropUtil:

    def __init__(
        self,
        axis,
        positions,
    ):
        """
        Parameters
        ----------
        axis : Lattice vectors arranged row-wise as a (3, 3)
        positions : parameters of atomic postions (N, 3)
        """
        self.axis = axis
        self.positions = positions

    def angle_between(self, v1, v2):
        """Calculate the angle (in degrees) between two vectors."""
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.degrees(np.arccos(np.clip(dot_product / norms, -1.0, 1.0)))

    @property
    def abc(self):
        """Convert lattice vectors to unit cell parameters."""
        a, b, c = np.array(self.axis)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        norm_c = np.linalg.norm(c)
        alpha = self.angle_between(b, c)
        beta = self.angle_between(a, c)
        gamma = self.angle_between(a, b)
        return np.array([norm_a, norm_b, norm_c, alpha, beta, gamma]).tolist()

    @property
    def volume(self):
        lat = self.axis
        coo = self.positions
        volume = np.linalg.det(lat) / coo.shape[0]
        return volume

    @property
    def least_distance(self):
        """Calculate the nearest neighbor atomic distance within a periodic lattice"""

        lat = self.axis
        coo = self.positions
        cartesian_coo = lat.T @ coo.T
        c1 = cartesian_coo  # (3, N)

        # Generate periodic image translations along x, y, and z
        image_x, image_y, image_z = np.meshgrid(
            np.arange(-1, 1.1), np.arange(-1, 1.1), np.arange(-1, 1.1), indexing="ij"
        )
        image_matrix = (
            np.stack([image_x, image_y, image_z], axis=-1).reshape(-1, 3).T
        )  # (3, num_images)

        # Compute the translations due to periodic images
        parallel_move = lat.T @ image_matrix
        parallel_move = np.tile(parallel_move[:, None, :], (1, c1.shape[-1], 1))
        c2_all = cartesian_coo[:, :, None] + parallel_move

        # Compute squared distances between all pairs of atoms in all periodic images
        z = (c1[:, None, :, None] - c2_all[:, :, None, :]) ** 2  # (3, N, N, num_images)
        _dist_mat = np.sqrt(np.sum(z, axis=0))  # (N, N, num_images)
        _dist_mat_refine = np.where(_dist_mat > 1e-10, _dist_mat, np.inf)

        # Find the minimum distance for each pair
        dist_mat = np.min(_dist_mat_refine, axis=-1)  # (N, N)
        distance_min = np.min(dist_mat)

        return distance_min

    def analyze_space_group(self, elements):
        comp_res = compute_composition(elements)
        polymlp_st = PolymlpStructure(
            self.axis.T,
            self.positions.T,
            comp_res.atom_counts,
            elements,
            comp_res.types,
        )

        spg_sets = []
        for tol in [1e-5, 1e-4, 1e-3, 1e-2]:
            try:
                sym = SymCell(st=polymlp_st, symprec=tol)
                spg = sym.get_spacegroup()
                spg_sets.append(spg)
            except TypeError:
                continue
            except IndexError:
                continue

        if spg_sets == []:
            print("Analyzing space group failed.")
        return spg_sets
