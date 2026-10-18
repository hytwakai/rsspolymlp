import numpy as np


def generate_lattice(crystal_system: str, volume: float, rng=None, axis_abc=None):
    """Generate a random lattice for a specified crystal system."""
    if rng is None:
        rng = np.random.default_rng()

    if axis_abc is None:
        a, b, c = rng.uniform(0.5, 1.5, size=3)
        alpha, beta, gamma = rng.uniform(70.0, 110.0, size=3)

        if crystal_system == "triclinic":
            pass
        elif crystal_system == "monoclinic":
            alpha, gamma = 90.0, 90.0
        elif crystal_system == "orthorhombic":
            alpha = beta = gamma = 90.0
        elif crystal_system == "tetragonal":
            b = a = 1.0
            alpha = beta = gamma = 90.0
        elif crystal_system == "hexagonal":
            b = a = 1.0
            alpha, beta, gamma = 90.0, 90.0, 120.0
        elif crystal_system == "rhombohedral":
            b = c = a = 1.0
            if np.isclose(alpha, 90.0):
                alpha += 1.0
            beta = gamma = alpha
        elif crystal_system == "cubic":
            b = c = a = 1.0
            alpha = beta = gamma = 90.0
        else:
            raise ValueError(f"Unknown crystal system: {crystal_system}")
    else:
        a, b, c, alpha, beta, gamma = axis_abc

    alpha, beta, gamma = np.deg2rad([alpha, beta, gamma])
    bx = b * np.cos(gamma)
    by = b * np.sin(gamma)
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(c**2 - cx**2 - cy**2)

    lattice = np.array(
        [
            [a, 0.0, 0.0],
            [bx, by, 0.0],
            [cx, cy, cz],
        ]
    )
    lattice *= (volume / np.linalg.det(lattice)) ** (1.0 / 3.0)
    return lattice
