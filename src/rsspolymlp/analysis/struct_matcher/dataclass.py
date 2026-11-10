from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class StructRep:
    axis: np.ndarray
    metric_tensor: np.ndarray
    reduced_posreps: dict
    all_posreps: dict
    elements: np.ndarray
    spg_number: int
    spg_list: list[str]
    original_axis: np.ndarray
    original_positions: np.ndarray
    original_elements: np.ndarray
    tol_sm: float
    tol_redpos: list[float]
    normalize_axis: bool
    struct_path: Optional[str]
    struct_tag: Optional[int]
    dupstr_paths: set[str]
