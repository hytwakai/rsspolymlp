import numpy as np
import pytest
import yaml

from rsspolymlp.analysis.struct_matcher.gen_struct_rep import (
    StructRepGenerator,
)
from rsspolymlp.analysis.struct_matcher.redrep_lattice import (
    get_reduced_lattice,
)
from rsspolymlp.analysis.struct_matcher.test.gen_lattice import (
    generate_lattice,
)
from rsspolymlp.api.rsspolymlp_utils import struct_matcher


@pytest.mark.parametrize(
    "crystal_system",
    [
        "triclinic",
        "monoclinic",
        "monoclinic",
        "tetragonal",
        "hexagonal",
        "rhombohedral",
        "cubic",
    ],
)
def test_sm_random_struct(tmp_path, crystal_system):
    print(f"----- crystal system: {crystal_system} -----")
    n_elements_list = [1, 5]
    for n_elements in n_elements_list:
        print(f"--- N_elements: {n_elements} ---")
        n_atom = 30 + 10 * (n_elements - 1)
        n_struct_reps = 10 * n_elements
        n_element_duplicates = 5
        n_duplicates = 5
        element_cands = ["A", "B", "C", "D", "E", "F"]

        axis = generate_lattice(crystal_system, n_atom * 10)
        _, axis_cands, signed_permutation_cands = get_reduced_lattice(
            axis, spg_number=1
        )

        struct_reps = []
        for i in range(n_struct_reps):
            if i % n_element_duplicates == 0:
                random_n = np.sort(
                    np.random.choice(
                        n_atom + n_elements - 1, size=n_elements - 1, replace=False
                    )
                )
                random_n = (
                    np.diff(np.concatenate(([0], random_n, [n_atom + n_elements]))) - 1
                )
                elements = []
                for i, n in enumerate(random_n):
                    e = [element_cands[i]] * n
                    elements.extend(e)
                elements = np.array(elements)
            random_positions = np.random.rand(n_atom, 3)

            for h in range(n_duplicates):
                pos_rep = random_positions.copy()
                _elements = elements.copy()

                random_translation = np.random.rand(1, 3)
                pos_rep = (pos_rep + random_translation) % 1.0

                random_permutation = np.random.permutation(n_atom)
                pos_rep = pos_rep[random_permutation]
                _elements = _elements[random_permutation]

                random_transformation = signed_permutation_cands[
                    np.random.randint(len(signed_permutation_cands))
                ]
                trans_pos_rep = np.zeros_like(pos_rep)
                for axis_idx, val in enumerate(random_transformation):
                    target_axis = np.where(val != 0)[0][0]
                    sign = val[target_axis]
                    if sign == 1:
                        trans_pos_rep[:, axis_idx] = pos_rep[:, target_axis]
                    else:
                        trans_pos_rep[:, axis_idx] = 1 - pos_rep[:, target_axis]
                pos_rep = trans_pos_rep
                struct_reps.append(
                    {
                        "axis": axis_cands[0],
                        "positions": pos_rep,
                        "elements": _elements.tolist(),
                        "struct_path": f"{i}_{h}",
                    }
                )
        generator = StructRepGenerator()
        struct_reps_list = generator.generate_struct_reps_parallel(
            struct_reps,
            verbose=True,
        )
        for assign_method in ["redrep", "greedy", "HK"]:
            print("assign_method:", assign_method)
            struct_matcher(
                struct_reps_list=struct_reps_list,
                assign_method=assign_method,
                output_file=f"{tmp_path}/{assign_method}.yaml",
            )
            with open(f"{tmp_path}/{assign_method}.yaml") as f:
                yaml_data = yaml.safe_load(f)
            for i, data in enumerate(yaml_data["unique_structures"]):
                assert data["n_duplicates"] == 5
