import numpy as np
import pytest
import yaml

from rsspolymlp.analysis.struct_matcher.gen_struct_rep import (
    StructRepGenerator,
)
from rsspolymlp.analysis.struct_matcher.test.gen_rand_lattice import (
    generate_random_lattice,
)
from rsspolymlp.analysis.struct_matcher.test.gen_rand_position import (
    apply_rand_disp,
    gen_random_elements,
)
from rsspolymlp.api.rsspolymlp_utils import struct_matcher


@pytest.mark.parametrize(
    "crystal_system",
    [
        "triclinic",
        "monoclinic",
        "orthorhombic",
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

        axis = generate_random_lattice(crystal_system, n_atom * 10)

        struct_reps = []
        for i in range(n_struct_reps):
            if i % n_element_duplicates == 0:
                elements = gen_random_elements(n_atom, n_elements)

            random_positions = np.random.rand(n_atom, 3)
            for h in range(n_duplicates):
                pos_rep = random_positions.copy()
                _elements = elements.copy()

                rand_pos, rand_elements = apply_rand_disp(
                    axis, pos_rep, _elements, spg_number=1, max_disp=0
                )
                struct_reps.append(
                    {
                        "axis": axis,
                        "positions": rand_pos,
                        "elements": rand_elements,
                        "struct_path": f"{i}_{h}",
                    }
                )
        generator = StructRepGenerator()
        struct_rep_list = generator.generate_struct_rep_parallel(
            struct_reps,
            verbose=True,
        )
        for assign_method in ["redrep", "greedy", "HK"]:
            print("assign_method:", assign_method)
            struct_matcher(
                struct_rep_list=struct_rep_list,
                assign_method=assign_method,
                output_file=f"{tmp_path}/{assign_method}.yaml",
            )
            with open(f"{tmp_path}/{assign_method}.yaml") as f:
                yaml_data = yaml.safe_load(f)
            for i, data in enumerate(yaml_data["unique_structures"]):
                assert data["n_duplicates"] == 5
