from rsspolymlp.analysis.unique_struct import (
    UniqueStructureAnalyzer,
    generate_unique_structs,
    log_unique_structures,
)


def struct_matcher(
    poscar_paths,
    use_joblib: bool = True,
    num_process: int = -1,
    backend: str = "loky",
    symprec_set: list[float] = [1e-5, 1e-4, 1e-3, 1e-2],
    output_file: str = "unique_struct.yaml",
):
    rss_results = []
    for poscar_path in poscar_paths:
        rss_results.append({"poscar": poscar_path})

    unique_structs = generate_unique_structs(
        rss_results,
        use_joblib=use_joblib,
        num_process=num_process,
        backend=backend,
        symprec_set=symprec_set,
    )

    analyzer = UniqueStructureAnalyzer()
    for unique_struct in unique_structs:
        analyzer.identify_duplicate_struct(
            unique_struct=unique_struct,
            axis_tol=0.05,
            pos_tol=0.03,
        )

    unique_structs = analyzer.unique_str
    unique_structs_sorted = sorted(
        unique_structs,
        key=lambda x: len(x.original_structure.positions.T),
        reverse=False,
    )

    with open(output_file, "w"):
        pass
    log_unique_structures(
        output_file,
        unique_structs_sorted,
    )
