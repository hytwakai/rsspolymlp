import numpy as np

from pypolymlp.core.data_format import PolymlpStructure


def polymlp_struct_to_dict(polymlp_st: PolymlpStructure):
    return {
        "axis": polymlp_st.axis.tolist(),
        "positions": polymlp_st.positions.tolist(),
        "n_atoms": list(polymlp_st.n_atoms),
        "elements": list(polymlp_st.elements),
        "types": list(polymlp_st.types),
        "volume": float(polymlp_st.volume) if polymlp_st.volume is not None else None,
        "supercell_matrix": (
            polymlp_st.supercell_matrix.tolist()
            if polymlp_st.supercell_matrix is not None
            else None
        ),
        "positions_cartesian": (
            polymlp_st.positions_cartesian.tolist()
            if polymlp_st.positions_cartesian is not None
            else None
        ),
        "valence": polymlp_st.valence,
        "n_unitcells": polymlp_st.n_unitcells,
        "axis_inv": (
            polymlp_st.axis_inv.tolist() if polymlp_st.axis_inv is not None else None
        ),
        "comment": polymlp_st.comment,
        "name": polymlp_st.name,
        "masses": polymlp_st.masses,
    }


def polymlp_struct_from_dict(polymlp_st_dict: dict):
    d = polymlp_st_dict
    return PolymlpStructure(
        axis=np.array(d["axis"]),
        positions=np.array(d["positions"]),
        n_atoms=d["n_atoms"],
        elements=d["elements"],
        types=d["types"],
        volume=d.get("volume"),
        supercell_matrix=(
            np.array(d["supercell_matrix"])
            if d.get("supercell_matrix") is not None
            else None
        ),
        positions_cartesian=(
            np.array(d["positions_cartesian"])
            if d.get("positions_cartesian") is not None
            else None
        ),
        valence=d.get("valence"),
        n_unitcells=d.get("n_unitcells"),
        axis_inv=np.array(d["axis_inv"]) if d.get("axis_inv") is not None else None,
        comment=d.get("comment"),
        name=d.get("name"),
        masses=d.get("masses"),
    )
