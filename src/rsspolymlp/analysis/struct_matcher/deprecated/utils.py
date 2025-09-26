import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from rsspolymlp.analysis.struct_matcher.irrep_position import IrrepPosition


class IrrepUtil:

    def __init__(self, positions, elements, symprec: list[float] = [1e-5, 1e-5, 1e-5]):
        """Init method."""
        self.positions = np.asarray(positions, dtype=float)
        self.elements = np.asarray(elements, dtype=str)
        _, idx = np.unique(self.elements, return_index=True)
        unique_ordered = self.elements[np.sort(idx)]
        self.types = np.array(
            [np.where(unique_ordered == el)[0][0] for el in self.elements]
        )
        self.irrep_pos = IrrepPosition(symprec)
        self.distance_cluster = None

    def inter_cluster_diffs(self):
        pos_cls_id, pos = self.irrep_pos.assign_clusters(self.positions, self.types)

        distance_cluster = []
        for ax in range(3):
            id_bins = np.bincount(pos_cls_id[:, ax])
            if len(id_bins) == 1:
                distance_cluster.append(np.array([1]))
                continue

            # Distance between consecutive centres (cyclic)
            centres = np.bincount(pos_cls_id[:, ax], weights=pos[:, ax]) / id_bins
            centres = np.sort(centres, kind="mergesort")
            centre_gap = np.roll(centres, -1) - centres
            centre_gap[-1] += 1.0
            distance_cluster.append(centre_gap)
        self.distance_cluster = distance_cluster

        return self.distance_cluster

    def recommended_symprec(self):
        """Determine and return two suitable symprec values derived from the distances
        between identified clusters."""
        if self.distance_cluster is None:
            self.distance_cluster = self.inter_cluster_diffs()

        recommend_orders = []
        for ax in range(3):
            diffs_ax = self.distance_cluster[ax]
            diffs_pos = diffs_ax[diffs_ax > 0]
            if diffs_pos.size == 0:
                continue
            orders = np.log10(diffs_pos)
            orders_valid = orders[~np.all(np.isnan(orders))]
            max_orders = np.nanmax(orders_valid)
            recommend_orders.append(max_orders)
        recommend_orders = np.array(recommend_orders)

        return [10 ** (recommend_orders - 1), 10 ** (recommend_orders - 2)]


def get_recommend_symprecs(
    primitive_st: PolymlpStructure,
    symprec_irrep: list[float] = [1e-5, 1e-5, 1e-5],
):
    _pos = primitive_st.positions.T
    _elements = primitive_st.elements
    irrep_util = IrrepUtil(_pos, _elements, symprec=symprec_irrep)
    recommend_symprecs = irrep_util.recommended_symprec()

    return recommend_symprecs


def get_distance_cluster(
    polymlp_st: PolymlpStructure,
    symprec_irrep: list[float] = [1e-5, 1e-5, 1e-5],
):
    _pos = polymlp_st.positions.T
    _elements = polymlp_st.elements
    irrep_util = IrrepUtil(_pos, _elements, symprec=symprec_irrep)
    distance_cluster = irrep_util.inter_cluster_diffs()

    return distance_cluster
