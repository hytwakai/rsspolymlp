import pandas as pd
from paretoset import paretoset


class Pareto:

    def __init__(
        self,
        x: list,
        y: list,
        index: list,
    ):
        self.x = x
        self.y = y
        self.index = index

    def get_pareto(self):
        pareto = pd.DataFrame({"x": self.x, "y": self.y}, index=self.index)
        point = paretoset(pareto, sense=["min", "min"])
        pareto_data = pareto[point]
        non_pareto_data = pareto[~point]
        pareto_dict = pareto_data.to_dict()
        non_pareto_dict = non_pareto_data.to_dict()

        pareto_point = {"x": [], "y": [], "index": []}
        for idx in pareto_dict["x"]:
            pareto_point["index"].append(idx)
            pareto_point["x"].append(pareto_dict["x"][idx])
            pareto_point["y"].append(pareto_dict["y"][idx])
        non_pareto_point = {"x": [], "y": [], "index": []}
        for idx in non_pareto_dict["x"]:
            non_pareto_point["index"].append(idx)
            non_pareto_point["x"].append(non_pareto_dict["x"][idx])
            non_pareto_point["y"].append(non_pareto_dict["y"][idx])

        return pareto_point, non_pareto_point


class Convex:

    def __init__(
        self,
        x: list,
        y: list,
        index: list,
    ):
        self.x = x
        self.y = y
        self.index = index

        sorted_data = sorted(zip(self.x, self.y, self.index), key=lambda t: t[0])
        self.x, self.y, self.index = map(list, zip(*sorted_data))

    def get_convex(self):
        point = 0
        convex_point = {"x": [self.x[0]], "y": [self.y[0]], "index": [self.index[0]]}
        while True:
            grad_all = []
            for i in range(len(self.index) - point):
                if not (self.x[i + point] - self.x[point]) == 0:
                    _grad = (self.y[i + point] - self.y[point]) / (
                        self.x[i + point] - self.x[point]
                    )
                    grad_all.append([_grad, i])
                else:
                    pass
            grad_all = sorted(grad_all, reverse=False, key=lambda x: x[0])
            convex_idx = point + grad_all[0][1]
            convex_point["x"].append(self.x[convex_idx])
            convex_point["y"].append(self.y[convex_idx])
            convex_point["index"].append(self.index[convex_idx])
            point += grad_all[0][1]
            if point == len(self.index) - 1:
                break

        return convex_point
