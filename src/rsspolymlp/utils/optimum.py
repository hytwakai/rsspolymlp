class Pareto:

    def __init__(
        self,
        x: list,
        y: list,
        label: list,
    ):
        self.x = x
        self.y = y
        self.label = label

    def get_pareto(self):
        data = sorted(zip(self.x, self.y, self.label), key=lambda t: (t[0], t[1]))
        pareto_point = {"x": [], "y": [], "label": []}
        non_pareto_point = {"x": [], "y": [], "label": []}

        best_y = float("inf")
        for x, y, idx in data:
            if y < best_y:
                pareto_point["x"].append(x)
                pareto_point["y"].append(y)
                pareto_point["label"].append(idx)
                best_y = y
            else:
                non_pareto_point["x"].append(x)
                non_pareto_point["y"].append(y)
                non_pareto_point["label"].append(idx)

        return pareto_point, non_pareto_point
