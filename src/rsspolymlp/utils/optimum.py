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


class Convex:

    def __init__(
        self,
        x: list,
        y: list,
        label: list,
    ):
        self.x = x
        self.y = y
        self.label = label

        sorted_data = sorted(zip(self.x, self.y, self.label), key=lambda t: t[0])
        self.x, self.y, self.label = map(list, zip(*sorted_data))

    def get_convex(self):
        point = 0
        convex_point = {"x": [self.x[0]], "y": [self.y[0]], "label": [self.label[0]]}
        while True:
            grad_all = []
            for i in range(len(self.label) - point):
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
            convex_point["label"].append(self.label[convex_idx])
            point += grad_all[0][1]
            if point == len(self.label) - 1:
                break

        return convex_point
