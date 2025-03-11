class ReadFile:

    def __init__(self, logfile):
        self.logfile = logfile

    def read_file(self):
        _res = {
            "potential": None,
            "spg": None,
            "res_f": None,
            "res_s": None,
            "time": None,
            "energy": None,
            "iter": 0,
            "fval": 0,
            "gval": 0,
            "dup_count": 1,
            "poscar": self.logfile.split("/")[-1].removesuffix(".log"),
        }

        keyword_parsers = {
            "Selected potential:": self.parse_potential,
            "Space group set": self.parse_spg,
            "Iterations": self.parse_iterations,
            "Function evaluations": self.parse_fval,
            "Gradient evaluations": self.parse_gval,
            "Maximum absolute value in Residuals (force)": lambda line, res: self.parse_numeric(
                line, "res_f", res
            ),
            "Maximum absolute value in Residuals (stress)": lambda line, res: self.parse_numeric(
                line, "res_s", res
            ),
            "Computational time": lambda line, res: self.parse_numeric(
                line, "time", res
            ),
            "Final function value (eV/atom):": lambda line, res: self.parse_numeric(
                line, "energy", res
            ),
        }

        with open(self.logfile) as f:
            lines = iter(line.strip() for line in f)

            for line in lines:
                for keyword, parser in keyword_parsers.items():
                    if keyword in line:
                        if keyword == "Space group set":
                            line = next(lines)
                        parser(line, _res)

                if "Maximum number of relaxation iterations has been exceeded" in line:
                    return _res, "iteration"

                if "Geometry optimization failed: Huge" in line:
                    return _res, (
                        "energy_zero" if abs(_res["energy"]) < 10**-3 else "energy_low"
                    )

        return _res, True

    def parse_potential(self, line, _res):
        _res["potential"] = line.split()[-1]

    def parse_spg(self, line, _res):
        try:
            _res["spg"] = eval(line)
        except Exception:
            _res["spg"] = None

    def parse_numeric(self, line, key, _res):
        _res[key] = float(line.split()[-1])

    def parse_iterations(self, line, _res):
        _res["iter"] += int(line.split()[-1])

    def parse_fval(self, line, _res):
        _res["fval"] += int(line.split()[-1])

    def parse_gval(self, line, _res):
        _res["gval"] += int(line.split()[-1])
