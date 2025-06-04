import ast
import re

import numpy as np


class LogfileLoader:

    def __init__(self, logfile):
        self.logfile = logfile

    def read_file(self):
        _res = {
            "potential": None,
            "pressure": None,
            "spg_list": None,
            "res_f": None,
            "res_s": None,
            "time": None,
            "energy": None,
            "iter": 0,
            "fval": 0,
            "gval": 0,
            "dup_count": 1,
            "poscar": self.logfile.split("/")[-1].removesuffix(".log"),
            "struct": None,
        }

        keyword_parsers = {
            "Selected potential:": self.parse_potential,
            "Pressure (GPa):": lambda line, res: self.parse_numeric(
                line, "pressure", res
            ),
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
            lines = [line.strip() for line in f]

        axis = []
        positions = []
        elements = []
        parse_struct = False
        in_axis = False
        in_frac_coords = False
        for i in range(len(lines)):
            line = lines[i]
            for keyword, parser in keyword_parsers.items():
                if keyword in line:
                    if keyword == "Space group set":
                        _line = lines[i + 1]
                    else:
                        _line = line
                    parser(_line, _res)

            if "Final structure" in line:
                parse_struct = True
            if "Axis basis vectors:" in line and parse_struct is True:
                in_axis = True
                in_frac_coords = False
                continue
            elif "Fractional coordinates:" in line and parse_struct is True:
                in_frac_coords = True
                in_axis = False
                continue

            if in_axis and line.startswith("- ["):
                vec = self.parse_vector_line(line[2:].strip())
                axis.append(vec)

            elif in_frac_coords and line.startswith("-"):
                match = re.match(r"-\s*(\w+)\s*(\[.*\])", line)
                if match:
                    el = match.group(1)
                    coord = self.parse_vector_line(match.group(2))
                    elements.append(el)
                    positions.append(coord)
                else:
                    raise ValueError(f"Could not parse line: {line}")

            judge = self.check_errors(line, _res)
            if judge is not True:
                return _res, judge

        _res["struct"] = {}
        _res["struct"]["axis"] = np.array(axis, dtype=np.float64)
        _res["struct"]["positions"] = np.array(positions, dtype=np.float64)
        _res["struct"]["elements"] = np.array(elements, dtype=str)
        if len(_res["struct"]["elements"]) == 0:
            _res["struct"] = None
            return _res, False

        return _res, True

    def check_errors(self, line, _res):
        if "Maximum number of relaxation iterations has been exceeded" in line:
            return "iteration"
        if "Geometry optimization failed: Huge" in line:
            return "energy_zero" if abs(_res["energy"]) < 10**-3 else "energy_low"
        if "Refining cell failed" in line:
            return "anom_struct"
        if "Analyzing space group failed" in line:
            return "anom_struct"
        return True

    def parse_potential(self, line, _res):
        try:
            _res["potential"] = ast.literal_eval(" ".join(line.split()[2:]))
        except Exception:
            _res["potential"] = None

    def parse_spg(self, line, _res):
        try:
            if _res["spg_list"] is not None:
                _res["spg_list"] = None
                return _res, False
            else:
                _res["spg_list"] = ast.literal_eval(line)
        except Exception:
            _res["spg_list"] = None

    def parse_numeric(self, line, key, _res):
        try:
            _res[key] = float(line.split()[-1])
        except Exception:
            _res[key] = None

    def parse_iterations(self, line, _res):
        try:
            _res["iter"] += int(line.split()[-1])
        except Exception:
            _res["iter"] = None

    def parse_fval(self, line, _res):
        try:
            _res["fval"] += int(line.split()[-1])
        except Exception:
            _res["fval"] = None

    def parse_gval(self, line, _res):
        try:
            _res["gval"] += int(line.split()[-1])
        except Exception:
            _res["gval"] = None

    def parse_vector_line(self, line):
        return [
            float(x)
            for x in re.findall(
                r"np\.float64\(([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)", line
            )
        ]
