import argparse


def generate_sp_shell_script(run_vaspmpi: str) -> str:
    """
    Generate a shell script for automated VASP execution and error recovery.

    Parameters
    ----------
    run_vaspmpi : str
        Command string to run VASP (e.g., "srun vasp_std > std.log").

    Returns
    -------
    str
        The full shell script as a string with placeholders replaced.
    """

    script = f"""
file="./OUTCAR"
incar="./INCAR"
output="vasp_stdout"

cp INCAR-sp INCAR

{run_vaspmpi} > "$output"

if grep -q -E "VERY BAD NEWS! internal error in subroutine (IBZKPT|SGRCON|PRICEL|INVGRP)" "$output"; then
    echo "" >> "$incar"
    echo "SYMPREC = 1e-08" >> "$incar"
    {run_vaspmpi} > "$output"
fi

nbands=0
if grep -q "band-crossings" "$file"; then
    nbands=$(grep "NBANDS=" "$file" | awk '{{print $NF}}')
    nbands=$((nbands * 2))
fi

if [ "$nbands" -ne 0 ]; then
    echo "" >> "$incar"
    echo "NBANDS = $nbands" >> "$incar"
    {run_vaspmpi} > "$output"
fi

incar_file="INCAR"
oszicar_file="OSZICAR"
status_file="calc_status_sp.txt"

nelm=$(awk '/^\s*NELM\s*=/ {{print $3}}' "$incar_file")
if [ -z "$nelm" ]; then
    nelm=60
fi

n_steps=$(awk '/^[^ ]+:/ {{n++}} END {{print n}}' "$oszicar_file")

if [ "$n_steps" -eq "$nelm" ]; then
    echo "max_iteration" >> "$status_file"
elif grep -q "E0=" "$oszicar_file"; then
    echo "success" >> "$status_file"
else
    echo "fail" >> "$status_file"
fi

grep "TITEL" ./POTCAR > ./POTCAR_compress
sed '/band No\\.  band energies     occupation/,/^$/d' ./OUTCAR > ./OUTCAR_compress
rm ./POTCAR
rm ./OUTCAR
if find "$PWD" -type f -name "*std*" -size +100M | grep -q .; then
    sed -i '/WARNING: Sub-Space-Matrix is not hermitian in DAV/,/DAV:/{{/DAV:/!d}}' "$output"
fi
"""
    return script.strip()


def generate_opt_shell_script(run_vaspmpi: str) -> str:
    """
    Generate a shell script for geometry optimization using VASP with iterative recovery.

    Parameters
    ----------
    run_vaspmpi : str
        Command string to run VASP.
    stdout_name : str
        Name of stdout file (e.g., "std.log").
    finish_path : str
        Path to append successful case names.
    fail_path : str
        Path to append failed case names.

    Returns
    -------
    str
        Shell script string.
    """

    script = f"""
cp POSCAR POSCAR.init

file="./OUTCAR"
incar="./INCAR"
oszicar="./OSZICAR"
output="vasp_stdout"

iteration_pre=0
relax_state=0
counter=1
while [ $counter -le 10 ]; do
    cp INCAR-first INCAR
    {run_vaspmpi} > "$output"

    if find "$PWD" -type f -name "*std*" -size +100M | grep -q .; then
        sed -i '/WARNING: Sub-Space-Matrix is not hermitian in DAV/,/DAV:/{{/DAV:/!d}}' "$output"
    fi

    cp INCAR-relax INCAR

    nbands=0
    if grep -q "band-crossings" "$file"; then
        nbands=$(grep "NBANDS=" "$file" | awk '{{print $NF}}')
        nbands=$((nbands * 2))
        echo "" >> "$incar"
        echo "NBANDS = $nbands" >> "$incar"
    fi

    vector=$(grep "search vector abs. value=" "$output" | awk 'END{{print $NF}}')
    vector=$(printf "%f" "$vector")
    if [ "$(echo "$vector > 10" | bc)" -eq 1 ]; then
        potim=$(echo "scale=6; 5 / $vector" | bc)
        echo "" >> "$incar"
        echo "POTIM = $potim" >> "$incar"
        cp INCAR INCAR_potim_${{counter}}
    fi

    if grep -q -E "VERY BAD NEWS! internal error in subroutine (IBZKPT|SGRCON|PRICEL|INVGRP)" "$output"; then
        echo "" >> "$incar"
        echo "SYMPREC = 1e-08" >> "$incar"
        cp "$output" vasp_stdout_failsym_${{counter}}
        cp INCAR INCAR_sym_${{counter}}
    fi

    {run_vaspmpi} > "$output"

    cp vasprun.xml vasprun_${{counter}}.xml
    sed -i '/band No\\.  band energies     occupation/,/^$/d' ./OUTCAR
    cp OUTCAR OUTCAR_${{counter}}
    cp OSZICAR OSZICAR_${{counter}}
    cp CONTCAR CONTCAR_${{counter}}
    cp "$output" vasp_stdout_${{counter}}
    cp CONTCAR POSCAR

    iteration=$(grep "F=" "$oszicar" | wc -l)

    if [ $counter -gt 1 ] && grep -q "reached required accuracy - stopping structural energy minimisation" "$output"; then
        if [ $iteration -eq 1 ] || ( [ $counter -gt 2 ] && [ $iteration -ge $iteration_pre ]; ); then
            relax_state=1
            break
        fi
    fi

    if grep -q "ERROR FEXCP: supplied Exchange-correletion table" "$output"; then
        relax_state=0
        break
    fi

    iteration_pre=$iteration
    ((counter++))
done

status_file="calc_status_opt.txt"
if [ $relax_state -eq 1 ]; then
    echo "success" >> "$status_file"
else
    echo "fail" >> "$status_file"
fi
"""
    return script.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a shell script for VASP run."
    )

    parser.add_argument("--sp", action="store_true")
    parser.add_argument("--opt", action="store_true")

    parser.add_argument(
        "--run_vaspmpi",
        type=str,
        required=True,
        help="VASP execution command, e.g., 'srun vasp_std > std.log'",
    )
    parser.add_argument(
        "--script_name",
        type=str,
        default="run_vasp.sh",
        help="Output shell script filename (default: run_vasp.sh)",
    )

    args = parser.parse_args()

    if args.sp:
        script = generate_sp_shell_script(
            run_vaspmpi=args.run_vaspmpi,
        )
    if args.opt:
        script = generate_opt_shell_script(
            run_vaspmpi=args.run_vaspmpi,
        )

    with open(args.script_name, "w") as f:
        f.write(script + "\n")
    print(f"Shell script written to: {args.script_name}")
