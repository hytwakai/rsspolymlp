import subprocess

CLI_COMMANDS = [
    "rsspolymlp",
    "rsspolymlp-devkit",
    "rsspolymlp-plot",
    "rsspolymlp-utils",
]


def _run_help(cmd: str):
    return subprocess.run(
        [cmd, "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_cli_help_exit_code_is_zero():
    for cmd in CLI_COMMANDS:
        res = _run_help(cmd)
        assert res.returncode == 0, (
            f"'{cmd} -h' failed.\n"
            f"returncode={res.returncode}\n"
            f"stdout:\n{res.stdout}\n"
            f"stderr:\n{res.stderr}\n"
        )


def test_cli_help_contains_usage():
    for cmd in CLI_COMMANDS:
        res = _run_help(cmd)
        out = (res.stdout + "\n" + res.stderr).lower()
        assert "usage" in out, (
            f"'{cmd} -h' did not print usage.\n"
            f"stdout:\n{res.stdout}\n"
            f"stderr:\n{res.stderr}\n"
        )
