import subprocess
import sys


def test_cli_module_imports_in_fresh_interpreter():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from s2rspc.data.io import load_raw_point_cloud; "
                "from s2rspc.cli.toy_demo import main; "
                "print('ok')"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
