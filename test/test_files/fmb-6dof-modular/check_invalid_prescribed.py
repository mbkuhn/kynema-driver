import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    structure = Path("fmb-6dof-modular.yaml").resolve()
    load = {"node": 0, "start_step": 1, "force": [0.0] * 6}
    cases = [
        ({"target_fmb": 2, "loads": [load]}, "target_fmb is out of range"),
        ({"target_fmb": 1, "loads": []}, "loads must be a nonempty sequence"),
        (
            {"target_fmb": 1, "loads": [dict(load, node=-1)]},
            "node is out of range",
        ),
        (
            {"target_fmb": 1, "loads": [dict(load, start_step=0)]},
            "start_step must be positive",
        ),
        (
            {"target_fmb": 1, "loads": [dict(load, force=[".nan"] * 6)]},
            "forces must be finite",
        ),
        (
            {"target_fmb": 1, "loads": [load, load]},
            "Duplicate prescribed node/start_step",
        ),
    ]
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        config = {
            "kynema_driver": {
                "kynema_fmb_inp": str(structure),
                "prescribed_inp": "prescribed.yaml",
                "num_timesteps": 1,
                "nonlinear_iterations": 2,
            }
        }
        (work / "driver.yaml").write_text(json.dumps(config))
        for prescribed, message in cases:
            (work / "prescribed.yaml").write_text(json.dumps(prescribed))
            result = subprocess.run(
                sys.argv[1:] + ["driver.yaml"],
                cwd=work, capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 or message not in result.stdout + result.stderr:
                raise AssertionError(
                    f"Expected explicit failure '{message}':\n"
                    f"{result.stdout}\n{result.stderr}"
                )
    print(f"Verified {len(cases)} invalid prescribed inputs are rejected")


if __name__ == "__main__":
    main()
