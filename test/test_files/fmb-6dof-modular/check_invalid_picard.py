import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    structure = str(Path("fmb-6dof-modular.yaml").resolve())
    cases = [
        (1, "must be a mapping"),
        ({"fmb": [1, 0]}, "one count per selected instance"),
        ({"fmb": [-1]}, "counts must be nonnegative"),
        ({"unknown": [1]}, "Unknown additional_picard_iterations module"),
        ({"ugf": [1]}, "one count per selected instance"),
        ({"prescribed": [1]}, "refers to a missing prescribed module"),
    ]
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        for picard, message in cases:
            config = {
                "kynema_driver": {
                    "kynema_fmb_inp": structure,
                    "num_timesteps": 1,
                    "additional_picard_iterations": picard,
                }
            }
            (work / "driver.yaml").write_text(json.dumps(config))
            result = subprocess.run(
                sys.argv[1:] + ["driver.yaml"],
                cwd=work, capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 or message not in result.stdout + result.stderr:
                raise AssertionError(
                    f"Expected explicit failure '{message}':\n"
                    f"{result.stdout}\n{result.stderr}"
                )
    print(f"Verified {len(cases)} invalid Picard inputs are rejected")


if __name__ == "__main__":
    main()
