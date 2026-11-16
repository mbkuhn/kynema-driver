import csv
import math
from pathlib import Path
import subprocess
import sys


def read_results(path):
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows or rows[0] != [
        "step", "time", "node", "x", "y", "z", "qw", "qx", "qy", "qz"
    ]:
        raise ValueError(f"{path}: invalid CSV header")
    if len(rows) < 2:
        raise ValueError(f"{path}: no results")
    if any(len(row) != 10 for row in rows[1:]):
        raise ValueError(f"{path}: invalid CSV row")
    values = [[float(value) for value in row] for row in rows[1:]]
    if not all(math.isfinite(value) for row in values for value in row):
        raise ValueError(f"{path}: nonfinite result")
    return values


def main():
    reference, output = map(Path, sys.argv[1:3])
    output.unlink(missing_ok=True)
    subprocess.run(sys.argv[3:], check=True)
    expected = read_results(reference)
    actual = read_results(output)
    if len(actual) != len(expected):
        raise ValueError(f"Expected {len(expected)} result rows, got {len(actual)}")
    for row_index, (gold, result) in enumerate(zip(expected, actual), start=2):
        for column, (want, got) in enumerate(zip(gold, result)):
            tolerance = 0.0 if column in (0, 2) else 1.0e-10
            if abs(want - got) > tolerance:
                raise ValueError(
                    f"Row {row_index}, column {column}: expected {want}, got {got}"
                )
    print(f"Verified {len(actual)} FMB rows against saved reference (atol=1e-10)")


if __name__ == "__main__":
    main()
