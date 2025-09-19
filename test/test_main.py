import subprocess
import re
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SCRIPT_PATH = os.path.join(BASE_DIR, "test", "run_opt.py")
EXPECTED_RESIDUAL = 0.027169734053930056
TOL = 1e-12

def parse_residual(output):
    """Extract residual value from script stdout."""
    match = re.search(r"FINAL_RESIDUAL=([0-9.eE+-]+)", output)
    if not match:
        raise ValueError(f"Could not parse residual from output:\n{output}")
    return float(match.group(1))

def test_serial_vs_mpi_consistency():
    """Ensure running serial and with mpirun -np 2 produces identical residuals."""

    # 1. Run in serial
    serial_proc = subprocess.run(
        [sys.executable, SCRIPT_PATH],
        capture_output=True,
        text=True
    )
    print("STDOUT:\n", serial_proc.stdout)
    print("STDERR:\n", serial_proc.stderr)
    serial_proc.check_returncode() 

    # 2. Run with MPI (2 processes)
    mpi_proc = subprocess.run(
        ["mpirun", "-np", "2", sys.executable, SCRIPT_PATH],
        capture_output=True,
        text=True,
        check=True
    )
    mpi_output = mpi_proc.stdout
    mpi_residual = parse_residual(mpi_output)

    # 3. Validate correctness and equality
    assert abs(serial_residual - EXPECTED_RESIDUAL) < TOL, \
        f"Serial residual {serial_residual} does not match expected {EXPECTED_RESIDUAL}"

    assert abs(mpi_residual - EXPECTED_RESIDUAL) < TOL, \
        f"MPI residual {mpi_residual} does not match expected {EXPECTED_RESIDUAL}"

    assert abs(serial_residual - mpi_residual) < TOL, \
        f"Serial residual {serial_residual} != MPI residual {mpi_residual}"