import h5py
from dolfin import *
from dolfin_adjoint import *
import numpy as np
import moola
import pandas as pd

import subprocess
import os
import gmsh
import matplotlib.pyplot as plt

from HH_shape_opt import *

# Ensure this can be run from root dir
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


frequency = 5.5e9
amp = 1
offset = 1e-8

wave_sources = [(-0.1, -offset), (0.1, -offset)]

msh_file_path = "meshes/torso_sim_mesh_initial.msh"
markers_dict = {
    "obstacle": OBSTACLE_MARKER,
    "side_wall": SIDE_WALL_MARKER,
    "bottom_wall": RECEIVER_EDGE_MARKER,
    "obstacle_opt": None
}
obstacle_stiffness = 25

initial_guess_mesh_util = MeshUtil(
    msh_file_path, markers_dict, obstacle_stiffness)

result_path = "outputs/result_5per_noise.h5"
matlab_fullfield_csv_path = "measurements/scattering_results.csv"

df_exact_measurement = pd.read_csv(matlab_fullfield_csv_path)

# Define set up of the transmitter at (-0.1, 0)
for tx_i, tx in enumerate(wave_sources):
    inc_wave_setup = IncidentWaveSetup(frequency, cylindrical_wave(amp, tx))
    
    tol = 1e-7
    mask = (
        np.isclose(df_exact_measurement["tx_x"].to_numpy(), tx[0], atol=tol, rtol=0.0) &
        np.isclose(df_exact_measurement["tx_y"].to_numpy(), tx[1], atol=tol, rtol=0.0) &
        (df_exact_measurement["x"] >= -0.1) &
        (df_exact_measurement["x"] <= 0.1)
    )
    df_tx = df_exact_measurement.loc[mask]
    inc_wave_setup = IncidentWaveSetup(frequency, cylindrical_wave(amp, wave_sources[tx_i]))

    # Find data frame of the first transmitter

    results = calculate_magnitude_and_phase_error_from_dataframe(df_tx, result_path,
                                            initial_guess_mesh_util, inc_wave_setup, True, 1)

    plot_projected_errors(results, f"outputs/error_5per_noise_{tx_i}.png", True, 1)
