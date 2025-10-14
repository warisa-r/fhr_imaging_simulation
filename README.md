# Installation step that seems to work

## Using conda:

1. Get miniforge: follow this tutorial step 1.
2. Use this setting in you `.condarc` file:
   ```
   channels:
    - conda-forge
      offline: false
      channel_priority: strict
      auto_activate_base: false
   ```
4. Set conda forge as your priority channel
   ```
   conda config --add channels conda-forge
   conda config --set channel_priority strict
   ```
6. Load the MPI environment: `module load GCC/11.3.0 OpenMPI/4.1.4`
7. Create a conda environment for your dolfin adjoint and install the software
  `conda create -n dolfin_adj_fixed -c conda-forge python=3.10 dolfin-adjoint mpi4py`
8. Check if your dolfin is working by
    `python -c "import dolfin_adjoint"`
9. If this is successful, install the rest of the dependencies:
  `conda install conda-forge python-gmsh meshio pandas matplotlib`


# TODO
- [x] We know the entire complex number. Change the objective functional to be |u-u_meas| instead of |u| - |u_meas| (commit before this change is https://github.com/warisa-r/fhr_imaging_simulation/commit/9bc010abf12aa56abeb6709c26be723efc1488c8)

- [] Create a mesh that supports receivers that are not directly next to each other.

Thesis
- [] Motivation
- [] Statement of the problem: Surface Radar imaging. And surface reconstruction. Torso paper vs 
- [] Method My approach
- [] Numerical experiments
- [] Outlook on refraction... limitations