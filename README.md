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
