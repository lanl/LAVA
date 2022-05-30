#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --ntasks=40
#SBATCH --partition=C5
#SBATCH --job-name=Lava_Wrapper

module purge
module load gnu8
module load openmpi3


mpirun lammps_executable -in lammps.in -sf opt
