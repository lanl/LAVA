#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --ntasks=40
#SBATCH --partition=C5
#SBATCH --job-name=Lava_Wrapper

module purge
module load gnu8
module load openmpi3


mpirun /home/kqdang/LAVA_with_user_defined_phase/Lava_latest_03_09_22/lmp_mpi -in Lammps_Uniaxial_Deform.in -sf opt
