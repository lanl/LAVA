# LAVA

**1. Overview**

The Lava Wrapper is a general-purpose calculator that provides a python interface to enable one-click calculation of the many common properties with lammps and vasp. The name Lava is derived from the “La” in Lammps and “va” in vasp. It provides a set of classes and functions to generate configurations, run lammps/vasp calculation, retrieve the output, postprocess and plot the results. All the above tasks are hard-coded into the script, without the need to call additional libraries.

**2. Features**

Lava Wrapper comprises of eight major python modules. The following flow chart gives an overall idea of the workings of Lava Wrapper. Lava_Wrapper.py provides the outmost layer of abstraction, where the users can specify the set of calculations to perform, invoking the corresponding functions in Lava_Calculator.py, which reads in configurations generated by Lava_Generator.py, submits lammps/vasp run, extracts and postprocess the output, invoking utility functions in Lava_Utility.py. In this process, two other modules, Lava_Lammps _Header.py and Lava_Vasp_Header.py, are responsible for generating Lammps input script and vasp INCAR file, respectively. Finally, the results are plotted with Lava_Plotter.py. The last module, Lava_Config.py is used for storing global settings.

![image](https://user-images.githubusercontent.com/106281982/171067764-f44ea45b-d270-4b85-93c4-c9e56a06004a.png)
Fig. 1  Flow-chart of Lava Wrapper.

The following is a summary of each module’s functionality:
Lava_Wrapper.py: The outmost wrapper where the users specify the type of calculations, and the parameters pertaining to these calculations.
Lava_Calculator.py: The core of Lava Wrapper, responsible for performing the various types of calculations through the following steps: generate the relevant input file, perform calculations, retrieve and postprocess the output, and plot the results.
Lava_Generator.py: Functions for the generation of various crystal structures and defect configurations. The following 8 bulk crystal structures are currently supported: Simple Cubic (SC), Body Centered Cubic (BCC), Face Centered Cubic (FCC), Hexagonal Close Packed (HCP), Diamond Cubic (DC), double HCP, A5_Beta_Sn, A15_Beta_W.  The following defect configurations: vacancy, interstitial, surface, stacking faults, twin faults can also be added to the generated crystal structures.
Lava_Utility.py: Utility functions for file management, data processing, etc.
Lava_Lammps_Header.py: Functions for generating lammps input script, depending on the type of calculation.
Lava_Vasp_Header.py: Functions for generating vasp input script, namely, the INCAR file, depending on the type of calculation.
Lava_Plotter.py: Functions for plotting the results.
Lava_Config.py: Specification the lammps executable, the lammps potential file, as well as the element, provided by the user. The first two files are only required for running the wrapper in lammps mode.

**3. Types of Calculations****

Currently, Lava Wrapper incorporates the following types of calculations. For some calculations the output a written to “Summary.dat” file that serves as the general output for Lava_Wrapper, and for other types of calculation where the results are tabulable, then it is written to a separate data file and plotted as well.

![image](https://user-images.githubusercontent.com/106281982/171067562-aa8b6181-480c-459a-84b6-aac9ec9d4e02.png)

**4. Running LAVA**

To run Lava Wrapper, load python3 first and then issue the following command on the terminal:

python3 Lava_Wrapper.py mode

wherein the highlighted mode can either be Lammps or Vasp, depending on which mode you wish to run.
Example scripts and output of Lava Wrapper are given below, when invoked for Al using Mishin-Farkas potential. This can be set up by specifying the lammps executable as well as potential file name in Lava_Config.py in the following way:

lammps_executable = "lmp_mpi"
element = "Al"
potential_file = 'Al99.eam.alloy'


Make sure that the lammps executable and the potential file is in the same folder as the Lava Wrapper scripts.


© 2022. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
