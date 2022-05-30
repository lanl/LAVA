###############################################################################
# Lava Wrapper, where 'Lava' comes from 'La' in lammps and 'va' in vasp
# Provides a one-click interface for calculating common properties in Lammps and Vasp
# Invoke it this way: python3 Lava_Wrapper Lammps/Vasp 
# The last argument being Lammps or Vasp 
# The results are plotted in JPEG format as well
# Refer to "Summary.dat" for more output
# The types of calculations currently supported are
# 1. get_cohesive_energy: calculate lattice constants and cohesive energy for a list of phases
# 2. get_cold_curve_V: calculate cold curve for a range of volumes for a list of phases
# 3. get_cold_curve_R: calculate cold curve for a list of phases, where the range of volumes are
#    centered around the lattice constant of each phases
# 4. get_liquid_rdf: calculate RDF for a list of phases with a range of temperatures
# 5. get_melting_point: calculate melting point using 2-phase method for a list of phases
# 6. get_elastic_properties: calculate elastic properties for a list of phases
# 7. get_thermal_expansion: calculate thermal expansion for a list of phases with a range of temperatures
# 8. get_deformation_path: 1-D mesh volume-conserving Bain/Trigonal path, varying c/a ratio
# 9. get_deformation_path_2D: 2-D contour plot Bain/Trigonal path, varying c/a ratio and volume
# 10. get_vacancy_interstitial_energy: calculate vacancy/interstitial formation energy
#     for a list of phases
# 11. get_surface_energy: calculate surface energy for a list of phases with a list of surface indices
# 12. get_stacking_fault: calculate stacking fault energy with a 2-D mesh for FCC/BCC/HCP phase
# 13. get_general_stacking_fault: calculate generalized stacking fault energy for FCC/BCC
###############################################################################

import numpy as np
import os
from os import sys
import Lava_Calculator
import Lava_Plotter

###############################################################################

def Calculations(mode=None):   
   
    ### Cold curve based on V
    # V_start, V_stop, and number of points
    # A mass is supplied in unit of 10^-23 g
    # for the purpose of calculating density
    # npoints is the number of points for the cold curve
    phase_list = ["FCC"]
    npoints = 30   
    #Lava_Calculator.get_cold_curve_V(phase_list, V_start=20, V_stop=40, npoints=npoints, mass=4.48, mode=mode)
     
    ### Cold curve based on R
    ## alat is lattice constant
    ## Expanse of lattice (from alat-alat_expanse to alat+alat_expanse)
    phase_list = ["FCC"]
    npoints = 30
    #Lava_Calculator.get_cold_curve_R(phase_list, alat_expanse=1, npoints=npoints, mass=4.48, mode=mode)
    
    ### Radial distribution function for liquid at different T (Lammps only)
    # T_dict is a dictionary with the keys corresponding to a phase
    # and values corresponding to a list of temperature
    T_dict = {"FCC": range(300,1500,300)}
    #Lava_Calculator.get_liquid_rdf(T_dict, mode=mode)

    ### Thermal expansion (Lammps only)
    # Refer to "get_liquid_rdf" for T_dict
    T_dict = {"FCC": range(100,1100,100)}
    #Lava_Calculator.get_thermal_expansion(T_dict, mode=mode)	

    ### Melting point using 2 phase method (Lammps only)
    ## get_melting_point(Tm, T_error): Tm is trial melting point, T_error is convergence criteria
    phase_list = ["FCC"]
    #Lava_Calculator.get_melting_point(phase_list, 400,1000, 10, mode=mode)
    
    ### Elastic properties
    phase_list = ["FCC"]
    #Lava_Calculator.get_elastic_properties(phase_list, mode=mode)
    
    ### Volume-conserving Bain path (Default mesh passes right through BCC and FCC)
    npoints = 24
    #Lava_Calculator.get_deformation_path(0.5, 1.2, npoints, orz=[0,0,1], path_name="Bain", mode=mode) 
    
    ### Volume-conserving Bain path (2D)
    # Number of simulations will be npoints^2
    npoints = 24
    #Lava_Calculator.get_deformation_path_2D(0.5, 1.2, npoints, 0.8, 1.2, npoints, orz=[0,0,1], path_name="Bain", mode=mode)
    
    ### Volume-conserving Trigonal path (Default mesh passes right through SC, BCC and FCC)
    npoints = 24
    #Lava_Calculator.get_deformation_path(0.175, 1.175, npoints, orz=[1,1,1], path_name="Trigonal", mode=mode)     
    
    ### Trigonal path 
    npoints = 24
    #Lava_Calculator.get_deformation_path_2D(0.175, 1.175, npoints, 0.8, 1.2, npoints, orz=[1,1,1], path_name="Trigonal", mode=mode)
        
    ### Vacancy/interstitial formation energy
    ## The unit cells are replicated to lat_dim
    phase_list = ["FCC"]
    #Lava_Calculator.get_vacancy_interstitial_energy(phase_list, lat_dim=[12,12,12], mode=mode)
        
    ### Surface energy
    # Miller-Bravais indices (uvtw) to Miller indices (v, u-v, w)
    # For example, (11-2n) to (10n), (10-1n) to (01n), (0001) to (001)
    # surface_dict is a dictionary with the keys corresponding to a phase
    # and values corresponding to a list of surface miller indices
    surface_dict = {"FCC": [(0,0,1), (1,1,0), (1,1,1)]} 
    #Lava_Calculator.get_surface_energy(surface_dict, mode=mode)
    
    ### Stacking fault energy: 2-D grid, very expensive for VASP
    # Current supports the following three Gamma surfaces system
    # FCC -> X: [11-2], Y: [1-10], Z:[111]
    # BCC -> X: [111], Y: [-110], Z: [11-2]
    # HCP -> X: [11-20], Y: [10-10], Z: [0001]
    phase = "FCC"
    sf_mesh = [24,24]
    #Lava_Calculator.get_stacking_fault(phase, sf_mesh, mode=mode)
    
    ### General stacking fault curve (full slip vs twin)
    # sf_mesh[0] must be a multiply of 3, and sf_mesh[1] must be 1
    phase = "FCC"
    sf_mesh = [24,1]
    #Lava_Calculator.get_general_stacking_fault(phase, sf_mesh, mode=mode)
    
    # Uniaxial deformation (Lammps_serial only)
    # Perform uniaxial deformation along desired direction with erate/trate via a fix/deform command
    # deform_mode indicates which deformation mode is performed
    # 1=x 2=y,3=z, 4=yz, 5=xz, 6=xy
    phase_list = ["A5_Beta_Sn","A5_Gamma_Sn"]
    temp = 300
    lat_rep=[15,15,15]
    deform_mode = [1,2,3,4,5,6]
    rate = 0.01
    final_strain = 0.25
    Lava_Calculator.uniaxial_deform(phase_list, temp, deform_mode , rate, final_strain, lat_rep,mode=mode)

def main():  
    # Specify mode: Lammps or Vasp
    list_of_mode = ["Lammps", "Vasp"]
    if len(sys.argv) != 2:
        print ("Please specify one of the following two modes: Lammps or Vasp")
        sys.exit()
    else:
        mode = sys.argv[1]
        if mode not in list_of_mode:
            print ("Wrong mode specified: %s!" %(mode))
            print ('Must be either "Lammps" or "Vasp"!')
            print ("Invoke it this way: 'python3 Lava_Wrapper Lammps/Vasp'")
            sys.exit()
        
    # Check whether there is a lattice data file for use in lammps
    if mode == "Lammps":
        if not os.path.isfile('Lammps_latt.dat'):
            # Lattice constants and cohesive energy
            # List of phases supported are the following 
            phase_list = ["FCC"]
            Lava_Calculator.get_cohesive_energy(phase_list, mode=mode)

        Calculations(mode=mode)
            
    # Check whether there is a lattice data file for use in vasp    
    elif mode == "Vasp":
        if not os.path.isfile('Vasp_latt.dat'):
            ### Lattice constants and cohesive energy
            phase_list =["FCC"]
            Lava_Calculator.get_cohesive_energy(phase_list, mode=mode)
        else:
            Calculations(mode=mode)  
            

if __name__ == "__main__":
    main()
