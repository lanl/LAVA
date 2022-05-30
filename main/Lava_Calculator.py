###############################################################################
# This module provides functions for setting up all kinds of calculations
# These functions are invoked in "Lava_Wrapper.py"
# Refer to the definition of these functions 
# if you wish to have more control over the calculations
###############################################################################

import numpy as np
import os
from os import sys
from shutil import copyfile
import math
from Lava_Lammps_Header import *
from Lava_Vasp_Header import *
from Lava_Utility import *
from Lava_Generator import *
from Lava_Plotter import *

###############################################################################

# Output directory and name
filedir = os.path.dirname(os.path.realpath('__file__'))
filename= os.path.join(filedir,"Summary.dat")
fout = open(filename,"a+")


def get_phase_params(phase_list):
    # Scaling constants for the initial a, b/a, and c/a and number of atoms per cell
    # Calculate the volume for FCC phase (V_base=1^3/4=0.25)
    # scale the lattice parameters of all phases to that volume
    # scale with a base value of 4.0 for a
    # This provodes a good initial lattice parameters
    a_base, V_base = 5, 0.25
    phase_params = {"SC": [1, 1, 1, 1], \
                    "BCC": [1, 1, 1, 2], \
                    "FCC": [1, 1, 1, 4], \
                    "HCP": [1, math.sqrt(3), 1.6, 4], \
                    "DC": [1, 1, 1, 8], \
                    "DHCP": [1, math.sqrt(3), 3.2, 8], \
                    "A5_Beta_Sn": [1, 1, 0.5, 4], \
                    "A5_Gamma_Sn": [1, 0.9, 0.5, 4], \
                    "A15_Beta_W": [1, 1, 1, 8], \
                    "L12": [1, 1, 1, 4], \
                    "9R": [1, 1/math.sqrt(3), 3*math.sqrt(2), 18]}
    for phase in phase_params:
        alat, blat, clat, atoms_per_cell = phase_params[phase]
        alat = (V_base*atoms_per_cell/blat/clat)**(1/3)
        phase_params[phase][0] *= alat*a_base
    return {phase:phase_params[phase] for phase in phase_list}


# Lattice constants and cohesive energy
def get_cohesive_energy(phase_list, NSW=100, IBRION=1, ISIF=3, lat_rep=[2,2,2], orz=[0,0,1], mode=None):
    # The directory where all the calculations are run
    dirname = "get_cohesive_energy_%s" %mode
    fout.write("############ Lattice constants ############\n")
    phase_params = get_phase_params(phase_list)
    if os.path.isfile("user_provided.data"): phase_list.append("user_provided.data")
    # Write out a lattice parameter file for later use     
    def write_latt_file_initialize(): 
        print ("Writing out lattice parameter file: %s_latt.dat" %mode)         
        filename_latt = os.path.join(filedir,"%s_latt.dat" %mode)
        fout_latt = open(filename_latt,"w")
        fout_latt.write("phase  a  b/a  c/a  atoms_per_cell  Ecoh\n") 
        return fout_latt
    def write_latt_data():
        fout.write("%s: alat = %.4f, blat = %.4f, clat = %.4f, atoms_per_cell = %d, Ecoh = %.4f\n" \
                   %(phase, a, b/a, c/a, natoms, pe))
    def write_delta_FCC_HCP():
        if "FCC" in phase_list and "HCP" in phase_list:
            fout.write("DE(FCC->HCP) = %.4f\n" %(phase_lat["FCC"][-1] - phase_lat["HCP"][-1])) 
    if mode == "Lammps": get_lammps_script(Lammps_file="Lammps_Relax_Box.in").Relax_Box()            
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
        for phase in phase_list:
            output_dir = make_dir(os.path.join(filedir, dirname, phase))
            if phase != "user_provided.data":
                # Initial a, b/a, c/a and number of atoms per cell
                alat, blat, clat, atoms_per_cell = phase_params[phase]
                # Write lammps data file "Temp.data" if mode == Lammps 
                # Qrite POSCAR file "POSCAR" if mode == Vasp
                data, NAN, box_2d = Build_Crystal(phase, alat, clat, lat_rep, orz).Build()
                Write_Data_wrapper(data, NAN, box_2d, output_dir=output_dir, mode=mode)
            else:
                copyfile('user_provided.data', '%s/Temp.data' %(output_dir))
            submit_job_wrapper(output_dir, filedir, NSW, IBRION, ISIF, lammps_file="Lammps_Relax_Box.in", mode=mode)              
    # Extract results now if on Lammps mode
    # Else wait until the folders are created and simulations are finished
    if mode == "Lammps" or os.path.isdir(dirname):
        fout_latt = write_latt_file_initialize()
        phase_lat = {}
        for phase in phase_list:
            # Get a, b/a, c/a and Ecoh and append to phase_lat
            target_file = target_file_wrapper(filedir, dirname, phase, mode=mode)
            pe = get_output(target_file, mode=mode).grep_E_per_atom()
            lat_rep_temp = lat_rep if phase != "user_provided.data" else [1,1,1]
            a, b, c = get_output(target_file, mode=mode).grep_latt(lat_rep=lat_rep_temp)
            natoms = phase_params[phase][3] if phase != "user_provided.data" else get_output(target_file, mode=mode).grep_natoms()
            phase_lat[phase] = (a, b/a, c/a, natoms, pe)        
            write_latt_data()
            fout_latt.write("%s  %.4f  %.4f  %.4f  %d  %.4f\n" %(phase, *phase_lat[phase])) 
        fout_latt.close()
        write_delta_FCC_HCP()
               

# Cold curve with varying V
def get_cold_curve_V(phase_list, V_start=None, V_stop=None, npoints=None, NSW=1, IBRION=1, ISIF=2, \
                     lat_rep=[1,1,1], orz=[0,0,1], V_list=None, mass=None, mode=None, \
                     dirname=None, output_name=None, plot_axis_x="volume", plot_axis_y="energy"): 
    if not dirname:
        dirname = "get_cold_curve_V_%s" %mode
        fout.write("############ Cold curve with varying V ###########\n")	
    if not V_list: V_list = {phase:np.linspace(V_start,V_stop,npoints+1,endpoint=True) for phase in phase_list}
    if not output_name: output_name = "EOS_V_%s" %mode
    def get_density(mass, V):
        return mass*1e-23/(V*1e-24)
    def write_EOS_initialize():
        filename_EOS = os.path.join(filedir,output_name+'.dat')
        fout_EOS = open(filename_EOS,"w")
        for phase in phase_list: fout_EOS.write("#%s          " %phase)
        fout_EOS.write("\n")   
        return fout_EOS
    def write_EOS_final():
        for i in range(npoints):
            for phase in phase_list:
                fout_EOS.write("%.4f  %.4f  %.4f  " %(phase_output[phase][i]))
            fout_EOS.write("\n")
        fout_EOS.close()
    if mode == "Lammps": get_lammps_script(Lammps_file="Lammps_Minimize.in").Minimize()
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
        phase_lat = read_latt_file(os.path.join(filedir,"%s_latt.dat" %mode))
        for phase in phase_list:
            output_dir = make_dir(os.path.join(filedir, dirname, phase))
            _, blat, clat, atoms_per_cell, _ = phase_lat[phase]
            for count, V in enumerate(V_list[phase]):
                V = round_digits(V)
                alat = (atoms_per_cell*V/blat/clat)**(1/3)
                output_dir = make_dir(os.path.join(filedir, dirname, phase, str(V)))
                data, NAN, box_2d = Build_Crystal(phase, alat, clat, lat_rep, orz).Build()
                Write_Data_wrapper(data, NAN, box_2d, output_dir=output_dir, mode=mode)
                submit_job_wrapper(output_dir, filedir, NSW, IBRION, ISIF, lammps_file="Lammps_Minimize.in", mode=mode)
    if mode == "Lammps" or os.path.isdir(dirname):                    
        fout_EOS = write_EOS_initialize()
        phase_output = {phase:[] for phase in phase_list}
        for phase in phase_list:
            for count, V in enumerate(V_list[phase]):
                V = round_digits(V)
                target_file = target_file_wrapper(filedir, dirname, phase, str(V), mode=mode)
                pe = get_output(target_file, mode=mode).grep_E_per_atom()
                density = get_density(mass, V)
                phase_output[phase].append((density, V, pe))
        write_EOS_final() 
        #plot_EOS(output_name+'.dat', output_name, plot_axis_x=plot_axis_x, plot_axis_y=plot_axis_y)


# Cold curve with varying R
def get_cold_curve_R(phase_list, alat_expanse=None, npoints=None, NSW=1, IBRION=1, ISIF=2, \
                     lat_rep=[1,1,1], orz=[0,0,1], mass=None, mode=None, \
                     plot_axis_x="volume", plot_axis_y="energy"):
    dirname = "get_cold_curve_R_%s" %mode
    fout.write("############ Cold curve with varying R ############\n")	
    phase_lat = read_latt_file(os.path.join(filedir,"%s_latt.dat" %mode))
    V_list = {}
    for phase in phase_list:  
        alat, blat, clat, atoms_per_cell, _ = phase_lat[phase]
        alat_list = np.linspace(alat-alat_expanse,alat+alat_expanse,npoints+1,endpoint=True)       
        V_list[phase] = [alat**3*blat*clat/atoms_per_cell for alat in alat_list]
    # Invoke get_cold_curve_V with V_list 
    get_cold_curve_V(phase_list, npoints=npoints, lat_rep=lat_rep, orz=orz, V_list=V_list, \
                     mass=mass, mode=mode, dirname=dirname, output_name="EOS_R_%s" %mode, \
                     plot_axis_x=plot_axis_x, plot_axis_y=plot_axis_y)


# Thermal expansion
def get_thermal_expansion(T_dict, lat_rep=[10,10,10], orz=[0,0,1], mode=None):
    dirname = "get_thermal_expansion_%s" %mode
    fout.write("############ Thermal expansion ############\n")    
    def write_thermal_initialize():
        fout_thermal = open(os.path.join(filedir,"Thermal_expansion_%s_%s.dat" %(phase, mode)),"w")
        fout_thermal.write("#Temp  alat  blat  clat  average_lat  d_lat(%)\n")
        return fout_thermal
    def write_thermal():
        average_lat = (lx+ly+lz)/3
        fout_thermal.write("%.1f  %.4f  %.4f  %.4f  %.4f  %.4f\n" %(T, lx, ly, lz, \
                           average_lat, 100*(average_lat/average_lat_ini-1)))
    if mode == "Vasp":
        print ("Thermal expansion not supported for Vasp !!!")
        return        
    phase_lat = read_latt_file(os.path.join(filedir,"%s_latt.dat" %mode))
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
        for phase in T_dict:
            alat, blat, clat, _, _ = phase_lat[phase]   
            data, NAN, box_2d = Build_Crystal(phase, alat, clat, lat_rep, orz).Build()
            for T in T_dict[phase]:
                output_dir = make_dir(os.path.join(filedir, dirname, phase+"_"+str(T))) 
                Write_Data_wrapper(data, NAN, box_2d, output_dir=output_dir, mode=mode)
                get_lammps_script(Lammps_file="Lammps_Fix_npt.in").Fix_npt(T)  
                run_lammps(output_dir, filedir, lammps_file="Lammps_Fix_npt.in")  
    if os.path.isdir(dirname):                
        for phase in T_dict:
            alat, blat, clat, _, _ = phase_lat[phase] 
            average_lat_ini = alat*(1+blat+clat)/3
            fout_thermal = write_thermal_initialize()
            for T in T_dict[phase]:
                target_file = target_file_wrapper(filedir, dirname, phase+"_"+str(T), mode=mode)
                lx, ly, lz = get_output(target_file, mode=mode).grep_latt(lat_rep=lat_rep)
                write_thermal()
            fout_thermal.close()
            #plot_thermal_expansion("Thermal_expansion_%s_%s.dat" %(phase, mode), "Thermal_expansion_%s_%s" %(phase, mode))


# Radial distribution function for liquid at different temperature (Lammps only)
def get_liquid_rdf(T_dict, lat_rep=[40,40,40], orz=[0,0,1], mode=None):
    dirname = "get_liquid_rdf_%s" %mode
    fout.write("########## Radial distribution function for liquid ##########\n")
    def write_RDF_initialize():
        fout_RDF = open(os.path.join(filedir,"Liquid_RDF_%s_%s.dat" %(phase, mode)),"w")
        for T in T_dict[phase]: fout_RDF.write("#%d          " %T)
        fout_RDF.write("\n")   
        return fout_RDF
    def write_RDF_final():
        for i in range(npoints):
            for T in T_dict[phase]:
                fout_RDF.write("%.3f  %.3f  " %(RDF_data[T][i][0], RDF_data[T][i][1]))
            fout_RDF.write("\n")
        fout_RDF.close() 
    if mode == "Vasp":
        print ("RDF not supported for Vasp !!!")
        return       
    RDF_data = {}             
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
        phase_lat = read_latt_file(os.path.join(filedir,"%s_latt.dat" %mode))
        for phase in T_dict:
            alat, _, clat, _, _ = phase_lat[phase]   
            data, NAN, box_2d = Build_Crystal(phase, alat, clat, lat_rep, orz).Build()
            for T in T_dict[phase]:
                output_dir = make_dir(os.path.join(filedir, dirname, phase+"_"+str(T))) 
                Write_Data_wrapper(data, NAN, box_2d, output_dir=output_dir, mode=mode)
                get_lammps_script(Lammps_file="Lammps_liquid_RDF.in").Fix_liquid_rdf(T)
                run_lammps(output_dir, filedir, lammps_file="Lammps_liquid_RDF.in")                      
    if os.path.isdir(dirname):
        for phase in T_dict:
            fout_RDF = write_RDF_initialize()
            for T in T_dict[phase]:
                # Read in RDF output file
                target_file = os.path.join(filedir, dirname, phase+"_"+str(T), "Liquid_RDF_%d.dat" % T)
                _, array_list = enumerate_files(target_file, skip_line=4, delimiter=' ')
                # Extract column 2 (r) and 3 (g(r))
                RDF_data[T] = np.array(array_list[0])[:,1:3]
            npoints = RDF_data[T].shape[0]
            write_RDF_final()
            #plot_RDF("Liquid_RDF_%s_%s.dat" %(phase, mode), "Liquid_RDF_%s_%s" %(phase, mode))   

		
# Melting point Tm with 2-phase method (Lammps_serial only)
# The idea is to start at the estimated melting point T1 between (Tleft,Tright)
# and run nph to get the solid to melt, which means the temperature (T1) is above the melting temperature. It means the T1_new will be between (Tleft,T1).
# or to get the liquid to solidify, which means the temperature (T1) is below the melting temperature. It means the T1_new will be between (T1,Tright).
# In either scenario, fix/nph will bring the overall T closer to Tm
# Thus one can keep on rerunning from the final temperature of the previous run till it converges
def get_melting_point(phase_list, Tleft, Tright, T_error, lat_rep=[10,10,50], orz=[0,0,1], mode=None):
    dirname = "get_melting_temperature_%s" %mode
    fout.write("############ Melting temperature ############\n")
    if mode == "Vasp":
        print ("2-phase melting not supported for Vasp !!!")
        return    
    if not os.path.isdir(dirname):
        os.mkdir(dirname)         
        phase_lat = read_latt_file(os.path.join(filedir,"%s_latt.dat" %mode))
        for phase in phase_list:
            alat, _, clat, _, _ = phase_lat[phase] 
            data, NAN, box_2d = Build_Crystal(phase, alat, clat, lat_rep, orz).Build()
            delta_T1, T1_new = sys.float_info.max, (Tleft+Tright)/2
            while delta_T1 > T_error:
                T1 = (Tleft+Tright)/2
                output_dir = make_dir(os.path.join(filedir, dirname, phase+"_"+str(T1)))
                Write_Data_wrapper(data, NAN, box_2d, output_dir=output_dir, mode=mode)
                target_file = target_file_wrapper(output_dir, mode=mode)
                get_lammps_script(Lammps_file="Lammps_2_phase_melt.in").Two_phase_melt(T1)
                run_lammps(output_dir, filedir, lammps_file="Lammps_2_phase_melt.in")  
                T1_new = get_output(target_file, mode=mode).grep_temp()
                if (T1_new>=T1):
                        Tleft=T1
                else:
                     	Tright=T1
                delta_T1 = abs(Tright-Tleft)/2
                fout.write("Left T: %.1f K, Right T: %.1f K, delta_T: %.1f K\n" %(Tleft,Tright,delta_T1))
            fout.write("delta_T = %.1f < %.1f (T_error), convergence achieved.\n" %(delta_T1,T_error))
            fout.write("Melting point: %.1f +- %.1f K\n" %((Tleft+Tright)/2,T_error))  		
	   
# Elastic constants
def get_elastic_properties(phase_list, NSW=100, IBRION=6, ISIF=3, NFREE=2, \
                           lat_rep=[2,2,2], orz=[0,0,1], mode=None, use_built_in_method=False):
    dirname = "get_elastic_properties_%s" %mode
    fout.write("############ Elastic constants ############\n")
    deform_list = ["x", "y", "z", "yz", "xz", "xy"]
    delta_list_scale = np.array([0, 1e-4, -1e-4])
    pxx, pyy, pzz, pyz, pxz, pxy = [np.zeros((6,3)) for _ in range(6)]
    # Write elastic moduli for cubic and non-cubic systems
    # More terms are non-zero due to reduced symmetry
    def write_elastic_moduli():
        if phase.upper() in ["SC", "BCC", "FCC", "DC", "DIAMOND"]:
            fout.write("%s:\nC11 = %.3f GPa\nC12 = %.3f GPa\nC44 = %.3f GPa\n" %(phase, C[0,0], C[0,1], C[3,3]))
        else:
            fout.write("%s:\nC11 = %.3f GPa\nC12 = %.3f GPa\nC13 = %.3f GPa\nC33 = %.3f GPa\nC44 = %.3f GPa\nC55 = %.3f GPa\nC66 = %.3f GPa\n" \
                        %(phase, C[0,0], C[0,1], C[0,2], C[2,2], C[3,3], C[4,4], C[5,5]))
        fout.write("Bulk modulus = %.3f GPa\nShear modulus = %.3f GPa\nPoisson Ratio = %.3f\n" %(B, G, Po)) 
    if not os.path.isdir(dirname):       
        os.mkdir(dirname)
        phase_lat = read_latt_file(os.path.join(filedir,"%s_latt.dat" %mode))
        for phase in phase_list:
            output_dir = make_dir(os.path.join(filedir, dirname, phase))
            alat, _, clat, _, _ = phase_lat[phase]
            data, NAN, box_2d = Build_Crystal(phase, alat, clat, lat_rep, orz).Build()
            lx0, ly0, lz0 = [box_2d[i,1]-box_2d[i,0] for i in range(3)]
            delta_list = np.array([lx0, ly0, lz0, lz0, lz0, ly0])    
            for i in range(6):  # Loop over six deformation paths
                for j in range(3):  # Loop over 0, +delta, -delta 
                    runpath = "deform_%s_%f" %(deform_list[i], delta_list_scale[j])
                    output_dir = make_dir(os.path.join(filedir, dirname, phase, runpath))
                    if not use_built_in_method: NSW, IBRION, ISIF = 1, 1, 2
                    if use_built_in_method and mode == "Lammps":
                        get_lammps_script(Lammps_file="Lammps_Deform.in").Deform(deform=deform_list[i], \
                                          delta=delta_list_scale[j]*delta_list[i])
                        Write_Data_wrapper(data, NAN, box_2d, output_dir=output_dir, mode=mode)
                        run_lammps(output_dir, filedir, lammps_file="Lammps_Deform.in")                     
                    else:
                        if mode == "Lammps": get_lammps_script(Lammps_file="Lammps_Minimize.in").Minimize()
                        data, NAN, box_2d = Build_Crystal(phase, alat, clat, lat_rep, orz).Build()
                        Write_Data_wrapper(data, NAN, box_2d, output_dir=output_dir, add_tilt=deform_list[i], \
                                           tilt_factor=delta_list_scale[j]*delta_list[i], mode=mode)
                        submit_job_wrapper(output_dir, filedir, NSW, IBRION, ISIF, lammps_file="Lammps_Minimize.in", mode=mode)
    if os.path.isdir(dirname):                              
        for phase in phase_list:
            for i in range(6):  # Loop over six deformation paths
                for j in range(3):  # Loop over 0, +delta, -delta 
                    runpath = "deform_%s_%f" %(deform_list[i], delta_list_scale[j])
                    target_file = target_file_wrapper(filedir, dirname, phase, runpath, mode=mode)
                    if mode == "Vasp" and use_built_in_method:
                        C = get_elastic_moduli_vasp(target_file)/10 # Convert from kbar to GPa
                    else:
                        pxx[i,j], pyy[i,j], pzz[i,j], pyz[i,j], pxz[i,j], pxy[i,j] = \
                        get_output(target_file, mode=mode).grep_stress() 
            # C[i,j] represents Cij, Bulk modulus (B), sheat modulus (G), and poisson_ratio (Po)
            if mode == "Vasp" and use_built_in_method:
                B, G, Po = (C[0,0]+C[1,1]+C[2,2]+2*C[0,1]+2*C[1,2]+2*C[0,2])/9, (C[0,0]-C[0,1])/2, 1/(1.0+C[0,0]/C[0,1])
            else:
                C, B, G, Po = get_elastic_moduli(pxx, pyy, pzz, pyz, pxz, pxy)
            write_elastic_moduli()


# Deformation path (Bain/trigonal path)
# More details see: Mishin, Y., et al. (2001). "Structural stability and lattice defects 
# in copper:Ab initio, tight-binding, and embedded-atom calculations." Physical Review B 63(22).
def get_deformation_path(clat_start, clat_end, npoints_c, Vol_scale=1, NSW=1, IBRION=1, ISIF=2, \
                         lat_rep=[1,1,1], orz=None, path_name=None, dirname=None, from_2D=False, mode=None): 
    if not from_2D:
        dirname = "get_%s_path_%s" %(path_name, mode)
        fout.write("######## Volume-conserving %s deformation path ########\n" %path_name.lower())
    phase = "FCC"   
    # Initialize deformation path output data
    def get_path_initialize(filedir, path_name, mode):
        fout_path = open(os.path.join(filedir,"%s_%s.dat" %(path_name, mode)),"w")
        fout_path.write("#V    c/a    E\n")
        return fout_path   
    if mode == "Lammps": get_lammps_script(Lammps_file="Lammps_Static.in").Static()
    Vol, V_cube_root = round_digits(Vol_scale), round_digits(Vol_scale**(1/3))
    if from_2D or not os.path.isdir(dirname):
        if not from_2D: os.mkdir(dirname)
        phase_lat = read_latt_file(os.path.join(filedir,"%s_latt.dat" %mode))
        alat0, _, clat0, _, _ = phase_lat[phase]
        for clat in np.linspace(clat_start,clat_end,npoints_c+1,endpoint=True):
            alat = get_params_alat(clat, alat0, clat0)
            applied_strain = [alat/alat0*V_cube_root-1, alat/alat0*V_cube_root-1, alat/alat0*clat/clat0*V_cube_root-1]
            clat = round_digits(clat)
            output_dir = make_dir(os.path.join(filedir, dirname, str(Vol)+"-"+str(clat)))
            data, NAN, box_2d = Build_Crystal(phase, alat0, clat0, lat_rep, orz).Build()
            Write_Data_wrapper(data, NAN, box_2d, output_dir=output_dir, applied_strain=applied_strain, mode=mode)
            submit_job_wrapper(output_dir, filedir, NSW, IBRION, ISIF, lammps_file="Lammps_Static.in", mode=mode)
    if not from_2D and os.path.isdir(dirname):
        fout_path = get_path_initialize(filedir, path_name, mode)
        for clat in np.linspace(clat_start,clat_end,npoints_c+1,endpoint=True):
            clat = round_digits(clat)
            target_file = target_file_wrapper(filedir, dirname, str(Vol)+"-"+str(clat), mode=mode)
            pe = get_output(target_file, mode=mode).grep_E_per_atom()
            fout_path.write("%.4f  %.4f  %.4f\n" %(Vol, clat, pe))
        fout_path.close()
        #plot_Bain("%s_%s.dat" %(path_name, mode),"%s_%s" %(path_name, mode))   
        

# Bain/trigonal deformation path
def get_deformation_path_2D(clat_start, clat_end, npoints_c, Vol_start, Vol_end, npoints_V, NSW=1, IBRION=1, ISIF=2, \
                            lat_rep=[1,1,1], orz=None, path_name=None, mode=None): 
    dirname = "get_%s_path_2D_%s" %(path_name, mode)
    fout.write("######## %s deformation path ########\n" %path_name)   
    outfile_list = ["2D_%s_V_%s.dat" %(path_name, mode), "2D_%s_c_a_%s.dat" %(path_name, mode), \
                    "2D_%s_E_%s.dat" %(path_name, mode)]  
    phase = "FCC"
    if not os.path.isdir(dirname):
        os.mkdir(dirname)    
        phase_lat = read_latt_file(os.path.join(filedir,"%s_latt.dat" %mode))
        alat0, _, clat0, _, _ = phase_lat[phase]      
        for Vol in np.linspace(Vol_start,Vol_end,npoints_V+1,endpoint=True):
            Vol = round_digits(Vol)
            # Invoke get_deformation_path with a given volume
            get_deformation_path(clat_start, clat_end, npoints_c, Vol_scale=Vol, NSW=NSW, IBRION=IBRION, ISIF=ISIF,\
                                 lat_rep=lat_rep, orz=orz, path_name=path_name, dirname=dirname, from_2D=True, mode=mode) 
    if os.path.isdir(dirname):
        fout_path_X, fout_path_Y, fout_path_Z = open_files_for_write(outfile_list, filedir)            
        for count, Vol in enumerate(np.linspace(Vol_start,Vol_end,npoints_V+1,endpoint=True)):
            Vol = round_digits(Vol)
            if count > 0: write_to_file_chars([fout_path_X,"\n"], [fout_path_Y, "\n"], [fout_path_Z, "\n"])
            for clat in np.linspace(clat_start,clat_end,npoints_c+1,endpoint=True):
                clat = round_digits(clat) 
                target_file = target_file_wrapper(filedir, dirname, str(Vol)+"-"+str(clat), mode=mode)
                pe = get_output(target_file, mode=mode).grep_E_per_atom()
                write_to_file_chars([fout_path_X, "%.4f  " %Vol], [fout_path_Y, "%.4f  " %clat],\
                                    [fout_path_Z, "%.4f  " %pe])
        close_files(fout_path_X, fout_path_Y, fout_path_Z)
        #plot_Bain_2D(*outfile_list, "2D_%s_%s" %(path_name, mode))
        
    
# Vacancy/interstitial formation energy
# lat_dim is the desired box size used to decide lat_rep
def get_vacancy_interstitial_energy(phase_list, NSW=100, IBRION=1, ISIF=2, \
                                    lat_rep=None, orz=[0,0,1], lat_dim=[12,12,12], mode=None):
    dirname = "get_vacancy_interstitial_energy_%s" %mode
    fout.write("########## Vacancy/interstitial formation energetics ########\n")  
    if mode in ["Lammps_serial", "Lammps"]:
        get_lammps_script(Lammps_file="Lammps_Minimize.in").Minimize()    
    # A utility function for calculate vacancy/interstitial energy
    # where pe is the total energy of the system with vacancy/intersitial
    # natom, ecoh are the total number of atoms and cohesive energy
    def get_E(pe, natom, ecoh): return pe-natom*ecoh   
    def write_def_energy():
        Evac = get_E(pe['Add_Vacancy'], natom['Add_Vacancy'], pe['Add_None']/natom['Add_None'])
        Eint = get_E(pe['Add_Interstitial'], natom['Add_Interstitial'], pe['Add_None']/natom['Add_None'])   
        fout.write("%s:\nVacancy formation energy:  %.3f eV\n" %(phase, Evac)) 
        fout.write("Interstitial formation energy:  %.3f eV\n" %Eint)
    natom, pe = {}, {}   
    if not os.path.isdir(dirname):
        os.mkdir(dirname)    
        phase_lat = read_latt_file(os.path.join(filedir,"%s_latt.dat" %mode)) 
        for phase in phase_list:
            alat, blat, clat, _, _ = phase_lat[phase]
            lat_rep_temp = set_lat_rep(lat_rep, lat_dim, alat, alat*blat, alat*clat)
            data0, NAN0, box_2d0 = Build_Crystal(phase, alat, clat, lat_rep_temp, orz).Build()
            for def_type in ('Add_None', 'Add_Vacancy', 'Add_Interstitial'):
                output_dir = make_dir(os.path.join(filedir, dirname, phase+"_"+def_type)) 
                def_instance = getattr(Introduce_Defects(nvac=1, nint=1), def_type)
                data, NAN, box_2d = def_instance(data=data0, NAN=NAN0, box_2d=box_2d0)
                Write_Data_wrapper(data, NAN, box_2d, output_dir=output_dir, mode=mode)
                submit_job_wrapper(output_dir, filedir, NSW, IBRION, ISIF, lammps_file="Lammps_Minimize.in", mode=mode) 
    if os.path.isdir(dirname):
            for phase in phase_list:
                for def_type in ('Add_None', 'Add_Vacancy', 'Add_Interstitial'):
                    target_file = target_file_wrapper(filedir, dirname, phase+"_"+def_type, mode=mode)
                    natom[def_type], pe[def_type] = get_output(target_file, mode=mode).grep_natoms(), \
                                                    get_output(target_file, mode=mode).grep_E()
                write_def_energy()    


# Surface energy
def get_surface_energy(surface_dict, NSW=100, IBRION=1, ISIF=2, \
                       lat_rep=None, lat_dim=[3,3,20], vac=10, mode=None):
    dirname = "get_surface_energy_%s" %mode
    fout.write(" ############ Surface energetics ############\n")
    if mode == "Lammps": get_lammps_script(Lammps_file="Lammps_Minimize.in").Minimize()      
    phase_lat = read_latt_file(os.path.join(filedir,"%s_latt.dat" %mode))
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
        for phase in surface_dict:   
            alat, blat, clat, _, ecoh = phase_lat[phase]
            for surface in surface_dict[phase]:
                surface_ind = ''.join([str(i) for i in surface])
                output_dir = make_dir(os.path.join(filedir, dirname, phase+"_"+surface_ind))
                if phase not in ["HCP", "DHCP"] or surface == (0,0,0,1):
                    if surface == (0,0,0,1): surface = (0,0,1)   
                    lat_rep_temp = get_lat_rep_fcc(phase, alat, clat, surface, lat_rep, lat_dim)
                    data, NAN, box_2d = Build_Crystal(phase, alat, clat, lat_rep_temp, surface, vac=vac).Build()
                else:
                    surface = to_miller_indices(*surface)
                    nlayer, lat_rep_temp = get_lat_rep_hcp(phase, alat, clat, surface, lat_rep, lat_dim)
                    data, NAN, box_2d = Build_Crystal(phase, alat, clat, lat_rep_temp, surface, nlayer=nlayer, vac=vac).Build()
                Write_Data_wrapper(data, NAN, box_2d, output_dir=output_dir, mode=mode)
                submit_job_wrapper(output_dir, filedir, NSW, IBRION, ISIF, lammps_file="Lammps_Minimize.in", mode=mode) 
    if os.path.isdir(dirname):
        for phase in surface_dict: 
            *_, ecoh = phase_lat[phase]
            for surface in surface_dict[phase]:
                surface_ind = ''.join([str(i) for i in surface])
                target_file = target_file_wrapper(filedir, dirname, phase+"_"+surface_ind, mode=mode)
                natom, pe = get_output(target_file, mode=mode).grep_natoms(), \
                get_output(target_file, mode=mode).grep_E()
                a, b, _ = get_output(target_file, mode=mode).grep_latt()
                Esuf = get_E_surface(pe, natom, ecoh, 2*a*b)
                fout.write("(%s) Surface energy:  %.3f mJ/m2\n" %(surface_ind, Esuf))


# Stacking fault energy
def get_stacking_fault(phase, sf_mesh, NSW=100, IBRION=1, ISIF=2, mode=None):
    dirname = "get_stacking_fault_%s" %mode
    fout.write("############ Stacking fault energy ############\n")
    # Construct gamma surface
    # FCC -> X: [11-2], Y: [1-10], Z:[111]
    # BCC -> X: [111], Y: [-110], Z: [11-2]
    # HCP -> X: [11-20], Y: [10-10], Z: [0001]
    # A5_Beta_Sn -> X: [001], Y: [010], Z: [100]
    orz_map = {"FCC": [1,1,1], "BCC": [1,1,-2], "HCP": [0,0,1], "A5_Beta_Sn":[1,0,0]}
    # Replicate the cell by 5x10x12 for lammps, and 1x2x6 for vasp
    lat_rep = [5,10,12] if mode in ["Lammps_serial", "Lammps"] else [1,2,6]
    Esf = np.zeros((sf_mesh[0]+1,sf_mesh[1]+1))
    Eusf, Essf = sys.float_info.min, sys.float_info.max
    outfile_list = ["SFE_X_%s.dat" %mode, "SFE_Y_%s.dat" %mode, "SFE_Z_%s.dat" %mode]
    if mode == "Lammps": get_lammps_script(Lammps_file="Lammps_SFE.in").Freeze_X_Y() 

    if not os.path.isdir(dirname):
        os.mkdir(dirname)
        # Store all the stacking fault configurations in the subfolder "Configurations"
        file_store = make_dir(os.path.join(filedir, dirname, "Configurations"))
        phase_lat = read_latt_file(os.path.join(filedir,"%s_latt.dat" %mode))
        alat, _, clat, _, _ = phase_lat[phase]
        data, NAN, box_2d = Build_Crystal(phase, alat, clat, lat_rep, orz_map[phase], vac=10).Build()
        Write_Data_wrapper(data, NAN, box_2d, output_dir=dirname, output_file="SFE.base", freeze_Z=True, mode=mode) 
        input_format = "data" if mode == "Lammps" else "poscar"
        data, NAN, box_2d = Introduce_Defects(input_dir=dirname, input_file="SFE.base", sf_mesh=sf_mesh \
                                              ).Stacking_fault(input_format=input_format, lat_rep=lat_rep)
        # Copy files to the subfolder "Configurations"
        output_file = "Temp.data" if mode == "Lammps" else "POSCAR"
        for j in range(sf_mesh[1]+1):  # Y
            for i in range(sf_mesh[0]+1):  # X
                output_dir = make_dir(os.path.join(filedir, dirname, "SFE_%d_%d" %(j,i)))
                Write_Data_wrapper(data[i,j,:,:], NAN, box_2d, output_dir=output_dir, freeze_Z=True, mode=mode)
                copyfile('%s/%s' %(output_dir, output_file), '%s/SFE_%d_%d' %(file_store, j, i))
                submit_job_wrapper(output_dir, filedir, NSW, IBRION, ISIF, lammps_file="Lammps_SFE.in", mode=mode)
    if os.path.isdir(dirname):
        fout_SFE_X, fout_SFE_Y, fout_SFE_Z = open_files_for_write(outfile_list, filedir)
        for j in range(sf_mesh[1]+1):  # Y
            if j > 0: write_to_file_chars([fout_SFE_X,"\n"], [fout_SFE_Y, "\n"], [fout_SFE_Z, "\n"])
            for i in range(sf_mesh[0]+1):  # X
                target_file = target_file_wrapper(filedir, dirname, "SFE_%d_%d" %(j,i), mode=mode)
                pe = get_output(target_file, mode=mode).grep_E()
                a, b, _ = get_output(target_file, mode=mode).grep_latt()
                if i==0 and j==0: pe0 = pe
                Esf[i,j] = get_SFE(pe, pe0, a*b)
                write_to_file_chars([fout_SFE_X, "%.3f  " %(i/sf_mesh[0])], [fout_SFE_Y, "%.3f  " %(j/sf_mesh[1])],\
                                    [fout_SFE_Z, "%.3f  " %Esf[i,j]])
                # Get Eusf and Esf
                Eusf, Essf = get_usf_sf(sf_mesh, Esf, Eusf, Essf, i, j)
        fout.write("Eus = %.3f mJ/m2, Esf = %.3f mJ/m2\n" %(Eusf, Essf))
        close_files(fout_SFE_X, fout_SFE_Y, fout_SFE_Z)  
        #plot_Gamma_2D(*outfile_list, "2D_Gamma_%s" %mode) 
        #plot_Gamma_3D(*outfile_list, "3D_Gamma_%s" %mode) 


def get_general_stacking_fault(phase, sf_mesh, NSW=100, IBRION=1, ISIF=2, mode=None):
    dirname = "get_general_stacking_fault_%s" %mode
    fout.write("############ General stacking fault energy ############\n")
    orz_map = {"FCC": [1,1,1], "BCC": [1,1,-2], "HCP": [0,0,1],"A5_Beta_Sn":[1,0,0]}
    lat_rep = [5,10,12] if mode == "Lammps" else [1,2,6]
    Esf = np.zeros((sf_mesh[0]+1,sf_mesh[1]+1))
    Eusf, Essf, Eutf = sys.float_info.min, sys.float_info.max, sys.float_info.min
    # Initialize Bain path output data
    def get_GSFE_initialize(filedir, mode):
        fout_GSFE = open(os.path.join(filedir,"GSFE_%s.dat" %mode),"w")
        fout_GSFE.write("X     E_slip     E_twin\n")
        return fout_GSFE  
    def write_GSFE_final():
        for i in range(sf_mesh[0]+1):
            fout_GSFE.write("%.3f  %.3f  %.3f\n" %(i/sf_mesh[0], Esf[i,0], Esf[i,1]))
        fout_GSFE.close()
        fout.write("Eus = %.3f mJ/m2, Esf = %.3f mJ/m2, Eutf = %.3f mJ/m2\n" %(Eusf, Essf, Eutf))
    if mode == "Lammps": get_lammps_script(Lammps_file="Lammps_SFE.in").Freeze_X_Y() 
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
        # Store all the stacking fault configurations in the subfolder "Configurations"
        file_store = make_dir(os.path.join(filedir,dirname,"Configurations"))
        phase_lat = read_latt_file(os.path.join(filedir,"%s_latt.dat" %mode))  
        alat, _, clat, _, _ = phase_lat[phase]
        # Construct surface
        data, NAN, box_2d = Build_Crystal(phase, alat, clat, lat_rep, orz_map[phase], vac=10).Build()
        Write_Data_wrapper(data, NAN, box_2d, output_dir=dirname, output_file="SFE.base", freeze_Z=True, mode=mode) 
        input_format = "data" if mode == "Lammps" else "poscar"
        # Stacking fault (Full slip)
        data, NAN, box_2d = Introduce_Defects(input_dir=dirname, input_file="SFE.base", sf_mesh=sf_mesh, \
                                              shift_plane=0).Stacking_fault(input_format=input_format, lat_rep=lat_rep)
        # Copy files to the subfolder "Configurations"
        output_file = "Temp.data" if mode == "Lammps" else "POSCAR"
        for i in range(sf_mesh[0]+1):  # X
            output_dir = make_dir(os.path.join(filedir, dirname, "SFE.slip.%d" %i))
            Write_Data_wrapper(data[i,0,:,:], NAN, box_2d, output_dir=output_dir, freeze_Z=True, mode=mode) 
            copyfile('%s/%s' %(output_dir, output_file), '%s/SFE.slip.%d' %(file_store, i))
            submit_job_wrapper(output_dir, filedir, NSW, IBRION, ISIF, lammps_file="Lammps_SFE.in", mode=mode) 
        Write_Data_wrapper(data[int(sf_mesh[0]/3),0,:,:], NAN, box_2d, output_dir=dirname, \
                           output_file="SFE.base.I1", freeze_Z=True, mode=mode)
        # Twin growth
        for k in range(1,3):  # Different planes
            data, NAN, box_2d = Introduce_Defects(input_dir=dirname, input_file="SFE.base.I1", sf_mesh=[int(sf_mesh[0]/3),1], \
                                                  shift_plane=k).Stacking_fault(input_format=input_format, lat_rep=lat_rep)     
            for i in range(1,int(sf_mesh[0]/3)+1):  # X
                j = int(sf_mesh[0]/3)*k+i
                output_dir = make_dir(os.path.join(filedir,dirname,"SFE.twin.%d" %j))
                Write_Data_wrapper(data[i,0,:,:], NAN, box_2d, output_dir=output_dir, freeze_Z=True, mode=mode) 
                copyfile('%s/%s' %(output_dir, output_file), '%s/SFE.twin.%d' %(file_store, j))                    
                submit_job_wrapper(output_dir, filedir, NSW, IBRION, ISIF, lammps_file="Lammps_SFE.in", mode=mode)
            copyfile('%s/%s' %(output_dir, output_file), '%s/SFE.base.I1' %dirname)  
                
    if os.path.isdir(dirname):
        fout_GSFE = get_GSFE_initialize(filedir, mode)     
        for i in range(sf_mesh[0]+1):  # X
            target_file = target_file_wrapper(filedir, dirname, "SFE.slip.%d" %i, mode=mode)           
            pe = get_output(target_file, mode=mode).grep_E()
            a, b, _ = get_output(target_file, mode=mode).grep_latt()
            if i==0: pe0 = pe            
            Esf[i,0] = get_SFE(pe, pe0, a*b)            
            Esf[i,1] = Esf[i,0]
            Eusf, Essf = get_usf_sf(sf_mesh, Esf, Eusf, Essf, i, 0)               
        # Twin growth
        for k in range(1,3):  # Different planes
            for i in range(1,int(sf_mesh[0]/3)+1):  # X
                j = int(sf_mesh[0]/3)*k+i
                target_file = target_file_wrapper(filedir, dirname, "SFE.twin.%d" %j, mode=mode) 
                pe = get_output(target_file, mode=mode).grep_E()
                a, b, _ = get_output(target_file, mode=mode).grep_latt()
                Esf[i+k*int(sf_mesh[0]/3),1] = get_SFE(pe, pe0, a*b)  
        # Get Eutf
        Eutf = get_utf(sf_mesh, Esf, Eutf) 
        write_GSFE_final()   
        plot_GSFE("GSFE_%s.dat" %mode,"GSFE_%s" %mode)  


# Uniaxial deformation (Lammps_serial only)
# Perform uniaxial deformation along desired direction with erate/trate via a fix/deform command
def uniaxial_deform(phase_list, temp, deform_mode , rate, final_strain,lat_rep, orz=[0,0,1], mode=None):
    dirname = "uniaxial_deform_%s" %mode
    fout.write("############ Uniaxial deformation ############\n")
    if mode == "Vasp":
        print ("Uniaxial deformation not supported for Vasp !!!")
        return
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
        phase_lat = read_latt_file(os.path.join(filedir,"%s_latt.dat" %mode))
        for phase in phase_list:
            alat, _, clat, _, _ = phase_lat[phase]
            data, NAN, box_2d = Build_Crystal(phase, alat, clat, lat_rep, orz).Build()
            for d_mode in deform_mode:
                output_dir = make_dir(os.path.join(filedir, dirname, phase+"_"+str(temp)+"_"+str(d_mode)+"_"+str(rate)))
                Write_Data_wrapper(data, NAN, box_2d, output_dir=output_dir, mode=mode)
                target_file = target_file_wrapper(output_dir, mode=mode)
                get_lammps_script(Lammps_file="Lammps_Uniaxial_Deform.in").uniaxial_deform(phase,temp,d_mode,rate,final_strain)
                run_lammps(output_dir, filedir, lammps_file="Lammps_Uniaxial_Deform.in")
                current_file= os.getcwd()
                copyfile('%s/Stress_strain_%s_%s.txt' %(output_dir,d_mode,phase),'%s/Stress_strain_%s_%s.txt' %(current_file,d_mode,phase))
                plot_stress_strain('%s/Stress_strain_%s_%s.txt' %(current_file,d_mode,phase),d_mode, 'Stress_strain_%s_%s' %(d_mode,phase))
if __name__ == "__main__":
    main()
