###############################################################################
# This module provides utility classes/functions 
# for extracting output from lammps/vasp runs 
# and other functions for file mamagement, data processing, etc
###############################################################################

import numpy as np
from numpy import genfromtxt
import os
from os import sys
import math
import re
from io import BytesIO
from shutil import copyfile
import subprocess
from Lava_Vasp_Header import *
from Lava_Generator import *
from Lava_Config import *

###############################################################################
#                          General utility functions                          #
###############################################################################

# Submit lammps/vasp job
def submit_job_wrapper(output_dir, filedir, NSW, IBRION, ISIF, lammps_file=None, mode=None):
    if mode == "Lammps":
        # Copy all files to output_dir and submit lammps run
        run_lammps(output_dir, filedir, lammps_file=lammps_file)
    else:
        # Generate INCAR file, copy all files to output_dir, and submit vasp run
        run_vasp(output_dir, filedir, NSW, IBRION, ISIF)

# Create directory and return the name
def make_dir(dirname):
    os.mkdir(dirname)
    return (dirname)
    	
# Copy lammps/vasp input files.
def copy_file(destination, *argv):
    for file in argv:
        copyfile('%s' %(file), '%s/%s' %(destination,file)) 

# Open a list of files in filedir for writing
def open_files_for_write(filelist, filedir):      
    return [open(os.path.join(filedir,file),"w") for file in filelist]

# Close a list of files (filelist)    
def close_files(*filelist):
    for file in filelist: file.close() 

# Write a list of characters to a list of files
# file_char_list is a list of pairs of file and char
def write_to_file_chars(*file_char_list):
    for file, char in file_char_list: file.write(char)      
    
# Read in Lammps_latt.dat or Vasp_latt.dat
def read_latt_file(filename):
    phase_lat = {}
    with open(filename) as infile:    
        for i, line in enumerate(infile):
            if i > 0:
                latt = re.split('\s+', line.strip())
                phase_lat[latt[0]] = tuple(float(entry) for entry in latt[1:])
    return phase_lat

# This wrapper selects the write method (write_datafile or write_poscar) based on the mode
def Write_Data_wrapper(data, NAN, box_2d, output_dir=None, output_file=None, \
                       add_tilt=None, tilt_factor=None, freeze_Z=None, applied_strain=None, mode=None):
    if not output_file: output_file="POSCAR" if mode == "Vasp" else "Temp.data"
    if mode == "Lammps":
        Write_Data(data, NAN, box_2d, output_dir=output_dir, output_file=output_file, \
                   add_tilt=add_tilt, tilt_factor=tilt_factor).write_datafile(applied_strain=applied_strain)
    else:
        Write_Data(data, NAN, box_2d, output_dir=output_dir, output_file=output_file, \
                   add_tilt=add_tilt, tilt_factor=tilt_factor).write_poscar(freeze_Z=freeze_Z,applied_strain=applied_strain)

# This wrapper retrives the output path and name (log.lammps or OUTCAR) based on the mode
def target_file_wrapper(*subdir, mode=None):
    fname = {"Lammps": "log.lammps", "Vasp": "OUTCAR"}
    return os.path.join(*subdir,fname[mode])

# Get the replication number
# lat_dim is the desired dimension
# alat, blat and clat are x, y and z dimention   
def set_lat_rep(lat_rep, lat_dim, alat, blat, clat): 
    if lat_rep: 
        return lat_rep 
    else:       
        return [math.ceil(lat_dim[i]/lat) for i, lat in enumerate([alat, blat, clat])]

# Get surface energy scaled by area
def get_E_surface(pe, natom, ecoh, area): 
    return (pe-natom*ecoh)/area*1.60217657e-16*1.0e20 # Convert from eV/Ang to mJ/m2 

# Get stacking fault energy scaled by area
def get_SFE(pe, pe0, area): 
    return (pe-pe0)/area*1.60217657e-16*1.0e20 # Convert from eV/Ang to mJ/m2

# Get unstable stacking fault energy Eusf and stable stacking fault energy Essf
def get_usf_sf(sf_mesh, Esf, Eusf, Essf, i, j):
    if j == 0:
        if i/sf_mesh[0] < 0.34 and Esf[i,j] > Eusf:  # Eusf usually occurs at 0.208 (5/24)
            Eusf = Esf[i,j]
        if i/sf_mesh[0] > 0.25 and i/sf_mesh[0] < 0.50 and Esf[i,j] < Essf:  # Esf usually occurs at 0.333
            Essf = Esf[i,j] 
    return (Eusf, Essf)

# Get unstable twinning energy Eutf
def get_utf(sf_mesh, Esf, Eutf):
    for i in range(sf_mesh[0]+1):
        if i/sf_mesh[0] > 0.40 and i/sf_mesh[0] < 0.60 and Esf[i,1] > Eutf:  # Eutf usually occurs at 0.5
            Eutf = Esf[i,1] 
    return Eutf

# Convert bravais-miller indice (hcp) to bravais indice
def to_miller_indices(u, v, t, w):
    # u: (11-20) -> (100), v: (10-10) -> (010), w: (0001) -> (001)
    # Convert from Miller-Bravais indices to Miller indices
    # (u, v, t, w) = v*(11-20) + (u-v)*(10-10) + w*(0001)  
    #              = v*(100) + (u-v)*(010) + w*(001)
    #              = (v, u-v, w)
    if (u+v+t):
        print ("Invalid Miller-Bravais indices!")
        sys.exit()
    else:
        return [v, u-v, w]

# Get cell replication based on a give dimension   
def get_lat_rep_fcc(phase, alat, clat, surface, lat_rep, lat_dim):
    _, _, box_2d = Build_Crystal(phase, alat, clat, [1,1,1], surface).Build()
    XL, XH, YL, YH, ZL, ZH = box_2d.flatten()
    return set_lat_rep(lat_rep, lat_dim, XH-ZL, YH-YL, ZH-ZL)

# Get cell replication based on a give dimension (hcp)    
def get_lat_rep_hcp(phase, alat, clat, surface, lat_rep, lat_dim):
    nlayer=24
    _, _, box_2d = Build_Crystal(phase, alat, clat, [1,1,1], surface, nlayer=nlayer).Build()
    XL, XH, YL, YH, ZL, ZH = box_2d.flatten()
    nlayer = math.ceil(nlayer/((ZH-ZL)/lat_dim[2]))   
    # Make sure nlayer is a multiplier of 4
    nlayer = 4*math.ceil(nlayer/4)
    return (nlayer, set_lat_rep(lat_rep, lat_dim, XH-ZL, YH-YL, ZH-ZL))

# Get the corresponding alat at the same volume for a given clat/alat
def get_params_alat(clat, alat0, clat0):
    return alat0*(clat0/clat)**(1/3)

# Round floating points to the corresponding significant digits 
def round_digits(param, significant_digits=4):
    return round(param, significant_digits)
    
###############################################################################
#                      Lammps-specific utility functions                      #
###############################################################################        
# Submit Lammps job
def run_lammps(output_dir, filedir, lammps_file):
    lammps_infile = ''.join(open('RunLammps.inp',"r").readlines())
    for str_set in (("lammps_executable", os.path.join(filedir,lammps_executable)), ("lammps.in", lammps_file)):
        lammps_infile = lammps_infile.replace(*str_set)
    with open("RunLammps.sh", "w") as outfile:
        outfile.write(lammps_infile)
    #copy_file(output_dir, lammps_file, potential_file, "library-Al.meam", "RunLammps.sh")
    copy_file(output_dir, lammps_file, "RunLammps.sh")
    for file in potential_file: copy_file(output_dir, file)
    os.chdir(output_dir)
    subprocess.call("chmod +x RunLammps.sh", shell=True)
    subprocess.call("./RunLammps.sh", shell=True)
    os.chdir(filedir)
    print (r"Lammps job submitted at %s: %s" %(output_dir, lammps_file))

###############################################################################
#                       Vasp-specific utility functions                       #
###############################################################################   
# Submit vasp job
def run_vasp(output_dir, filedir, *argv):
    get_vasp_script(*argv)
    copy_file(output_dir,'INCAR','KPOINTS','POTCAR','RunVasp.sh')
    os.chdir(output_dir)
    subprocess.call("sbatch RunVasp.sh", shell=True)
    os.chdir(filedir) 
    print (r"Vasp job submitted at %s" %output_dir)

# Extracte elastic moduli from vasp OUTCAR file
def get_elastic_moduli_vasp(filename):
    dp, C = np.zeros((6,6)), np.zeros((6,6))
    with open(filename) as infile:
        data = infile.read().split('\n')
        if ' TOTAL ELASTIC MODULI (kBar)' in data: 
            index = data.index(' TOTAL ELASTIC MODULI (kBar)')
        for i, line in enumerate(data[index+3:index+9]):
            line = re.split('\s+', line.strip())
            dp[i,:] = [float(entry) for entry in line[1:]]
    for i in range(6):
        for j in range(i,6):
            C[i,j] = (dp[i,j] + dp[j,i])/2
    return C

###############################################################################
#                    Utility functions for extracting output                  #
###############################################################################   
# Extract lattice parameter, energy from lammps under serial mode
# utilizing lammps-python interface
class get_lammps_output_serial:
    def __init__(self, lmp):
        self.lmp = lmp 
        
    def grep_E(self):
        return self.lmp.get_thermo("pe")
                
    def grep_natoms(self):                
        return self.lmp.get_natoms()
    
    def grep_E_per_atom(self):
        return self.lmp.get_thermo("pe")/self.lmp.get_natoms()
        
    def grep_latt(self, lat_rep=[1,1,1]):
        latt = ["lx", "ly", "lz"]
        return [self.lmp.get_thermo(latt[i])/lat_rep[i] for i in range(3)]

    def grep_stress(self):
        p_list = ["pxx", "pyy", "pzz", "pyz", "pxz", "pxy"]
        return [lmp.get_thermo(p) for p in p_list]

# Extract lattice parameter, energy from lammps/vasp output
class get_output:
    # Lammps
    # thermo: step pe temp press pxx pyy pzz pyz pxz pxy lx ly lz
    #        0    1  2    3     4   5   6   7   8   9  10  11 12
    # Vasp
    # pressure: pxx, pyy, pzz, pxy, pyz, pxz
    def __init__(self, target_file, mode=None):
        self.target_file = target_file
        self.mode = mode
   
    def grep_E(self):
        def grep_E_lammps():
            with open(self.target_file) as infile:
                for line in infile:
                    if re.match(r"Loop time", line): 
                        E = re.split('\s+', line_hist.strip())[1] 
                        return float(E)
                    else:
                        line_hist = line                   
        def grep_E_vasp():
            with open(self.target_file) as infile:
                for line in infile:
                    if re.match(".*free  energy", line):
                        match_line = line
                E = re.split('\s+', match_line.strip())[4]
                return float(E)
        return (grep_E_lammps() if self.mode == "Lammps" else grep_E_vasp())
    
    def grep_latt(self, lat_rep=[1,1,1]):
        def grep_latt_lammps():
            with open(self.target_file) as infile:
                for line in infile:
                    if re.match(r"Loop time", line): 
                        latt = re.split('\s+', line_hist.strip())[-3:] 
                        return [float(latt[i])/lat_rep[i] for i in range(3)]
                    else:
                        line_hist = line    
        def grep_latt_vasp():
            with open(self.target_file) as infile:
                for line in infile:
                    if re.match(".*length of vectors", line):
                        match_line = next(infile)
                latt = re.split('\s+', match_line.strip())[:3]
                return [float(latt[i])/lat_rep[i] for i in range(3)]
        return (grep_latt_lammps() if self.mode == "Lammps" else grep_latt_vasp())
    
    def grep_natoms(self):
        def grep_natoms_lammps():
            with open(self.target_file) as infile:
                for line in infile:
                    if re.match(r"Loop time", line): 
                        natoms = re.search(r"with(.*?)atoms", line).group(1)
                        return int(natoms)
        def grep_natoms_vasp():
            with open(self.target_file) as infile:
                for line in infile:
                    if re.match(".*ions per type", line):
                        natoms = re.split('\s+', line.strip())[4]
                        return int(natoms)  
        return (grep_natoms_lammps() if self.mode == "Lammps" else grep_natoms_vasp())

    def grep_E_per_atom(self):
        return self.grep_E()/self.grep_natoms()
        
    def grep_stress(self):					
        def grep_stress_lammps():
            with open(self.target_file) as infile:
                for line in infile:
                    if re.match(r"Loop time", line): 
                        stress = re.split('\s+', line_hist.strip())[4:10] 
                        return ([float(entry) for entry in stress])
                    else:
                        line_hist = line  
        def grep_stress_vasp():
            with open(self.target_file) as infile:
                for line in infile:
                    if re.match(".*in kB", line):
                        match_line = line
                # Notice the order in p is different for vasp as compared to lammps
                pxx, pyy, pzz, pxy, pyz, pxz = re.split('\s+', match_line.strip())[2:8]
                press = [pxx, pyy, pzz, pyz, pxz, pxy]
                return ([float(p) for p in press])  
        return (grep_stress_lammps() if self.mode == "Lammps" else grep_stress_vasp())

    def grep_temp(self):
        temp=np.zeros(10)
        with open(self.target_file) as infile:
            for line in infile:
                if re.match(r"  250000", line): 
                    temp[0] = re.split('\s+', line.strip())[-3]
                if re.match(r"  245500", line):
       	       	    temp[1] = re.split('\s+', line.strip())[-3]
                if re.match(r"  246000", line):
       	       	    temp[2] = re.split('\s+', line.strip())[-3]
                if re.match(r"  246500", line):
       	       	    temp[3] = re.split('\s+', line.strip())[-3]
                if re.match(r"  247000", line):
       	       	    temp[4] = re.split('\s+', line.strip())[-3]
                if re.match(r"  247500", line):
       	       	    temp[5] = re.split('\s+', line.strip())[-3]
                if re.match(r"  248000", line):
       	       	    temp[6] = re.split('\s+', line.strip())[-3]
                if re.match(r"  248500", line):
       	       	    temp[7] = re.split('\s+', line.strip())[-3]
                if re.match(r"  249000", line):
       	       	    temp[8] = re.split('\s+', line.strip())[-3]
                if re.match(r"  249500", line):
       	       	    temp[9] = re.split('\s+', line.strip())[-3]
                else:
                    line_hist = line
            return float(np.average(temp))

# Calculate elastic moduli from pressure
# and other properties such as bulk modulus, etc.
def get_elastic_moduli(pxx, pyy, pzz, pyz, pxz, pxy):
    # C is elastic moduli
    dp, C = np.zeros((6,6)), np.zeros((6,6))
    for i in range(6):
        dp[i,:] = [ (p[i,2]-p[i,1])/2 for p in [pxx, pyy, pzz, pyz, pxz, pxy] ]
    for i in range(6):    
        for j in range(i,6):
            C[i,j] = (dp[i,j] + dp[j,i])/2
    # Bulk modulus (B), shear modulus (G), and poisson_ratio (Po)
    B, G, Po = (C[0,0]+C[1,1]+C[2,2]+2*C[0,1]+2*C[1,2]+2*C[0,2])/9, (C[0,0]-C[0,1])/2, 1.0/(1.0+C[0,0]/C[0,1])
    return (C, B, G, Po)
