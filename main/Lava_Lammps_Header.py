###############################################################################
# This module provides header file for Lammps input script
###############################################################################

import numpy as np
import math
import re
from Lava_Config import *

###############################################################################

lammps_potential_specification = \
"%s\n" %(potential_file_description) + \
"neighbor 2.0 bin\n" + \
"neigh_modify delay 10 check yes\n"

lammps_head = \
"units metal\n" + \
"boundary p p p\n" + \
"atom_style atomic\n\n" + \
"read_data Temp.data\n" + \
"timestep 0.002\n"

lammps_thermo = \
"thermo 50\n" + \
"thermo_style custom step pe temp press pxx pyy pzz pyz pxz pxy lx ly lz\n"

lammps_minimize = \
"min_style cg\n" + \
"minimize 1e-10 1e-10 5000 10000\n"


lammps_fix_box = \
"fix 1 all box/relax aniso 0.0 vmax 0.001\n"

"""
lammps_fix_box = \
"fix 1 all box/relax aniso 0.0 couple xy vmax 0.001\n"
"""

lammps_freeze_x_y = \
"fix 1 all setforce 0 0 NULL\n" + \
"fix 2 all temp/berendsen 0.0 0.0 .2\n" + \
"fix 3 all nve\n" + \
"run 500\n"

lammps_write_data = \
"write_data Relaxed.data\n"

lammps_compute_energy = \
"compute peratom all pe/atom\n"

lammps_thermo_melt = \
"thermo 500\n" + \
"thermo_style custom step etotal pe ke press pxx pyy pzz temp c_LiquidTemp c_SolidTemp\n"

class get_lammps_script:

    def __init__(self, Lammps_file = "Lammps.in"):
        self.Lammps_file = Lammps_file
        self.outfile = open(self.Lammps_file,"w")
        self.outfile.write("%s\n" %(lammps_head))
        self.outfile.write("%s\n" %(lammps_potential_specification))

    def Static(self):    
        self.outfile.write("%s\n" %(lammps_thermo)) 
        self.outfile.write("run 0\n")

    def Minimize(self):
        self.outfile.write("%s\n" %(lammps_thermo))
        self.outfile.write("%s\n" %(lammps_minimize))
        self.outfile.write("%s\n" %(lammps_write_data))

    def Relax_Box(self):
        self.outfile.write("%s\n" %(lammps_fix_box))
        self.outfile.write("%s\n" %(lammps_thermo))
        self.outfile.write("%s\n" %(lammps_minimize))
        self.outfile.write("%s\n" %(lammps_write_data))

    def Deform(self, deform = "Z", delta = 0):
        self.outfile.write("%s\n" %(lammps_thermo))
        if len(deform) == 1:  # x or y or z
            self.outfile.write("change_box all %s delta 0 %f remap units box\n" %(deform,delta))            
        else:  # yz or xz or xy
            self.outfile.write("change_box all %s delta %f remap units box\n" %(deform,delta))
        self.outfile.write("run 0\n")

    def Freeze_X_Y(self):
        self.outfile.write("%s\n" %(lammps_thermo))
        self.outfile.write("%s\n" %(lammps_freeze_x_y))
        self.outfile.write("%s\n" %(lammps_write_data))
       
    def Fix_npt(self, T):
        T = max(T,1)  # Target temperature cannot be zero in fix npt
        self.outfile.write("velocity all create %d 4928459 dist gaussian\n" %(2*T))      
        self.outfile.write("fix 1 all npt temp %d %d .1 iso 0 0 10\n" %(T, T))
        self.outfile.write("%s\n" %(lammps_thermo))
        self.outfile.write("run 25000  # equilibrate for 50 ps\n")
 
    def Fix_rdf(self):
        self.outfile.write("compute rdf all rdf 500\n")
        self.outfile.write("fix rdf all ave/time 1 1 1 c_rdf[*] ave running file RDF.dat mode vector\n")
        self.outfile.write("run 0\n")

    def Fix_liquid_rdf(self, T):
        self.outfile.write("variable Tm equal %d\n" %T)
        self.outfile.write("variable Tm2 equal 2*${Tm}\n\n")
        self.outfile.write("velocity all create ${Tm2} 4928459 dist gaussian\n")
        self.outfile.write("fix 1 all npt temp ${Tm} ${Tm} 0.1 iso 0 0 0.2\n\n")
        self.outfile.write("%s\n" %(lammps_thermo))
        self.outfile.write("run 25000 # equilibrate for 100 ps\n\n")
        self.outfile.write("compute rdf all rdf 500 cutoff 6.0\n")
        self.outfile.write("fix rdf all ave/time 1 1 1 c_rdf[*] ave running file Liquid_RDF_${Tm}.dat mode vector\n")
        self.outfile.write("run 0\n")

    def Two_phase_melt(self, T):
        self.outfile.write("variable Tm equal %d\n" %T)
        self.outfile.write("variable Tm2 equal 2*${Tm}\n")
        self.outfile.write("variable Tm_melt equal ${Tm}*1.2+300\n\n")
        self.outfile.write("variable lz equal \"lz\"\n")
        self.outfile.write("variable z_cen equal ${lz}/2\n")
        self.outfile.write("region liquid block INF INF INF INF INF ${z_cen} units box\n")
        self.outfile.write("group liquid region liquid\n")
        self.outfile.write("group solid subtract all liquid\n")
        self.outfile.write("#set group liquid type 2\n\n")
        self.outfile.write("compute 2 all pe/atom\n")
        self.outfile.write("compute LiquidTemp liquid temp\n")
        self.outfile.write("compute SolidTemp solid temp\n\n")
        self.outfile.write("%s\n" %(lammps_minimize))
        self.outfile.write("%s\n" %(lammps_thermo_melt))
        self.outfile.write("#--------Equilibrate at Tm--------#\n")
        self.outfile.write("reset_timestep 0\n")
        self.outfile.write("velocity all create ${Tm2} 4928459 dist gaussian\n")
        self.outfile.write("fix 1 all npt temp ${Tm} ${Tm} 0.1 iso 0 0 0.2\n\n")
        self.outfile.write("#dump dnve all custom 5000 Step1_Equi.* id type x y z c_2\n")
        self.outfile.write("run 10000\nunfix 1\n#undump dnve\n\n")
        self.outfile.write("#--------Melt liquid at Tm_melt--------#\n")
        self.outfile.write("reset_timestep 0\n")
        self.outfile.write("fix 1 liquid npt temp ${Tm_melt} ${Tm_melt} 2 x 0 0 1.0\n\n")
        self.outfile.write("#dump dnve all custom 5000 Step2_Heat.* id type x y z c_2\n")
        self.outfile.write("run 15000\nunfix 1\n#undump dnve\n\n")
        self.outfile.write("#--------Cool liquid to Tm--------#\n")
        self.outfile.write("reset_timestep 0\n")
        self.outfile.write("fix 1 liquid npt temp ${Tm} ${Tm} 2 x 0 0 1.0\n\n")
        self.outfile.write("#dump dnve all custom 5000 Step3_Cool.* id type x y z c_2\n")
        self.outfile.write("run 15000\nunfix 1\n#undump dnve\n\n")
        self.outfile.write("#--------Fix nph on entire system--------#\n")
        self.outfile.write("reset_timestep 0\n")
        self.outfile.write("fix 1 all nph iso 0.0 0.0 0.2\n\n")
        self.outfile.write("dump dnve all custom 25000 Melt_nph.* id type x y z c_2\n")
        self.outfile.write("run 250000\n")

    def uniaxial_deform(self,phase,temp,deform_mode,rate,final_strain):
        self.outfile.write("%s\n" %(lammps_compute_energy))
        self.outfile.write("variable Tm equal %d\n" %temp)
        self.outfile.write("variable phase string %s\n" %phase)
        self.outfile.write("variable mode equal %s\n" %deform_mode)
        self.outfile.write("variable strainf equal %f\n" %final_strain)
        self.outfile.write("variable rate equal %f\n" %rate)
        self.outfile.write("variable iter equal v_strainf/(0.002*v_rate)\n")
        self.outfile.write("#--------Equilibrate at Tm--------#\n")
        self.outfile.write("reset_timestep 0\n")
        self.outfile.write("velocity all create ${Tm} 4928459 dist gaussian\n")
        self.outfile.write("fix 1 all npt temp ${Tm} ${Tm} 0.1 x 0 0 0.1 y 0 0 0.1 z 0 0 0.1\n\n")
        self.outfile.write("thermo 1000\n")
        self.outfile.write("thermo_style custom step lx ly lz press pxx pyy pzz pe temp\n")
        self.outfile.write("#--------Equilibration--------#\n")
        self.outfile.write("run 10000\nunfix 1\n\n")
        self.outfile.write("fix 1 all npt temp ${Tm} ${Tm} 0.1 x 0 0 0.1 y 0 0 0.1 z 0 0 0.1\n\n")
        self.outfile.write("run 10000\nunfix 1\n\n")
        self.outfile.write("#--------Store final cell length for strain calculations--------#\n")
        self.outfile.write("variable tmp equal lx\n")
        self.outfile.write("variable Lx0 equal ${tmp}\n")
        self.outfile.write("variable tmp equal ly\n")
        self.outfile.write("variable Ly0 equal ${tmp}\n")
        self.outfile.write("variable tmp equal lz\n")
        self.outfile.write("variable Lz0 equal ${tmp}\n")
        self.outfile.write("print 'Initial Length, Lx0: ${Lx0}'\n")
        self.outfile.write("print 'Initial Length, Ly0: ${Ly0}'\n")
        self.outfile.write("print 'Initial Length, Lz0: ${Lz0}'\n")
        self.outfile.write("#--------DEFORMATION--------#\n")
        self.outfile.write("reset_timestep	0\n")
        if (deform_mode==1):
                self.outfile.write("fix		1 all npt temp ${Tm} ${Tm} 1 y 0 0 1 z 0 0 1\n")
                self.outfile.write("fix         2 all deform 1 x erate ${rate} units box remap x\n")
                self.outfile.write("variable strain equal (lx-v_Lx0)/v_Lx0\n")
        elif (deform_mode==2):
       	       	self.outfile.write("fix         1 all npt temp ${Tm} ${Tm} 1 x 0 0 1 z 0 0 1\n")
                self.outfile.write("fix         2 all deform 1 y erate ${rate} units box remap x\n")
                self.outfile.write("variable strain equal (ly-v_Ly0)/v_Ly0\n")
        elif (deform_mode==3):
                self.outfile.write("fix         1 all npt temp ${Tm} ${Tm} 1 x 0 0 1 y 0 0 1\n")
                self.outfile.write("fix         2 all deform 1 z erate ${rate} units box remap x\n")
                self.outfile.write("variable strain equal (lz-v_Lz0)/v_Lz0\n")
        elif (deform_mode==4):
                self.outfile.write("change_box	all triclinic\n")
                self.outfile.write("fix         1 all npt temp ${Tm} ${Tm} 1 x 0 0 1 y 0 0 1 z 0 0 1 xz 0 0 1 xy 0 0 1\n")
                self.outfile.write("fix         2 all deform 1 yz erate ${rate} units box remap x\n")
                self.outfile.write("variable strain equal yz/v_Lz0\n")
        elif (deform_mode==5):
                self.outfile.write("change_box  all triclinic\n")
                self.outfile.write("fix         1 all npt temp ${Tm} ${Tm} 1 x 0 0 1 y 0 0 1 z 0 0 1 yz 0 0 1 xy 0 0 1\n")
                self.outfile.write("fix         2 all deform 1 xz erate ${rate} units box remap x\n")      
                self.outfile.write("variable strain equal xz/v_Lz0\n")
        elif (deform_mode==6):
       	       	self.outfile.write("change_box  all triclinic\n")
       	       	self.outfile.write("fix         1 all npt temp ${Tm} ${Tm} 1 x 0 0 1 y 0 0 1 z 0 0 1 yz 0 0 1 xz 0 0 1\n")
                self.outfile.write("fix         2 all deform 1 xy erate ${rate} units box remap x\n")
                self.outfile.write("variable strain equal xy/v_Ly0\n")
        self.outfile.write("#--------Output strain and stress info to file--------#\n")
        self.outfile.write("#for units metal, pressure is in [bars] = 100 [kPa] = 1/10000 [GPa]#\n")
        self.outfile.write("#p2, p3, p4 are in GPa#\n")
        self.outfile.write("variable p1 equal v_strain\n")
        self.outfile.write("variable p2 equal -pxx/10000\n")
        self.outfile.write("variable p3 equal -pyy/10000\n")
        self.outfile.write("variable p4 equal -pzz/10000\n")
        self.outfile.write("variable p5 equal -pyz/10000\n")
        self.outfile.write("variable p6 equal -pxz/10000\n")
        self.outfile.write("variable p7 equal -pxy/10000\n")
        self.outfile.write("fix 3 all print 100 '${p1} ${p2} ${p3} ${p4} ${p5} ${p6} ${p7}' file Stress_strain_${mode}_${phase}.txt screen no\n")
        self.outfile.write("dump 1 all custom 250 uniaxial_deform id type x y z c_peratom\n")
        self.outfile.write("thermo_style	custom step v_strain temp v_p2 v_p3 v_p4 v_p5 v_p6 v_p7 ke pe press\n")  
        self.outfile.write("run		${iter}\n")        
        self.outfile.write("%s\n" %(lammps_write_data))
