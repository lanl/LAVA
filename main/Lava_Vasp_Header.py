###############################################################################
# This module provides header file for vasp INCAR
# Settings in incar_header are global settings that 
# are copied to every INCAR file in run time
# Simulation-speficic settings such as NSW, IBRION, ISIF
# are set independently for each type of calculation
# passed as parameters when invoking functions in Lava_Wrapper.py
###############################################################################

import numpy as np
import math
import re

###############################################################################

# VASP INCAR header
incar_header = \
"GGA=PE\n" + \
"ISMEAR=-5\n" + \
"POTIM=0.1\n" + \
"SYMPREC=1E-6\n" + \
"EDIFF=1E-6\n" + \
"EDIFFG=-0.01\n" + \
"ENCUT=500\n" + \
"ALGO=N\n" + \
"NPAR=4\n"
		
# This is for generating VASP INCAR file
def get_vasp_script(NSW,IBRION,ISIF,NFREE=0):
    incar_file = ''.join(open('INCAR.inp',"r").readlines())
    for str_set in (("NSW_val", str(NSW)), ("IBRION_val", str(IBRION)), ("ISIF_val", str(ISIF))):
        incar_file = incar_file.replace(*str_set)
    if NFREE != 0:
        incar_file += "NFREE=%d\n" %NFREE
    with open("INCAR","w") as outfile:
        outfile.write(incar_file)