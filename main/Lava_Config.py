###############################################################################
# Config file for Lava Wrapper
# Includes the lammps executable name and potential file name
###############################################################################

lammps_executable = "lmp_mpi"
element = "Al"
potential_file = ["Al99.eam.alloy"]


potential_file_description = \
"pair_style eam/alloy\n" + \
"pair_coeff * * %s %s\n" %(potential_file[0], element)


#potential_file_description = \
#"pair_style meam/c\n" + \
#"pair_coeff * * %s %s %s %s\n" %(potential_file[0],element,potential_file[1],element)


