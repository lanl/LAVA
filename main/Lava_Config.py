###############################################################################
# Config file for Lava Wrapper
# Includes the lammps executable name and potential file name
###############################################################################

lammps_executable = "lmp_mpi"
element = "Sn"
potential_file = ["library.meam","Sn.meam"]

"""
potential_file_description = \
"pair_style eam/alloy\n" + \
"pair_coeff * * %s %s\n" %(potential_file[0], element)


"""

potential_file_description = \
"pair_style meam/c\n" + \
"pair_coeff * * %s %s %s %s\n" %(potential_file[0],element,potential_file[1],element)


