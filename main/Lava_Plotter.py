###############################################################################
# This module provides utility functions/wrappers
# for plotting outputs as ".jpeg' format 
###############################################################################

import numpy as np
import os
import math
from numpy import genfromtxt
from io import BytesIO
import matplotlib.pyplot as plt
from numpy import ma
from matplotlib import ticker, cm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from Lava_Utility import *

###############################################################################

# This is for reading in a list of files. 
def enumerate_files(*filelist, skip_line=0, delimiter=None):
    filedir = os.path.dirname(os.path.realpath('__file__'))
    array_list, skipped_lines_list = [], []
    for file in filelist:
        with open (os.path.join(filedir,'%s' %file)) as infile:
            array, skipped_lines = [], []
            # Skip the first few lines specified by skip_line
            for i, line in enumerate(infile):
                if i >= skip_line: 
                    array.append(genfromtxt(BytesIO(line.encode('utf8')), delimiter=delimiter))
                else:
                    skipped_lines.append(line)
            array_list.append(array)
            skipped_lines_list.append(skipped_lines)
    return (skipped_lines_list, array_list)


def plot_1D_wrapper(plt, xlabel_params=None, ylabel_params=None, tick_params=None, \
                    legend_params=None, title_params=None, output_file=None):
    if xlabel_params: plt.xlabel(xlabel_params['title'], size=xlabel_params['size'])
    if ylabel_params: plt.ylabel(ylabel_params['title'], size=ylabel_params['size'])  
    if tick_params: plt.tick_params(labelsize=tick_params['labelsize'], pad=tick_params['pad'])   
    if legend_params: plt.legend(loc=legend_params['loc'], prop={'size': legend_params['size']})
    if title_params: plt.title(title_params['title'], size=title_params['size'])
    if output_file: plt.savefig(output_file)


def plot_2D_wrapper(fig, plt, ax, cs, xlabel_params=None, ylabel_params=None, tick_params=None, 
                cbar_params=None, title_params=None, output_file=None):
    if xlabel_params: ax.set_xlabel(xlabel_params['title'], size=xlabel_params['size'])
    if ylabel_params: ax.set_ylabel(ylabel_params['title'], size=ylabel_params['size'])  
    if tick_params: ax.tick_params(labelsize=tick_params['labelsize'], pad=tick_params['pad'])
    for tick in ax.xaxis.get_major_ticks(): tick.label.set_fontsize(26)
    for tick in ax.yaxis.get_major_ticks(): tick.label.set_fontsize(26)
    if cbar_params: 
        cbar = fig.colorbar(cs)
        cbar.ax.tick_params(labelsize=tick_params['labelsize'], pad=tick_params['pad'])
        cbar.set_label(cbar_params['title'], size = cbar_params['size'], \
                       labelpad = cbar_params['labelpad'], rotation=90)
    if title_params: plt.title(title_params['title'], size=title_params['size'])
    if output_file: plt.savefig(output_file) 


def get_color_list():
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728',
          '#9467bd','#8c564b','#e377c2','#7f7f7f']    
    return colors         


def plot_EOS(input_file, output_file, plot_axis_x="volume", plot_axis_y="energy"):
    # Load input
    skipped_lines, array_list = enumerate_files(input_file, skip_line=1, delimiter='  ')
    phase_list = skipped_lines[0][0].split ("      ")
    EOS_data = np.array(array_list[0])
    # Plot
    plot_axis_map = {"density":0, "volume":1, "energy":2}
    plot_axis_title_map = {"density": 'Density, (g/$\mathregular{{cm}^3}$)', 
                           "volume": 'Volume, ($\mathregular{Å^3}$)', 
                           "energy": 'Energy, (eV/atom)'}
    x_axis, y_axis = plot_axis_map[plot_axis_x], plot_axis_map[plot_axis_y]
    colors = get_color_list()
    fig = plt.figure(figsize=(16,12))
    for i in range(len(phase_list)-1):
        plt.plot(EOS_data[:,3*i+x_axis], EOS_data[:,3*i+y_axis],'-*',color=colors[i], 
                 markersize=16, linewidth=3, label=str(phase_list[i]).strip(' #'))
    plot_1D_wrapper(plt, 
                    xlabel_params={'title': plot_axis_title_map[plot_axis_x], 'size': 30}, 
                    ylabel_params={'title': plot_axis_title_map[plot_axis_y], 'size': 30}, 
                    tick_params={'labelsize': 28, 'pad': 12},
                    legend_params={'loc': 1, 'size': 26},
                    output_file="%s.jpeg" %(output_file))


def plot_RDF(input_file, output_file):
    # Load input
    skipped_lines, array_list = enumerate_files(input_file, skip_line=1, delimiter='  ')
    T_list = skipped_lines[0][0].split ("      ")
    RDF_data = np.array(array_list[0])
    # Plot
    colors = get_color_list()
    fig = plt.figure(figsize=(16,12))
    for i in range(len(T_list)-1):
        plt.plot(RDF_data[:,2*i], RDF_data[:,2*i+1],'-', color=colors[i], linewidth=3, \
                 label=str(T_list[i]).strip(' #')+" K")
    plot_1D_wrapper(plt, 
                    xlabel_params={'title': 'r, (Å)', 'size': 30}, 
                    ylabel_params={'title': 'g(r)', 'size': 30}, 
                    tick_params={'labelsize': 28, 'pad': 12},
                    legend_params={'loc': 1, 'size': 26},
                    output_file="%s.jpeg" %(output_file))    
    

def plot_thermal_expansion(input_file, output_file):
    # Load input
    skipped_lines, array_list = enumerate_files(input_file, skip_line=1, delimiter='  ')
    Thermal_expansion_data = np.array(array_list[0])
    # Plot
    colors = get_color_list()
    fig = plt.figure(figsize=(16,12))
    plt.plot(Thermal_expansion_data[:,0], Thermal_expansion_data[:,4],'-*', \
             markersize=20, color=colors[0], linewidth=4)
    plot_1D_wrapper(plt, 
                    xlabel_params={'title': 'Temperature, (K)', 'size': 30}, 
                    ylabel_params={'title': 'Lattice constant, (Å)', 'size': 30}, 
                    tick_params={'labelsize': 28, 'pad': 12},
                    legend_params={'loc': 1, 'size': 26},
                    output_file="%s.jpeg" %(output_file))    

	
def plot_Bain(input_file, output_file):
    # Load input
    _, array_list = enumerate_files(input_file, skip_line=1, delimiter='  ')
    Bain_data = np.array(array_list[0])
    # Plot
    colors = get_color_list()
    fig = plt.figure(figsize=(16,12))
    plt.plot(Bain_data[:,1], Bain_data[:,2],'-*', markersize=16, color=colors[0],linewidth=3)
    plot_1D_wrapper(plt, 
                    xlabel_params={'title': 'c/a', 'size': 30}, 
                    ylabel_params={'title': 'Energy, (eV/atom)', 'size': 30}, 
                    tick_params={'labelsize': 28, 'pad': 12},
                    legend_params={'loc': 1, 'size': 26},
                    output_file="%s.jpeg" %(output_file))
  
    
def plot_Bain_2D(input_X, input_Y, input_Z, output_file):
    # Load input
    _, array_list = enumerate_files(input_X, input_Y, input_Z, delimiter='  ')
    Mesh_X, Mesh_Y, Mesh_Z = np.array(array_list[0]), np.array(array_list[1]), np.array(array_list[2]) 
    # Plot
    fig, ax = plt.subplots()
    plt.gcf().set_size_inches(16, 12)
    levels = MaxNLocator(nbins=20).tick_values(Mesh_Z.min(), Mesh_Z.max())
    cs = ax.contourf(Mesh_X, Mesh_Y, Mesh_Z, levels=levels, cmap="RdBu_r")
    plot_2D_wrapper(fig, plt, ax, cs,
                    xlabel_params={'title': 'Volume', 'size': 30}, 
                    ylabel_params={'title': 'c/a', 'size': 30}, 
                    tick_params={'labelsize': 28, 'pad': 12},
                    cbar_params={'title': 'Energy, (eV)', 'size': 26, 'labelpad': 28},
                    output_file="%s.jpeg" %(output_file))
  
    
def plot_Gamma_2D(input_X, input_Y, input_Z, output_file):
      # Load input
    _, array_list = enumerate_files(input_X, input_Y, input_Z, delimiter='  ')
    Mesh_X, Mesh_Y, Mesh_Z = np.array(array_list[0]), np.array(array_list[1]), np.array(array_list[2]) 
    # Plot
    fig, ax = plt.subplots()
    plt.gcf().set_size_inches(16, 12)
    levels = MaxNLocator(nbins=20).tick_values(Mesh_Z.min(), Mesh_Z.max())
    cs = ax.contourf(Mesh_X, Mesh_Y, Mesh_Z, levels=levels, cmap="RdBu_r")
    plot_2D_wrapper(fig, plt, ax, cs,
                    xlabel_params={'title': 'Normalized <112> displacement', 'size': 30}, 
                    ylabel_params={'title': 'Normalized <110> displacement', 'size': 30}, 
                    tick_params={'labelsize': 28, 'pad': 16},
                    cbar_params={'title': 'Energy, (mJ/$\mathregular{m^{2}}$)', 'size': 26, 'labelpad': 26},
                    output_file="%s.jpeg" %(output_file))    


def plot_GSFE(input_file, output_file):
    # Load input   
    _, array_list = enumerate_files(input_file, skip_line=1, delimiter='  ')
    GSFE_data = np.array(array_list[0])
    # Plot
    colors = get_color_list()
    labels = ['Slip', 'Twin']
    labels = ['Slip']
    fig = plt.figure(figsize=(16,12))
    for i in range(1):
        plt.plot(GSFE_data[:,0], GSFE_data[:,i+1],'-', color=colors[i], label=labels[i], linewidth=3)    
    plot_1D_wrapper(plt, 
                    xlabel_params={'title': 'Normalized <001> displacement', 'size': 30}, 
                    ylabel_params={'title': 'Energy, (mJ/$\mathregular{m^{2}}$)', 'size': 30}, 
                    tick_params={'labelsize': 28, 'pad': 12},
                    legend_params={'loc': 'upper left', 'size': 26},
                    output_file="%s.jpeg" %(output_file))   


def plot_Gamma_3D(input_X, input_Y, input_Z, output_file):   
    # Load input
    _, array_list = enumerate_files(input_X, input_Y, input_Z, delimiter='  ')
    Mesh_X, Mesh_Y, Mesh_Z = np.array(array_list[0]), np.array(array_list[1]), np.array(array_list[2])   
    # Plot 
    fig = plt.figure(figsize=(16,12))
    ax = plt.axes(projection='3d')
    ax.plot_surface(Mesh_X, Mesh_Y, Mesh_Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Normalized <112> displacement', size = 20, labelpad=20)
    ax.set_ylabel('Normalized <110> displacement', size = 20, labelpad=20)
    ax.set_zlabel('Energy, (mJ/$\mathregular{m^{2}}$)', size = 20, labelpad=28)
    ax.tick_params(labelsize=28, pad=8)
    for tick in ax.xaxis.get_major_ticks(): tick.label.set_fontsize(20) 
    for tick in ax.yaxis.get_major_ticks(): tick.label.set_fontsize(20) 
    for tick in ax.zaxis.get_major_ticks(): tick.label.set_fontsize(20)     
    plt.savefig("%s.jpeg" %(output_file))


def plot_stress_strain(input_file,mode, output_file):
    # Load input
    skipped_lines, array_list = enumerate_files(input_file, skip_line=1, delimiter=' ')
    stress_strain_data = np.array(array_list[0])
    strain = np.array(stress_strain_data[:,0])
    if (mode==1):
        stress = np.array(stress_strain_data[:,1])
    elif (mode==2):
        stress = np.array(stress_strain_data[:,2])
    elif (mode==3):
        stress = np.array(stress_strain_data[:,3])
    elif (mode==4):
       	stress = np.array(stress_strain_data[:,4])
    elif (mode==5):
       	stress = np.array(stress_strain_data[:,5])
    elif (mode==6):
       	stress = np.array(stress_strain_data[:,6])
    # Plot
    colors = get_color_list()
    fig = plt.figure(figsize=(16,12))
    plt.plot(strain, stress,'-*', \
             markersize=20, color=colors[0], linewidth=4)
    plot_1D_wrapper(plt,
                    xlabel_params={'title': 'Strain', 'size': 30},
                    ylabel_params={'title': 'Stress (GPa)', 'size': 30},
                    tick_params={'labelsize': 28, 'pad': 12},
                    legend_params={'loc': 1, 'size': 26},
                    output_file="%s.jpeg" %(output_file))
