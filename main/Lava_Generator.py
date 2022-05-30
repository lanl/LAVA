###############################################################################
# This module provides classes for generating bulk and defect configurations
# for performing the calculations
# Supports three file formats: lammps data file, POSCAR file, and extended xyz file
###############################################################################

import numpy as np
import os
from os import sys
import math
import random
import re
from numpy.linalg import inv
from Lava_Config import *

###############################################################################

class Build_Crystal:
    def __init__(self, phase, alat, clat, lat_rep, orz, nlayer=24, vac=0, d_tol=0.01, verbose=False):
        self.phase = phase
        self.alat = alat
        self.clat = clat
        self.lat_rep = lat_rep
        self.orz = orz
        self.nlayer = nlayer
        self.vac = vac
        self.d_tol = d_tol
        self.verbose = verbose
        self.nx, self.ny, self.nz = self.lat_rep
            
    def Build(self):
        phase_list = ["SC", "BCC", "FCC", "HCP", "DC", "DIAMOND", "DHCP", \
                      "W", "OMEGA", "A5_BETA_SN", "A5_GAMMA_SN", "A15_BETA_W", "L12", "9R"]
        if self.phase.upper() in phase_list:
            if self.phase.upper() == "DIAMOND": self.phase = "DC"
            if self.phase.upper() == "OMEGA": self.phase = "W"
            ncell,basis,atype,cube = getattr(self, "Build_"+self.phase)()
        else:
            print ("{0} is not implemented.".format(self.phase))
            sys.exit()                      
        angle,nrep,zrep = self.Calculate_Angle()
        data_rep = self.Replicate_Cell(ncell,basis*cube,atype,cube,nrep,nrep,nrep)
        data_build, NAN, box = self.Build_Cell(data_rep,angle,zrep) 
        data = self.Replicate_Cell(NAN,data_build[:,-3:],data_build[:,1],box,self.nx,self.ny,self.nz)
        data[:,-3:] = data[:,-3:] * self.alat
        # Add surface, half on each side
        data[:,-1] = data[:,-1] + self.vac/2
        NAN *= self.nx*self.ny*self.nz
        box_2d = np.zeros((3,2))
        box_2d[:,1] = np.multiply(box, [self.alat*self.nx,self.alat*self.ny,self.alat*self.nz]) + [0,0,self.vac]  
        return (data, NAN, box_2d)   
    
    def Calculate_Angle(self):
        nrep, zrep = 20, True 
        blat, multilplier = 1, 1
        if self.phase.upper() in ["HCP", "DHCP", "W", "OMEGA"]:
            blat, multilplier = 1.73205, 2
            if self.orz[0] != 0 or self.orz[1] != 0:
                nrep, zrep = math.ceil(self.nlayer/math.sqrt(self.orz[0]**2+self.orz[1]**2+self.orz[2]**2)), False
                assert self.nlayer > 0, "Number of layers must be defined for HCP along none-[001] direction"
                if self.nz > 1:
                    self.nz = 1
                    print ("Warning: No replication allowed for HCP along none-[001] direction")  
        if self.orz[2] != 0:
            angle = math.atan(self.clat*np.sqrt(self.orz[0]**2+(self.orz[1]/blat)**2)/self.orz[2]*multilplier)
        else:
            angle = math.pi/2
        return (angle, nrep, zrep)   
       
    def get_atype(self,ncell):
        return np.array([1]*ncell,dtype=np.int)

    def Build_SC(self):
        ncell = 1
        basis = np.array([0,0,0])
        cube = np.array([1,1,self.clat])
        return (ncell,basis,self.get_atype(ncell),cube)
    
    def Build_BCC(self):
        ncell = 2
        basis = np.array([[0,0,0], [0.5,0.5,0.5]])
        cube = np.array([1,1,self.clat])            
        return (ncell,basis,self.get_atype(ncell),cube)    
        
    def Build_FCC(self):
        ncell = 4
        basis = np.array([[0,0,0], [0,0.5,0.5], [0.5,0,0.5],[0.5,0.5,0]])
        cube = np.array([1,1,self.clat]) 
        return (ncell,basis,self.get_atype(ncell),cube)
      
    def Build_HCP(self):
        ncell = 4
        basis = np.array([[0,0,0], [0.5,0.5,0], [0.5,5./6,0.5],[0,1./3,0.5]])
        cube = np.array([1,1.73205,self.clat])
        return (ncell,basis,self.get_atype(ncell),cube) 
    
    def Build_DC(self):   
        ncell = 8
        basis = np.array([[0,0,0], [0,0.5,0.5], [0.5,0,0.5],[0.5,0.5,0],\
                          [0.25,0.25,0.25], [0.25,0.75,0.75], [0.75,0.25,0.75], [0.75,0.75,0.25]])
        cube = np.array([1,1,self.clat]) 
        return (ncell,basis,self.get_atype(ncell),cube) 

    def Build_DHCP(self):
        ncell = 8
        basis = np.array([[0,0,0], [0.5,0.5,0], [0.5,5./6,0.25],[0,1./3,0.25],\
                          [0,0,0.5], [0.5,0.5,0.5], [0.5,1./6,0.75], [0,2./3,0.75]])
        cube = np.array([1,1.73205,self.clat])
        return (ncell,basis,self.get_atype(ncell),cube)
    
    def Build_W(self):
       ncell = 6
       basis = np.array([[0,0,0], [0.5,0.5,0], [0.5,1./6,0.5],\
                         [0,1./3,0.5], [0,2./3,0.5], [0.5,5./6,0.5]])
       cube = np.array([1,1.73205,self.clat])
       return (ncell,basis,self.get_atype(ncell),cube)

    def Build_A5_Beta_Sn(self):
       ncell = 4
       basis = np.array([[0,0,0], [0.5,0.5,0.5], [0,0.5,0.25], [0.5,0,0.75]])
       cube = np.array([1,1,self.clat])
       return (ncell,basis,self.get_atype(ncell),cube)

    def Build_A5_Gamma_Sn(self):
       ncell = 4
       basis = np.array([[0,0,0], [0.5,0,0], [0,0.5,0.5], [0.5,0.5,0.5]])
       cube = np.array([1,0.9,self.clat])
       return (ncell,basis,self.get_atype(ncell),cube)

    def Build_A15_Beta_W(self):
      ncell = 8
      basis = np.array([[0,0,0], [0.5,0.25,0], [0.5,0.75,0],[0,0.5,0.25],\
                          [0.5,0.5,0.5], [0.25,0,0.5], [0.75,0,0.5], [0,0.5,0.75]])
      cube = np.array([1,1,self.clat])
      return (ncell,basis,self.get_atype(ncell),cube)
  
    def Build_L12(self):
        ncell = 3
        basis = np.array([[0,0.5,0.5], [0.5,0,0.5],[0.5,0.5,0]])
        cube = np.array([1,1,self.clat]) 
        return (ncell,basis,self.get_atype(ncell),cube)
    
    def Build_9R(self):
        ncell = 18
        basis = np.array([[0,0,0], [0.5,0.5,0], [1./6,0.5,1./9],[2./3,0,1./9],\
                          [1./3,0,2./9], [5./6,0.5,2./9], [2./3,0,3./9], [1./6,0.5,3./9],\
                          [5./6,0.5,4./9], [1./3,0,4./9], [0,0,5./9], [0.5,0.5,5./9],\
                          [1./3,0,6./9], [5./6,0.5,6./9], [0.5,0.5,7./9], [0,0,7./9],\
                          [2./3,0,8./9], [1./6,0.5,8./9]])
        cube = np.array([1,1/1.73205,self.clat])
        return (ncell,basis,self.get_atype(ncell),cube) 

    # Replicate the cell by nx * ny * nz times
    def Replicate_Cell(self,ncell,basis,atype,cube,nrep_x,nrep_y,nrep_z):
        NAN = ncell*nrep_x*nrep_y*nrep_z
        data_rep = np.zeros((NAN,5))
        data_rep[:ncell,-3:] = basis
        for i in range(nrep_x):
            for j in range(nrep_y):
                for k in range(nrep_z):
                    inc = i*nrep_y*nrep_z + j*nrep_z + k
                    data_rep[inc*ncell:(inc+1)*ncell,1] = atype[:]
                    data_rep[inc*ncell:(inc+1)*ncell,-3:] = data_rep[:ncell,-3:] + np.multiply(cube, np.array([i, j, k]))                  
        return (data_rep)
        
    # Cut an orthorgonal box based on search_id, xlen_id, ylen_id, zlen_id
    def Build_Cell(self,data_rep,angle,zrep=True):
        data_build, NAN, atom_layer, search_id, xlen_id, ylen_id, zlen_id, \
         = self.Find_Orthogonal_Cell(data_rep,angle,zrep)
        x0, y0, z0 = data_build[search_id,-3], data_build[search_id,-2], data_build[search_id,-1]
        xlo, xhi = x0-self.d_tol, data_build[xlen_id,-3]-self.d_tol
        ylo, yhi = y0-self.d_tol, data_build[ylen_id,-2]-self.d_tol
        zlo, zhi = z0-self.d_tol, data_build[zlen_id,-1]-self.d_tol
        if self.phase.upper() == "DHCP":
            zhi = zhi + zhi - zlo
        elif self.phase.upper() == "9R":
            zhi = zhi + (zhi-zlo)/5*4
        elif self.phase.upper() == "A5_GAMMA_SN":
            xhi = xhi + xhi - xlo
        atom_count = 0
        # By default, xlo, ylo and zlo are set to 0
        for i in range(NAN):
            if data_build[i,-3] > xlo and data_build[i,-3] < xhi and data_build[i,-2] > ylo \
            and data_build[i,-2] < yhi and data_build[i,-1] > zlo and data_build[i,-1] < zhi:
                data_build[atom_count,0] = atom_count + 1
                data_build[atom_count,1] = data_build[i,1]
                data_build[atom_count,2:] = data_build[i,2:] - [x0,y0,z0]
                atom_count+=1
        box = [xhi-xlo, yhi-ylo, zhi-zlo]        
        return (data_build[:atom_count,:], atom_count, box)
      
    # Build the cell by constructing a orthogonal box
    def Find_Orthogonal_Cell(self,data_rep,angle,zrep):
        data_build = self.Rotate_Cell(data_rep,angle) 
        data_build = data_build[np.argsort(data_build[:, -1])]
        y_min, y_max = np.amin(data_build[:,-2],axis=0), np.amax(data_build[:,-2],axis=0)
        y_cen = (y_min+y_max)/2
        NAN = data_build.shape[0]
        atom_layer = np.zeros(NAN,dtype=np.int)
        atom_sorted, current_layer, z_min = 0, 0, data_build[0,-1]
        # Sort the atom to different layers based on Z coordinates
        for i in range(NAN):
            dz = np.abs(data_build[i,-1]-z_min)
            if dz < self.d_tol:
                atom_layer[i] = current_layer + 1
                atom_sorted+=1
            else:
                current_layer+=1
                atom_layer[i] = current_layer + 1
                z_min = data_build[i,-1]
        if self.verbose:
            print ("All atoms sorted: {0} layers located!".format(current_layer))            
            print ("Now searching for periodic images of atoms in X, Y and Z direction")
        search_id, search_layer = np.argmin(data_build[:,-1],axis=0), 1
        for j in range(NAN):
            # x0, y0, z0 defines the origin for the search
            x0, y0, z0 = data_build[search_id,-3], data_build[search_id,-2], data_build[search_id,-1]
            search_x_bound = sys.float_info.max
            if self.verbose:
                print ("Searching start from atom {0}: {1},{2},{3} in layer {4}".format(search_id,x0,y0,z0,search_layer))
            image_x, image_y, image_z, xlen_id, ylen_id, zlen_id = \
            self.Image_Search(data_build,atom_layer,search_layer,NAN,x0,y0,z0,zrep)
            if image_x and image_y and image_z:
                if self.verbose:
                    print ("Image atom {0} found in X direction: {1},{2},{3}"\
                           .format(xlen_id,data_build[xlen_id,-3],data_build[xlen_id,-2],data_build[xlen_id,-1]))
                    print ("Image atom {0} found in Y direction: {1},{2},{3}"\
                           .format(ylen_id,data_build[ylen_id,-3],data_build[ylen_id,-2],data_build[ylen_id,-1]))
                    if zrep:
                        print ("Image atom {0} found in Z direction: {1},{2},{3}"\
                               .format(zlen_id,data_build[zlen_id,-3],data_build[zlen_id,-2],data_build[zlen_id,-1]))
                    else:
                        print ("{0} layers cut in Z direction".format(self.nlayer))
                break
            else:
                search_layer+=1
                if search_layer > current_layer:
                    break
                for i in range(NAN):
                    if atom_layer[i] == search_layer:
                        if data_build[i,-3] < search_x_bound and np.abs(data_build[i,-2]-y_cen) < 2*self.alat:
                            search_x_bound = data_build[i,-3]
                            search_id = i  
        if not (image_x and image_y and image_z):
            print ("No unit cell found!: {0}, {1}, {2}".format(image_x, image_y, image_z))
            sys.exit()
        return (data_build, NAN, atom_layer, search_id, xlen_id, ylen_id, zlen_id)   

    # Rotate the coordinates based on orz
    def Rotate_Cell(self,data_rep,angle):
        if self.verbose:
            print ("Rotation angle = {0} --> {1} degree".format(angle,angle*180/math.pi))
        #Construct the Rodrigues rotation formula
        #[0 -rz ry
        #rz 0 -rx
        #-ry rx 0]
        if self.orz[0] != 0 or self.orz[1] != 0:
            orz_R = math.sqrt(self.orz[0]**2+self.orz[1]**2)
            orz_norm = np.array([self.orz[1], -self.orz[0], 0])/orz_R
            rx,ry,rz = orz_norm[0], orz_norm[1], orz_norm[2]
            Q_s = np.array([[0,-rz,ry], [rz,0,-rx], [-ry,rx,0]])
            Q = np.eye(3) + math.sin(angle)*Q_s + 2*(math.sin(angle/2))**2*(Q_s.dot(Q_s))
            Q_r = np.array([[-ry,rx,0], [rx,ry,0], [0,0,1]])
        else:
            Q, Q_r = np.eye(3), np.eye(3)
        if self.verbose:
            print ("Rotation matrix Q = [[{0},{1},{2}], [{3},{4},{5}], [{6},{7},{8}]]"\
                   .format(Q[0,0],Q[0,1],Q[0,2],Q[1,0],Q[1,1],Q[1,2],Q[2,0],Q[2,1],Q[2,2]))
            print ("Rotation matrix Q_r= [[{0},{1},{2}], [{3},{4},{5}], [{6},{7},{8}]]"\
                   .format(Q_r[0,0],Q_r[0,1],Q_r[0,2],Q_r[1,0],Q_r[1,1],Q_r[1,2],Q_r[2,0],Q_r[2,1],Q_r[2,2]))
        data_build = np.zeros((data_rep.shape[0],5))
        data_build[:,:2] = data_rep[:,:2]   
        data_build[:,-3:] = data_rep[:,-3:].dot(Q).dot(inv(Q_r))     
        return (data_build)
            
    # Search for image atoms along all dimensions
    def Image_Search(self,data_build,atom_layer,search_layer,NAN,x0,y0,z0,zrep):
        image_x, image_y, image_z = False, False, False
        xlen_id, ylen_id, zlen_id = 0, 0, 0
        xlen, ylen, zlen = sys.float_info.max, sys.float_info.max, sys.float_info.max
        for i in range(NAN):
            dx, dy, dz = np.abs(data_build[i,-3]-x0), np.abs(data_build[i,-2]-y0), np.abs(data_build[i,-1]-z0)
            if atom_layer[i] == search_layer:
                if data_build[i,-3] > x0 and dx > self.d_tol and dy < self.d_tol and dz < self.d_tol:
                    if xlen > dx:
                        xlen, xlen_id, image_x = dx, i, True
                if data_build[i,-2] > y0 and dx < self.d_tol and dy > self.d_tol and dz < self.d_tol:
                    if ylen > dy:
                        ylen, ylen_id, image_y = dy, i, True
            if zrep:    # non-hcp, and hcp along none-[0001]
                if atom_layer[i] > search_layer and data_build[i,-1] > z0 and dx < self.d_tol and dy < self.d_tol and dz > self.d_tol:
                    if zlen > dz:
                        zlen, zlen_id, image_z = dz, i, True
            else:       # others
                if atom_layer[i]-search_layer == self.nlayer:
                    if zlen > dz:
                        zlen, zlen_id, image_z = dz, i, True                               
        return(image_x,image_y,image_z,xlen_id,ylen_id,zlen_id)
                        

class Introduce_Defects:
    def __init__(self, input_dir=None, input_file=None, nvac=1, nint=1, sf_mesh=[12,12], \
                 shift_plane=0, Rc=3.46, max_neigh=50, verbose=False):
        self.input_dir = input_dir
        self.input_file = input_file
        self.nvac = nvac
        self.nint = nint
        self.sf_mesh = sf_mesh
        self.shift_plane = shift_plane
        self.Rc = Rc
        self.max_neigh = max_neigh
        self.verbose = verbose
    
    def Read_Input_Data(self):
        filename = os.path.join(self.input_dir,'%s' % self.input_file)
        infile = open(filename, "r")
        header, num_header = [], 9
        for i, line in enumerate(infile):
            header.append(line)
            if i == num_header-1: break
        NAN = int(np.fromstring(header[1],sep=' ',count=1))
        data = np.zeros((NAN,5))
        for i,line in enumerate(infile):
            data[i,:] = np.fromstring(line,  sep=' ', count=5)
        XL, XH = float(re.split('\s+', header[3])[1]), float(re.split('\s+', header[3])[2])
        YL, YH = float(re.split('\s+', header[4])[1]), float(re.split('\s+', header[4])[2])
        ZL, ZH = float(re.split('\s+', header[5])[1]), float(re.split('\s+', header[5])[2])       
        box_2d = np.array([[XL,XH], [YL,YH], [ZL,ZH]])
        return (data, NAN, box_2d)

    def Read_Input_POSCAR(self, freeze_Z=False):
        filename = os.path.join(self.input_dir,'%s' % self.input_file)
        infile = open(filename, "r")
        header = []
        num_header = 8 if freeze_Z else 7
        for i, line in enumerate(infile):
            header.append(line)
            if i == num_header-1:
                break
        NAN = int(np.fromstring(header[5],sep=' ',count=1))
        data = np.zeros((NAN,5))
        for i,line in enumerate(infile):
            data[i,:2] = i+1, 1
            data[i,2:] = np.fromstring(line,  sep=' ', count=3)
        XL, XH = 0, float(re.split('\s+', header[2])[1])
        YL, YH = 0, float(re.split('\s+', header[3])[2])
        ZL, ZH = 0, float(re.split('\s+', header[4])[3])
        box_2d = np.array([[XL,XH], [YL,YH], [ZL,ZH]])
        return (data, NAN, box_2d)
        
    def neigh_build_wrapper(self, data, NAN, box_2d):
        XL, XH, YL, YH, ZL, ZH = box_2d.flatten()
        XBIN, YBIN, ZBIN = int(math.floor((XH-XL)/self.Rc)), int(math.floor((YH-YL)/self.Rc)), \
                            int(math.floor((ZH-ZL)/self.Rc))
        max_atoms = max(3*int((NAN/(XBIN*YBIN*ZBIN))),20)
        if self.verbose:
            XBINSIZE, YBINSIZE, ZBINSIZE = (XH-XL)/XBIN, (YH-YL)/YBIN, (ZH-ZL)/ZBIN
            print ("For link cell, bin size = {0}, {1}, {2}, number of bins = {3}, {4}, {5} in x, y and z direction." \
                   .format(XBINSIZE,YBINSIZE,ZBINSIZE,XBIN,YBIN,ZBIN))
        neigh_list, neigh_id_list, neigh_count = self.neigh_build(data,NAN,XL,XH,XBIN,YL,YH,YBIN,ZL,ZH,ZBIN,max_atoms)        
        return (data, NAN, box_2d, neigh_list, neigh_id_list, neigh_count)

    # Introduce vacancies   
    def Add_Vacancy(self, input_format='data', data=None, NAN=None, box_2d=None):
        if NAN:
            pass
        elif input_format == 'data':
            data, NAN, box_2d = self.Read_Input_Data()
        elif input_format == 'poscar':
            data, NAN, box_2d = self.Read_Input_POSCAR()
        else:
            print ("Wrong input type!")
            sys.exit() 
        atom_delete = np.zeros(NAN, dtype=bool)
        atom_delete[random.sample(range(1, NAN), self.nvac)] = True
        data_out = np.zeros((NAN-self.nvac,5))
        atom_count = 0
        for i in range(NAN):
            if not atom_delete[i]:
                data_out[atom_count,:] = data[i,:]
                atom_count+=1
        return (data_out, atom_count, box_2d)
 
    # Introduce interstitials    
    def Add_Interstitial(self, input_format=None, data=None, NAN=None, box_2d=None):
        if NAN:
            data, NAN, box_2d, neigh_list, neigh_id_list, neigh_count = self.neigh_build_wrapper(data, NAN, box_2d)
        elif input_format == 'data':
            data, NAN, box_2d, neigh_list, neigh_id_list, neigh_count = self.neigh_build_wrapper(self.Read_Input_Data())
        elif input_format == 'poscar':
            data, NAN, box_2d, neigh_list, neigh_id_list, neigh_count = self.neigh_build_wrapper(self.Read_Input_POSCAR())
        else:
            print ("Wrong input type!")
            sys.exit()
        atom_insert = np.zeros(NAN, dtype=bool)
        atom_insert[random.sample(range(NAN), self.nint)] = True
        data_out = np.zeros((NAN+self.nint,5))   
        data_out[:NAN,:] = data[:,:]
        atom_count = NAN
        box_len = [box_2d[0,1] - box_2d[0,0], box_2d[1,1] - box_2d[1,0], box_2d[2,1] - box_2d[2,0]]
        for i in range(NAN):
            if atom_insert[i]:
                j = neigh_id_list[i,random.randint(1,neigh_count[i])]
                dd = data[i,:] - data[j,:]
                for k in range(2,5):
                    if abs(dd[k]) < box_len[k-2]/2:
                        data_out[atom_count,k] = (data[i,k] + data[j,k])/2
                    else:
                        data_out[atom_count,k] = (data[i,k] + data[j,k] - box_len[k-2])/2
                        if data_out[atom_count,k] < 0: data_out[atom_count,k] += box_len[k-2]
                data_out[atom_count,0], data_out[atom_count,1] = atom_count, 1 
                atom_count+=1
                if atom_count == NAN+self.nint: break
        return (data_out, atom_count, box_2d)
    
    def Add_None(self, data=None, NAN=None, box_2d=None):
        return (data, NAN, box_2d)
                    
    # Displace to generate stacking fault
    def Stacking_fault(self, input_format='data', lat_rep=[1,1,1], freeze_Z=True):
        # x [1 1 -2], y [1 -1 0], z [1 1 1]
        # dx = math.sqrt(6)/2*alat, dy = math.sqrt(2)*alat
        if input_format == 'data':
            data, NAN, box_2d = self.Read_Input_Data()
        elif input_format == 'poscar':
            data, NAN, box_2d = self.Read_Input_POSCAR(freeze_Z=True)
        else:
            print ("Wrong input type!")
            sys.exit()
        XL, XH, YL, YH, ZL, ZH = box_2d.flatten()
        box_len = [XH-XL, YH-YL, ZH-ZL]
        z_cutoff = (ZL+ZH)/2-0.1 + self.shift_plane*(YH-YL)/lat_rep[1]*math.sqrt(2)/math.sqrt(3)
        dx, dy = -(XH-XL)/lat_rep[0]/self.sf_mesh[0], (YH-YL)/lat_rep[1]/self.sf_mesh[1]
        if self.shift_plane > 0:
            dx, dy = dx/3, dy/3   
        data_out = np.zeros((self.sf_mesh[0]+1,self.sf_mesh[1]+1,NAN,5))
        for i in range(self.sf_mesh[0]+1):
            for j in range(self.sf_mesh[1]+1):
                data_out[i,j,:,:] = data[:,:]
                for k in range(NAN):
                    if data[k,-1] > z_cutoff:
                        data_out[i,j,k,-3] += i*dx
                        data_out[i,j,k,-2] += j*dy
                        for kk in range(-3,-1):
                            if data_out[i,j,k,kk] < box_2d[kk,0]:
                                data_out[i,j,k,kk] += box_len[kk]
                            elif data_out[i,j,k,kk] > box_2d[kk,1]:
                                data_out[i,j,k,kk] -= box_len[kk]                                                   
        return (data_out, NAN, box_2d)
        
    # Build linked cells in all direction
    def linked_cell(self,data,NAN,XL,XH,XBIN,YL,YH,YBIN,ZL,ZH,ZBIN,max_atoms):
        CELL_X = np.linspace(XL,XH,XBIN+1,endpoint=True)
        CELL_Y = np.linspace(YL,YH,YBIN+1,endpoint=True)
        CELL_Z = np.linspace(ZL,ZH,ZBIN+1,endpoint=True)
        Bin_count = np.zeros((XBIN,YBIN,ZBIN),dtype=np.int)
        Bin_list = np.zeros((XBIN,YBIN,ZBIN,max_atoms),dtype=np.int)
        for i in range(NAN):
            for kx in range(XBIN):
                if CELL_X[kx] <= data[i,2] < CELL_X[kx+1]:
                    break
            for ky in range(YBIN):
                if CELL_Y[ky] <= data[i,3] < CELL_Y[ky+1]:
                    break
            for kz in range(ZBIN):
                if CELL_Z[kz] <= data[i,4] < CELL_Z[kz+1]:
                    break
            Bin_list[kx,ky,kz,Bin_count[kx,ky,kz]] = i
            Bin_count[kx,ky,kz]+=1
        return (Bin_count, Bin_list)

    # Create a neighbor list
    def neigh_build(self,data,NAN,XL,XH,XBIN,YL,YH,YBIN,ZL,ZH,ZBIN,max_atoms):
        Bin_count, Bin_list = self.linked_cell(data,NAN,XL,XH,XBIN,YL,YH,YBIN,ZL,ZH,ZBIN,max_atoms)
        neigh_count = np.zeros(NAN,dtype=np.int)
        neigh_id_list = np.zeros((NAN,self.max_neigh),dtype=np.int)       
        neigh_list = -np.zeros((NAN,self.max_neigh))
        for kx in range(XBIN):
            XBIN_LIST = range(XBIN) if XBIN <= 2 else set([kx-1,kx,kx-XBIN+1 and kx+1])
            for ky in range(YBIN):
                YBIN_LIST = range(YBIN) if YBIN <= 2 else set([ky-1,ky,ky-YBIN+1 and ky+1])
                for kz in range(ZBIN):
                    ZBIN_LIST = range(ZBIN) if ZBIN <= 2 else set([kz-1,kz,kz-ZBIN+1 and kz+1])
                    for i in range(Bin_count[kx,ky,kz]):
                        index1 = Bin_list[kx,ky,kz,i]
                        for kkx in XBIN_LIST:
                            for kky in YBIN_LIST:
                                for kkz in ZBIN_LIST:
                                    for j in range(Bin_count[kkx,kky,kkz]):
                                        index2 = Bin_list[kkx,kky,kkz,j]
                                        dx = abs(data[index1,2]-data[index2,2])
                                        ddx = dx if dx < (XH-XL)/2 else XH-XL-dx
                                        if ddx <= self.Rc:
                                            dy = abs(data[index1,3]-data[index2,3])
                                            ddy = dy if dy < (YH-YL)/2 else YH-YL-dy
                                            if ddy <= self.Rc:
                                                dz = abs(data[index1,4]-data[index2,4])
                                                ddz = dz if dz < (XH-ZL)/2 else ZH-ZL-dz
                                                if ddz <= self.Rc:
                                                    R = math.sqrt(ddx**2+ddy**2+ddz**2)
                                                    if R > 0.01 and R < self.Rc:
                                                        neigh_list[index1,neigh_count[index1]] = R
                                                        neigh_id_list[index1,neigh_count[index1]] = index2
                                                        neigh_count[index1]+=1                                                   
        return (neigh_list, neigh_id_list, neigh_count)


class Write_Data:
    # add_tilt: xy, xz, yz, or None
    def __init__(self, data, NAN, box_2d, output_dir, output_file, add_tilt=None, tilt_factor=0):
        self.data = data
        self.NAN = NAN
        self.box_2d = box_2d
        self.output_dir = output_dir
        self.output_file = output_file       
        self.add_tilt = add_tilt
        self.tilt_factor = tilt_factor
        self.filename= os.path.join(self.output_dir,'%s' %(self.output_file))
        self.fout = open(self.filename,"w")  
        self.xlo, self.xhi = self.box_2d[0]
        self.ylo, self.yhi = self.box_2d[1]
        self.zlo, self.zhi = self.box_2d[2]
        # Transform coordinates based on tilt
        self.tilt = dict.fromkeys(("x", "y", "z", "xy", "xz", "yz"), 0)
        if self.add_tilt: 
            self.tilt[self.add_tilt] = self.tilt_factor
            self.transform_cord()    
    
    def transform_cord(self):
        dx, dy, dz = np.zeros(self.NAN), np.zeros(self.NAN), np.zeros(self.NAN)
        for i in range(self.NAN):
            xx, yy, zz = self.data[i,2:5]
            if self.add_tilt == True:
                pass
            elif self.add_tilt == "x":
                dx[i] = (xx-self.xlo)/(self.xhi-self.xlo)
            elif self.add_tilt == "y":
                dy[i] = (yy-self.ylo)/(self.yhi-self.ylo)
            elif self.add_tilt == "z":
                dz[i] = (zz-self.zlo)/(self.zhi-self.zlo)
            elif self.add_tilt == "xy":
                dx[i] = (yy-self.ylo)/(self.yhi-self.ylo)
            elif self.add_tilt == "xz":
                dx[i] = (zz-self.zlo)/(self.zhi-self.zlo)
            elif self.add_tilt == "yz":
                dy[i] = (zz-self.zlo)/(self.zhi-self.zlo)
            else:
                print ("Error: Wrong tilt!")
                sys.exit()
        self.data[:,2] += dx*self.tilt_factor
        self.data[:,3] += dy*self.tilt_factor
        self.data[:,4] += dz*self.tilt_factor
        self.xhi += int(self.add_tilt=="x")*self.tilt_factor
        self.yhi += int(self.add_tilt=="y")*self.tilt_factor
        self.zhi += int(self.add_tilt=="z")*self.tilt_factor  
        
    def apply_strain(self, applied_strain):
        xs, ys, zs = applied_strain
        self.xhi *= (1+xs)
        self.yhi *= (1+ys)
        self.zhi *= (1+zs)
        for i in range(self.NAN):
            self.data[i,2] *= (1+xs)
            self.data[i,3] *= (1+ys)
            self.data[i,4] *= (1+zs)        
    
    def write_datafile(self, applied_strain=None):
        if applied_strain: self.apply_strain(applied_strain)
        self.fout.write("# LAMMPS data file: %s\n" %element)
        self.fout.write("%d atoms\n" %(self.NAN))
        self.fout.write("%d atom types\n" %(np.amax(self.data[:,1])))
        self.fout.write(" %22.16f  %22.16f   xlo xhi\n" %(self.xlo,self.xhi))
        self.fout.write(" %22.16f  %22.16f   ylo yhi\n" %(self.ylo,self.yhi))
        self.fout.write(" %22.16f  %22.16f   zlo zhi\n" %(self.zlo,self.zhi))
        if self.add_tilt in ["xy", "xz", "yz"]:
            self.fout.write(" %22.16f  %22.16f  %22.16f  xy xz yz" % (self.tilt["xy"], self.tilt["xz"], self.tilt["yz"]))
        self.fout.write("\nAtoms\n\n")
        for i in range(self.NAN):
            self.fout.write("%4d %3d %22.16f %22.16f %22.16f\n" %(i+1, \
                            self.data[i,1], self.data[i,2], self.data[i,3], self.data[i,4]))
        self.fout.close()
        
    def write_poscar(self, freeze_Z=False, applied_strain=None):
        if applied_strain: self.apply_strain(applied_strain)
        self.fout.write("# %s structure\n" %element)
        self.fout.write("1.00\n")
        # Box: [[xx,yx,zx], [xy,yy,zy], [xz,yz,zz]]  
        self.fout.write("%12.6f %12.6f %12.6f\n" %(self.xhi,0,0))
        self.fout.write("%12.6f %12.6f %12.6f\n" %(self.tilt["xy"],self.yhi,0))
        self.fout.write("%12.6f %12.6f %12.6f\n" %(self.tilt["xz"],self.tilt["yz"],self.zhi))
        self.fout.write("%d\n" %(self.NAN))
        if freeze_Z: self.fout.write("Selective Dynamics\n")
        self.fout.write("Cartesian\n")
        for i in range(self.NAN):
            if freeze_Z:
                self.fout.write("%22.16f %22.16f %22.16f  F  F  T\n" %(self.data[i,2], self.data[i,3], self.data[i,4]))
            else:
                self.fout.write("%22.16f %22.16f %22.16f\n" %(self.data[i,2], self.data[i,3], self.data[i,4]))
        self.fout.close()

    def write_extended_xyz(self, applied_strain=None):
        if applied_strain: self.apply_strain(applied_strain)
        self.fout.write("%d\n" %(self.NAN))
        tilt_xy, tilt_xz, tilt_yz = self.tilt_factor
        self.fout.write("Lattice=\"%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\" \
                        Properties=species:S:1:pos:R:3 Time=0.0\n" %(self.xhi,0,0,\
                        self.tilt["xy"],self.yhi,0,self.tilt["xz"],self.tilt["yz"],self.zhi))
        for i in range(self.NAN):
            self.fout.write("%s\t%12.6f\t%12.6f\t%12.6f\n" %(element, \
                            self.data[i,2], self.data[i,3], self.data[i,4]))
        self.fout.close()       
  

def main():
    global filedir
    filedir = os.path.dirname(os.path.realpath('__file__')) 
    # Generate bulk lattice: phase alat clat orz replication vacuum tilt
    if len(sys.argv) >= 15:
        latt=sys.argv[1]
        phase=sys.argv[2]
        alat=float(sys.argv[3])
        clat=float(sys.argv[4])
        orz=sys.argv[5] 
        orz_1=int(sys.argv[6])  
        orz_2=int(sys.argv[7])  
        orz_3=int(sys.argv[8]) 
        replicate=sys.argv[9]        
        nx=int(sys.argv[10])
        ny=int(sys.argv[11])
        nz=int(sys.argv[12]) 
        add_vac=sys.argv[13]          
        vac=float(sys.argv[14])
        add_tilt=False
        if len(sys.argv) >= 17:
            tilt=sys.argv[15]
            add_tilt=bool(sys.argv[16])      
        # Build lattice
        data, NAN, box_2d = Build_Crystal(phase, alat, clat, rep=[nx,ny,nz], orz=[orz_1,orz_2,orz_3], vac=vac).Build()
        Write_Data(data, NAN, box_2d, output_dir=filedir, output_file="Temp.data", add_tilt=add_tilt).write_datafile()

    # Generate defect configurations
    elif len(sys.argv) == 4:
        input_file=sys.argv[1]
        defect=sys.argv[2]
        n_def=int(sys.argv[3])          
        # Build defects      
        if defect.upper() == "VACANCY":
            data, NAN, box_2d = Introduce_Defects(input_dir=filedir, input_file="Temp.data", nvac=n_def).Add_Vacancy()
            Write_Data(data, NAN, box_2d, output_dir=filedir, output_file="Temp.data").write_datafile()
        # Add interstitial
        if defect.upper() == "INTERSTITIAL":
            data, NAN, box_2d = Introduce_Defects(input_dir=filedir, input_file="Temp.data", nint=n_def).Add_Interstitial()
            Write_Data(data, NAN, box_2d, output_dir=filedir, output_file="Temp.data").write_datafile()
    
    # Stacking fault curve, support only FCC lattice for now
    elif len(sys.argv) == 6:
        sf_name=sys.argv[1]
        alat=float(sys.argv[2])
        mesh_x=int(sys.argv[3])
        mesh_y=int(sys.argv[4])
        shift_plane = int(sys.argv[5])     
        # Displace to build stacking faults
        data, NAN, box_2d = Build_Crystal("FCC", alat, 1.000, rep=[5,5,12], orz=[1,1,1],vac=10).Build()
        Write_Data(data, NAN, box_2d, output_dir=filedir, output_file="Temp.data").write_datafile() 
        data, NAN, box_2d = Introduce_Defects(input_dir=filedir, input_file="Temp.data",sf_mesh=[mesh_x,mesh_y],shift_plane=shift_plane).Stacking_fault(rep=rep)  
        for i in range(mesh_x+1):
            for j in range(mesh_y+1):
                Write_Data(data[i,j,:,:], NAN, box_2d, output_dir=filedir, output_file="Temp.data").write_datafile()
  
    else:
        print ("Error: Wrong number of arguments!!!")
        sys.exit()
        
        
if __name__ == "__main__":
    main()
