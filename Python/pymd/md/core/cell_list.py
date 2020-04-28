# Copyright 2020 Rastko Sknepnek, University of Dundee, r.skepnek@dundee.ac.uk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions 
#  of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED 
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

# Class handling the cell list building an maintenace

from copy import deepcopy
from .cell import Cell
    
class CellList:
  """ Class that handles creation and mainenance of the cell list."""
  
  def __init__(self, box, r_cut = 1.0):
    """
      Construct a gird of cells and set their neighbours.
      Parameters
      ----------
      box : Box
         Simulation box object
      r_cut : float
         Cutoff distance (size of each cell)
    """
    self.box = box
    self.Lx = self.box.Lx 
    self.Ly = self.box.Ly
    self.r_cut = r_cut
    self.cell_indices = {}  # Dictionary that maps a particle into the corresponding cell
    self.nx = int(self.Lx/r_cut)
    self.ny = int(self.Ly/r_cut)
    self.dx = self.Lx/float(self.nx)
    self.dy = self.Ly/float(self.ny)
    # total number of cells
    n_cell = self.nx*self.ny
    # Cell list is a Python list
    self.cell_list = [None for i in range(n_cell)]
    for i in range(self.nx):
      x = -0.5*self.Lx + float(i)*self.dx
      for j in range(self.ny):
        y = -0.5*self.Ly + float(j)*self.dy
        # Cell labelling scheme: for each x, do all y
        idx = self.ny*i + j
        # Create new cell with index, position and size
        self.cell_list[idx] = Cell(idx,[x,y])
        for ix in [-1,0,1]:
          for iy in [-1,0,1]:
            iix, iiy = i + ix, j + iy
            if iix == self.nx: iix = 0
            elif iix < 0: iix = self.nx - 1
            if iiy == self.ny: iiy = 0
            elif iiy < 0: iiy = self.ny - 1
            self.cell_list[idx].neighbours.append(self.ny*iix + iiy)
			
  def get_cell_idx(self, r):
    """
      Return index of a cell a vector falls into.
      Parameters
      ----------
        r : Vec
          Position vector 
    """
    i, j = int((r.x-self.box.xmin)/self.dx), int((r.y-self.box.ymin)/self.dy)
    cell_idx = self.ny*i + j 
    return cell_idx
  
  def add_particle(self, p, cell_idx = None):
    """
      Add a particle to a cell
      Parameters
      ----------
        p : Particle
          Particle to be added to the cell list
        cell_idx : int
          Index of a cell to add to. If None (default), compute the cell index based on the particle's
          position. 
    """
    if cell_idx == None:
      cell_idx = self.get_cell_idx(p.r)
    else:
      cell_idx = cell_index    
    self.cell_list[cell_idx].add_particle(p)
    self.cell_indices[p.id] = cell_idx
    
  def wipe(self):
    """
      Clear the entire cell list
    """
    for cell in self.cell_list:
      cell.wipe()
      
  def get_neighbours(self, p):
    """
      Return ids off all particles that are neighbours of a given particle. This includes all particles
      in the cell p belongs to and all particles in the 8 neighbouring cells.
    """
    cell_index = self.get_cell_idx(p.r)
    neighbours = []
    for idx in self.cell_list[cell_index].neighbours:
      neighbours.extend(deepcopy(self.cell_list[idx].particles))
    return neighbours
    
    
  
      