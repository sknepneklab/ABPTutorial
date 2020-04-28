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

# Class handling neighbour list

from .cell_list import CellList
from copy import copy

class NeighbourList:
  """
    This class handles building and maintenance of the Verlet neighbour list.
    Note: In this implementation performance is sacrificed for simplicity
  """
  def __init__(self, sys, rcut, pad):
    """
      Initialise the neighbour list object.
      Parameter
      ---------
        sys : Particles
          Simulation system
        rcut : float
          Cutoff distance for the neighbours
        pad : float
          Padding distance for the neighbour list
    """
    self.sys = sys 
    self.rcut = rcut 
    self.pad = pad
    self.cell_list = CellList(self.sys.box, self.rcut + self.pad)

  def build(self):
    """
      Build the neighbour list aided by the cell list.
    """
    # Store current positions of all particles
    self.old_pos = []
    for p in self.sys.particles:
      self.old_pos.append(copy(p.r))
    
    # Set up the cell list
    self.cell_list.wipe()
    for p in self.sys.particles:
      self.cell_list.add_particle(p)

    # Build the list 
    self.neighbours = []
    for p in self.sys.particles:
      neighbours = []
      for n in self.cell_list.get_neighbours(p):
        pn = self.sys.particles[n]
        if pn.id > p.id:
          dr = pn.r - p.r 
          dr.apply_periodic(self.sys.box)
          if dr.length() < self.rcut + self.pad:
            neighbours.append(n)
      self.neighbours.append(neighbours)
    
    self.sys.has_nl = True

  def needs_rebuild(self):
    """
      Check if the neighbour list needs to be rebuilt.
      Note
      ----
        A rebuild is done if one of the particles has moved more than 0.5*pad
    """
    for p in self.sys.particles:
      dr = p.r - self.old_pos[p.id]
      dr.apply_periodic(self.sys.box)
      if dr.length() >= 0.5*self.pad:
        return True 
    return False