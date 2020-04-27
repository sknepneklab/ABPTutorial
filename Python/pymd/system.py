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

# Class handling the simulated system

from .vec import Vec
from .particle import Partice
from .neighbour_list import NeighbourList
from random import uniform
from math import pi, sin, cos
import json

class System:
  """This class stores and handles the simulated system."""

  def __init__(self, box, rcut, pad):
    """
      Construct the system object.
      Parameters
      ----------
        box : Box
          Simulation box
        rcut : float
          Neighbour list cutoff distance
        pad : float
          Neighbour list padding distance
    """
    self.box = box 
    self.particles = []
    self.neighbour_list = NeighbourList(self, rcut, pad)
    self.has_nl = False

  def random_init(self, phi, rcut = None):
    """
      Create a random initial condition.
      Parameters
      ----------
        phi : float
          Number density defined as phi = N/(Lx*Ly), where N is the number of parcticles 
          and Lx and Ly are dimesions of the simulation box.
        rcut : float
          Distance cutoff for building a nonoverlapping initial configuration. If None (default), 
          particles can overlap.
    """
    self.particles = []  # Store all particles in a list 
    self.N = int(round(phi*self.box.A))
    if rcut == None:
      for i in range(self.N):
        x = uniform(self.box.xmin, self.box.xmax)
        y = uniform(self.box.ymin, self.box.ymax)
        theta = uniform(-pi, pi)
        # Add a particle at position x,y with the director pointing in the random direction
        self.particles.append(Partice(i, Vec(x,y,self.box), n = Vec(cos(theta),sin(theta))))
    else:
      # Here we implement avoidance of overlaps 
      i = 0
      max_attempts = 10 # set the limit of the maximum attempts to avoid getting stuck in an endless loop
      while i < self.N:
        attempt = 0
        while attempt < max_attempts:
          x = uniform(self.box.xmin, self.box.xmax)
          y = uniform(self.box.ymin, self.box.ymax)
          r = Vec(x, y, self.box)
          if self.__can_add(r, rcut):
            theta = uniform(-pi, pi)
            # Add a particle at position x,y with the director pointing in the random direction
            self.particles.append(Partice(i, r, n = Vec(cos(theta),sin(theta))))
            i += 1
            break 
          else:
            attempt += 1

  def read_init(self, initfile):
    """
      Read the initial configuration from a JSON file.
      Parameter
      ---------
        initfile : str
          Name of the input JSON file
    """
    try:
      with open(initfile) as f:
        data = json.load(f)
        if 'box' in data['system']:
          Lx = data["system"]["box"]["Lx"]
          Ly = data["system"]["box"]["Ly"]
          self.box = Box(Lx, Ly)
        else:
          raise Exception('Input JSON file has to include system box section.')
        if 'particles' in data['system']:
          self.particles = []
          for p in data['system']['particles']:
            idx = p['id']
            x, y = p['r']
            theta = uniform(-pi,pi)
            nx, ny = cos(theta), sin(theta)
            vx, vy = 0.0, 0.0
            fx, fy = 0.0, 0.0
            if 'n' in p:  nx, ny = p['n']
            if 'v' in p:  vx, vy = p['v']
            if 'f' in p:  fx, fy = p['f']
            self.particles.append(Partice(idx, Vec(x,y, self.box), v = Vec(vx, vy), n = Vec(nx,ny), f = Vec(fx, fy)))
          self.N = len(self.particles)          
        else:
          raise Exception('Input JSON file has to include particles section.')
    except IOError:
      print('Could not open {:s} for reading.'.format(initfile))

  def reset_forces(self):
    """
      Reset forces and torques on all particles to zero. 
    """
    for p in self.system.particles:
      p.f = Vec(0.0, 0.0)
      p.tau = Vec(0.0, 0.0)

  # Private auxiliary functions
  def __can_add(self, r, rcut):
    """
      Auxiliary private function that checks is a particle can be added at a given position
      Parameters
      ----------
        r : Vec
          Position of the particle to be added
        rcut : float
          Minimum overlap distance
      Note
      ----
        This function is very inefficient and is implemented this way for simplicity and clarity
    """   
    for p in self.particles:
      dr = p.r - r
      if dr.length() < rcut:
        return False 
    return True
      
