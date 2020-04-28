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

# Class handling the particles

from .vec import Vec
from .box import Box
from .particle import Partice
from .neighbour_list import NeighbourList
from random import uniform
from math import pi, sin, cos
import json

class System:
  """This class stores and handles simulated particles."""

  def __init__(self, rcut = 2.0, pad = 0.5):
    """
      Construct the object holding simulated system.
      rcut : flaot
        Neighbour list cutoff
      pad : float
        Neighbour list padding distance
    """
    self.particles = []
    self.rcut = rcut 
    self.pad = pad
    
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
            self.particles.append(Partice(idx, Vec(x,y), v = Vec(vx, vy), n = Vec(nx,ny), f = Vec(fx, fy)))
          self.N = len(self.particles)  
          self.neighbour_list = NeighbourList(self, self.rcut, self.pad) 
          self.neighbour_list.build()       
        else:
          raise Exception('Input JSON file has to include particles section.')    
    except IOError:
      print('Could not open {:s} for reading.'.format(initfile))

  def apply_periodic(self):
    """
      Apply period boundary conditions to all particles in the simulation box.
    """
    for p in self.particles:
      p.r.apply_periodic(self.box)       

  def reset_forces(self):
    """
      Reset forces and torques on all particles to zero. 
    """
    for p in self.particles:
      p.f = Vec(0.0, 0.0)
      p.tau = 0.0

  
