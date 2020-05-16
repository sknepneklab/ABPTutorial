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

# This script generates random initial configuration and saves it as a json file

from math import sin, cos, sqrt, pi
from random import uniform
import json

def random_init(phi, Lx, Ly = None, rcut = None, outfile = 'test.json', max_attempts = 10):
  """
    Create a random initial condition.
    Parameters
    ----------
      phi : float
        Number density defined as phi = N/(Lx*Ly), where N is the number of parcticles 
        and Lx and Ly are dimesions of the simulation box.
      Lx : float
        Width of the simulation box
      Ly : float
        Height of the simulation box. If None (default) Ly = Lx
      rcut : float
        Distance cutoff for building a nonoverlapping initial configuration. If None (default), 
        particles can overlap.
      outfile : str
        Name of the output JSON file. Caller has to ensure correct extension
      max_attempts : int
        Maximum number of attempts to place each particle
  """
  particles = []  # Store all particles in a list 
  lx = Lx 
  ly = Ly if Ly != None else Lx
  A = lx*ly 
  N = int(round(phi*A))
  xmin, xmax = -0.5*lx, 0.5*lx 
  ymin, ymax = -0.5*ly, 0.5*ly
  if rcut == None:
    for i in range(N):
      r = [uniform(xmin, xmax), uniform(ymin, ymax)]
      theta = uniform(-pi, pi)
      n = [cos(theta), sin(theta)]
      v = [0.0, 0.0]
      f = [0.0, 0.0]
      # Add a particle at position x,y with the director pointing in the random direction
      particles.append({'id': i, 'r': r, 'n': n, 'v': v, 'f': f, 'radius': 0.5*rcut})
  else:
    # Here we implement avoidance of overlaps 
    i = 0
    while i < N:
      attempt = 0
      while attempt < max_attempts:
        r = [uniform(xmin, xmax), uniform(ymin, ymax)]
        can_add = True 
        for p in particles:
          if sqrt((p['r'][0]-r[0])**2 + (p['r'][1]-r[1])**2) < rcut:
            can_add = False 
            break
        if can_add:
          theta = uniform(-pi, pi)
          n = [cos(theta), sin(theta)]
          v = [0.0, 0.0]
          f = [0.0, 0.0]
          # Add a particle at position x,y with the director pointing in the random direction
          particles.append({'id': i, 'r': r, 'n': n, 'v': v, 'f': f, 'radius': 0.5*rcut})
          i += 1
          break 
        else:
          attempt += 1
  jsonData = {}
  jsonData["system"] = {}
  jsonData["system"]["box"] = {"Lx": lx, "Ly": ly}
  jsonData["system"]["particles"] = particles
  with open(outfile, 'w') as out:
    json.dump(jsonData, out, indent = 4)


