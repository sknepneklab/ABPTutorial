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

# Class handling individual particles

from .vec import Vec

class Partice:
  """Class that handles individual particles"""

  def __init__(self, idx, r, v = Vec(0.0, 0.0), n = Vec(0.0, 0.0), f = Vec(0.0, 0.0)):
    """
      Create a particle 
      Parameters
      ----------
        r : Vec 
          Particle position inside the simulation box
        v : Vec 
          Particle velocity 
        n : Vec
          Particle director
        f : Vec
          Total force on the particle
    """
    self.id = idx
    self.r = r 
    self.v = v
    self.n = n
    self.f = f
    self.tau = 0.0  # Torque on the particle (note that since we are in 2d, torque is a scalar)
    self.m = 1.0  # We add mass just in case we want to implement inertia