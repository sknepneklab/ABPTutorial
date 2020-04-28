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

# Class for handling polar alignment
class PolarAlign:
  """
    Computes torque between pairs of particles in order to achieve polar alignment.
  """
  def __init__(self, sys, J = 1.0, a = 2.0):
    """
      Create an object that handles polar alignment between pairs of partiles.
      Parameters
      ----------
        sys : System
          Simulation system
        J : float
          Alignment strength 
        a : float
          Cutoff distance
    """
    self.sys = sys
    self.J = J 
    self.a = a
  
  def compute(self):
    for pi in self.sys.particles:
      ri = pi.r 
      for n in self.sys.neighbour_list.neighbours[pi.id]:
        pj = self.sys.particles[n]
        rj = pj.r 
        dr = rj - ri 
        dr.apply_periodic(self.sys.box)
        lendr = dr.length()
        if lendr <= self.a:
          tau_z = self.J*(pi.n.x*pj.n.y - pi.n.y*pj.n.x)
          pi.tau += tau_z 
          pj.tau -= tau_z