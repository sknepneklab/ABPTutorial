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

# Class handling Brownian integrator for rotation of particle directors

from random import gauss
from math import sqrt

class BrownianRotIntegrator:
  """
    Class that implements first order Brownian integrator for rotating direction 
    of each particle.
  """
  def __init__(self, sys, T = 0.0, gamma = 1.0):
    """
      Construct a BrownianRotIntegrator object
      Parameter
      ---------
        sys : System
          Simulation system
        T : float
          Temperature 
        gamma : float
          Friction coefficient 
      Note
      ----
        Rotation diffusion constant is Dr = T/gamma
    """
    self.sys = sys 
    self.T = T 
    self.gamma = gamma

  def prestep(self, dt):
    """
      Performs step before force is computed.
      Parameter
      ---------
        dt : float
          step size
    """
    pass 
  
  def poststep(self, dt):
    """
      Perform actual integration step
      Parameter
      ---------
        dt : float
          step size
    """
    Dr = self.T/self.gamma 
    B = sqrt(2*Dr*dt)
    for p in self.sys.particles:
      theta = (dt/self.gamma)*p.tau 
      if self.T > 0:
        theta += B*gauss(0,1)
      p.n.rotate(theta)


    

