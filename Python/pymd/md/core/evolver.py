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

# Class handling time evolution of the system

class Evolver:
  """
    This class handles time evolution of the system.
  """
  def __init__(self, sys):
    """
      Initialise the Evolver object.
      Parameter
      ---------
        sys : System
          Object containt the simulate system
    """
    self.sys = sys 
    self.integrators = []
    self.force_computes = []
    self.torque_computes = []

  def evolve(self, dt):
    """
      Perform one time step of the simulation.
      Parameter
      ---------
        dt : float
          Time step
    """
    # Check is neighbour list needs rebuilding
    if self.sys.neighbour_list.needs_rebuild():
      self.sys.neighbour_list.build()

    # Perform the preintegration step, i.e., step before forces and torques are computed
    for integ in self.integrators:
      integ.prestep(dt)

    # Apply period boundary conditions
    self.sys.apply_periodic()
    
    # Reset all forces and toques
    self.sys.reset_forces()

    # Compute all forces and torques
    for fc in self.force_computes:
      fc.compute()

    for tc in self.torque_computes:
      tc.compute()

    # Perform the second step of integration
    for integ in self.integrators:
      integ.poststep(dt)
    
    # Apply period boundary conditions
    self.sys.apply_periodic()

  def add_integrator(self, integ):
    """
      Add an integrator to the list of all integrators.
    """
    self.integrators.append(integ)
  
  def add_force(self, force):
    """
      Add force class to the list of all force computes.
    """
    self.force_computes.append(force)
  
  def add_torque(self, torque):
    """
      Add torque class to to the list of all torque computes.
    """
    self.torque_computes.append(torque)
  
  
 
    