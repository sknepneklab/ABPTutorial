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

# Class handling the simulation box

class Box:

  def __init__(self, Lx, Ly = None):
    """
    Construct simulation box.
      Parameters
      ----------
      Lx : float
         Size of the simulation box in x direction
      Ly : float
         Size of the simulation box in y direction (if None, same as Lx, i.e., square box)
      Note
      ----
        Simulation box is centred as (0,0), i.e., x is in (-Lx/2,Lx/2] and y is in (-Ly/2,Ly/2]
    """
    if Lx < 0.0:
      raise ValueError('Simulation box has to have length larger than 0.')
    self.Lx = Lx 
    self.Ly = Lx if (Ly == None or Ly < 0.0) else Ly 
    self.xmin = -0.5*self.Lx
    self.xmax =  0.5*self.Lx 
    self.ymin = -0.5*self.Ly
    self.ymax =  0.5*self.Ly
    self.A = self.Lx*self.Ly