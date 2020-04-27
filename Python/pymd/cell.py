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

# Class containing a single cell of the cell list

class Cell:
  """Cell class holds ids of particles the belong to a given cell. Instances 
     of this class are actual cells."""

  def __init__(self,id,rc):
    """
        Parameters
        ----------
        id : int
            Unique identified of the cell (set by CellList)
        rc : numpy array or list 
            Coordinates of the cell's centre
        """
    self.id = id
    self.rc = rc 
    self.particles = []      # List of ids of all particles in the cell
    self.neighbours = []     # List of ids of all neighbouring cells

  def add_particle(self, p):
    """
      Add particle id to the cell
      Parameters
      ----------
      p : Particle
          Particle to be added to the list
    """
    self.particles.append(p.id)
  
  def wipe(self):
    """
      Clear all particles from the cell.
    """
    self.particles = []
    