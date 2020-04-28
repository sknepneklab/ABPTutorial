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

# Class handling 2d vector algebra in periodic boundaries

from math import sqrt, sin, cos

class Vec:
  """Class that handles 2D vector algebra in periodic boundaries."""

  def __init__(self, x, y):
    """
      Construct a vector in 2d.
      Parameters
      ----------
        x : float
          x coorsiante of the vector
        y : float
          y coordinate of the vector
    """
    self.x = x 
    self.y = y 
      
  def __add__(self, r):
    """
      Define vector addition.
      Parameters
      ----------
        r : Vec
          Vector to add to self
    """
    return Vec(self.x + r.x, self.y + r.y)

  def __sub__(self, r):
    """
      Define vector subtraction.
      Parameters
      ----------
        r : Vec
          Vector to subtract from self
    """
    return Vec(self.x - r.x, self.y - r.y)

  def __mul__(self, s):
    """
      Define vector scaling (from left).
      Parameters
      ----------
        s : float
          Scaling factor
    """
    return Vec(s*self.x, s*self.y)

  def __rmul__(self, s):
    """
      Define vector scaling (from right).
      Parameters
      ----------
        s : float
          Scaling factor
    """
    return Vec(s*self.x, s*self.y)

  def __iadd__(self, r):
    """
      Increment the vector by another vector.
      Parameters
      ----------
        r : Vec
          Vector to increment by
    """
    self.x += r.x 
    self.y += r.y 
    return self

  def __isub__(self, r):
    """
      Decrement the vector by another vector.
      Parameters
      ----------
        r : Vec
          Vector to decrement by
    """
    self.x -= r.x 
    self.y -= r.y 
    return self

  def __repr__(self):
    """
      Return vector components as a tuple
    """
    return (self.x, self.y) 
  
  def __str__(self):
    """
      Return vector components as a string for printing.
    """
    return "({:.6f}, {:.6f})".format(self.x, self.y)

  def dot(self, r):
    """
      Compute dot product between two vectors.
      Parameters
      ----------
        r : Vec
          Vector to dot product with
    """
    return self.x*r.x + self.y*r.y

  def length(self):
    """ 
      Compute length of the vector.
    """
    return sqrt(self.dot(self))

  def unit(self):
    """ 
      Return a unit-length vector in the direction of this vector.
    """
    l = self.length()
    if l > 0.0:
      return (1.0/l)*self 

  def rotate(self, phi):
    """
      Rotate the vector in plane.
      Parameter
      ---------
        phi : rotaton angle
    """
    c = cos(phi)
    s = sin(phi)
    x = c*self.x - s*self.y 
    y = s*self.x + c*self.y 
    self.x = x
    self.y = y

  def to_list(self):
    """
      Return a list with x an y components of the vector.
    """
    return [self.x, self.y]
  
  def apply_periodic(self, box):
    """
      Apply periodic boundary conditions to a vector.
    """
    if self.x < box.xmin:
        self.x += box.Lx
    elif self.x > box.xmax:
        self.x -= box.Lx
    if self.y < box.ymin:
        self.y += box.Ly
    elif self.y > box.ymax:
        self.y -= box.Ly