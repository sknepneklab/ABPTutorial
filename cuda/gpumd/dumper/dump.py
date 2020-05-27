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

# Class handling simulation output

import json
import vtk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

class Dump:
  """
    This class handles multiple types of simulation output.
  """

  def __init__(self, sys):
    """
      Construct a Dump object.
      Parameter
      ---------
        sys : System
          Simulated system
    """
    self.sys = sys

  def dump_data(self, outfile):
    """
      Output the system as simple text table with a header and each particle date being in a separate row.
      Columns are: id x y ipx ipy nx ny vx vy fx fy
      Parameter
      ---------
        outfile : string
          Name of the output file
    """
    with open(outfile, 'w') as out:

      out.write("#  id x y ipx ipy nx ny vx vy fx fy\n")
      for p in self.sys.get():
        out.write('{:4d}  {:.6f}  {:.6f}  {:.4d} {:.4d}  {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}\n'.format(
            p.id, p.r.x, p.r.y, p.ip.x, p.ip.y, p.n.x, p.n.y, p.v.x, p.v.y, p.f.x, p.f.y))

  def dump_json(self, outfile):
    """
      Output the system as a JSON file.
      Parameter
      ---------
        outfile : string
          Name of the output file. 
      Note
      ----
        Caller should ensure that the file has correct extension
    """
    jsonData = {}
    jsonData["system"] = {}
    jsonData["system"]["particles"] = []
    jsonData["system"]["box"] = {
        "Lx": self.system.box.L.x, "Ly": self.system.box.L.y}
    for p in self.system.particles:
      pd = {}
      pd["id"] = p.id
      pd["r"] = [p.r.x, p.r.y]
      pd["ip"] = [p.ip.i, p.ip.y]
      pd["n"] = [p.n.x, p.n.y]
      pd["v"] = [p.v.x, p.v.y]
      pd["f"] = [p.forceC.x, p.forceC.y]
      jsonData["system"]["particles"].append(pd)
    with open(outfile, 'w') as out:
      json.dump(jsonData, out, indent=4)

  def dump_vtp(self, outfile):
    """
      Output the system as a VTP file for direct visualisation in ParaView.
      Parameter
      ---------
        outfile : string
          Name of the output file. 
      Note
      ----
        Caller should ensure that the file has correct extension
    """
    points = vtk.vtkPoints()
    ids = vtk.vtkIntArray()
    n = vtk.vtkDoubleArray()
    v = vtk.vtkDoubleArray()
    f = vtk.vtkDoubleArray()
    radius = vtk.vtkDoubleArray()
    ids.SetNumberOfComponents(1)
    n.SetNumberOfComponents(3)
    v.SetNumberOfComponents(3)
    f.SetNumberOfComponents(3)
    radius.SetNumberOfComponents(1)
    ids.SetName("id")
    n.SetName("director")
    v.SetName("velocity")
    f.SetName("force")
    radius.SetName("radius")
    for p in self.sys.get_particles():
      points.InsertNextPoint([p.r.x, p.r.y, 0.0])
      ids.InsertNextValue(p.id)
      n.InsertNextTuple([p.n.x, p.n.y, 0.0])
      v.InsertNextTuple([p.v.x, p.v.y, 0.0])
      f.InsertNextTuple([p.forceC.x, p.forceC.y, 0.0])
      radius.InsertNextValue(p.radius)
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.GetPointData().AddArray(ids)
    polyData.GetPointData().AddArray(n)
    polyData.GetPointData().AddArray(v)
    polyData.GetPointData().AddArray(f)
    polyData.GetPointData().AddArray(radius)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(outfile)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polyData)
    else:
        writer.SetInputData(polyData)
    writer.SetDataModeToAscii()
    writer.Write()

  def show(self, notebook=False):
    box = self.sys.box()
    fig, ax = plt.subplots()
    ax.add_patch(plt.Rectangle([box.Llo.x, box.Llo.x], box.L.x, box.L.y, fill=False))
    for p in self.sys.get_particles():
        ax.quiver(p.r.x, p.r.y, p.n.x, p.n.y,  angles='xy', scale_units='xy', scale=0.8, color='black')
        ax.add_patch(plt.Circle([p.r.x, p.r.y], p.radius, color='salmon'))
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    if notebook:
      fig.canvas.layout.width = '150%'
      fig.canvas.layout.height = '150%'
    else:
      plt.show()
