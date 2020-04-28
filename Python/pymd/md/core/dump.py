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
      Columns are: id x y nx ny vx ny fx fy
      Parameter
      ---------
        outfile : string
          Name of the output file
    """
    with open(outfile, 'w') as out:
      out.write("#  id  x  y  nx  ny  vx  vy  fx  fy\n")
      for p in self.sys.particles:
        idx = p.id
        x, y = p.r.to_list()
        nx, ny = p.n.to_list()
        vx, vy = p.v.to_list()
        fx, fy = p.f.to_list()
        out.write('{:4d}  {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}\n'.format(idx, x, y, nx, ny, vx, vy, fx, fy))

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
    jsonData["system"]["box"] = {"Lx": self.system.box.Lx, "Ly": self.system.box.Ly}
    for p in self.system.particles:
      pd = {}
      pd["id"] = p.id
      pd["r"] = p.r.to_list()
      pd["n"] = p.n.to_list()
      pd["v"] = p.v.to_list()
      pd["f"] = p.f.to_list()
      jsonData["system"]["particles"].append(pd)
    with open(outfile, 'w') as out:
      json.dump(jsonData, out, indent = 4)

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
    ids.SetNumberOfComponents(1)
    n.SetNumberOfComponents(3)
    v.SetNumberOfComponents(3)
    f.SetNumberOfComponents(3)
    ids.SetName("id")
    n.SetName("director")
    v.SetName("velocity")
    f.SetName("force")
    for p in self.sys.particles:
      points.InsertNextPoint([p.r.x, p.r.y, 0.0])
      ids.InsertNextValue(p.id)
      n.InsertNextTuple([p.n.x, p.n.y, 0.0])
      v.InsertNextTuple([p.v.x, p.v.y, 0.0])
      f.InsertNextTuple([p.f.x, p.f.y, 0.0])
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.GetPointData().AddArray(ids)
    polyData.GetPointData().AddArray(n)
    polyData.GetPointData().AddArray(v)
    polyData.GetPointData().AddArray(f)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(outfile)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polyData)
    else:
        writer.SetInputData(polyData)
    writer.SetDataModeToAscii()
    writer.Write()
