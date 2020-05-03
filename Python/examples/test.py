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

# Example of running a simple simulation

from pymd.md import * 


s = System(rcut = 3.0, pad = 0.5)
s.read_init('test.json')

e = Evolver(s)
d = Dump(s)

hf = HarmonicForce(s, 10.0, 2.0)
sp = SelfPropulsion(s, 1.0)
pa = PolarAlign(s, 1.0, 2.0)

pos_integ = BrownianIntegrator(s, T = 0.0, gamma = 1.0)
rot_integ = BrownianRotIntegrator(s, T = 0.1, gamma = 1.0)

e.add_force(hf)
e.add_force(sp)
e.add_torque(pa)
e.add_integrator(pos_integ)
e.add_integrator(rot_integ)

dt = 0.01
for t in range(1000):
  print("Time step : ", t)
  e.evolve(dt)
  if t % 10 == 0:
    d.dump_vtp('test_{:05d}.vtp'.format(t))




