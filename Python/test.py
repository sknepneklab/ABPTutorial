import pymd 

from pymd.harmonic_force import HarmonicForce
from pymd.self_propulsion import SelfPropulsion
from pymd.polar_align import PolarAlign
from pymd.brownian_integrator import BrownianIntegrator
from pymd.brownian_rot_integrator import BrownianRotIntegrator

b = pymd.Box(50)
s = pymd.System(b, 2.5, 0.5)
s.random_init(0.5, rcut=1.0)

e = pymd.Evelover(s)
d = pymd.Dump(s)

hf = HarmonicForce(s, 1.0, 2.0)
sp = SelfPropulsion(s, 1.0)
pa = PolarAlign(s, 1.0, )

pos_integ = BrownianIntegrator(s, T=0.0, gamma = 1.0)
rot_integ = BrownianRotIntegrator(s, T = 0.1, gamma=1.0)

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




