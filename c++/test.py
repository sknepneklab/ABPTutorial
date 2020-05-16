import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import abpcpp as mm



def minimum_image(ri, rj, box):

    rij = rj - ri
    if box.periodic.x == True :
        if rij.x > box.Lhi.x:
            rij.x -= box.L.x
        elif rij.x < box.Llo.x:
            rij.x += box.L.x
    if box.periodic.y:
        if rij.y > box.Lhi.y:
            rij.y -= box.L.y
        elif rij.y < box.Llo.y :
            rij.y += box.L.y

    return rij


phi = 0.4
L = 50
a = 1.0
mm.random_init(phi, L, rcut=a, outfile='init.json')
particles, box = mm.read_json("init.json")
system = mm.System(particles, box)

dump = mm.Dump(system)
dump.dump_vtp("test.vtp")

dump.show()
#plt.show()

evolver = mm.Evolver(system)


"""

evolver.create_neighbourlist(2.0)
evolver.fill_neighbourlist()


neighbourlist = evolver.get_neighbourlist()

particles = system.get_particles()

print("brute force")
#for pi in particles:
pi = particles[10]
for pj in particles:
    if pi.id!=pj.id:
        rij = minimum_image(pi.r, pj.r, system.box())
        if rij*rij < 2.6*2.6:
            print(pj.id)

print("neighbourlist")
for c in range(pi.coordination):
    ng = c + 100*pi.id
    print(neighbourlist[ng])

"""


#add the forces and torques
evolver.add_force("Harmonic Force", {"k":10.0, "a":2.0})
evolver.add_force("Self Propulsion", {"alpha":1.0})
evolver.add_torque("Polar Align", {"k":1.0, "a":2.0})

#add integrators
evolver.add_integrator("Brownian Positions", {"T":0.0, "gamma":1.0, "seed":10203})
evolver.add_integrator("Brownian Rotation", {"T":0.1, "gamma":1.0, "seed":10203})

evolver.set_time_step(1e-2)

for t in range(1000):
    #print("Time step : ", t)
    evolver.evolve()
    if t % 10 == 0:    # Produce snapshot of the simulation once every 10 time steps
        dump.dump_vtp('test_{:05d}.vtp'.format(t))

dump.show()
#plt.show()
