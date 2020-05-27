#####################################################################################
# MIT License                                                                       #
#                                                                                   #
# Copyright (c) 2020 Dr. Daniel Alejandro Matoz Fernandez                           #
#               fdamatoz@gmail.com                                                  #
#                    Dr. Rastko Sknepnek                                            #
#               r.sknepnek@dundee.ac.uk                                             #
# Permission is hereby granted, free of charge, to any person obtaining a copy      #
# of this software and associated documentation files (the "Software"), to deal     #
# in the Software without restriction, including without limitation the rights      #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell         #
# copies of the Software, and to permit persons to whom the Software is             #
# furnished to do so, subject to the following conditions:                          #
#                                                                                   #
# The above copyright notice and this permission notice shall be included in all    #
# copies or substantial portions of the Software.                                   #
#                                                                                   #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR        #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,          #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE       #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER            #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,     #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE     #
# SOFTWARE.                                                                         #
#####################################################################################
import matplotlib.pyplot as plt
import gpumd as md

reader = md.fast_read_json("initphi=0.4L=50.json")
system = md.System(reader.particles, reader.box)

dump = md.Dump(system)          # Create a dump object
dump.show()                     # Plot the particles with matplotlib
plt.show()                      

evolver = md.Evolver(system)    # Create a system evolver object

#add the forces and torques
# Create pairwise repulsive interactions with the spring contant k = 10 and range a = 2.0
evolver.add_force("Harmonic Force", {"k":10.0, "a":2.0})
# Create self-propulsion, self-propulsion strength alpha = 1.0
evolver.add_force("Self Propulsion", {"alpha":1.0})
# Create pairwise polar alignment with alignment strength J = 1.0 and range a = 2.0
evolver.add_torque("Polar Align", {"k":1.0, "a":2.0})

#add integrators
# Integrator for updating particle position, friction gamma = 1.0 , "random seed" seed = 10203 and no thermal noise
evolver.add_integrator("Brownian Positions", {"T":0.0, "gamma":1.0, "seed":10203})
# Integrator for updating particle orientation, friction gamma = 1.0, "rotation" T = 0.1, D_r = 0.0, "random seed" seed = 10203
evolver.add_integrator("Brownian Rotation", {"T":0.1, "gamma":1.0, "seed":10203})

evolver.set_time_step(1e-2) # Set the time step for all the integrators


for t in range(1000):
    print("Time step : ", t)
    evolver.evolve()    # Evolve the system by one time step
    if t % 10 == 0:     # Produce snapshot of the simulation once every 10 time steps
        dump.dump_vtp('test_{:05d}.vtp'.format(t))

dump.show()
plt.show()


