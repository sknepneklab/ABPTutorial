from gpumd.md import *
import json
from random import uniform
from math import pi, sin, cos

def read_json(filename):
    """
      Read the initial configuration from a JSON file.
      Parameter
      ---------
        filename : str
          Name of the input JSON file
    """
    try:
      with open(filename) as f:
        data = json.load(f)
        if 'box' in data['system']:
          Lx = data["system"]["box"]["Lx"]
          Ly = data["system"]["box"]["Ly"]
          box = Box(Lx, Ly)
        else:
          raise Exception('Input JSON file has to include system box section.')
        if 'particles' in data['system']:
            particles = []
            for pf in data['system']['particles']:
                p = Particle()
                p.id = pf['id']
                p.radius = pf['radius']
                p.r.x = pf['r'][0]
                p.r.y = pf['r'][1]
                theta = uniform(-pi, pi)
                p.n.x, p.n.y = cos(theta), sin(theta)
                vx, vy = 0.0, 0.0
                fx, fy = 0.0, 0.0
                if 'n' in pf:  nx, ny = pf['n']
                if 'v' in pf:  vx, vy = pf['v']
                if 'f' in pf:  fx, fy = pf['f']
                p.forceC.x = fx
                p.forceC.y = fy
                p.v.x = vx
                p.v.y = vy
                particles.append(p)
        else:
          raise Exception('Input JSON file has to include particles section.')  
        return particles, box
    except IOError:
      print('Could not open {:s} for reading.'.format(filename))
    
