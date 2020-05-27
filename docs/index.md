---
layout: default
---

**MembraneEvolver14** is a python interactive program for the study solid elastic thin sheets. The code is written in c++14 
with a python 3+ interface provided by [pybind11](https://pybind11.readthedocs.io/en/stable/). The main object of the code is the _surface_
which is represented as a collection of [2-simplexes](http://mathworld.wolfram.com/Simplex.html) (triangles) that form the **mesh**. 

As the name suggests, the main objective of **MembraneEvolver14** is to evolve the _surface_ to a configuration that satisfies the minimum of energy. 
The energy minimization is performed by a Monte Carlo simulation as is described in Ref. [1,2]. The package allows to define different type of energies such as
harmonic, elastic, bending.


# Index

* [Compiling the code](./README.html)
* [The basic](./docs/datastructures.html)
    * [Reading a mesh](./docs/datastructures.html)
    * [Visualization](./docs/datastructures.html)
    * [Data structures](./docs/datastructures.html)
* [Defined Potentials]
* [Monte Carlo Methods]
    * [Vertex move]
    * [Edge swap]
    * [Face swap]
* [Measurements]
    * [Energy]
    * [Curvature]
* [Advance methods]
    * [Parallel tempering](./docs/tempering.md)
* [Examples]
    * [Two components edge swap](./docs/edgeswap.md)
    * [Three components vertex swap](./docs/vertexswap.md)
* [References](./docs/references.html)








