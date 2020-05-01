# KITP ACTIVE20 Tutorial on Simulating Active Brownian Particles

This is the GitHub repository for a series of three tutorial on implementing a simulation of Active Brownian Particles (ABP) in two dimensions. These tutorial were delivered as a part of the KITP ACTIVE20 virtual programme.

## Layout

This tutorial consists of three sessions:

### Session 1: Overview of particle-based simulations and implementation of ABP in Python

In the first session, we cover the key components of a particle-based simulation. We then implement in Python a simple 2D simulation of self-propelled soft particles subject to polar alignment. 

Accompanied course material can be found in Tutorials/Session_1

### Session 2: Improving performance by mixing C++ and Python

In this tutorial, we cover how to significantly improve performance of the ABP simulation we developed in Session 1 by implementing the computation-heavy parts of the code in C++. We also show how to use pybind11 library to expose the low-level C++ code to Python, without loosing the convenience of working with Python.

Accompanied course material can be found in Tutorials/Session_2

### Session 3: Porting the code to GPUs

In the final tutorial, we cover basics of GPGPU programming and show how to further improve the performance of the code by utilising the immense computational power of modern GPUs.

## Getting started

**Note:** For this tutorial we recommend using Unix-like environment (e.g., Linux or Mac OS X)

Please clone the repository by using 

git clone git@github.com:sknepneklab/ABPTutorial.git

This creates a local copy of the repository on your computer. It is called ABPTutorial. 

The ABPTutorial directory has the following structure

    Tutorial  -  Jupyter notebooks and slides for the tutorial
    Python    -  Python modules that implement a fully-functional ABP simulation in Python (Session 1)
    c++       -  C++ and CUDA codes for Sessions 2 and 3
    conda     -  an example of the working conda environment 




## Lecturers
Rastko Sknepnek, University of Dundee, United Kingdom
Daniel Matoz-Fernandez, Northwestern University, USA

