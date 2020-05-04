# Tutorial on Simulating Active Brownian Particles

This is the GitHub repository for a series of three tutorial on implementing a simulation of Active Brownian Particles (ABP) in two dimensions. These tutorial were delivered as a part of the [KITP ACTIVE20](https://www.kitp.ucsb.edu/activities/active20)  virtual programme.

## Layout 

This tutorial consists of three sessions:

### Session 1: Overview of particle-based simulations and implementation of ABP in Python

In the first session, we cover the key components of a particle-based simulation. We then implement in Python a simple 2D simulation of self-propelled soft particles subject to polar alignment. 

Accompanied course material can be found in Tutorials/Session_1

### Session 2: Improving performance by mixing C++ and Python

In this tutorial, we cover how to significantly improve performance of the ABP simulation we developed in Session 1 by implementing the computation-heavy parts of the code in C++. We also show how to use pybind11 library to expose the low-level C++ code to Python, without loosing the convenience of working with Python.

Accompanied course material can be found in Tutorials/Session_2

### Session 3: Porting the code to GPUs

In the final tutorial, we cover basics of GPGPU programming and show how to further improve the performance of the code by utilising the immense computational power of modern GPUs.

Accompanied course material can be found in Tutorials/Session_3

## Getting started

### Cloning the code repository

**Note:** For this tutorial we recommend using Unix-like environment (e.g., Linux or Mac OS X). You will need to have git installed. 

Open a terminal and lease clone the repository with 

`git clone git@github.com:sknepneklab/ABPTutorial.git`

This command will create a local copy of the repository on your computer. It will be stored in a directory called *ABPTutorial*. 

The *ABPTutorial* directory has the following structure:

    Tutorial  -  Jupyter notebooks and slides for the tutorial
    Python    -  Python modules that implement a fully-functional ABP simulation in Python (Session 1)
    c++       -  C++ and CUDA codes for Sessions 2 and 3
    conda     -  an example of a working conda environment 

### Creating conda environment 

We recommend using Anaconda and creating a ABP environment. An example of a suitable conda environment can be found in the *conda* directory (file: *ABP.yml*).

To create a conda environment with using provided yml file, from the *ABPTutorial* directory type:

`conda env create --file conda/ABP.yml`

If you encounter problems with using the provided yml file (some users have reported the "Solving environment: failed" error), you can always create a new conda environment, e.g. by typing:

`conda create -n ABP python=3.7 numpy scipy matplotlib jupyter jupyterlab vtk pip`

This should install all packages needed to run Session 1 of the Tutorial. 

You can make sure that the environment was properly installed by typing:

`conda activate ABP`


### Installing pymd module

In order to make the Python modules visible, please install the *pymd* module. From the *ABPTutorial* directory type:

`pip install Python/`

In order to test the installation, you can type

`python -c 'from pymd.md import *; b = Box(10); print(b.xmax)'`

If the pymd module has been properly installed, the output of the previous line should be 5.0.

### Additional software 

We also recommend installing [Paraview](https://www.paraview.org/download/) for visualisation.

**Note:** It is also possible to install Paraview via the conda-forge channel. However, this has not been tested and one may encounter a number of package dependency issues if trying to install Paraview in the ABP conda environment provided above. 

## Lecturers
Rastko Sknepnek, University of Dundee, United Kingdom

Daniel Matoz-Fernandez, Northwestern University, USA

