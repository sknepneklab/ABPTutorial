### Creating conda environment 

Here, we assume that you have a working C/C++ compiler that it supports the **C++14** or newer standard, and GPU capable to run **cuda9.x** or above. In the existing conda environment (ABP) please install these additional packages:

* cmake
* ipympl
* nodejs

``conda install -c anaconda cmake ``

``conda install -c conda-forge ipympl``

``conda install -c conda-forge nodejs``



Please also install the following extensions for **JupyterLab**:



``jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib``



##  Installing cppmd module

In order to make the Python modules visible, please install the cppmd module. From the **``ABPTutorial/c++``**  directory type:

``python setup.py install``

In order to test the installation, you can type 



``python -c 'from gpumd.md import *; b = Box(10.0, 10.0); print(b)'`` 



If the cppmd module has been properly installed, the output of the previous line should look like:

```python
<box Lx = 10 Ly = 10 
<box Lx = (-5, 5)
<box Ly = (-5, 5)
periodic_Lx = 1 periodic_Ly = 1 >
```

