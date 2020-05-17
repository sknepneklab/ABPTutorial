### Creating conda environment 

In the existent environment (ABP) install these additional packages:

* cmake
* ipympl
* nodejs

``conda install -c anaconda cmake ``

``conda install -c conda-forge ipympl``

``conda install -c conda-forge nodejs``



and then install the following extension for **JupyterLab**:



``jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib``



##  Installing cppmd module

In order to make the Python modules visible, please install the cpmd module. From the **``ABPTutorial/c++``**  directory type:

``python setup.py install``

In order to test the installation, you can type 



``python -c 'from cppmd.md import *; b = Box(10.0, 10.0); print(b)'`` 



If the cppmd module has been properly installed, the output of the previous line should looks like:

```python
<box Lx = 10 Ly = 10 
<box Lx = (-5, 5)
<box Ly = (-5, 5)
periodic_Lx = 1 periodic_Ly = 1 >
```

