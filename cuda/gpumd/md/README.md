# README

If you are using conda or other enviroment in which you have more than one GCC compiler the make sure
that the nvcc compiler is compatible with gcc. 

[System Requirements](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements)

For example, I am running gcc-9 but cuda 10.2 is only compatible with gcc-8 then before compile the code I need to export the ``CC`` and ``CXX`` variables to be compatible with nvcc:

``export CC=/usr/bin/gcc-8``
``export CXX=/usr/bin/g++-8``

