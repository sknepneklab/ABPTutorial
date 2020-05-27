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
import subprocess
import setuptools
from setuptools.command.install import install

class InstallLocalPackage(install):
    def run(self):
        print()
        install.run(self)
        subprocess.call("cd gpumd/md/build && cmake .. && make -j2 && cp nvccmodule.so ../", shell=True)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gpumd", 
    version="1.0",
    author="Daniel Matoz Fernandez",
    author_email="fdamatoz@gmail.com",
    description="A simple 2D simulation of Active Brownian particles with cuda and c++ interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sknepneklab/ABPTutorial",
    cmdclass=dict(install=InstallLocalPackage), 
    packages=setuptools.find_packages(),
    package_data={'': ['nvccmodule.so']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
