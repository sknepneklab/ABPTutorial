import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pymd", 
    version="0.9",
    author="Rastko Sknepnek",
    author_email="r.sknepnek@dundee.ac.uk",
    description="A simple 2D simulation of Active Brownian particles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sknepneklab/ABPTutorial",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.4',
)