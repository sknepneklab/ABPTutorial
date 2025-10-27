# setup.py
import os
import shutil
import subprocess
import sysconfig
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py

class build_py(_build_py):
    def run(self):
        # Build pure-Python bits first so self.build_lib exists
        super().run()

        build_dir = os.path.join("cppmd", "md", "build")
        src_dir = os.path.join("cppmd", "md")
        os.makedirs(build_dir, exist_ok=True)

        # Configure & build with CMake
        subprocess.check_call(["cmake", "-S", src_dir, "-B", build_dir])
        subprocess.check_call(["cmake", "--build", build_dir, "-j", "2"])

        # Figure out the platform-specific extension suffix (e.g. .so, .cpython-311-darwin.so, .pyd)
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

        # If your CMake target produces exactly "cppmodule<ext_suffix>", this will work.
        # Adjust the name if your CMake target is different.
        built_name = "cppmodule" + ext_suffix
        src_path = os.path.join(build_dir, built_name)

        if not os.path.exists(src_path):
            # Fallbacks that sometimes occur depending on CMake configs
            # Try plain ".so" on mac even if EXT_SUFFIX has ABI tags, or try ".dylib"
            alt_candidates = [
                os.path.join(build_dir, "cppmodule.so"),
                os.path.join(build_dir, "cppmodule.dylib"),
            ]
            for cand in alt_candidates:
                if os.path.exists(cand):
                    src_path = cand
                    break
            else:
                raise FileNotFoundError(
                    f"Could not find built extension: tried {src_path} and {alt_candidates}"
                )

        dst_dir = os.path.join(self.build_lib, "cppmd", "md")
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cppmd",
    version="1.0",
    author="Daniel Matoz Fernandez",
    author_email="fdamatoz@gmail.com",
    description="A simple 2D simulation of Active Brownian particles with c++ interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sknepneklab/ABPTutorial",
    packages=find_packages(),
    include_package_data=True,
    # IMPORTANT: key must be the *package path* that contains the file
    package_data={"cppmd.md": ["cppmodule*"]},  # glob to catch .so/.pyd with ABI tags
    cmdclass={"build_py": build_py},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)

