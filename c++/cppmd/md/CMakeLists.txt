#####################################################################################
# MIT License                                                                       #
#                                                                                   #
# Copyright (c) 2020 Dr. Daniel Alejandro Matoz Fernandez                           #
#               fdamatoz@gmail.com                                                  #
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

CMAKE_MINIMUM_REQUIRED(VERSION 3.4 FATAL_ERROR)
project (ABPCPP C CXX)
set(CMAKE_BUILD_TYPE "Release")
# place all executables in the build directory 
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

########### ADD C++ SUPPORT  ##############
find_program(CMAKE_C_COMPILER NAMES $ENV{CC} gcc PATHS ENV PATH NO_DEFAULT_PATH)
find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++ PATHS ENV PATH NO_DEFAULT_PATH)
########### SET C++ COMPILER FLAGS  ############
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -Wall -DNDEBUG" CACHE STRING "Flags used by the compiler during minimum size release builds." FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -ffast-math -DNDEBUG -std=c++14 -fvisibility=hidden -fpermissive" CACHE STRING "Flags used by the compiler during release builds." FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O -funroll-loops -ffast-math -DNDEBUG -Wall" CACHE STRING "Flags used by the compiler during release builds with debug info." FORCE)
############################################

########### ADD PYBIND SUPPORT  ###############
add_subdirectory(external/pybind11)
set(PYBIND11_PYTHON_VERSION 3.7 CACHE STRING "")
set(PYBIND11_CPP_STANDARD -std=c++14)
############################################

############################################
SET(INCLUDE_DIRS "${ABPCPP_SOURCE_DIR}/src")
include_directories(${INCLUDE_DIRS})
file(GLOB INCLUDE_DIRS_SRCS ${INCLUDE_DIRS}/*.cpp)
file(GLOB INCLUDE_DIRS_HDRS ${INCLUDE_DIRS}/*.hpp)
## rng ##
file(GLOB types_HDRS ${INCLUDE_DIRS}/rng/*.hpp)
file(GLOB types_SRCS ${INCLUDE_DIRS}/rng/*.cpp)
list(APPEND INCLUDE_DIRS_HDRS ${types_HDRS})
list(APPEND INCLUDE_DIRS_HDRS ${types_SRCS})
## types ##
file(GLOB types_HDRS ${INCLUDE_DIRS}/types/*.hpp)
file(GLOB types_SRCS ${INCLUDE_DIRS}/types/*.cpp)
list(APPEND INCLUDE_DIRS_HDRS ${types_HDRS})
list(APPEND INCLUDE_DIRS_HDRS ${types_SRCS})
## system ##
file(GLOB system_HDRS ${INCLUDE_DIRS}/system/*.hpp)
file(GLOB system_SRCS ${INCLUDE_DIRS}/system/*.cpp)
list(APPEND INCLUDE_DIRS_HDRS ${system_HDRS})
list(APPEND INCLUDE_DIRS_SRCS ${system_SRCS})
## box ##
file(GLOB box_HDRS ${INCLUDE_DIRS}/box/*.hpp)
file(GLOB box_SRCS ${INCLUDE_DIRS}/box/*.cpp)
list(APPEND INCLUDE_DIRS_HDRS ${box_HDRS})
list(APPEND INCLUDE_DIRS_SRCS ${box_SRCS})
## neighbourlist ##
file(GLOB neighbourlist_HDRS ${INCLUDE_DIRS}/neighbourlist/*.hpp)
file(GLOB neighbourlist_SRCS ${INCLUDE_DIRS}/neighbourlist/*.cpp)
list(APPEND INCLUDE_DIRS_HDRS ${neighbourlist_HDRS})
list(APPEND INCLUDE_DIRS_SRCS ${neighbourlist_SRCS})
## potentials ##
file(GLOB potentials_HDRS ${INCLUDE_DIRS}/potentials/*.hpp)
file(GLOB potentials_SRCS ${INCLUDE_DIRS}/potentials/*.cpp)
list(APPEND INCLUDE_DIRS_HDRS ${potentials_HDRS})
list(APPEND INCLUDE_DIRS_SRCS ${potentials_SRCS})
## integrators ##
file(GLOB integrators_HDRS ${INCLUDE_DIRS}/integrators/*.hpp)
file(GLOB integrators_SRCS ${INCLUDE_DIRS}/integrators/*.cpp)
list(APPEND INCLUDE_DIRS_HDRS ${integrators_HDRS})
list(APPEND INCLUDE_DIRS_SRCS ${integrators_SRCS})
## evolver ##
file(GLOB evolver_HDRS ${INCLUDE_DIRS}/evolver/*.hpp)
file(GLOB evolver_SRCS ${INCLUDE_DIRS}/evolver/*.cpp)
list(APPEND INCLUDE_DIRS_HDRS ${evolver_HDRS})
list(APPEND INCLUDE_DIRS_SRCS ${evolver_SRCS})
############################################

set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )

MESSAGE("\n") 
MESSAGE("sources : " ${INCLUDE_DIRS_SRCS}) 
MESSAGE("\n") 
MESSAGE("headers : " ${INCLUDE_DIRS_HDRS})
MESSAGE("\n") 

########### LINK THE LIBRARIES  ###############
pybind11_add_module(cppmodule ${INCLUDE_DIRS_SRCS})
set_target_properties(cppmodule PROPERTIES OUTPUT_NAME cppmodule.so SUFFIX "")
