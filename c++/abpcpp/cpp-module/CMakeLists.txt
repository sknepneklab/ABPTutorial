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
## particle ##
file(GLOB particle_HDRS ${INCLUDE_DIRS}/particle/*.hpp)
file(GLOB particle_SRCS ${INCLUDE_DIRS}/particle/*.cpp)
list(APPEND INCLUDE_DIRS_HDRS ${particle_HDRS})
list(APPEND INCLUDE_DIRS_SRCS ${particle_SRCS})
## box ##
file(GLOB box_HDRS ${INCLUDE_DIRS}/box/*.hpp)
file(GLOB box_SRCS ${INCLUDE_DIRS}/box/*.cpp)
list(APPEND INCLUDE_DIRS_HDRS ${box_HDRS})
list(APPEND INCLUDE_DIRS_SRCS ${box_SRCS})
############################################

set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )

MESSAGE("C++ DIR: "  ${INCLUDE_DIRS_SRCS}) 
MESSAGE("CUDA DIR: " ${INCLUDE_DIRS_HDRS})

pybind11_add_module(abpcpp ${INCLUDE_DIRS_SRCS})
