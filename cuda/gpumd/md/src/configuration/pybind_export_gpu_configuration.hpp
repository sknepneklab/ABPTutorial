#ifndef __pybind_export_gpu_configuration_hpp__
#define __pybind_export_gpu_configuration_hpp__

#include "execution_policy.hpp"

void export_ExecutionPolicyGPU(py::module& m)
{
    py::class_<ExecutionPolicy>(m, "GPUExecutionPolicy")
        .def(py::init<>(), 
        "Fully automatic: The default device is 0, the number of blocks is set to 10*(Streaming Multiprocessors) and the number of Threads to 256")
        .def("getConfigState", &ExecutionPolicy::getConfigState)
        .def("getGridSize", &ExecutionPolicy::getGridSize)
        .def("getBlockSize", &ExecutionPolicy::getBlockSize)
        .def("getDeviceNumber", &ExecutionPolicy::getDeviceNumber)
        .def("setDevice", &ExecutionPolicy::setDevice, py::arg("Device").noconvert(), "Set the device globally")
        .def("setGridSize", &ExecutionPolicy::setGridSize, py::arg("GridSize").noconvert(), "Set the Grid Size for CUDA kernels executions")
        .def("setBlockSize", &ExecutionPolicy::setBlockSize, py::arg("BlockSize").noconvert(), "Set the Block Size for CUDA kernels executions")
        .def("getDeviceProperty", &ExecutionPolicy::getUseDeviceProperty);
}

#endif