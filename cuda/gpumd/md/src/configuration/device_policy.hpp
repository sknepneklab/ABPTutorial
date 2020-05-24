///////////////////////////////////////////////////////////////////////////////
// 
// "Hemi" CUDA Portable C/C++ Utilities
// 
// Copyright 2012-2015 NVIDIA Corporation
//
// License: BSD License, see LICENSE file in Hemi home directory
//
// The home for Hemi is https://github.com/harrism/hemi
//
///////////////////////////////////////////////////////////////////////////////
// Please see the file README.md (https://github.com/harrism/hemi/README.md) 
// for full documentation and discussion.
///////////////////////////////////////////////////////////////////////////////

#ifndef __device_policy_hpp__
#define __device_policy_hpp__

#include <iostream>
#include <string>
#include "dev_types.hpp"

class DevicePolicy 
{
public:

    DevicePolicy(): mDevice(0) {}
    ~DevicePolicy() {}

    void _setDevice(const int CUDA_DEVICE)
    { 
        #ifdef __CUDACC__
        mDevice = CUDA_DEVICE;
        cudaSetDevice(CUDA_DEVICE); 
        #endif
    }
    int multiProcessorCount()
    { 
        #ifdef __CUDACC__
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, mDevice);
        return(deviceProp.multiProcessorCount);
        #else
        return 0;
        #endif
    }

    std::string getDeviceProperty(const int CUDA_DEVICE)
    {
        #ifdef __CUDA_ENABLE__
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, CUDA_DEVICE);

        auto device = "Device " + std::to_string(CUDA_DEVICE) + " " + deviceProp.name;
        // Console log
        int driverVersion = 0, runtimeVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        auto driver_version = std::to_string((driverVersion/1000)) + "." + std::to_string(((driverVersion%100)/10));
        auto run_time_version = std::to_string((runtimeVersion/1000)) + "." + std::to_string(((runtimeVersion%100)/10));
        auto version = "CUDA Driver Version / Runtime Version " + driver_version + "/" + run_time_version;
        auto capability = "CUDA Capability Major/Minor version number: " + std::to_string(deviceProp.major) + "." + std::to_string(deviceProp.minor);
        
        auto total_memory = "Total amount of global memory " + std::to_string(((int)((double)deviceProp.totalGlobalMem/1048576.0f/1024))) + " GB";

        auto multiprocessors = std::to_string(deviceProp.multiProcessorCount) + " Multiprocessors ";
        auto CUDA_Cores_per_sm = std::to_string(_ConvertSMVer2Cores(deviceProp.major,deviceProp.minor)) + " CUDA Cores/MP ";
        auto CUDA_Cores = std::to_string((_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount)) + " CUDA Cores ";
        auto Clock_rate = "GPU Clock rate " + std::to_string((deviceProp.clockRate * 1e-6f)) + " GHz ";

        //auto line_str = "\n-------------------------------------------------------------------------------------------------------\n";
        auto line_str = "\n";
        std::string out_string = line_str;
        out_string+= device + "\n";
        out_string+= version + "\n";
        out_string+= capability + "\n";
        out_string+= total_memory + "\n";
        out_string+= multiprocessors + CUDA_Cores_per_sm + CUDA_Cores + "\n";
        out_string+= Clock_rate + "\n";
        out_string+= line_str;
        return out_string;
        #else
        std::string out_string = "No cuda capable device in use\n";
        #endif
    }

private:

    // Beginning of GPU Architecture definitions
    inline int _ConvertSMVer2Cores(const int major, const int minor)
    {
        // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
        typedef struct
        {
            int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
            int Cores;
        } sSMtoCores;

        sSMtoCores nGpuArchCoresPerSM[] =
        {
            { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
            { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
            { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
            { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
            { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
            { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
            { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
            { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
            { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
            { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
            { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
	       {0x52, 128},
	      {0x53, 128},
	      {0x60,  64},
	      {0x61, 128},
	      {0x62, 128},
	      {0x70,  64},
	      {0x72,  64},
	      {0x75,  64},
	      {-1, -1}};

        int index = 0;

        while (nGpuArchCoresPerSM[index].SM != -1)
        {
            if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
            {
                return nGpuArchCoresPerSM[index].Cores;
            }
            index++;
        }
        // If we don't find the values, we default use the previous one to run properly
        //printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
        return nGpuArchCoresPerSM[index-1].Cores;
    }

protected:
    //Variables
    int mDevice;
};



#endif






