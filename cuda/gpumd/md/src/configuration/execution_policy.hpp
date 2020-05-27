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

#ifndef __execution_policy_hpp__
#define __execution_policy_hpp__

#include "device_policy.hpp"

/**
 * @class ExecutionPolicy
 * @brief ExecutionPolicy class manage how the cuda kernels are executed 
 */
class ExecutionPolicy : public DevicePolicy
{
public:
    ExecutionPolicy()
    {
        mmultiProcessorCount = multiProcessorCount();
        if (mmultiProcessorCount > 0)
            mGridSize = 10 * mmultiProcessorCount;
        else
            mGridSize = 10 * 16;
        mBlockSize = 256;
        mStream = 0;
        mSharedMemBytes = 0;
    }
    ~ExecutionPolicy() {}
    int getDeviceNumber() const { return mDevice; }
    int getConfigState() const { return mState; }
    int getGridSize() const { return mGridSize; }
    int getBlockSize() const { return mBlockSize; }
    int getMaxBlockSize() const { return mMaxBlockSize; }
    int getMultiProcessorCount() const { return mmultiProcessorCount; }
    size_t getSharedMemBytes() const { return mSharedMemBytes; }
    devStream_t getStream() const { return mStream; }
    std::string getUseDeviceProperty() { return getDeviceProperty(mDevice); }

    void setGridSize(int arg)
    {
        mGridSize = arg;
    }
    void setBlockSize(int arg)
    {
        mBlockSize = arg;
    }
    void setMaxBlockSize(int arg)
    {
        mMaxBlockSize = arg;
    }
    void setSharedMemBytes(size_t arg)
    {
        mSharedMemBytes = arg;
    }
    void setStream(devStream_t stream)
    {
        mStream = stream;
    }
    void setDevice(const int CUDA_DEVICE_ID)
    {
        _setDevice(CUDA_DEVICE_ID);
        if (mmultiProcessorCount > 0)
            mmultiProcessorCount = multiProcessorCount();
        else
            mmultiProcessorCount = 1;
    }

private:
    //Variables
    int mState;
    int mGridSize;
    int mBlockSize;
    int mMaxBlockSize;
    int mmultiProcessorCount;
    size_t mSharedMemBytes;
    devStream_t mStream;
};

#endif
