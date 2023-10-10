#pragma once

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Metal.hpp>

const unsigned long int ARRAY_LENGTH = 100000000;
const unsigned long int BUFFER_SIZE = ARRAY_LENGTH * sizeof(int);

class TensorMetalWrapper {
public:
    TensorMetalWrapper();

    MTL::Device* mDevice;
    MTL::ComputePipelineState* mComputeFunction;
    MTL::CommandQueue* mCommandQueue;

    MTL::Buffer* mBufferA;
    MTL::Buffer* mBufferB;
    MTL::Buffer* mBufferResult;

    void initDevice(MTL::Device* device);
    void prepareData();
    void generateRandomIntData(MTL::Buffer* buffer);
    void sendComputeCommand();
    void encodeComputeCommand(MTL::ComputeCommandEncoder* computeEncoder);
    void verifyResults();
};

#include "tensor_metal.ipp"
