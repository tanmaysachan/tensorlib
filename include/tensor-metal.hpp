#pragma once

#include <Metal/Metal.hpp>

class TensorMetalWrapper {
public:
    MTL::Device mDevice;
    MTL::ComputePipelineState* mComputeFunction;
    MTL::CommandQueue* commandQueue;

    MTL::Buffer* inA;
    MTL::Buffer* inB;
    MTL::Buffer* result;

    void initDevice(MTL::Device* device);
}
