#pragma once

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Metal.hpp>

#include <map>
#include <memory>

const int ARRAY_LENGTH = 1024;
const int BUFFER_SIZE = 1024;

template <typename T>
class TensorMetalWrapper {
public:
    MTL::Device* device;
    NS::AutoreleasePool* pool;

    TensorMetalWrapper(MTL::Device* device = nullptr);

    ~TensorMetalWrapper();

    MTL::CommandQueue* command_queue;
    std::map<std::string, MTL::ComputePipelineState*> compute_functions;
    std::map<std::string, MTL::CommandBuffer*> command_buffers;

    MTL::Buffer* mBufferA;
    MTL::Buffer* mBufferB;
    MTL::Buffer* mBufferResult;

    std::map<std::string, MTL::Buffer*> tensor_buffer_map;
    void assign(tensorlib::Tensor<T>* const tensor_ptr);
    void enqueue_kernel(std::string tuid1, std::string tuid2, std::string func);

    void initDevice();
    void prepareData();
    void generateRandomIntData(MTL::Buffer* buffer);
    void sendComputeCommand();
    void encodeComputeCommand(MTL::ComputeCommandEncoder* computeEncoder);
    void verifyResults();
};

// TODO: Unable to separately compile
// Appending the file as a quickfix, should be .cpp ideally
#include "tensor_metal.tpp"
