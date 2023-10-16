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
    /* std::map<std::string, MTL::CommandBuffer*> command_buffers; */

    MTL::Buffer* mBufferA;
    MTL::Buffer* mBufferB;
    MTL::Buffer* mBufferResult;

    // Maps tensor uid to its memory buffer
    std::map<std::string, MTL::Buffer*> tensor_membuf_map;
    // Maps tensor uid to its command buffer, to be executed on realization
    std::map<std::string, MTL::CommandBuffer*> tensor_cmdbuf_map;
    void assign(tensorlib::Tensor<T>* const tensor_ptr);
    // single tensor functions
    void enqueue_kernel(std::string tuid,
            std::string rtuid,
            std::string fn_name);
    // two tensor functions
    void enqueue_kernel(std::string tuid1,
            std::string tuid2,
            std::string rtuid,
            std::string fn_name);
    void schedule_realize(std::string tuid);
    void wait_for(std::string tuid);
    void copy_to_host(tensorlib::Tensor<T>* const tensor_ptr);

    tensorlib::BUFFER_STATUS get_cmdbuf_status(std::string tuid);
};

// TODO: Unable to separately compile
// Appending the file as a quickfix, should be .cpp ideally
#include "tensor_metal.tpp"
