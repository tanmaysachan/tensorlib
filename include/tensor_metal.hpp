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
    std::map<const std::string, MTL::ComputePipelineState*> compute_functions;

    // Maps tensor uid to its memory buffer
    std::map<const std::string, MTL::Buffer*> tensor_membuf_map;
    // Maps tensor uid to its command buffer, parent tensors and function
    // to be executed on realization.
    // NOTE: Tracking parent tensors because metal doesn't support
    // reusing command buffers. To recreate them, we need the parent ids,
    // the function name, all that good info.
    typedef struct {
        std::vector<const std::string> parent_tuids;
        const std::string rtuid;
        const std::string fn_name;
        MTL::CommandBuffer* cmd_buf;
    } kernel_info;
    std::map<const std::string, kernel_info> tensor_cmdbuf_map;
    void assign(tensorlib::Tensor<T>* const tensor_ptr);
    // tensor functions
    void enqueue_kernel(
            const std::vector<const std::string>& tuids,
            const std::string& rtuid,
            const std::string& fn_name);
    std::vector<kernel_info> to_requeue;
    void requeue();

        void schedule_realize(const std::string& tuid);
    void wait_for(const std::string& tuid);
    void copy_to_host(tensorlib::Tensor<T>* const tensor_ptr);

    tensorlib::BUFFER_STATUS get_cmdbuf_status(const std::string& tuid);
};

// TODO: Unable to separately compile
// Appending the file as a quickfix, should be .cpp ideally
#include "tensor_metal.tpp"
