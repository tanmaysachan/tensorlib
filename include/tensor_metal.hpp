#pragma once

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <tensor_device_wrapper.hpp>
#include <Metal.hpp>

#include <map>

#include <utils.hpp>

class TensorMetalWrapper : public TensorDeviceWrapper {
private:
    MTL::Device* device;
    NS::AutoreleasePool* pool;

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

    std::vector<kernel_info> to_requeue;
    void requeue();

public:
    TensorMetalWrapper(MTL::Device* device = nullptr);
    ~TensorMetalWrapper();

    void enqueue_kernel(
            const std::vector<const std::string>& tuids,
            const std::string& rtuid,
            const std::string& fn_name);
    void assign(const std::string& tuid, void* data, size_t mem_size);
    void copy_to_host(const std::string& tuid, void* data, size_t mem_size);
    void wait_for(const std::string& tuid);
    void schedule_realize(const std::string& tuid);
    int get_cmdbuf_status(const std::string& tuid);
};

// TODO: Unable to separately compile
// Appending the file as a quickfix, should be .cpp ideally
#include "tensor_metal.tpp"
