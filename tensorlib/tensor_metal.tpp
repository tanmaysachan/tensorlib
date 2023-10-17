#include <iostream>

template <typename T>
TensorMetalWrapper<T>::TensorMetalWrapper(MTL::Device* device)
{
    DBOUT << "Initializing TensorMetalWrapper" << std::endl;
    pool = NS::AutoreleasePool::alloc()->init();
    if (device)
        this->device = device;
    else
        this->device = MTL::CreateSystemDefaultDevice();

    auto defaultLibrary = this->device->newDefaultLibrary();
    if (!defaultLibrary) {
        VOUT << "Failed to find the default library"
            << "\nCannot run on current device, change to cpu." << std::endl;
        exit(-1);
    }
    // Initialize shader functions into pipelines
    // TODO: write more kernels
    std::vector<const std::string> shader_functions = {
        "mul_v_f32",
        "mul_v_i32",
        "mul_v_i64",
        "add_v_f32",
        "add_v_i32",
        "add_v_i64",
        "sub_v_f32",
        "sub_v_i32",
        "sub_v_i64",
    };
    NS::Error* error;
    for (auto func: shader_functions) {
        auto _f = NS::String::string(func.c_str(), NS::ASCIIStringEncoding);
        auto f = defaultLibrary->newFunction(_f);
        auto pipeline_state = this->device->newComputePipelineState(f, &error);
        compute_functions.insert(std::make_pair(func, pipeline_state));
    }
    command_queue = this->device->newCommandQueue();
}

template <typename T>
TensorMetalWrapper<T>::~TensorMetalWrapper() {
    // print typename
    pool->release();
}

template <typename T>
void TensorMetalWrapper<T>::assign(tensorlib::Tensor<T>* const tensor_ptr) {
    // Allot memory, map to tuid, copy data
    std::string tuid = tensor_ptr->tuid;
    if (tensor_membuf_map.find(tuid) != tensor_membuf_map.end()) return;
    long long int mem_reqd = tensor_ptr->get_mem_size();

    MTL::Buffer* newbuf = device->newBuffer(mem_reqd, MTL::ResourceStorageModeShared);
    if (!newbuf) {
        VOUT << "Failed to allocate memory for tensor id \"" << tuid << "\"" << std::endl;
    }
    tensor_membuf_map.insert(std::make_pair(tuid, newbuf));

    T* data = (T*) newbuf->contents();
    for (unsigned long int index = 0; index < tensor_ptr->data.size(); ++index)
        data[index] = tensor_ptr->data[index];
}

template <typename T>
void TensorMetalWrapper<T>::enqueue_kernel(
        const std::string& tuid1,
        const std::string& tuid2,
        const std::string& rtuid,
        const std::string& fn_name) {
    MTL::CommandBuffer* command_buf = command_queue->commandBuffer();
    if (!command_buf)
        VOUT << "Failed to create command buffer on metal gpu." << std::endl;
    MTL::ComputeCommandEncoder* encoder = command_buf->computeCommandEncoder();
    if (!encoder)
        VOUT << "Failed to create command encoder on metal gpu." << std::endl;

    auto fn = compute_functions[fn_name];
    encoder->setComputePipelineState(fn);
    // 3rd argument is the index of the buffer in the shader arguments
    encoder->setBuffer(tensor_membuf_map[tuid1], 0, 0);
    encoder->setBuffer(tensor_membuf_map[tuid2], 0, 1);
    encoder->setBuffer(tensor_membuf_map[rtuid], 0, 2);

    // NOTE: Length of the result tensor handled by the TensorLibrary,
    // Not the wrappers.
    int tensorR_size = tensor_membuf_map[rtuid]->length();

    MTL::Size grid_size = MTL::Size(tensorR_size, 1, 1);
    // Calculate a threadgroup size.
    NS::UInteger maxthreads = fn->maxTotalThreadsPerThreadgroup();
    MTL::Size thread_group_size = MTL::Size(
                maxthreads > tensorR_size ? tensorR_size : maxthreads,
                1, 1);
    encoder->dispatchThreads(grid_size, thread_group_size);
    encoder->endEncoding();
    tensor_cmdbuf_map.insert(std::make_pair(rtuid, command_buf));
}

template <typename T>
void TensorMetalWrapper<T>::schedule_realize(std::string tuid) {
    try {
        auto cmd_buf = tensor_cmdbuf_map.at(tuid);
        cmd_buf->commit();
        // Requeue, because committed buffers can't be reused apparently.

    } catch (std::out_of_range& e) {
        VOUT << "Stray tensor being realized? tuid - " << tuid << std::endl;
    }
}

template <typename T>
void TensorMetalWrapper<T>::wait_for(std::string tuid) {
    try {
        auto cmd_buf = tensor_cmdbuf_map.at(tuid);
        cmd_buf->waitUntilCompleted();
    } catch(std::out_of_range& e) {
        // Ignore on miss
    }
}

template <typename T>
void TensorMetalWrapper<T>::copy_to_host(tensorlib::Tensor<T>* const tensor_ptr) {
    MTL::Buffer* buf = tensor_membuf_map.at(tensor_ptr->tuid);
    T* data = (T*) buf->contents();
    for (unsigned long int index = 0; index < tensor_ptr->data.size(); ++index)
        tensor_ptr->data[index] = data[index];
}

template <typename T>
tensorlib::BUFFER_STATUS TensorMetalWrapper<T>::get_cmdbuf_status(std::string tuid) {
    auto cmd_buf = tensor_cmdbuf_map.at(tuid);
    // Convert MTL statuses to tensorlib statuses
    switch (cmd_buf->status()) {
        case MTL::CommandBufferStatusNotEnqueued:
            return tensorlib::BUFFER_STATUS::IDLE;
        case MTL::CommandBufferStatusEnqueued:
            return tensorlib::BUFFER_STATUS::BUSY;
        case MTL::CommandBufferStatusCommitted:
            return tensorlib::BUFFER_STATUS::BUSY;
        case MTL::CommandBufferStatusScheduled:
            return tensorlib::BUFFER_STATUS::BUSY;
        case MTL::CommandBufferStatusCompleted:
            return tensorlib::BUFFER_STATUS::DONE;
        case MTL::CommandBufferStatusError:
            return tensorlib::BUFFER_STATUS::ERROR;
        default:
            return tensorlib::BUFFER_STATUS::ERROR;
    }
}
