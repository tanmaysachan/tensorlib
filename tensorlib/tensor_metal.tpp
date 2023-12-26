#include <iostream>

TensorMetalWrapper::TensorMetalWrapper(MTL::Device* device) {
    DBOUT << "Initializing TensorMetalWrapper" << std::endl;
    pool = NS::AutoreleasePool::alloc()->init();
    if (device)
        this->device = device;
    else
        this->device = MTL::CreateSystemDefaultDevice();

    auto default_library = this->device->newDefaultLibrary();
    if (!default_library) {
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
        "mul_m_f32",
        "mul_m_i32",
        "mul_m_i64",
    };
    NS::Error* error;
    for (auto func: shader_functions) {
        auto _f = NS::String::string(func.c_str(), NS::ASCIIStringEncoding);
        auto f = default_library->newFunction(_f);
        auto pipeline_state = this->device->newComputePipelineState(f, &error);
        compute_functions.insert(std::make_pair(func, pipeline_state));
    }
    command_queue = this->device->newCommandQueue();
}

TensorMetalWrapper::~TensorMetalWrapper() {
    // print typename
    pool->release();
}

void TensorMetalWrapper::assign(const std::string& tuid,
                                void* raw_data,
                                size_t mem_size) {
    // Make sense of the raw pointer
    uint8_t* data = static_cast<uint8_t*>(raw_data);

    // Allot memory, map to tuid, copy data
    if (tensor_membuf_map.find(tuid) != tensor_membuf_map.end()) return;

    MTL::Buffer* newbuf = device->newBuffer(mem_size, MTL::ResourceStorageModePrivate);
    if (!newbuf) {
        throw std::runtime_error("Failed to allocate memory for a tensor");
    }
    tensor_membuf_map.insert(std::make_pair(tuid, newbuf));

    size_t bytes = mem_size / sizeof(uint8_t);
    uint8_t* device_data = (uint8_t*) newbuf->contents();
    for (unsigned long int index = 0; index < bytes; ++index)
        device_data[index] = data[index];
}

void TensorMetalWrapper::copy_to_host(const std::string& tuid,
                                      void* raw_data,
                                      size_t mem_size) {
    uint8_t* data = static_cast<uint8_t*>(raw_data);
    MTL::Buffer* buf = tensor_membuf_map.at(tuid);
    size_t bytes = mem_size / sizeof(uint8_t);
    uint8_t* device_data = (uint8_t*) buf->contents();
    for (unsigned long int index = 0; index < bytes; ++index)
        data[index] = device_data[index];
}

void TensorMetalWrapper::enqueue_kernel(
        const std::vector<const std::string>& tuids,
        const std::string& rtuid,
        const std::string& fn_name) {
    MTL::CommandBuffer* cmd_buf = command_queue->commandBuffer();
    if (!cmd_buf) throw std::runtime_error("Failed to create command buffer on gpu.");

    MTL::ComputeCommandEncoder* encoder = cmd_buf->computeCommandEncoder();
    if (!encoder) throw std::runtime_error("Failed to create command encoder on gpu.");

    auto fn = compute_functions[fn_name];
    encoder->setComputePipelineState(fn);
    // 3rd argument is the index of the buffer in the shader arguments
    for (unsigned long int i = 0; i < tuids.size(); ++i)
        encoder->setBuffer(tensor_membuf_map.at(tuids[i]), 0, i);
    encoder->setBuffer(tensor_membuf_map[rtuid], 0, tuids.size());

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

    tensor_cmdbuf_map.insert(std::make_pair(rtuid, (kernel_info){
        // Struct to hold command buffer info
        // See tensor_metal.hpp for details
        tuids,
        rtuid,
        fn_name,
        cmd_buf
    }));
}

void TensorMetalWrapper::requeue() {
    for (auto it : to_requeue) {
        enqueue_kernel(it.parent_tuids, it.rtuid, it.fn_name);
    }
}

void TensorMetalWrapper::schedule_realize(const std::string& tuid) {
    try {
        auto kinfo = tensor_cmdbuf_map.at(tuid);
        kinfo.cmd_buf->commit();
        to_requeue.push_back(kinfo);
    } catch (std::out_of_range& e) {
        VOUT << "Stray tensor being realized? tuid - " << tuid << std::endl;
    }
}

void TensorMetalWrapper::wait_for(const std::string& tuid) {
    try {
        auto cmd_buf = tensor_cmdbuf_map.at(tuid).cmd_buf;
        cmd_buf->waitUntilCompleted();
    } catch(std::out_of_range& e) {
        // Ignore on miss
    }
}

int TensorMetalWrapper::get_cmdbuf_status(const std::string& tuid) {
    auto cmd_buf = tensor_cmdbuf_map.at(tuid).cmd_buf;
    // 0 - Completed
    // 1 - Idle
    // 2 - Busy
    // -1 - Error
    switch (cmd_buf->status()) {
        case MTL::CommandBufferStatusNotEnqueued:
            return 1;
        case MTL::CommandBufferStatusEnqueued:
            return 2;
        case MTL::CommandBufferStatusCommitted:
            return 2;
        case MTL::CommandBufferStatusScheduled:
            return 2;
        case MTL::CommandBufferStatusCompleted:
            return 0;
        case MTL::CommandBufferStatusError:
            return -1;
        default:
            return -1;
    }
}
