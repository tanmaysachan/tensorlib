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
        "mul_v_i32",
        "mul_v_i64",
        "mul_v_f32",
        "add_v_i32",
        "add_v_i64",
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
    DBOUT << "TensorMetalWrapper<" << typeid(T).name() << ">" << " released" << std::endl;
    pool->release();
}

template <typename T>
void TensorMetalWrapper<T>::assign(tensorlib::Tensor<T>* const tensor_ptr) {
    // Allot memory, map to tuid, copy data
    std::string tuid = tensor_ptr->tuid;
    long long int mem_reqd = tensor_ptr->get_mem_size();

    MTL::Buffer* newbuf = device->newBuffer(mem_reqd, MTL::ResourceStorageModeShared);
    if (!newbuf) {
        VOUT << "Failed to allocate memory for tensor id \"" << tuid << "\"" << std::endl;
    }
    tensor_membuf_map.insert(std::make_pair(tuid, newbuf));

    T* data = (T*) newbuf->contents();
    for (unsigned long int index = 0; index < tensor_ptr->data.size(); ++index)
        data[index] = tensor_ptr->data[index];

    VOUT << "Assigned tensor id \"" << tuid << "\" to device \""
        << tensor_ptr->device << "\"" << std::endl;
}

template <typename T>
void TensorMetalWrapper<T>::enqueue_kernel(
        std::string tuid1,
        std::string tuid2,
        std::string rtuid,
        std::string fn_name) {
    MTL::CommandBuffer* command_buf = command_queue->commandBuffer();
    if (!command_buf) {
        VOUT << "Failed to create command buffer on metal gpu" << std::endl;
    }
    MTL::ComputeCommandEncoder* encoder = command_buf->computeCommandEncoder();
    if (!encoder) {
        VOUT << "Failed to create command encoder on metal gpu" << std::endl;
    }

    auto fn = compute_functions[fn_name];

    encoder->setComputePipelineState(fn);

    // 3rd argument is the index of the buffer in the shader arguments
    encoder->setBuffer(tensor_membuf_map[tuid1], 0, 0);
    encoder->setBuffer(tensor_membuf_map[tuid2], 0, 1);
    encoder->setBuffer(tensor_membuf_map[rtuid], 0, 2);
    
    int tensor1_size = tensor_membuf_map[tuid1]->length();
    int tensor2_size = tensor_membuf_map[tuid2]->length();
    int tensorR_size = tensor_membuf_map[rtuid]->length();

    MTL::Size grid_size = MTL::Size(tensorR_size, 1, 1);
    
    // Calculate a threadgroup size.
    NS::UInteger maxthreads = fn->maxTotalThreadsPerThreadgroup();
    MTL::Size thread_group_size = MTL::Size(
                maxthreads > tensorR_size ? tensorR_size : maxthreads,
                1, 1);
    encoder->dispatchThreads(grid_size, thread_group_size);
    encoder->endEncoding();
}

template <typename T>
void TensorMetalWrapper<T>::realize(std::string tuid) {
    try {
        auto gpu_command_buf = tensor_cmdbuf_map.at(tuid);
        gpu_command_buf->commit();
    } catch (std::out_of_range& e) {
        DBOUT << "Failed to realize tensor id \"" << tuid << "\"" << std::endl;
    }
}


/* Sample code, for help */
/* To be removed */

template <typename T>
void TensorMetalWrapper<T>::prepareData() {
    // Allocate three buffers to hold our initial data and the result.
    mBufferA = device->newBuffer(BUFFER_SIZE, MTL::ResourceStorageModePrivate);
    mBufferB = device->newBuffer(BUFFER_SIZE, MTL::ResourceStorageModePrivate);
    mBufferResult = device->newBuffer(BUFFER_SIZE, MTL::ResourceStorageModePrivate);

    generateRandomIntData(mBufferA);
    generateRandomIntData(mBufferB);
}

template <typename T>
void TensorMetalWrapper<T>::generateRandomIntData(MTL::Buffer * buffer) {
    int* dataPtr = (int*) buffer->contents();

    for(unsigned long int index = 0; index < ARRAY_LENGTH; index++)
        dataPtr[index] = 2;
}

template <typename T>
void TensorMetalWrapper<T>::sendComputeCommand() {
    // Create a command buffer to hold commands.
    MTL::CommandBuffer* commandBuffer = command_queue->commandBuffer();
    assert(commandBuffer != nullptr);
    
    // Start a compute pass.
    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);
    
    encodeComputeCommand(computeEncoder);
    
    // End the compute pass.
    computeEncoder->endEncoding();
    
    // Execute the command.
    commandBuffer->commit();
    
    // Normally, you want to do other work in your app while the GPU is running,
    // but in this example, the code simply blocks until the calculation is complete.
    commandBuffer->waitUntilCompleted();
    
    verifyResults();
}

template <typename T>
void TensorMetalWrapper<T>::encodeComputeCommand(MTL::ComputeCommandEncoder * computeEncoder) {
    // Encode the pipeline state object and its parameters.
    computeEncoder->setComputePipelineState(compute_functions["mul_v_i32"]);
    computeEncoder->setBuffer(mBufferA, 0, 0);
    computeEncoder->setBuffer(mBufferB, 0, 1);
    computeEncoder->setBuffer(mBufferResult, 0, 2);
    
    MTL::Size gridSize = MTL::Size(ARRAY_LENGTH, 1, 1);
    
    // Calculate a threadgroup size.
    NS::UInteger threadGroupSize = compute_functions["mul_v_i32"]->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > ARRAY_LENGTH)
    {
        threadGroupSize = ARRAY_LENGTH;
    }
    MTL::Size threadgroupSize = MTL::Size(threadGroupSize, 1, 1);

    // Encode the compute command.
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
}

template <typename T>
void TensorMetalWrapper<T>::verifyResults(){
    int* a = (int*) mBufferA->contents();
    int* b = (int*) mBufferB->contents();
    int* result = (int*) mBufferResult->contents();
    
    for(unsigned long int index = 0; index < ARRAY_LENGTH; index++)
        if(result[index] != 4) {
            std::cout << "Compute ERROR: index= " << index << " result= " << result[index] <<  " vs " << a[index] * b[index] << " = a * b\n";
        }
    
    std::cout << "Compute results as expected.\n";
}
