#include <iostream>

TensorMetalWrapper::TensorMetalWrapper() {
    // init class
    std::cout << "init class" << std::endl;
}

void TensorMetalWrapper::initDevice(MTL::Device* device) {
    mDevice = device;
    NS::Error* error;

    auto defaultLibrary = mDevice->newDefaultLibrary();

    if (!defaultLibrary) {
        std::cerr << "Failed to find the default library.\n";
        exit(-1);
    }
    
    auto _mul_v_i32 = NS::String::string("mul_v_i32", NS::ASCIIStringEncoding);

    auto mul_v_i32 = defaultLibrary->newFunction(_mul_v_i32);

    mComputeFunction = mDevice->newComputePipelineState(mul_v_i32, &error);

    mCommandQueue = mDevice->newCommandQueue();
}

void TensorMetalWrapper::prepareData() {
    // Allocate three buffers to hold our initial data and the result.
    mBufferA = mDevice->newBuffer(BUFFER_SIZE, MTL::ResourceStorageModeShared);
    mBufferB = mDevice->newBuffer(BUFFER_SIZE, MTL::ResourceStorageModeShared);
    mBufferResult = mDevice->newBuffer(BUFFER_SIZE, MTL::ResourceStorageModeShared);

    generateRandomIntData(mBufferA);
    generateRandomIntData(mBufferB);
}

void TensorMetalWrapper::generateRandomIntData(MTL::Buffer * buffer) {
    int* dataPtr = (int*) buffer->contents();
    
    for(unsigned long int index = 0; index < ARRAY_LENGTH; index++)
        dataPtr[index] = 2;
}

void TensorMetalWrapper::sendComputeCommand() {
    // Create a command buffer to hold commands.
    MTL::CommandBuffer* commandBuffer = mCommandQueue->commandBuffer();
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

void TensorMetalWrapper::encodeComputeCommand(MTL::ComputeCommandEncoder * computeEncoder) {
    // Encode the pipeline state object and its parameters.
    computeEncoder->setComputePipelineState(mComputeFunction);
    computeEncoder->setBuffer(mBufferA, 0, 0);
    computeEncoder->setBuffer(mBufferB, 0, 1);
    computeEncoder->setBuffer(mBufferResult, 0, 2);
    
    MTL::Size gridSize = MTL::Size(ARRAY_LENGTH, 1, 1);
    
    // Calculate a threadgroup size.
    NS::UInteger threadGroupSize = mComputeFunction->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > ARRAY_LENGTH)
    {
        threadGroupSize = ARRAY_LENGTH;
    }
    MTL::Size threadgroupSize = MTL::Size(threadGroupSize, 1, 1);

    // Encode the compute command.
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
}

void TensorMetalWrapper::verifyResults(){
    int* a = (int*) mBufferA->contents();
    int* b = (int*) mBufferB->contents();
    int* result = (int*) mBufferResult->contents();
    
    for(unsigned long int index = 0; index < ARRAY_LENGTH; index++)
        if(result[index] != 4) {
            std::cout << "Compute ERROR: index= " << index << " result= " << result[index] <<  " vs " << a[index] * b[index] << " = a * b\n";
        }
    
    std::cout << "Compute results as expected.\n";
}
