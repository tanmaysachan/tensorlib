#include "tensor-metal.hpp"

void TensorMetalWrapper::initDevice(MTL::Device* device) {
    mDevice = device;
    NSError* error;

    auto defaultLibrary = mDevice->newDefaultLibrary();

    if (!defaultLibrary) {
        std::cerr << "Failed to find the default library.\n";
        exit(-1);
    }
    
    auto _mul_v_f32 = NS::String::string("mul_v_f32", NS::ASCIIStringEncoding);
    auto _mul_v_f64= NS::String::string("mul_v_f32", NS::ASCIIStringEncoding);

    auto mul_v_f32 = defaultLibrary->newFunction(_mul_v_f32);
    auto mul_v_f64 = defaultLibrary->newFunction(_mul_v_f64);
}
