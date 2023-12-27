#pragma once

#include <map>
#include <memory>

#include <device.hpp>
#include <dtype.hpp>

namespace tensorlib {

class Tensor;


// Minimal context for tensor passing
struct TensorPassingContext {
    // Just bytes
    std::vector<uint8_t> data;
    DType dtype;
    std::string tuid;
    std::vector<int> shape;
    std::vector<int> strides;
    std::vector<std::string> parents;
    Device* device;
};

// Device interfaces
//
// Exact interfaces are defined in device.hpp
// Initialized on first use.
std::unordered_map<std::string, Device*> device_interfaces;

// Keep track of all tensors created, and useful for unique id generation
long long int global_tensor_count = 0;

// Helpful for ownership management.
// For passing tensors around, we use strings
std::map<std::string, std::unique_ptr<Tensor>> global_tensor_map;

} // namespace tensorlib
