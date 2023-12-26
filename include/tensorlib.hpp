#include <map>
#include <memory>
#include <device.hpp>

namespace tensorlib {
    template <typename T>
    class Tensor;

    // Minimal context for tensor passing
    /* struct TensorContext { */
    /*     public: */
    /*         std::string tuid; */
    /*         std::string dtype; */
    /*         std::vector<int> shape; */
    /*         std::vector<int> strides; */
    /*         std::vector<std::string> parents; */
    /*         bool requires_grad; */
    /*         bool realized; */
    /*         bool queued_realization; */
    /*         std::vector<long long int> data_ptr; */
    /*         std::vector<long long int> grad_ptr; */
    /*         std::vector<long long int> grad_fn_ptr; */
    /* } */
    // Device interfaces
    //
    // Exact interfaces are defined in device.hpp
    // Initialized on first use.
    std::unordered_map<std::string, Device*> device_interfaces;

    // Keep track of all tensors created, and useful for unique id generation
    long long int global_tensor_count = 0;

    // Helpful for ownership management.
    // For passing tensors around, we use strings
    template <typename T>
    std::map<std::string, std::unique_ptr<Tensor<T>>> global_tensor_map;
}
