#pragma once
#include <vector>
#include <functional>
#include <ostream>
#include <memory>
#include <map>

// Initialize cout as debug stream
// DBOUT << "statement"...
#ifdef DEBUG
    #define DBOUT std::cout << "DEBUG: "
#else
    #define DBOUT 0 && std::cout
#endif

// Initialize cout as verbose stream
#ifdef VERBOSE
    #define VOUT std::cout << "VERBOSE: "
#else
    #define VOUT 0 && std::cout
#endif

// CPU parallelization
#ifndef NTHREADS
    #define NTHREADS 1
#endif

namespace tensorlib {
    template <typename T>
    class Tensor;
    // Used for GPU status tracking
    // Wrappers to convert their status to these values
    enum class BUFFER_STATUS {
        IDLE,
        BUSY,
        DONE,
        ERROR,
    };
    // Keep track of all tensors created, and useful for unique id generation
    long long int __global_tensor_count = 0;
    // Helpful for ownership management.
    // Tensors offer their ownership to this map through unique_ptr
    // (However the unique_ptr must be released if a tensor is destroyed prematurely,
    // see the destructor tensorlib::Tensor<T>::~Tensor())
    // For passing tensors around, we use strings
    template <typename T>
    std::map<std::string, std::unique_ptr<Tensor<T>>> __global_tensor_map;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const tensorlib::Tensor<T>& tensor);

#ifdef RUN_METAL
#include "tensor_metal.hpp"
#endif

namespace tensorlib {

// GPU interfaces
// TODO: Make this generic
// I don't have any other device, but would be good if its extensible.
#ifdef RUN_METAL
template <typename T>
std::shared_ptr<TensorMetalWrapper<T>> __global_metal_interface;
#endif

template <typename T>
class Tensor {
    std::vector<int> strides;
    std::vector<T> grad;
    std::function<T(T)> grad_fn;
    std::vector<std::string> parents;

#ifdef RUN_METAL
    std::shared_ptr<TensorMetalWrapper<T>> local_metal_interface;
#endif

public:
    std::vector<T> data;
    std::vector<int> shape;
    std::string dtype;
    std::string device;
    std::string tuid; // tensor unique id
    bool requires_grad;
    // By default user-created tensors are fully "realized", i.e.
    // They do not need any processing. However, tensors created
    // through operations are not, they need to be realized to have
    // value. This is done to make gpu ops easily parallelizable.
    bool realized = true;
    // Helpful for GPU scheduling
    bool queued_realization = false;

    Tensor(std::vector<T>const& data,
        std::vector<int>const& shape,
        bool requires_grad = true,
        const std::string& dtype = "none",
        const std::string& device = "cpu");

    ~Tensor();

    /* Tensor repr */
    template <typename U>
    friend std::ostream& ::operator<<(std::ostream& os, tensorlib::Tensor<U>& tensor);

    /* Tensor ops */
    inline Tensor<T> __unaryop_boilerplate(
            Tensor<T>& a,
            const std::string& op_name);
    inline Tensor<T> __binop_boilerplate(
            Tensor<T>& a,
            Tensor<T>& b,
            const std::string& op_name);
    Tensor<T> operator+(Tensor<T>& other);
    Tensor<T> operator-(Tensor<T>& other);
    Tensor<T> operator*(Tensor<T>& other);
    Tensor<T> operator/(Tensor<T>& other);
    Tensor<T> operator-() const;
    Tensor<T> operator[](int index) const;
    Tensor<T> operator[](std::vector<int> index) const;
    Tensor<T> matmul(std::vector<T> index) const;

    /* Tensor utils */
    long long int get_mem_size();
    void to(const std::string& device);
    // Realize the tensor.
    // force = true makes the thread wait
    // until the tensor is realized.
    void realize(bool force = false);
};

} // namespace TensorLib

#include "tensor.tpp"
