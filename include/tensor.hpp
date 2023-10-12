#pragma once
#include <vector>
#include <functional>
#include <ostream>

// Initialize cout as debug stream
// DBOUT << "statement"...
#ifdef DEBUG
    #define DBOUT std::cout
#else
    #define DBOUT 0 && std::cout
#endif

namespace tensorlib {
    template <typename T>
    class Tensor;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const tensorlib::Tensor<T>& tensor);

#ifdef RUN_METAL
#include "tensor_metal.hpp"
#endif

namespace tensorlib {

long long int global_tensor_count = 0;

#ifdef RUN_METAL
template <typename T>
TensorMetalWrapper<T>* metal_interface = nullptr;
#endif

template <typename T>
class Tensor {
    std::vector<int> strides;
    std::vector<T> grad;
    std::function<T(T)> grad_fn;

public:
    std::vector<T> data;
    std::vector<int> shape;
    std::string dtype;
    std::string device;
    std::string tuid; // tensor unique id
    bool requires_grad;

    Tensor(std::vector<T>& data,
        std::vector<int>& shape,
        bool requires_grad = false,
        std::string dtype = "none",
        std::string device = "cpu");

    /* Tensor repr */
    template <typename U>
    friend std::ostream& ::operator<<(std::ostream& os, const tensorlib::Tensor<U>& tensor);

    /* Tensor ops */
    Tensor<T> operator+(const Tensor<T>& other) const;
    Tensor<T> operator-(const Tensor<T>& other) const;
    Tensor<T> operator*(const Tensor<T>& other) const;
    Tensor<T> operator/(const Tensor<T>& other) const;
    Tensor<T> operator-() const;
    Tensor<T> operator[](int index) const;
    Tensor<T> operator[](std::vector<int> index) const;
    Tensor<T> matmul(std::vector<T> index) const;

    /* Tensor utils */
    long long int get_mem_size();
    void to(const std::string& device);
};

} // namespace TensorLib

#include "tensor.tpp"
