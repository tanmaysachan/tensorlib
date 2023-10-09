#pragma once

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal.hpp>

#include <vector>
#include <functional>
#include <iostream>

#ifdef RUN_METAL
#include "tensor-metal.hpp"
#endif

namespace tensorlib {
    template <typename T>
    class Tensor;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const tensorlib::Tensor<T>& tensor);

namespace tensorlib {

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
    bool requires_grad; Tensor(std::vector<T>& data,
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
};

} // namespace TensorLib

#include "tensor.tpp"
