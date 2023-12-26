#pragma once

#define TENSORLIB_HPP

#include <tensorlib.hpp>
#include <device.hpp>
#include <utils.hpp>

#include <vector>
#include <functional>
#include <ostream>
#include <memory>
#include <map>

namespace tensorlib {

template <typename T>
class Tensor {
    std::vector<int> strides;
    std::vector<T> grad;
    std::function<T(T)> grad_fn;
    std::vector<std::string> parents;

    // UNSAFE
    void* get_raw_data_ptr();
public:
    std::vector<T> data;
    std::vector<int> shape;
    std::string dtype;
    Device* device;

    // tensor unique id
    std::string tuid;
    bool requires_grad;

    // By default user-created tensors are fully "realized", i.e.
    // They do not need any processing. However, tensors created
    // through operations are not, they need to be realized to have
    // value.
    bool realized = true;
    // Helpful for GPU scheduling
    bool queued_realization = false;

    Tensor(std::vector<T>const& data,
           std::vector<int>const& shape,
           bool requires_grad = true,
           const std::string& dtype = "none",
           const std::string& device_name = "cpu");

    Tensor(Tensor<T>& other); // Copy constructor

    ~Tensor();

    /* Tensor repr */
    template <typename U>
    friend std::ostream& ::operator<<(std::ostream& os, tensorlib::Tensor<U>& tensor);

    /* Equality */
    /* TODO: to be used for testing only right now. Equality checks on tensor
     * would be much more complicated than this. */
    friend bool operator==(const Tensor<T>& a, const Tensor<T>& b) {
        return a.data == b.data && a.shape == b.shape && a.dtype == b.dtype;
    }

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
    void to(const std::string& device_name);
    // Realize the tensor.
    // force = true makes the thread wait
    // until the tensor is realized.
    void realize(bool force = false);
    void switch_device_to(const std::string& device_name);
};

} // namespace TensorLib

#include "tensor.tpp"
