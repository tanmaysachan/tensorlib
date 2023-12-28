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

class Tensor {
    void* get_raw_data_ptr();
    static TensorPassingContext init_context(
            const void* data,
            std::vector<int>const& shape,
            const std::string& dtype,
            size_t dtype_size_in_bytes,
            size_t num_elements,
            const std::string& device_name);

    template <typename T>
    DType infer_dtype(const std::string& input_dtype);
public:
    TensorPassingContext context;

    // tensor unique id
    bool requires_grad = true;

    // By default user-created tensors are fully "realized", i.e.
    // They do not need any processing. However, tensors created
    // through operations are not, they need to be realized to have
    // value.
    bool realized = true;
    // Helpful for GPU scheduling
    bool queued_realization = false;

    template <typename T>
    Tensor(std::vector<T>const& data,
           std::vector<int>const& shape,
           bool requires_grad = true,
           const std::string& dtype = "none",
           const std::string& device_name = "cpu");

    Tensor(Tensor& other); // Copy constructor

    ~Tensor();

    /* Tensor repr */
    friend std::ostream& ::operator<<(std::ostream& os, tensorlib::Tensor& tensor);

    /* Equality */
    /* TODO: to be used for testing only right now. Equality checks on tensor
     * would be much more complicated than this. */
    friend bool operator==(const Tensor& a, const Tensor& b) {
        return a.context.data == b.context.data
            && a.shape() == b.shape();
    }

    /* getters */
    const std::string& tuid() const { return context.tuid; }
    const DType& dtype() const { return context.dtype; }
    const std::vector<int>& shape() const { return context.shape; }
    const std::vector<int>& strides() const { return context.strides; }

    /* Tensor ops */
    // boilerplates
    inline Tensor unaryop_boilerplate(
            Tensor& a,
            const std::string& op_name);
    inline Tensor binop_boilerplate(
            Tensor& a,
            Tensor& b,
            const std::string& op_name);

    Tensor operator+(Tensor& other);
    Tensor operator-(Tensor& other);
    Tensor operator*(Tensor& other);
    Tensor operator/(Tensor& other);
    Tensor operator-() const;
    Tensor operator[](int index) const;
    Tensor operator[](std::vector<int> index) const;
    Tensor matmul(std::vector<int> index) const;

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
