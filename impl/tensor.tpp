#include <iostream>
#include <cassert>
#include <numeric>
#include <type_traits>

/* Tensor repr */
/* global namespace */
template<typename T>
void __print_util(std::ostream& os,
        const tensorlib::Tensor<T>& tensor,
        int shape_idx,
        int offset,
        int par_size) {
    int dim_i = tensor.shape[shape_idx];
    int elems_per_vec = par_size/dim_i;
    os << "[";
    if (shape_idx == tensor.shape.size() - 1) {
        for (int i = 0; i < dim_i; ++i) {
            os << tensor.data[offset + i];
            if (i != dim_i - 1) {
                os << ",";
            }
        }
    } else {
        for  (int i = 0; i < dim_i; ++i) {
            __print_util(os, tensor, shape_idx+1, offset+elems_per_vec*i, elems_per_vec);
            if (i != dim_i - 1) {
                os << ",";
            }
        }
    }
    os << "]";
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const tensorlib::Tensor<T>& tensor) {
    os << "Tensor(";
    __print_util(os, tensor, 0, 0, tensor.data.size());
    os << ", dtype=" << tensor.dtype << ", device=" << tensor.device << ")";
    return os;
}

/* Tensor implementation */
namespace tensorlib {

template <typename T>
tensorlib::Tensor<T>::Tensor(
    std::vector<T>& data,
    std::vector<int>& shape,
    bool requires_grad,
    std::string dtype,
    std::string device)
    :   data(std::move(data)),
        shape(std::move(shape)),
        requires_grad(requires_grad) {
    assert(this->data.size() ==
           std::accumulate(this->shape.begin(), this->shape.end(), 1, std::multiplies<int>()));
    this->device = device;
    // infer dtype
    if (dtype == "none") {
        if (std::is_same<T, int>::value ||
            std::is_same<T, long>:: value ||
            std::is_same<T, long long>::value) {
            this->dtype = "i" + std::to_string(sizeof(T)*8);
        } else if (std::is_same<T, float>::value ||
                    std::is_same<T, double>::value) {
            this->dtype = "f" + std::to_string(sizeof(T)*8);
        } else if (std::is_same<T, bool>::value) {
            this->dtype = "b" + std::to_string(sizeof(T)*8);
        }
    }
    if (this->dtype == "none") throw std::runtime_error("dtype not implemented");
}

/* Tensor ops */
template <typename T>
Tensor<T> tensorlib::Tensor<T>::operator+(const Tensor& other) const {
    assert(this->shape == other.shape);
#ifdef RUN_METAL
    
#else
    std::vector<T> newdata(this->data.size());
    std::vector<int> shape = this->shape;
    for (int i = 0; i < this->data.size(); ++i) {
        newdata[i] = this->data[i] + other.data[i];
    }
    return Tensor<T>(newdata, shape, requires_grad);
#endif
}

template <typename T>
Tensor<T> tensorlib::Tensor<T>::operator-(const Tensor& other) const {
    assert(this->shape == other.shape);
    std::vector<T> newdata(this->data.size());
    std::vector<int> shape = this->shape;
    for (int i = 0; i < this->data.size(); ++i) {
        newdata[i] = this->data[i] - other.data[i];
    }
    return Tensor<T>(newdata, shape, requires_grad);
}

template <typename T>
Tensor<T> tensorlib::Tensor<T>::operator*(const Tensor& other) const {
    // Hadamard product, NOT matmul
    assert(this->shape == other.shape);
    std::vector<T> newdata(this->data.size());
    std::vector<int> shape = this->shape;
    for (int i = 0; i < this->data.size(); ++i) {
        newdata[i] = this->data[i] * other.data[i];
    }
    return Tensor<T>(newdata, shape, requires_grad);
}

} // namespace tensorlib
