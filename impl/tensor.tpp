#include <iostream>
#include <cassert>
#include <numeric>
#include <type_traits>

/* ----------------------
 *     Tensor repr
 * ---------------------- */

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


/* ----------------------
 * Tensor Implementation
 * ---------------------- */
namespace tensorlib {

template <typename T>
tensorlib::Tensor<T>::Tensor(
    std::vector<T>const& data,
    std::vector<int>const& shape,
    bool requires_grad,
    std::string dtype,
    std::string device)
    :   data(std::move(data)),
        shape(std::move(shape)),
        requires_grad(requires_grad) {
    assert(this->data.size() ==
           std::accumulate(this->shape.begin(), this->shape.end(), 1, std::multiplies<int>()));
    // initialize on cpu, shift if anything else
    this->device = "cpu";
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
    // Assign incremental tensor id
    this->tuid = "Tensor_" + std::to_string(__global_tensor_count);
    __global_tensor_count++;

    // Offer ownership to global tensor map
    __global_tensor_map<T>[this->tuid].reset(this);

#ifdef RUN_METAL
    if (!__global_metal_interface<T>)
        __global_metal_interface<T>.reset(new TensorMetalWrapper<T>());
    local_metal_interface = __global_metal_interface<T>;
#endif

    if (device != "cpu") {
        to(device);
    }
}


template <typename T>
tensorlib::Tensor<T>::~Tensor() {
    DBOUT << "Tensor " << tuid << " destroyed" << std::endl;
    __global_tensor_map<T>[this->tuid].release();
}

/* ----------------------
 *     Tensor Ops
 * ---------------------- */
template <typename T>
Tensor<T> tensorlib::Tensor<T>::operator+(Tensor& other) {
    assert(shape == other.shape);
#ifdef RUN_METAL
    // Execute metal kernels if either tensor is on gpu
    if (this->device == "gpu" || other.device == "gpu") {
        this->to("gpu");
        other.to("gpu");
        try {
            // Generate a return tensor on device
            Tensor<T> result = Tensor<T>(
                    std::vector<int>(this->data.size()),
                    std::vector<int>(this->shape), requires_grad, dtype, "gpu");
            // Internal tensors are NOT realized instantly.
            result.realized = false;
            local_metal_interface->enqueue_kernel(this->tuid,
                    other.tuid, result.tuid, "add_v_i32");
            return result;
        } catch (std::runtime_error& e) {
            // On kernel failure, fall back to CPU
            this->to("cpu");
            other.to("cpu");
        }
    }
#endif
    std::vector<T> _data(data.size());
    std::vector<int> _shape = shape;
    for (int i = 0; i < data.size(); ++i) {
        _data[i] = data[i] + other.data[i];
    }
    return Tensor<T>(_data, _shape, requires_grad);
}

template <typename T>
Tensor<T> tensorlib::Tensor<T>::operator-(Tensor& other) {
    assert(shape == other.shape);
    std::vector<T> _data(data.size());
    std::vector<int> _shape = shape;
    for (int i = 0; i < data.size(); ++i) {
        _data[i] = data[i] - other.data[i];
    }
    return Tensor<T>(_data, _shape, requires_grad);
}

template <typename T>
Tensor<T> tensorlib::Tensor<T>::operator*(Tensor& other) {
    // Hadamard product, NOT matmul
    assert(shape == other.shape);
    std::vector<T> _data(data.size());
    std::vector<int> _shape = shape;
    for (int i = 0; i < data.size(); ++i) {
        _data[i] = data[i] * other.data[i];
    }
    return Tensor<T>(_data, _shape, requires_grad);
}

/* ----------------------
 *    Tensor Utils
 * ---------------------- */
template <typename T>
long long int tensorlib::Tensor<T>::get_mem_size() {
    return data.size() * sizeof(T);
}

template <typename T>
void tensorlib::Tensor<T>::to(const std::string& device) {
    if (device == this->device) return;
    if (device != "cpu" && device != "gpu")
        throw std::runtime_error("device not implemented");

    this->device = device;
    if (device == "gpu") {
#ifdef RUN_METAL
        try {
            local_metal_interface->assign(this);
        } catch (std::runtime_error& e) {
            std::cout << "Selected device \"" << device
                << "\" not found." << std::endl;
            std::cout << "Falling back to CPU." << std::endl;
            this->device = "cpu";
        }
#endif
    } else {
        //TODO: implement transfer to CPU
        //On movement to CPU, tensors must be fully realized.
    }
}

template <typename T>
void tensorlib::Tensor<T>::realize() {

}

} // namespace tensorlib
