#include <iostream>

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
std::ostream& operator<<(std::ostream& os, tensorlib::Tensor<T>& tensor) {
    // Only print if on CPU
    os << "Tensor(";
    if (tensor.device->name() == "cpu")
        __print_util(os, tensor, 0, 0, tensor.data.size());
    else
        os << "<unrealized>";
    os << ", dtype=" << tensor.dtype << ", device=" << tensor.device->name() << ")";
    return os;
}
