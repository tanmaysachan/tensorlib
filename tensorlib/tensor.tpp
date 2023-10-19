#include <iostream>
#include <cassert>
#include <numeric>
#include <type_traits>
#include <thread>
#include <queue>
#include <mutex>
#include <map>

/* ----------------------
 *     Tensor repr
 * ---------------------- */

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
    if (tensor.device == "cpu")
        __print_util(os, tensor, 0, 0, tensor.data.size());
    else
        os << "<unrealized>";
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
    const std::string& dtype,
    const std::string& device)
    :   data(std::move(data)),
        shape(std::move(shape)),
        requires_grad(requires_grad) {
    assert(this->data.size() ==
           std::accumulate(this->shape.begin(),
               this->shape.end(),
               1, std::multiplies<int>()));
    // initialize on cpu, shift if anything else
    this->device = "cpu";
    // infer dtype if none
    this->dtype = dtype;
    if (dtype == "none") {
        if (std::is_same<T, int>::value ||
            std::is_same<T, long>:: value ||
            std::is_same<T, long long>::value)
            this->dtype = "i" + std::to_string(sizeof(T)*8);
        else if (std::is_same<T, float>::value ||
                    std::is_same<T, double>::value)
            this->dtype = "f" + std::to_string(sizeof(T)*8);
        else if (std::is_same<T, bool>::value)
            this->dtype = "b" + std::to_string(sizeof(T)*8);
    }
    if (this->dtype == "none") throw std::runtime_error("dtype not implemented");
    // Assign incremental tensor id
    this->tuid = std::to_string(__global_tensor_count) +
        "_" + this->dtype +
        "_" + "Tensor";
    __global_tensor_count++;
    // Offer ownership to global tensor map
    __global_tensor_map<T>[this->tuid].reset(std::move(this));
#ifdef RUN_METAL
    if (!__global_metal_interface<T>)
        __global_metal_interface<T>.reset(new TensorMetalWrapper<T>());
    local_metal_interface = __global_metal_interface<T>;
#endif
    if (device != "cpu") to(device);
}

template <typename T>
tensorlib::Tensor<T>::Tensor(Tensor& other) {
    throw std::runtime_error("Copying a tensor signifies destroying its history.");
}

// Release from unique_ptr if being destroyed
template <typename T>
tensorlib::Tensor<T>::~Tensor() {
    __global_tensor_map<T>[this->tuid].release();
}

/* ----------------------
 *     Tensor Ops
 * ---------------------- */

// Code common to all binary operations.
// Shape conformity, new shape calculation, etc. handled here.
template <typename T>
inline Tensor<T> tensorlib::Tensor<T>::__binop_boilerplate(
        Tensor<T>& a,
        Tensor<T>& b,
        const std::string& op_name) {

    // Tensor for storing results
    Tensor<T> result = Tensor<T>(
        std::vector<T>(a.data.size()),
        std::vector<int>(a.shape),
        a.requires_grad, a.dtype, "cpu");
    result.parents = {a.tuid, b.tuid};
    result.realized = true;
    // If either tensor is on GPU, run the calculation on GPU
    if (a.device == "gpu" || b.device == "gpu") {
        a.to("gpu");
        b.to("gpu");
        // Allocate memory on gpu
        result.to("gpu");
        // Internal gpu tensors are NOT realized instantly.
        result.realized = false;
        try {
#ifdef RUN_METAL
            local_metal_interface->enqueue_kernel({a.tuid, b.tuid},
                    // Trying to follow a naming convention
                    // with the kernel names. The "_v_" is meant
                    // to indicate vector
                    result.tuid, op_name + "_v_" + dtype);
#else
            throw std::runtime_error("device not enabled");
#endif
            return result;
        } catch (std::runtime_error& e) {
            // On kernel failure, fall back to CPU
            a.to("cpu");
            b.to("cpu");
            result.to("cpu");
        }
    }
    return result;
}

template <typename T>
Tensor<T> tensorlib::Tensor<T>::operator+(Tensor& other) {
    Tensor<T> result = __binop_boilerplate(*this, other, "add");
    // If device is not cpu, assume execution handled
    if (device != "cpu") return result;
    // CPU default implementation
    for (int i = 0; i < data.size(); ++i) {
        result.data[i] = this->data[i] + other.data[i];
    }
    return result;
}

template <typename T>
Tensor<T> tensorlib::Tensor<T>::operator-(Tensor& other) {
    Tensor<T> result = __binop_boilerplate(*this, other, "sub");
    if (device != "cpu") return result;
    // CPU default implementation
    for (int i = 0; i < data.size(); ++i) {
        result.data[i] = this->data[i] - other.data[i];
    }
    return result;
}

template <typename T>
Tensor<T> tensorlib::Tensor<T>::operator*(Tensor& other) {
    Tensor<T> result = __binop_boilerplate(*this, other, "mul");
    // CPU default implementation
    for (int i = 0; i < data.size(); ++i) {
        result.data[i] = this->data[i] * other.data[i];
    }
    return result;
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
    if (this->device == device) return;
    if (device != "cpu" && device != "gpu")
        throw std::runtime_error("device not implemented");
    if (device == "gpu") {
        try {
#ifdef RUN_METAL
            local_metal_interface->assign(this);
#else
            throw std::runtime_error("device not enabled");
#endif
        } catch (std::runtime_error& e) {
            std::cout << "Selected device \"" << device
                << "\" not found." << std::endl;
            std::cout << "Falling back to CPU." << std::endl;
            this->device = "cpu";
            return;
        }
    } else {
        //On movement to CPU, tensors must be fully realized.
        if (realized == false) {
            // force realization
            realize(true);
        }
#ifdef RUN_METAL
        // Copy memory into data vector
        local_metal_interface->copy_to_host(this);
#endif
    }
    this->device = device;
}

template <typename T>
void tensorlib::Tensor<T>::realize(bool force) {
    if (realized == true) return;
    if (device == "cpu") {
        // TODO: can implement CPU parallelization
        // and buffer the calculations.
        // NOTE: Currently if not realized on CPU, it is an error
        throw std::runtime_error(
            "Tensor " + tuid + " not realized, \
            graph construction error."
        );
    }
    // Check if parents are realized.
    bool parents_realized = true;
    for (auto& parent : parents) {
        // Parents are tuids, get tensor ptr from global map
        auto& pt = __global_tensor_map<T>[parent];
        if (pt->realized == false) {
            parents_realized = false;
            break;
        }
    }
    /* Parallel Realization */
    // Make the threads continually make tree passes
    // until they find a tensor that has realized parents.
    //
    // TODO: Bit of a clusterfuck, refactor
    if (parents_realized == false) {
        std::mutex lock;
        auto tree_search = [&lock] (const std::string& tuid) {
            auto& cur = __global_tensor_map<T>[tuid];
            // Keep performing parallel BFS search on the graph.

            // Reference wrapper lets you store references in containers.
            std::map<
                const std::string,
                std::reference_wrapper<
                    std::unique_ptr<Tensor<T>>
                >
            > to_process;
            while(cur->realized == false) {
                to_process.clear();
                to_process.insert({tuid, cur});
                // Standard BFS just done with a map
                while(!to_process.empty()) {
                    auto front = *to_process.begin();
                    to_process.erase(to_process.begin());
                    auto& tensor_ptr = front.second.get();
                    if (tensor_ptr->realized == true) continue;
                    if (tensor_ptr->queued_realization == true) {
                        // Check if realization complete,
                        // if so, set to realized.
                        tensor_ptr->realize();
                    }
                    // Check if parents are realized.
                    bool parents_realized = true;
                    for (auto& parent : tensor_ptr->parents) {
                        auto& parent_pt = __global_tensor_map<T>[parent];
                        if (parent_pt->realized == false) {
                            parents_realized = false;
                            to_process.insert({parent, parent_pt});
                        }
                    }
                    // Realize tensor if:
                    // 1. It is not already queued
                    // 2. Parents are realized
                    // 3. No other thread is realizing it
                    if (tensor_ptr->queued_realization == false &&
                            parents_realized == true &&
                            lock.try_lock()) {
                        tensor_ptr->realize();
                        lock.unlock();
                    }
                }
            }
        };
        std::vector<std::thread> threads(NTHREADS);
        for (int t = 0; t < NTHREADS; ++t) {
            threads.push_back(std::thread(tree_search, this->tuid));
        }
        for (auto& thread : threads) {
            if (thread.joinable())
                thread.join();
        }
    } else {
        // Realize self -
        // schedule realize on gpu as soon as you find a candidate,
        // but don't block the thread on it.
#ifdef RUN_METAL
        switch (local_metal_interface->get_cmdbuf_status(this->tuid)) {
            case BUFFER_STATUS::IDLE:
                local_metal_interface->schedule_realize(this->tuid);
                this->queued_realization = true;
                // Wait the thread for realization if forced
                if (force == true) {
                    local_metal_interface->wait_for(this->tuid);
                    this->realized = true;
                }
                break;
            case BUFFER_STATUS::BUSY:
                break;
            case BUFFER_STATUS::DONE:
                this->realized = true;
                break;
            default:
                throw std::runtime_error("Device " + this->device + " processing error.");
        }
#else
        throw std::runtime_error("device not enabled");
#endif
    }
}

} // namespace tensorlib
