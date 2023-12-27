#include <iostream>
#include <cassert>
#include <numeric>
#include <type_traits>
#include <thread>
#include <queue>
#include <mutex>
#include <map>

namespace tensorlib {

// UNSAFE
template <typename T>
void* tensorlib::Tensor<T>::get_raw_data_ptr() {
    return static_cast<void*>(context->data.data());
}

template <typename T>
tensorlib::Tensor<T>::Tensor(
    std::vector<T>const& data,
    std::vector<int>const& shape,
    bool requires_grad,
    const std::string& dtype,
    const std::string& device_name)
    :   requires_grad(requires_grad) {
    // Check if data size matches shape
    assert(data.size() == std::accumulate(shape.begin(),
           shape.end(), 1, std::multiplies<int>()));
    // Initialize devices, switch current device
    switch_device_to(device_name);

    std::string _dtype = dtype;
    // infer dtype if none
    {
        if (_dtype == "none") {
            if (std::is_same<T, int>::value ||
                std::is_same<T, long>:: value ||
                std::is_same<T, long long>::value)
                _dtype = "i" + std::to_string(sizeof(T)*8);
            else if (std::is_same<T, float>::value ||
                        std::is_same<T, double>::value)
                _dtype = "f" + std::to_string(sizeof(T)*8);
            else if (std::is_same<T, bool>::value)
                _dtype = "b" + std::to_string(sizeof(T)*8);
        }
        if (_dtype == "none") throw std::runtime_error("dtype not implemented");
    }

    // Housekeeping
    {
        // Create new context, move ownership of important vectors.
        context.reset(new TensorPassingContext());

        context->dtype = _dtype;
        context->tuid = _dtype +
            "_" + std::to_string(global_tensor_count) +
            "_" + "Tensor";

        global_tensor_count++;
        // Offer ownership to global tensor map
        global_tensor_map<T>[this->tuid()].reset(std::move(this));

        context->shape = shape;
        // Default strides
        context->strides = std::vector<int>(shape.size(), 1);

        auto* raw_data = data.data();
        size_t bytes = data.size() * sizeof(T);
        context->data = std::vector<uint8_t>(bytes);
        std::memcpy(context->data.data(), raw_data, bytes);

        context->parents = std::vector<std::string>();
    }
}

template <typename T>
tensorlib::Tensor<T>::Tensor(Tensor& other) {
    throw std::runtime_error("Copying a tensor signifies destroying its history.");
    // TODO: it probably does not. How should a copy look like?
    //  Should the connections be backpropagated? Should the new tensor be a leaf?
}

// Release from unique_ptr if being destroyed
template <typename T>
tensorlib::Tensor<T>::~Tensor() {
    global_tensor_map<T>[this->tuid()].release();
}

/* ----------------------
 *     Tensor Ops
 * ---------------------- */

// Code common to all binary operations.
// Shape conformity, new shape calculation, etc. handled here.
template <typename T>
inline Tensor<T> tensorlib::Tensor<T>::binop_boilerplate(
        Tensor<T>& a,
        Tensor<T>& b,
        const std::string& op_name) {

    // Tensor for storing results
    int num_elements = std::accumulate(a.shape().begin(),
            a.shape().end(), 1, std::multiplies<int>());
    Tensor<T> result = Tensor<T>(
        std::vector<T>(num_elements),
        std::vector<int>(a.shape()),
        a.requires_grad, a.dtype(), "cpu");
    result.context->parents = {a.tuid(), b.tuid()};
    result.realized = true;
    // If either tensor is on GPU, run the calculation on GPU
    if (a.device->name() == "gpu" || b.device->name() == "gpu") {
        a.to("gpu");
        b.to("gpu");
        // Allocate memory on gpu
        result.to("gpu");
        // Internal gpu tensors are NOT realized instantly.
        result.realized = false;
        try {
#ifdef RUN_METAL
            device->get()->enqueue_kernel({a.tuid(), b.tuid()},
                    // Trying to follow a naming convention
                    // with the kernel names. The "_v_" is meant
                    // to indicate vector
                    result.tuid(), op_name + "_v_" + this->dtype());
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
    Tensor<T> result = binop_boilerplate(*this, other, "add");
    return result;
}

template <typename T>
Tensor<T> tensorlib::Tensor<T>::operator-(Tensor& other) {
    Tensor<T> result = binop_boilerplate(*this, other, "sub");
    return result;
}

template <typename T>
Tensor<T> tensorlib::Tensor<T>::operator*(Tensor& other) {
    Tensor<T> result = binop_boilerplate(*this, other, "mul");
    return result;
}

/* ----------------------
 *    Tensor Utils
 * ---------------------- */
template <typename T>
long long int tensorlib::Tensor<T>::get_mem_size() {
    return context->data.size();
}

template <typename T>
void tensorlib::Tensor<T>::switch_device_to(const std::string& device_name) {
    // Initialize once (TODO: can i remove this?)
    if (device_interfaces.empty()) {
        device_interfaces["cpu"] = new tensorlib::Device("cpu");
        device_interfaces["gpu"] = new tensorlib::Device("gpu");
    }
    if (device_interfaces.find(device_name) == device_interfaces.end())
        throw std::runtime_error("device not implemented");
    this->device = device_interfaces[device_name];
}

template <typename T>
void tensorlib::Tensor<T>::to(const std::string& device_name) {
    if (device_name != "cpu" && device_name != "gpu")
        throw std::runtime_error("device not implemented");
    if (device_name == "gpu") {
        try {
#ifdef RUN_METAL
            auto new_device = device_interfaces[device_name];
            new_device->get()->assign(this->tuid(), get_raw_data_ptr(), get_mem_size());
#else
            throw std::runtime_error("device not enabled");
#endif
        } catch (std::runtime_error& e) {
            std::cout << "Selected device \"" << device->name()
                << "\" not found." << std::endl;
            std::cout << "Falling back to CPU." << std::endl;
            std::cerr << "Error: " << e.what() << std::endl;
            switch_device_to("cpu");
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
        device->get()->copy_to_host(this->tuid(), get_raw_data_ptr(), get_mem_size());
#endif
    }
    switch_device_to(device_name);
}

template <typename T>
void tensorlib::Tensor<T>::realize(bool force) {
    if (realized == true) return;
    if (device->name() == "cpu") {
        // TODO: can implement CPU parallelization
        // and buffer the calculations.
        // NOTE: Currently if not realized on CPU, it is an error
        throw std::runtime_error(
            "Tensor " + this->tuid() + " not realized, \
            graph construction error."
        );
    }
    // Explicitly move tensor to device to force memory assignment
    this->to(device->name());
    // Check if parents are realized.
    bool parents_realized = true;
    for (auto& parent : context->parents) {
        // Parents are tuids, get tensor ptr from global map
        auto& pt = global_tensor_map<T>[parent];
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
            auto& cur = global_tensor_map<T>[tuid];
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
                    for (auto& parent : tensor_ptr->context->parents) {
                        auto& parent_pt = global_tensor_map<T>[parent];
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
        for (int t = 0; t < NTHREADS; ++t)
            threads.push_back(std::thread(tree_search, this->tuid()));

        for (auto& thread : threads)
            if (thread.joinable())
                thread.join();
    } else {
        // Realize self -
        // schedule realize on gpu as soon as you find a candidate,
        // but don't block the thread on it.
#ifdef RUN_METAL
        switch (device->get()->get_cmdbuf_status(this->tuid())) {
            case 1:
                device->get()->schedule_realize(this->tuid());
                this->queued_realization = true;
                // Wait the thread for realization if forced
                if (force == true) {
                    device->get()->wait_for(this->tuid());
                    this->realized = true;
                }
                break;
            case 2:
                break;
            case 0:
                this->realized = true;
                break;
            default:
                throw std::runtime_error("Device " + this->device->name() + " processing error.");
        }
#else
        throw std::runtime_error("device not enabled");
#endif
    }
}

} // namespace tensorlib
