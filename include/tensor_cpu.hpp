#include <tensor_device_wrapper.hpp>

#include <map>

class TensorCPUWrapper : public TensorDeviceWrapper {
private:

public:
    TensorCPUWrapper();
    ~TensorCPUWrapper();

    void enqueue_kernel(
            const std::vector<const std::string>& tuids,
            const std::string& rtuid,
            const std::string& fn_name);
    void assign(const std::string& tuid, void* data, size_t mem_size);
    void copy_to_host(const std::string& tuid, void* data, size_t mem_size);
    void wait_for(const std::string& tuid);
    void schedule_realize(const std::string& tuid);
    int get_cmdbuf_status(const std::string& tuid);
};
