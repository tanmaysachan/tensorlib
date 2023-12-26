#pragma once

#include <vector>
#include <string>

class TensorDeviceWrapper {
public:
    virtual void enqueue_kernel(
        const std::vector<const std::string>&,
        const std::string&,
        const std::string&) = 0;
    virtual void assign(const std::string&, void*, size_t) = 0;
    virtual void copy_to_host(const std::string&, void*, size_t) = 0;
    virtual void wait_for(const std::string&) = 0;
    virtual void schedule_realize(const std::string&) = 0;
    virtual int get_cmdbuf_status(const std::string&) = 0;
    virtual ~TensorDeviceWrapper() = default;
};
