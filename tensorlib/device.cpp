#include <device.hpp>

tensorlib::Device::Device(const std::string& name) : _name(name) {
    set_interface(_name);
}

void tensorlib::Device::set_interface(const std::string& device) {
    if (device == "cpu") {
        /* device_interface = new TensorCPUWrapper(); */
    } else if (device == "gpu") {
        auto gpu_wrapper = std::unique_ptr<TensorDeviceWrapper>();
#ifdef RUN_METAL
        gpu_wrapper.reset(new TensorMetalWrapper());
#else
        // Other devices...
#endif
        this->device_interface = std::move(gpu_wrapper);
    } else {
        throw std::runtime_error("Unknown device: " + device);
    }
}
