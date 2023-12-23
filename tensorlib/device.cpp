#include <device.hpp>

tensorlib::Device::Device(const std::string& name) : _name(name) {
    set_interface(_name);
}

void tensorlib::Device::set_interface(const std::string& device) {
    if (device == "cpu") {
        /* device_interface = new TensorCPUWrapper(); */
    } else if (device == "gpu") {
#ifdef RUN_METAL
        device_interface = new TensorMetalWrapper();
#else
        // Other devices...
#endif
    } else {
        throw std::runtime_error("Unknown device: " + device);
    }
}
