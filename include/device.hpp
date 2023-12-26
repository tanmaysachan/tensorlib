/* Define device interfaces here.
 *
 * Due to duplicate symbols, ensure the Metal.hpp file
 * is not included in the same translation unit as tensorlib.
 */
#pragma once
#include <memory>

#ifdef TENSORLIB_HPP
    #include <tensor_device_wrapper.hpp>
#else
    #include <tensor_cpu.hpp>
    #ifdef RUN_METAL
        #include <tensor_metal.hpp>
    #endif
#endif

namespace tensorlib {

class Device {
    const std::string _name;
    std::unique_ptr<TensorDeviceWrapper> device_interface;
public:
    Device(const std::string& name);
    std::string name() const { return _name; }
    std::unique_ptr<TensorDeviceWrapper>& get() { return device_interface; }
    void set_interface(const std::string& device);
};

} // namespace tensorlib
