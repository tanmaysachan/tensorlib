#pragma once

#include <string>

namespace tensorlib {

struct DType {
    enum class Type { Float, Int, Bool, None };
    size_t size; // in bits
    size_t bytes; // in bytes (helpful)
    std::string repr;
    DType::Type type;
    DType() : size(0), bytes(0), type(DType::Type::None) {}
    DType(DType::Type type, size_t size) : size(size), type(type) {}
    DType(const std::string& dtype) {
        if (dtype == "float32") {
            size = 32;
            bytes = 4;
            type = DType::Type::Float;
            repr = "f32";
        } else if (dtype == "int32") {
            size = 32;
            bytes = 4;
            type = DType::Type::Int;
            repr = "i32";
        } else if (dtype == "int64") {
            size = 64;
            bytes = 8;
            type = DType::Type::Int;
            repr = "i64";
        } else if (dtype == "bool") {
            size = 1;
            bytes = 1;
            type = DType::Type::Bool;
            repr = "b";
        } else {
            size = 0;
            bytes = 0;
            type = DType::Type::None;
            repr = "none";
        }
    }
    friend bool operator==(const DType& a, const DType& b) = default;
};

} // namespace tensorlib
