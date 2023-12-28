#pragma once

#include <string>
#include <map>

namespace tensorlib {

enum class Primitive { Float, Int, Bool, None };

static std::map<Primitive, std::string> primitive_repr = {
    {Primitive::Float, "f"},
    {Primitive::Int, "i"},
    {Primitive::Bool, "b"},
    {Primitive::None, "n"},
};

struct DType {
    size_t size; // in bits
    size_t bytes; // in bytes (helpful)
    std::string repr;
    Primitive type;
    DType() : size(0), bytes(0), repr("empty"), type(Primitive::None) {}
    DType(Primitive type, size_t size)
        : size(size), bytes(size/8), type(type) {
        this->repr = primitive_repr[type] + std::to_string(size);
    }
    friend bool operator==(const DType& a, const DType& b) = default;
};

static std::map<std::string, DType> dtypes_map = {
    {"float32", DType(Primitive::Float, 32)},
    {"f32", DType(Primitive::Float, 32)},
    {"int32", DType(Primitive::Int, 32)},
    {"i32", DType(Primitive::Int, 32)},
    {"int64", DType(Primitive::Int, 64)},
    {"i64", DType(Primitive::Int, 64)},
    {"bool", DType(Primitive::Bool, 1)},
    {"b1", DType(Primitive::Bool, 1)},
};

} // namespace tensorlib
