#include <metal_stdlib>
using namespace metal;

// Parallel vector multiplications
kernel void mul_v_f32 (device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] * inB[index];
}

kernel void mul_v_i32 (device const int* inA,
                       device const int* inB,
                       device int* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] * inB[index];
}

kernel void mul_v_i64 (device const long* inA,
                       device const long* inB,
                       device long* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] * inB[index];
}

// Parallel vector additions
kernel void add_v_f32 (device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] + inB[index];
}

kernel void add_v_i32 (device const int* inA,
                       device const int* inB,
                       device int* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] + inB[index];
}

kernel void add_v_i64 (device const long* inA,
                       device const long* inB,
                       device long* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] + inB[index];
}

// Parallel vector subtractions
kernel void sub_v_f32 (device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] - inB[index];
}

kernel void sub_v_i32 (device const int* inA,
                       device const int* inB,
                       device int* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] - inB[index];
}

kernel void sub_v_i64 (device const long* inA,
                       device const long* inB,
                       device long* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] - inB[index];
}

// Matrix multiplication
kernel void mul_m_f32 (device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint2 index [[thread_position_in_grid]],
                       uint2 size [[threads_per_grid]])
{
    uint row = index.x;
    uint col = index.y;
    uint width = size.x;
    uint height = size.y;

    float sum = 0.0f;
    for (uint i = 0; i < width; i++) {
        sum += inA[row * width + i] * inB[i * height + col];
    }
    result[row * height + col] = sum;
}

kernel void mul_m_i32 (device const int* inA,
                       device const int* inB,
                       device int* result,
                       uint2 index [[thread_position_in_grid]],
                       uint2 size [[threads_per_grid]])
{
    uint row = index.x;
    uint col = index.y;
    uint width = size.x;
    uint height = size.y;

    int sum = 0;
    for (uint i = 0; i < width; i++) {
        sum += inA[row * width + i] * inB[i * height + col];
    }
    result[row * height + col] = sum;
}

kernel void mul_m_i64 (device const long* inA,
                       device const long* inB,
                       device long* result,
                       uint2 index [[thread_position_in_grid]],
                       uint2 size [[threads_per_grid]])
{
    uint row = index.x;
    uint col = index.y;
    uint width = size.x;
    uint height = size.y;

    long sum = 0;
    for (uint i = 0; i < width; i++) {
        sum += inA[row * width + i] * inB[i * height + col];
    }
    result[row * height + col] = sum;
}
