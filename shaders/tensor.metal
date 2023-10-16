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
