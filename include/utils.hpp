#pragma once
// Initialize cout as debug stream
// DBOUT << "statement"...
#ifdef DEBUG
    #define DBOUT std::cout << "DEBUG: "
#else
    #define DBOUT 0 && std::cout
#endif

// Initialize cout as verbose stream
#ifdef VERBOSE
    #define VOUT std::cout << "VERBOSE: "
#else
    #define VOUT 0 && std::cout
#endif

// CPU parallelization
#ifndef NTHREADS
    #define NTHREADS 1
#endif

#include <iostream>
#include <map>

namespace tensorlib {
    template <typename T>
    class Tensor;
}

template<typename T>
void __print_util(std::ostream& os,
        const tensorlib::Tensor<T>& tensor,
        int shape_idx,
        int offset,
        int par_size);

template <typename T>
std::ostream& operator<<(std::ostream& os, tensorlib::Tensor<T>& tensor);

// Template file
#include <utils.tpp>