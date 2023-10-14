#include "tensor.hpp"

#include <iostream>


using namespace tensorlib;

int main() {
    /* std::vector<int> data = {1, 2, 3, 4, 5, 6}; */
    std::vector<int> shape = {2, 3};
    Tensor<int> t({1, 2, 3, 4, 5, 6}, shape);
    t.to("gpu");
    std::cout << t << std::endl;

    std::vector<float> data1 = {1.5, 2.2, 3.1, 4.5, 5.8, 6.0};
    std::vector<int> shape1 = {2, 3};
    Tensor<float> t1(data1, shape1);
    t1.to("gpu");
    std::cout << t1 << std::endl;

    std::vector<int> data2 = {5, 6, 7, 8, 10, 11};
    std::vector<int> shape2 = {2, 3};
    Tensor<int> t2(data2, shape2);
    std::cout << t2 << std::endl;

    Tensor<int> t3 = t + t2;
    Tensor<int> t4 = t * t2;
    std::cout << t3 << std::endl;
    std::cout << t << std::endl;
    std::cout << t2 << std::endl;

    std::cout << t.tuid << std::endl;
    std::cout << t2.tuid << std::endl;
    std::cout << t3.tuid << std::endl;

    /* std:: cout << metal_interface<int> << std::endl; */

#ifdef RUN_METAL
    /* TensorMetalWrapper<int>* computer = new TensorMetalWrapper<int>(); */
    /* // Create buffers to hold data */
    /* computer->prepareData(); */
    
    /* // Time the compute phase. */
    /* auto start = std::chrono::steady_clock::now(); */
    
    /* // Send a command to the GPU to perform the calculation. */
    /* computer->sendComputeCommand(); */
    
    /* // End of compute phase. */
    /* auto end = std::chrono::steady_clock::now(); */
    /* auto delta_time = end - start; */
    
    /* std::cout << "Computation completed in " */
    /*         << std::chrono::duration <double, std::milli> (delta_time).count() */
    /*         << " ms for array of size " */
    /*         << ARRAY_LENGTH */
    /*         <<".\n"; */
#endif
}
