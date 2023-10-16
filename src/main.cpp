#include "tensor.hpp"

#include <iostream>


using namespace tensorlib;

int main() {
    Tensor<float> t0({1.2, 2, 3, 4, 5, 6}, {2, 3});
    std::cout << "t0 " << t0 << std::endl;
    t0.to("gpu");

    Tensor<float> t1({1, 2.3, 3, 4, 5, 6}, {2, 3});
    std::cout << "t1 " << t1 << std::endl;
    t1.to("gpu");

    Tensor<float> t2({1, 2, 3, 4, 5.9, 6}, {2, 3});
    std::cout << "t2 " << t2 << std::endl;
    t2.to("gpu");

    Tensor<float> t3 = t0 + t1;
    /* t3.to("cpu"); */
    /* Expected t3 = {2, 4, 6, 8, 10, 12} */
    std::cout << t3 << std::endl;

    Tensor<float> t4 = t2 + t0;
    t4.to("cpu");
    /* Expected t4 = {2, 4, 6, 8, 10, 12} */
    std::cout << t4 << std::endl;

    Tensor<float> t5 = t3 * t4;
    /* t5.to("cpu"); */
    /* Expected t5 = {4, 16, 36, 64, 100, 144} */
    std::cout << t5 << std::endl;

    Tensor<float> t6 = t5 - t0;
    t6.to("cpu");
    /* Expected t6 = {3, 14, 33, 60, 95, 138} */
    std::cout << t6 << std::endl;

    t3.to("cpu");
    std::cout << t3 << std::endl;
    t5.to("cpu");
    std::cout << t5 << std::endl;
}
