#include "tensor.hpp"

#include <iostream>
using namespace tensorlib;

int main() {
    std::vector<int> data = {1, 2, 3, 4, 5, 6};
    std::vector<int> shape = {2, 3};
    Tensor<int> t(data, shape);
    std::cout << t << std::endl;

    std::vector<int> data2 = {5, 6, 7, 8, 10, 11};
    std::vector<int> shape2 = {2, 3};
    Tensor<int> t2(data2, shape2);
    std::cout << t2 << std::endl;

    Tensor<int> t3 = t * t2;
    std::cout << t3 << std::endl;
    std::cout << t << std::endl;
    std::cout << t2 << std::endl;
}
