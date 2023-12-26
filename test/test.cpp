#include <tensor.hpp>
#include <iostream>

using namespace std;
using namespace tensorlib;

#define IS_TRUE(x, fn) {                                            \
    if (!(x))                                                       \
        std::cout << fn << " \033[1;31mfailed\033[0m" << std::endl; \
    else                                                            \
        std::cout << fn << " \033[1;32mpassed\033[0m" << std::endl; \
}

/* include TEST files */
#include "test_arith.hpp"

int main() {
    RUN_ARITH_TESTS();
    return 0;
}
