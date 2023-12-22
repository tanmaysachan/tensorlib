#include <tensor.hpp>
#include <iostream>

using namespace tensorlib;

#define IS_TRUE(x) {                                                              \
    if (!(x))                                                                     \
        std::cout << __FUNCTION__ << " failed on line " << __LINE__ << std::endl; \
}

/* include TEST files */
#include "test_arith.hpp"

int main() {
    RUN_ARITH_TESTS();
    return 0;
}
