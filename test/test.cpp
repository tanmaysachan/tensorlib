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

#define DISABLE_TEST() { \
    std::cout << "\033[1;33m Warning - " << __func__ << " disabled\033[0m -- "; \
    return true; \
}

/* include TEST files */
#include "test_arith.hpp"

int main() {
    RUN_ARITH_TESTS();
    return 0;
}
