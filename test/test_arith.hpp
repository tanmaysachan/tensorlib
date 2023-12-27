bool test_add() {
    DISABLE_TEST();
    Tensor<int> t0({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor<int> t1({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor<int> t2 = t0 + t1;
    Tensor<int> t3({2, 4, 6, 8, 10, 12}, {2, 3});
    return t2 == t3;
}

bool test_add_gpu() {
    Tensor<int> t0({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor<int> t1({1, 2, 3, 4, 5, 6}, {2, 3});
    t0.to("gpu");
    t1.to("gpu");
    Tensor<int> t2 = t0 + t1;
    Tensor<int> t3({2, 4, 6, 8, 10, 12}, {2, 3});
    t2.to("cpu");
    return t2 == t3;
}

bool test_add_gpu_cpu() {
    Tensor<int> t0({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor<int> t1({1, 2, 3, 4, 5, 6}, {2, 3});
    t0.to("gpu");
    Tensor<int> t2 = t0 + t1;
    Tensor<int> t3({2, 4, 6, 8, 10, 12}, {2, 3});
    t2.to("cpu");
    return t2 == t3;
}

bool test_add_float() {
    DISABLE_TEST();
    Tensor<float> t0({1.2, 2, 3, 4, 5, 6}, {2, 3});
    Tensor<float> t1({1, 2.3, 3, 4, 5, 6}, {2, 3});
    Tensor<float> t2 = t0 + t1;
    Tensor<float> t3({2.2, 4.3, 6, 8, 10, 12}, {2, 3});
    return t2 == t3;
}

bool test_add_float_gpu() {
    Tensor<float> t0({1.2, 2, 3, 4, 5, 6}, {2, 3});
    Tensor<float> t1({1, 2.3, 3, 4, 5, 6}, {2, 3});
    t0.to("gpu");
    t1.to("gpu");
    Tensor<float> t2 = t0 + t1;
    Tensor<float> t3({2.2, 4.3, 6, 8, 10, 12}, {2, 3});
    t2.to("cpu");
    return t2 == t3;
}

bool test_sub_gpu() {
    Tensor<float> t0({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor<float> t1({1, 2, 3, 4, 5, 6}, {2, 3});
    t0.to("gpu");
    t1.to("gpu");
    Tensor<float> t2 = t0 - t1;
    Tensor<float> t3({0, 0, 0, 0, 0, 0}, {2, 3});
    t2.to("cpu");
    return t2 == t3;
}

// ADD TESTS TO THIS MACRO
#define RUN_ARITH_TESTS() \
    IS_TRUE(test_add(), "test_add"); \
    IS_TRUE(test_add_gpu(), "test_add_gpu"); \
    IS_TRUE(test_add_gpu_cpu(), "test_add_gpu_cpu"); \
    IS_TRUE(test_add_float(), "test_add_float"); \
    IS_TRUE(test_add_float_gpu(), "test_add_float_gpu"); \
    IS_TRUE(test_sub_gpu(), "test_sub_gpu"); \
    std::cout << "arith tests finished ✓" << std::endl;
