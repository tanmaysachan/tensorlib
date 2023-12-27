bool test_add() {
    DISABLE_TEST();
    Tensor t0(vector<int>{1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor t1(vector<int>{1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor t2 = t0 + t1;
    Tensor t3(vector<int>{2, 4, 6, 8, 10, 12}, {2, 3});
    return t2 == t3;
}

bool test_add_gpu() {
    Tensor t0(vector<int>{1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor t1(vector<int>{1, 2, 3, 4, 5, 6}, {2, 3});
    t0.to("gpu");
    t1.to("gpu");
    Tensor t2 = t0 + t1;
    Tensor t3(vector<int>{2, 4, 6, 8, 10, 12}, {2, 3});
    t2.to("cpu");
    return t2 == t3;
}

bool test_add_gpu_cpu() {
    Tensor t0(vector<int>{1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor t1(vector<int>{1, 2, 3, 4, 5, 6}, {2, 3});
    t0.to("gpu");
    Tensor t2 = t0 + t1;
    Tensor t3(vector<int>{2, 4, 6, 8, 10, 12}, {2, 3});
    t2.to("cpu");
    return t2 == t3;
}

bool test_add_float() {
    DISABLE_TEST();
    Tensor t0(vector<float>{1.2, 2, 3, 4, 5, 6}, {2, 3});
    Tensor t1(vector<float>{1, 2.3, 3, 4, 5, 6}, {2, 3});
    Tensor t2 = t0 + t1;
    Tensor t3(vector<float>{2.2, 4.3, 6, 8, 10, 12}, {2, 3});
    return t2 == t3;
}

bool test_add_float_gpu() {
    Tensor t0(vector<float>{1.2, 2, 3, 4, 5, 6}, {2, 3});
    Tensor t1(vector<float>{1, 2.3, 3, 4, 5, 6}, {2, 3});
    t0.to("gpu");
    t1.to("gpu");
    Tensor t2 = t0 + t1;
    Tensor t3(vector<float>{2.2, 4.3, 6, 8, 10, 12}, {2, 3});
    t2.to("cpu");
    return t2 == t3;
}

bool test_sub_gpu() {
    Tensor t0(vector<float>{1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor t1(vector<float>{1, 2, 3, 4, 5, 6}, {2, 3});
    t0.to("gpu");
    t1.to("gpu");
    Tensor t2 = t0 - t1;
    Tensor t3(vector<float>{0, 0, 0, 0, 0, 0}, {2, 3});
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
