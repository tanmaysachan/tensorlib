# Tensorpp

It's a tensor library in C++.

### Expectations
1. Somewhat fast, on macs atleast.
3. Somewhat educational, with clean code (gripe with ggml).

### Wants
1. Capable of loading an LLM.
2. Trainable, with 0 dependencies (Mostly for learning).

### Usage
1. Make sure metal compiler is installed.
2. `make DEBUG=1 RUN_METAL=1` or `make DEBUG=1 RUN_METAL=1 rebuild` for a fresh build.
3. `./run_tests`

### Notes
1. Tensors are by default lazy if not present on CPU. They can be realized and printed by moving to the CPU.
2. Example of a tensor addition -

```c++
#include "tensor.hpp"
using namespace tensorlib;

// Inside function body
//              --- Contents ---, -Shape-
Tensor t1(vector<int>{1, 2, 3, 4, 5, 6}, {2, 3});
t1.to("gpu");
Tensor t2(vector<int>{4, 5, 6, 7, 8, 9}, {2, 3});
t2.to("gpu");
// If tensors on GPU, result will be on GPU
Tensor result = t1 + t2;
result.to("cpu");
std::cout << result << std::endl;
// Expected out - "Tensor([[5,7,9],[11,13,15]], dtype=i32, device=cpu)"
```

### TODO
1. Shape tracking and inference.
2. More ops and efficient kernels.
3. Backprop.
4. Better kernels for metal.
5. Treat CPU as an accelerator, avx and all that [WIP].
6. Graph creation/opt - Ast2IR, DCE, CSE [WIP].
