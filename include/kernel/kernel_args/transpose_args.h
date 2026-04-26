//
// Created by hzw on 2026/4/23.
//

#ifndef MY_INFERENCE_TRANSPOSE_ARGS_H
#define MY_INFERENCE_TRANSPOSE_ARGS_H
#include <cstdint>

namespace my_inference {
    struct TransposeArgs {
        int64_t num_elem;
        int64_t num_dim;
        int64_t x_shape[8];
        int64_t x_strides[8];
        int64_t y_shape[8];
        int64_t y_strides[8];
        int64_t perm[8];
    };
}
#endif //MY_INFERENCE_TRANSPOSE_ARGS_H
