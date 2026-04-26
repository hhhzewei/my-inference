//
// Created by hzw on 2026/4/23.
//

#ifndef MY_INFERENCE_SLICE_ARGS_H
#define MY_INFERENCE_SLICE_ARGS_H
#include <cstdint>

namespace my_inference {
    struct SliceArgs {
        int64_t y_num_elem;
        int64_t num_dim;
        int64_t y_shape[8];
        int64_t x_stride[8];
    };
}
#endif //MY_INFERENCE_SLICE_ARGS_H
