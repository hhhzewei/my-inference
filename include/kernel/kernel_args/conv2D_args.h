//
// Created by hzw on 2026/4/23.
//

#ifndef MY_INFERENCE_CONV2D_ARGS_H
#define MY_INFERENCE_CONV2D_ARGS_H
#include <cstdint>

namespace my_inference {
    struct Conv2DArgs {
        int64_t N;
        int64_t C_IN;
        int64_t H_IN;
        int64_t W_IN;
        int64_t C_OUT;
        int64_t H_OUT;
        int64_t W_OUT;
        int64_t H_K;
        int64_t W_K;
        int64_t PAD_UP;
        int64_t PAD_DOWN;
        int64_t PAD_LEFT;
        int64_t PAD_RIGHT;
        int64_t STRIDE_H;
        int64_t STRIDE_W;
        int64_t DILATION_H;
        int64_t DILATION_W;
        int64_t GROUP;
    };
}

#endif //MY_INFERENCE_CONV2D_ARGS_H