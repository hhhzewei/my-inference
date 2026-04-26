//
// Created by hzw on 2026/4/23.
//

#ifndef MY_INFERENCE_BATCH_NORM_ARGS_H
#define MY_INFERENCE_BATCH_NORM_ARGS_H
#include <cstdint>

namespace my_inference {
    struct BatchNormalizeArgs {
        int64_t N;
        int64_t C;
        int64_t stride;
        float eps;
    };
}
#endif //MY_INFERENCE_BATCH_NORM_ARGS_H
