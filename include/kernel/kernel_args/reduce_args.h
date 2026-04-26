//
// Created by hzw on 2026/4/23.
//

#ifndef MY_INFERENCE_REDUCE_ARGS_H
#define MY_INFERENCE_REDUCE_ARGS_H
#include <cstdint>

namespace my_inference {
    struct ReduceArgs {
        int64_t Outer;
        int64_t Reduce;
        int64_t Inner;
    };
}

#endif //MY_INFERENCE_REDUCE_ARGS_H
