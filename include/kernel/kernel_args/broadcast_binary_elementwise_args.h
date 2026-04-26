//
// Created by hzw on 2026/4/23.
//

#ifndef MY_INFERENCE_BROADCAST_BINARY_ELEMENTWISE_ARGS_H
#define MY_INFERENCE_BROADCAST_BINARY_ELEMENTWISE_ARGS_H
#include <cstdint>

namespace my_inference {
    struct BroadcastBinaryElementwiseArgs {
        int64_t num_elem;
        int64_t num_dim;
        int64_t shape[8];
        int64_t a_strides[8];
        int64_t b_strides[8];
    };
}
#endif //MY_INFERENCE_BROADCAST_BINARY_ELEMENTWISE_ARGS_H