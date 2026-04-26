//
// Created by hzw on 2026/4/5.
//

#ifndef MY_INFERENCE_REDUCE_H
#define MY_INFERENCE_REDUCE_H
#include <cstdint>

#include "kernel/kernel_args/reduce_args.h"

namespace my_inference::cpu::generic::primitive {

    template<typename T, typename ReducePolicy>
    void reduce(T *input, T *output,
                const ReduceArgs args) {
        typename ReducePolicy::BinaryFunctor binary_functor;
        typename ReducePolicy::PostFunctor post_functor;
        constexpr T InitValue = ReducePolicy::InitValue;
        const int64_t Reduce = args.Reduce;
        const int64_t Inner = args.Inner;
        const int64_t Outer = args.Outer;
        const int64_t stride[2] = {Reduce * Inner, Inner};
        for (int64_t outer = 0; outer < Outer; ++outer) {
            for (int64_t inner = 0; inner < Inner; ++inner) {
                T ret = InitValue;
                for (int reduce = 0; reduce < Reduce; ++reduce) {
                    ret = binary_functor(ret, input[outer * stride[0] + reduce * stride[1] + inner]);
                }
                output[outer * Inner + inner] = post_functor(ret, Reduce);
            }
        }
    }
}
#endif //MY_INFERENCE_REDUCE_H
