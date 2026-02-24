//
// Created by hzw on 2026/2/20.
//

#ifndef MY_INFERENCE_TENSOR_TYPE_H
#define MY_INFERENCE_TENSOR_TYPE_H
#include <cstdint>

namespace my_inference{
    enum class TensorType:uint8_t {
        INPUT = 1 << 0,
        WEIGHT = 1 << 1,
        INTERNAL = 1 << 2,
        OUTPUT = 1 << 3,
        ALL = 0xff,
    };

    template<TensorType target>
    constexpr bool is(TensorType o) {
        return static_cast<uint8_t>(o) & static_cast<uint8_t>(target);
    }

    constexpr TensorType operator |(TensorType o1, TensorType o2) {
        return static_cast<TensorType>(static_cast<uint8_t>(o1) | static_cast<uint8_t>(o2));
    }
}

#endif //MY_INFERENCE_TENSOR_TYPE_H
