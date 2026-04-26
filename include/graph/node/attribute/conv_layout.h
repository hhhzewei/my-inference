//
// Created by hzw on 2026/4/11.
//

#ifndef MY_INFERENCE_CONV_LAYOUT_H
#define MY_INFERENCE_CONV_LAYOUT_H
#include <cstdint>

namespace my_inference::ConvLayout {
    constexpr inline int64_t NCHW = 0;
    constexpr inline int64_t NHWC = 1;
}
#endif //MY_INFERENCE_CONV_LAYOUT_H
