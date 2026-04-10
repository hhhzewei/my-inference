//
// Created by hzw on 2026/4/4.
//

#ifndef MY_INFERENCE_CLIP_H
#define MY_INFERENCE_CLIP_H
#include <cstdint>

namespace my_inference::cpu::generic::primitive {
    template<typename T>
    void clip(T *input, T *min, T *max,
              T *output,
              const int64_t N) {
        T min_value = min[0], max_value = max[0];
        for (int64_t i = 0; i < N; ++i) {
            T input_value = input[i];
            output[i] = input_value < min_value ? min_value : input_value > max_value ? max_value : input_value;
        }
    }
}
#endif //MY_INFERENCE_CLIP_H
