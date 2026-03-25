//
// Created by hzw on 2026/2/22.
//

#ifndef MY_INFERENCE_UTIL_H
#define MY_INFERENCE_UTIL_H
#include <vector>
#include "graph/node/tensor_dim.h"

namespace my_inference {
    template<typename T>
    void swapAndPop(std::vector<T> &vec, const T &target) {
        for (int i = 0; i < vec.size();) {
            if (vec[i] == target) {
                vec[i] = vec.back();
                vec.pop_back();
            } else {
                ++i;
            }
        }
    }

    template<typename T, typename Matcher>
    void swapAndPop(std::vector<T> &vec, const Matcher &matcher) {
        for (int i = 0; i < vec.size();) {
            if (matcher(vec[i])) {
                vec[i] = vec.back();
                vec.pop_back();
            } else {
                ++i;
            }
        }
    }

    std::vector<int64_t> toValue(const std::vector<TensorDim> &vec);
}

#endif //MY_INFERENCE_UTIL_H
