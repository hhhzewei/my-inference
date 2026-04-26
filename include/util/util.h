//
// Created by hzw on 2026/2/22.
//

#ifndef MY_INFERENCE_UTIL_H
#define MY_INFERENCE_UTIL_H
#include <cassert>
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

    inline int64_t alignUp(const int64_t num, const int64_t alignment) {
        assert(alignment > 0 && (alignment & (alignment - 1)) == 0);
        return (num + alignment - 1) & (~(alignment - 1));
    }

    std::vector<void *> batchMalloc(const std::vector<size_t> &sizes);

    void batchFree(const std::vector<void *> &ptrs);
}

#endif //MY_INFERENCE_UTIL_H
