//
// Created by hzw on 2026/2/28.
//
#pragma once
#include <cstdint>

namespace my_inference::kernel::primitive::cpu {
    template<typename T, typename Func>
    void elementWiseWithStrides1D(
        // 输入数据指针
        const T *a, const int64_t *a_strides,
        const T *b, const int64_t *b_strides,
        // 输出数据指针 (Output)
        T *c,
        const int64_t N) {
        Func func;
        for (int64_t i = 0; i < N; ++i) {
            c[i] = func(a[i * a_strides[0]], b[i * b_strides[0]]);
        }
    }

    template<typename T, typename Func>
    void elementWiseWithStrides2D(
        const T *a, const int64_t *a_strides,
        const T *b, const int64_t *b_strides,
        T *c, const int64_t *c_shape, const int64_t *c_strides) {
        Func func;
        for (int64_t i = 0; i < c_shape[0]; ++i) {
            for (int64_t j = 0; j < c_shape[1]; ++j) {
                c[i * c_strides[0] + j * c_strides[1]] = func(a[i * a_strides[0] + j * a_strides[1]],
                                                              b[i * b_strides[0] + j * b_strides[1]]);
            }
        }
    }
}
