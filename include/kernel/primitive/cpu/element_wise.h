//
// Created by hzw on 2026/2/28.
//
#pragma once
#include <cstdint>
#include <vector>

namespace my_inference::cpu::primitive {
    template<typename T, typename Func>
    void binaryElementWise(
        // 输入数据指针
        const T *a, const T *b,
        // 输出数据指针 (Output)
        T *c,
        const int64_t N) {
        Func func;
        for (int64_t i = 0; i < N; ++i) {
            c[i] = func(a[i], b[i]);
        }
    }

    template<typename T, typename Func>
    void binaryElementWiseWithStrides1D(
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
    void binaryElementWiseWithStrides2D(
        const T *a, const int64_t *a_strides,
        const T *b, const int64_t *b_strides,
        T *c, const int64_t M, const int64_t N) {
        Func func;
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                c[i * N + j] = func(a[i * a_strides[0] + j * a_strides[1]],
                                    b[i * b_strides[0] + j * b_strides[1]]);
            }
        }
    }

    template<typename T, typename Func>
    void binaryElementWiseWithStridesND(
        const T *a, const int64_t *a_strides,
        const T *b, const int64_t *b_strides,
        T *c, const int64_t *c_strides,
        const int64_t *shape,
        const int64_t num_data, const int64_t num_dim) {
        Func func;
        std::vector<int64_t> coords(num_dim, 0);
        for (int64_t i = 0; i < num_data; ++i) {
            // update coords
            for (int dim_i = num_dim - 1; dim_i >= 0; --dim_i) {
                ++coords[dim_i];
                if (coords[dim_i] < shape[dim_i]) {
                    break;
                }
                coords[dim_i] = 0;
            }
            // offset
            int64_t a_offset = 0, b_offset = 0, c_offset = 0;
            for (int dim_i = num_dim - 1; dim_i >= 0; --dim_i) {
                a_offset += coords[dim_i] * a_strides[i];
                b_offset += coords[dim_i] * b_strides[i];
                c_offset += coords[dim_i] * c_strides[i];
            }
            c[c_offset] = func(a[a_offset], b[b_offset]);
        }
    }
}
