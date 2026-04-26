//
// Created by hzw on 2026/4/13.
//
#pragma once
#include <cstdint>
#include <vector>

#include "kernel/kernel_args/transpose_args.h"

namespace my_inference::cpu::generic::primitive {
    template<typename T>
    void transpose(const T *x, T *y,
                   const TransposeArgs args) {
        const int64_t num_dim = args.num_dim;
        const int64_t *x_strides = args.x_strides;
        int64_t coord[8]{};
        int64_t x_offset = 0;
        for (int64_t y_offset = 0; y_offset < args.num_elem; ++y_offset) {
            y[y_offset] = x[x_offset];
            for (int64_t y_dim = num_dim - 1; y_dim >= 0; --y_dim) {
                const int64_t x_stride = x_strides[args.perm[y_dim]];
                ++coord[y_dim];
                x_offset += x_stride;
                if (coord[y_dim] != args.y_shape[y_dim]) {
                    break;
                }
                x_offset -= coord[y_dim] * x_stride;
                coord[y_dim] = 0;
            }
        }
    }
}
