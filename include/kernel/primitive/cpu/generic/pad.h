//
// Created by hzw on 2026/4/23.
//

#ifndef MY_INFERENCE_PAD_H
#define MY_INFERENCE_PAD_H

#include "kernel/kernel_args/pad_args.h"

namespace my_inference::cpu::generic::primitive {
    template<typename T>
    void pad(const T *x, const int64_t *pads,
             T *y,
             const PadArgs args) {
        const int64_t num_dim = args.num_dim;
        int64_t coord[8]{};
        const int64_t *x_stride = args.x_stride;
        int64_t x_offset = 0;
        for (int64_t i = 0; i < num_dim; ++i) {
            x_offset -= pads[i] * x_stride[i];
        }
        for (int64_t y_offset = 0; y_offset < args.y_num_elem; ++y_offset) {
            bool is_pad = false;
            for (int64_t i = 0; i < num_dim; ++i) {
                is_pad |= coord[i] < pads[i] || coord[i] >= pads[i] + args.x_shape[i];
                if (is_pad) {
                    break;
                }
            }
            y[y_offset] = is_pad ? 0 : x[x_offset];
            for (int64_t i = num_dim - 1; i >= 0; --i) {
                ++coord[i];
                x_offset += x_stride[i];
                if (coord[i] != args.y_shape[i]) {
                    break;
                }
                x_offset -= coord[i] * x_stride[i];
                coord[i] = 0;
            }
        }
    }
}
#endif //MY_INFERENCE_PAD_H
