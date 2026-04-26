//
// Created by hzw on 2026/4/2.
//

#pragma once
#include <cstdint>

namespace my_inference::cpu::generic::primitive {
    template<typename T>
    void standard_conv2D(
        const T *input, const T *kernel, const T *bias,
        T *output,
        const Conv2DArgs args) {
        const int64_t C_IN = args.C_IN;
        const int64_t H_IN = args.H_IN;
        const int64_t W_IN = args.W_IN;
        const int64_t C_OUT = args.C_OUT;
        const int64_t H_OUT = args.H_OUT;
        const int64_t W_OUT = args.W_OUT;
        const int64_t K_H = args.H_K;
        const int64_t K_W = args.W_K;
        const int64_t i_stride[4] = {C_IN * H_IN * W_IN, H_IN * W_IN, W_IN, 1};
        const int64_t k_stride[4] = {C_IN * K_H * K_W, K_H * K_W, K_W, 1};
        const int64_t o_stride[4] = {C_OUT * H_OUT * W_OUT, H_OUT * W_OUT, W_OUT, 1};
        for (int64_t n = 0; n < args.N; ++n) {
            for (int64_t c_out = 0; c_out < C_OUT; ++c_out) {
                for (int64_t h_out = 0; h_out < H_OUT; ++h_out) {
                    for (int64_t w_out = 0; w_out < W_OUT; ++w_out) {
                        T ret = bias != nullptr ? bias[c_out] : 0;
                        for (int64_t c_in = 0; c_in < C_IN; ++c_in) {
                            for (int64_t h_k = 0; h_k < K_H; ++h_k) {
                                const int64_t h_in = h_out * args.STRIDE_H + h_k * args.DILATION_H - args.PAD_UP;
                                if (h_in < 0 || h_in >= H_IN)continue;
                                for (int64_t w_k = 0; w_k < K_W; ++w_k) {
                                    const int64_t w_in = w_out * args.STRIDE_W + w_k * args.DILATION_W - args.PAD_LEFT;
                                    if (w_in < 0 || w_in >= W_IN)continue;
                                    ret += input[
                                                n * i_stride[0] + c_in * i_stride[1] + h_in * i_stride[2] + w_in *
                                                i_stride[3]] *
                                            kernel[c_out * k_stride[0] + c_in * k_stride[1] + h_k * k_stride[2] + w_k *
                                                   k_stride[3]];
                                }
                            }
                        }
                        output[n * o_stride[0] + c_out * o_stride[1] + h_out * o_stride[2] + w_out * o_stride[3]] = ret;
                    }
                }
            }
        }
    }
}
