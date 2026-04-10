//
// Created by hzw on 2026/4/4.
//

#ifndef MY_INFERENCE_GROUPED_CONV2D_H
#define MY_INFERENCE_GROUPED_CONV2D_H
#include <cstdint>

namespace my_inference::cpu::generic::primitive {
    template<typename T>
    void grouped_conv2D(
        T *input, T *kernel, T *bias, T *output,
        const int64_t N, const int64_t C_IN, const int64_t H_IN, const int64_t W_IN,
        const int64_t C_OUT, const int64_t H_OUT, const int64_t W_OUT,
        const int64_t K_H, const int64_t K_W,
        const int64_t PAD_UP, int64_t PAD_DOWN, const int64_t PAD_LEFT, int64_t PAD_RIGHT,
        const int64_t STRIDE_H, const int64_t STRIDE_W,
        const int64_t DILATION_H, const int64_t DILATION_W,
        const int64_t GROUP) {
        // group
        const int64_t C_IN_PER_GROUP = C_IN / GROUP;
        const int64_t C_OUT_PER_GROUP = C_OUT / GROUP;
        // tensor stride
        const int64_t i_s[4] = {C_IN * H_IN * W_IN, H_IN * W_IN, W_IN, 1};
        const int64_t k_s[4] = {C_IN_PER_GROUP * K_H * K_W, K_H * K_W, K_W, 1};
        const int64_t o_s[4] = {C_OUT * H_OUT * W_OUT, H_OUT * W_OUT, W_OUT, 1};
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c_out = 0; c_out < C_OUT; ++c_out) {
                const int64_t group_idx = c_out / C_OUT_PER_GROUP;
                const int64_t c_in_begin = group_idx * C_IN_PER_GROUP;
                for (int64_t h_out = 0; h_out < H_OUT; ++h_out) {
                    for (int64_t w_out = 0; w_out < W_OUT; ++w_out) {
                        T ret = bias != nullptr ? bias[c_out] : 0;
                        for (int64_t c_k = 0; c_k < C_IN_PER_GROUP; ++c_k) {
                            const int64_t c_in = c_in_begin + c_k;
                            for (int64_t h_k = 0; h_k < K_H; ++h_k) {
                                const int64_t h_in = h_out * STRIDE_H + h_k * DILATION_H - PAD_UP;
                                if (h_in < 0 || h_in >= H_IN)continue;
                                for (int64_t w_k = 0; w_k < K_W; ++w_k) {
                                    const int64_t w_in = w_out * STRIDE_W + w_k * DILATION_W - PAD_LEFT;
                                    if (w_in < 0 || w_in >= W_IN)continue;
                                    ret += input[n * i_s[0] + c_in * i_s[1] + h_in * i_s[2] + w_in] *
                                            kernel[c_out * k_s[0] + c_k * k_s[1] + h_k * k_s[2] + w_k];
                                }
                            }
                        }
                        output[n * o_s[0] + c_out * o_s[1] + h_out * o_s[2] + w_out] = ret;
                    }
                }
            }
        }
    }
}
#endif //MY_INFERENCE_GROUPED_CONV2D_H
