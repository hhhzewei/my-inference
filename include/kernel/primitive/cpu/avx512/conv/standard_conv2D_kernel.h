//
// Created by hzw on 2026/4/24.
//
#pragma once

#include "kernel/kernel_args/conv2D_args.h"
#include "backend/isa_traits/avx512_traits.h"

namespace my_inference::cpu::avx512::primitive {
    template<typename T>
    void standard_conv2D(T *x, T *weight, T *bias,
                         T *y,
                         const Conv2DArgs args) {
        using traits = Traits<T>;
        constexpr int64_t NUM_PER_VEC = traits::NumPerVec;
        for (int64_t n = 0; n < args.N; ++n) {
            const int64_t n_in_offset = n * args.H_IN * args.W_IN * args.C_IN;
            const int64_t n_out_offset = n * args.H_OUT * args.W_OUT * args.C_OUT;
            for (int64_t h_out = 0; h_out < args.H_OUT; ++h_out) {
                int64_t h_out_offset = h_out * args.W_OUT * args.C_OUT;
                for (int64_t w_out = 0; w_out < args.W_OUT; ++w_out) {
                    int64_t w_out_offset = w_out * args.C_OUT;
                    for (int64_t c_out_offset = 0; c_out_offset < args.C_OUT; c_out_offset += NUM_PER_VEC) {
                        auto y_vec = traits::setzero();
                        for (int64_t h_k = 0; h_k < args.H_K; ++h_k) {
                            const int64_t h_k_offset = h_k * args.W_K * args.C_IN * args.C_OUT;
                            const int64_t h_in = h_out * args.STRIDE_H + h_k * args.DILATION_H - args.PAD_UP;
                            if (h_in < 0 || h_in >= args.H_IN) {
                                continue;
                            }
                            const int64_t h_in_offset = h_in * args.W_IN * args.C_IN;
                            for (int64_t w_k = 0; w_k < args.W_K; ++w_k) {
                                const int64_t w_k_offset = w_k * args.C_IN * args.C_OUT;
                                const int64_t w_in = w_out * args.STRIDE_W + w_k * args.DILATION_W - args.PAD_LEFT;
                                if (w_in < 0 || w_in >= args.W_IN) {
                                    continue;
                                }
                                const int64_t w_in_offset = w_in * args.C_IN;
                                for (int64_t c_in = 0; c_in < args.C_IN; ++c_in) {
                                    const int64_t c_in_k_offset = c_in * args.C_OUT;
                                    auto x_vec = traits::set1(x[n_in_offset + h_in_offset + w_in_offset + c_in]);
                                    auto weight_vec = traits::load(
                                        weight + h_k_offset + w_k_offset + c_in_k_offset + c_out_offset);
                                    y_vec = traits::fmadd(x_vec, weight_vec, y_vec);
                                }
                            }
                        }
                        if (bias) {
                            auto bias_vec = traits::load(bias + c_out_offset);
                            y_vec = traits::add(y_vec, bias_vec);
                        }
                        traits::store(y + n_out_offset + h_out_offset + w_out_offset + c_out_offset, y_vec);
                    }
                }
            }
        }
    }
}
