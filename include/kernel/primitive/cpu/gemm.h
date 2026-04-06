//
// Created by hzw on 2026/4/5.
//

#ifndef MY_INFERENCE_GEMM_H
#define MY_INFERENCE_GEMM_H
#include <cstdint>

namespace my_inference::cpu::primitive {
    template<typename T, bool transA, bool transB>
    void gemm(const T *a, const T *b,
              const T *bias, const int64_t bias_stride_0, int64_t bias_stride_1,
              T *y,
              const int64_t M, const int64_t K, const int64_t N,
              const float alpha, const float beta) {
        int64_t strideA[2] = {K, 1};
        if (transA) {
            strideA[0] = 1;
            strideA[1] = M;
        }
        int64_t strideB[2] = {N, 1};
        if (transB) {
            strideB[0] = 1;
            strideB[1] = K;
        }
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < N; ++k) {
                    y[i * N + k] = alpha * a[i * strideA[0] + k * strideA[1]] * b[k * strideB[0] + j * strideB[1]] +
                                   beta * bias[i * bias_stride_0 + j * bias_stride_1];
                }
            }
        }
    }
}
#endif //MY_INFERENCE_GEMM_H
